
import os
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from evaluation import evaluate_accuracy_and_loss
from efficiency_utils import Stopwatch, summarize_training_times


def train_hesitant_fuzzy(ensemble_model, train_loader, val_loader, num_epochs, lr,
                        device, save_dir, model_names):
    os.makedirs(save_dir, exist_ok=True)
    hesitant_net = ensemble_model.hesitant_fuzzy
    optimizer = torch.optim.AdamW(hesitant_net.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.BCEWithLogitsLoss()
    best_val_acc = 0.0

    total_train_samples = len(train_loader.dataset)

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }

    epoch_times, train_times, val_times = [], [], []
    total_sw = Stopwatch(device).start()

    print("="*70)
    print("Training Fuzzy Hesitant Network (Single GPU)")
    print("="*70)
    print(f"Trainable params: {sum(p.numel() for p in hesitant_net.parameters()):,}")
    print(f"Epochs: {num_epochs} | Initial LR: {lr}")
    print(f"Hesitant memberships per model: {hesitant_net.num_memberships}")
    print(f"Number of models: {len(model_names)}")
    print(f"Total training samples: {total_train_samples:,}")
    print(f"Validation samples: {len(val_loader.dataset):,}")
    print("(all routing statistics below are computed on VALIDATION)\n")

    for epoch in range(num_epochs):
        epoch_sw = Stopwatch(device).start()
        ensemble_model.train()

        train_loss = 0.0
        train_correct = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device).float()
            optimizer.zero_grad()

            outputs, _, _, _ = ensemble_model(images, return_details=True)
            loss = criterion(outputs.squeeze(1), labels)
            loss.backward()
            optimizer.step()

            batch_size = images.size(0)
            train_loss += loss.item() * batch_size
            pred = (outputs.squeeze(1) > 0).long()
            train_correct += pred.eq(labels.long()).sum().item()

        t_train = epoch_sw.stop()
        train_loss = train_loss / total_train_samples
        train_acc  = 100. * train_correct / total_train_samples

        val_sw = Stopwatch(device).start()
        val_acc, val_loss, vstats = evaluate_accuracy_and_loss(
            ensemble_model, val_loader, device, criterion)
        t_val = val_sw.stop()
        t_epoch = epoch_sw.stop()
        epoch_times.append(t_epoch); train_times.append(t_train); val_times.append(t_val)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"{'='*70}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | LR: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"Time: epoch {t_epoch:.1f}s (train {t_train:.1f}s | val {t_val:.1f}s)")

        print(f"\n{'Hesitancy (Variance) per Model:':^70}")
        print(f"{'-'*70}")
        for i, name in enumerate(model_names):
            print(f"  {i+1:2d}. {name:<30}: {vstats['per_model_hesitancy'][i]:.6f}")
        print(f"{'-'*70}")
        print(f"  {'Overall Mean Hesitancy:':<30}  {vstats['overall_hesitancy']:.6f}")

        print(f"\n{'Cumulative Weight Threshold (Cumsum) Analysis:':^70}")
        print(f"{'-'*70}")
        print(f"  {'Cumsum activated in:':<30}  {vstats['cumsum_used_pct']:.2f}% of samples")
        if vstats['cumsum_used_pct'] > 50:
            print(f"  {'Status:':<30}  ✓ Efficient")
        elif vstats['cumsum_used_pct'] > 20:
            print(f"  {'Status:':<30}  ≈ Moderate")
        else:
            print(f"  {'Status:':<30}  ✗ Low efficiency")
        print(f"  {'All models used in:':<30}  {100 - vstats['cumsum_used_pct']:.2f}% of samples")

        # ================== NEW: Active Models per Sample ==================
        print(f"\n{'Active Models per Sample (Validation):':^70}")
        print(f"{'-'*70}")
        print(f"  {'Mean active models:':<30}  {vstats['mean_active']:.2f}")
        print(f"  {'Distribution of active count:':<30}")
        for k in sorted(vstats['dist_counts'].keys()):
            cnt = vstats['dist_counts'][k]
            pct = vstats['dist_pcts'][k]
            bar = '█' * int(pct / 2)
            print(f"    {k} model(s): {cnt:6d} samples ({pct:6.2f}%) {bar}")
        print(f"  {'Only 1 active model:':<30}  {vstats['pct_1_model']:.2f}% of samples")

        print(f"\n{'Model Activation Frequency:':^70}")
        print(f"{'-'*70}")
        for i, name in enumerate(model_names):
            activation_pct = vstats['per_model_active_pct'][i]
            bar_length = int(activation_pct / 2)
            bar = '█' * bar_length
            print(f"  {i+1:2d}. {name:<30}: {activation_pct:5.1f}% {bar}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(save_dir, 'best_hesitant_fuzzy.pt')
            torch.save({
                'hesitant_state_dict': hesitant_net.state_dict(),
                'best_val_acc': val_acc,
                'best_val_loss': val_loss,
                'best_epoch': epoch + 1,
                'val_routing_stats': vstats,
                'T': ensemble_model.cum_weight_threshold,
                'H': ensemble_model.hesitancy_threshold,
            }, save_path)
            print(f"\n✓ Best model saved → {val_acc:.2f}%")

        print()

    total_time = total_sw.stop()
    timing = summarize_training_times(epoch_times, train_times, val_times, total_time)
    print(f"Total training time: {total_time/60:.2f} min "
          f"(mean epoch: {timing['mean_epoch_time_s']:.1f}s)")

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), history['train_loss'], label='train', color='blue', linewidth=2)
    plt.plot(range(1, num_epochs + 1), history['val_loss'], label='validation', color='orange', linewidth=2)
    plt.title('Loss', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), history['train_acc'], label='train', color='blue', linewidth=2)
    plt.plot(range(1, num_epochs + 1), history['val_acc'], label='validation', color='orange', linewidth=2)
    plt.title('Accuracy', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)

    plt.tight_layout()
    curves_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(curves_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training curves (Loss & Acc) saved to: {curves_path}")

    return best_val_acc, timing
