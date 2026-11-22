import os
import json
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from models.fuzzy_ensemble import MultiModelNormalization
from dataset_utils import get_sample_info
from visualization_utils import generate_visualizations


@torch.no_grad()
def evaluate_single_model_exact(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    name: str,
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
) -> float:
    """Evaluate a single base model accuracy on a loader."""
    model.eval()
    normalizer = MultiModelNormalization([mean], [std]).to(device)
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc=f"Evaluating {name}"):
        images, labels = images.to(device), labels.to(device).float()
        images = normalizer(images, 0)
        out = model(images)
        if isinstance(out, (tuple, list)):
            out = out[0]
        pred = (out.squeeze(1) > 0).long()
        correct += pred.eq(labels.long()).sum().item()
        total += labels.size(0)

    acc = 100.0 * correct / total
    print(f" {name}: {acc:.2f}%")
    return acc


@torch.no_grad()
def evaluate_accuracy_and_loss(model, loader, device, criterion):
    """Compute accuracy, loss and routing statistics on a loader."""
    model.eval()
    correct = 0
    total_loss = 0.0
    total_samples = len(loader.dataset)
    num_models = model.num_models

    sum_per_model_hesitancy = torch.zeros(num_models, device=device)
    sum_active_models = torch.zeros(num_models, device=device)
    sum_cumsum_used = 0
    active_counts_all = []

    for images, labels in tqdm(loader, desc="Validation"):
        images, labels = images.to(device), labels.to(device).float()
        outputs, weights, memberships, _ = model(images, return_details=True)

        loss = criterion(outputs.squeeze(1), labels)
        total_loss += loss.item() * images.size(0)
        pred = (outputs.squeeze(1) > 0).long()
        correct += pred.eq(labels.long()).sum().item()

        per_model_hes = memberships.var(dim=2, unbiased=False)
        sum_per_model_hesitancy += per_model_hes.sum(dim=0)

        active_mask = (weights > 1e-4).float()
        num_active_per_sample = active_mask.sum(dim=1)
        active_counts_all.append(num_active_per_sample.cpu())
        sum_cumsum_used += (num_active_per_sample < num_models).sum().item()
        sum_active_models += active_mask.sum(dim=0)

    acc = 100.0 * correct / total_samples
    avg_loss = total_loss / total_samples

    per_model_hes_avg = (sum_per_model_hesitancy / total_samples).cpu().tolist()
    overall_hes_avg = (sum_per_model_hesitancy / total_samples).mean().item()
    cumsum_used_pct = 100.0 * sum_cumsum_used / total_samples
    per_model_active_pct = (100.0 * sum_active_models / total_samples).cpu().tolist()

    ac = torch.cat(active_counts_all).long()
    mean_active = ac.float().mean().item()

    dist_counts = {}
    dist_pcts = {}
    for k in range(1, num_models + 1):
        if k < num_models:
            cnt = (ac == k).sum().item()
        else:
            cnt = (ac >= k).sum().item()
        dist_counts[k] = cnt
        dist_pcts[k] = 100.0 * cnt / total_samples

    pct_1_model = dist_pcts.get(1, 0.0)

    stats = {
        "per_model_hesitancy": per_model_hes_avg,
        "overall_hesitancy": overall_hes_avg,
        "cumsum_used_pct": cumsum_used_pct,
        "per_model_active_pct": per_model_active_pct,
        "mean_active": mean_active,
        "dist_counts": dist_counts,
        "dist_pcts": dist_pcts,
        "pct_1_model": pct_1_model,
    }
    return acc, avg_loss, stats


def final_evaluation_unified(model, test_loader, device, save_dir, model_names, args):
    """Full sample-by-sample evaluation + metrics + visualizations."""
    model.eval()

    test_dataset = test_loader.dataset
    num_samples = len(test_dataset)

    lines = []
    lines.append("=" * 100)
    lines.append("SAMPLE-BY-SAMPLE PREDICTIONS (For McNemar Test Comparison):")
    lines.append("=" * 100)
    header = f"{'ID':<10} {'Filename':<60} {'True':<10} {'Pred':<10} {'Correct':<10}"
    lines.append(header)
    lines.append("-" * 100)

    TP, TN, FP, FN = 0, 0, 0, 0

    all_y_true = []
    all_y_score = []
    all_y_pred = []
    all_model_weights = []

    print(f"\nRunning Unified Final Evaluation on {num_samples} samples...")

    with torch.no_grad():
        for i in tqdm(range(num_samples), desc="Final Eval"):
            try:
                image, label = test_dataset[i]
                original_idx = test_dataset.indices[i]
                path, _ = get_sample_info(test_dataset.dataset, original_idx)
            except Exception as e:
                print(f"Error loading sample {i}: {e}")
                continue

            image = image.unsqueeze(0).to(device)
            label_int = int(label)

            output, final_weights, _, _ = model(image, return_details=True)

            prob = torch.sigmoid(output.squeeze()).item()

            all_model_weights.append(final_weights.squeeze(0).cpu().numpy())

            pred_int = int(prob > 0.5)

            all_y_true.append(label_int)
            all_y_score.append(prob)
            all_y_pred.append(pred_int)

            if label_int == 1:  # Real
                if pred_int == 1:
                    TP += 1
                else:
                    FN += 1
            else:  # Fake
                if pred_int == 1:
                    FP += 1
                else:
                    TN += 1

            correct_str = "Yes" if pred_int == label_int else "No"
            filename = os.path.basename(path)
            if len(filename) > 55:
                filename = filename[:25] + "..." + filename[-27:]
            line = f"{i+1:<10} {filename:<60} {label_int:<10} {pred_int:<10} {correct_str:<10}"
            lines.append(line)

    total_samples = len(all_y_true)
    correct_count = TP + TN
    acc = 100.0 * correct_count / total_samples if total_samples > 0 else 0.0

    precision_real = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall_real = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_real = (
        2 * precision_real * recall_real / (precision_real + recall_real)
        if (precision_real + recall_real) > 0
        else 0
    )

    # Fake class (Negative class = 0)
    precision_fake = TN / (TN + FN) if (TN + FN) > 0 else 0
    recall_fake = TN / (TN + FP) if (TN + FP) > 0 else 0
    f1_fake = (
        2 * precision_fake * recall_fake / (precision_fake + recall_fake)
        if (precision_fake + recall_fake) > 0
        else 0
    )

    # Macro averages
    macro_precision = (precision_real + precision_fake) / 2
    macro_recall = (recall_real + recall_fake) / 2
    macro_f1 = (f1_real + f1_fake) / 2

    # Balanced Accuracy
    balanced_acc = ((recall_real + recall_fake) / 2) * 100

    # ================== Confusion Matrix Plot ==================
    cm = np.array([[TN, FP], [FN, TP]])

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Predicted Fake', 'Predicted Real'],
        yticklabels=['Actual Fake', 'Actual Real'],
        cbar=True,
        annot_kws={"size": 16},
    )
    plt.title(
        f'Confusion Matrix\nAccuracy: {acc:.2f}% | Balanced Acc: {balanced_acc:.2f}% | Macro-F1: {macro_f1:.4f}',
        fontsize=13,
        fontweight='bold',
    )
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()

    cm_path = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion Matrix saved to: {cm_path}")

    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")
    print(f"Accuracy          : {acc:.2f}%")
    print(f"Balanced Accuracy : {balanced_acc:.2f}%")
    print(f"Macro-F1          : {macro_f1:.4f}")
    print()
    print(f"{'Class':<12} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
    print("-" * 45)
    print(f"{'Fake (0)':<12} {precision_fake:10.4f} {recall_fake:10.4f} {f1_fake:10.4f}")
    print(f"{'Real (1)':<12} {precision_real:10.4f} {recall_real:10.4f} {f1_real:10.4f}")
    print("-" * 45)
    print(f"{'Macro Avg':<12} {macro_precision:10.4f} {macro_recall:10.4f} {macro_f1:10.4f}")
    print(f"\nCorrect Predictions  : {correct_count} ({acc:.2f}%)")
    print(f"Incorrect Predictions: {total_samples - correct_count} ({100 - acc:.2f}%)")
    print("=" * 70)

    if len(all_model_weights) > 0:
        weights_np = np.array(all_model_weights)

        print(f"\n{'='*70}")
        print("MODEL ACTIVATION STATISTICS (Final Evaluation)")
        print(f"{'='*70}")

        activation_threshold = 1e-4
        active_mask = (weights_np > activation_threshold).astype(float)

        activation_rates = active_mask.mean(axis=0) * 100
        avg_weights = weights_np.mean(axis=0)
        max_weights = weights_np.max(axis=0)

        print(f"\n  {'Model':<35} {'Active %':>10} {'Avg Weight':>12} {'Max Weight':>12}")
        print(f"  {'-'*70}")
        for i, name in enumerate(model_names):
            act_pct = activation_rates[i]
            avg_w = avg_weights[i]
            max_w = max_weights[i]
            bar_length = int(act_pct / 2)
            bar = '█' * bar_length + '░' * (50 - bar_length)
            print(f"  {i+1:2d}. {name:<30}: {act_pct:6.2f}%  | {avg_w:.6f}  | {max_w:.6f}")
            print(f"      [{bar}]")

        print(f"  {'-'*70}")
        print(f"  {'Overall Average':<35} {activation_rates.mean():6.2f}%")

        active_per_sample = active_mask.sum(axis=1)
        print(f"\n  Active Models Per Sample:")
        print(f"    Min:    {int(active_per_sample.min())}")
        print(f"    Max:    {int(active_per_sample.max())}")
        print(f"    Mean:   {active_per_sample.mean():.2f}")
        print(f"    Median: {np.median(active_per_sample):.1f}")

        unique_counts, count_freq = np.unique(
            active_per_sample.astype(int), return_counts=True
        )
        print(f"\n  Distribution of Active Model Count:")
        for count, freq in zip(unique_counts, count_freq):
            pct = 100.0 * freq / len(all_model_weights)
            bar = '█' * int(pct / 2)
            print(f"    {count} model(s): {freq:5d} samples ({pct:5.2f}%) {bar}")

        print(f"{'='*70}\n")

        activation_stats = {
            "activation_rates_pct": {
                name: float(activation_rates[i]) for i, name in enumerate(model_names)
            },
            "average_weights": {
                name: float(avg_weights[i]) for i, name in enumerate(model_names)
            },
            "max_weights": {
                name: float(max_weights[i]) for i, name in enumerate(model_names)
            },
            "active_per_sample_stats": {
                "min": int(active_per_sample.min()),
                "max": int(active_per_sample.max()),
                "mean": float(active_per_sample.mean()),
                "median": float(np.median(active_per_sample)),
            },
            "per_sample_weights": weights_np.tolist(),
        }
        activation_json_path = os.path.join(save_dir, "model_activation_stats.json")
        with open(activation_json_path, 'w', encoding='utf-8') as f:
            json.dump(activation_stats, f, indent=2, ensure_ascii=False)
        print(f"Model activation stats saved: {activation_json_path}")

    output_str = []

    output_str.append("-" * 100)
    output_str.append("SUMMARY STATISTICS:")
    output_str.append("-" * 100)
    output_str.append(f"Overall Accuracy: {acc:.2f}%")
    output_str.append(f"Balanced Accuracy: {balanced_acc:.2f}%")
    output_str.append(f"Macro F1-Score: {macro_f1:.4f}")

    output_str.append("\n--- Real (1) Class Metrics ---")
    output_str.append(f"Precision (Real): {precision_real:.4f}")
    output_str.append(f"Recall (Real):    {recall_real:.4f}")
    output_str.append(f"F1-Score (Real):  {f1_real:.4f}")

    output_str.append("\n--- Fake (0) Class Metrics ---")
    output_str.append(f"Precision (Fake): {precision_fake:.4f}")
    output_str.append(f"Recall (Fake):    {recall_fake:.4f}")
    output_str.append(f"F1-Score (Fake):  {f1_fake:.4f}")

    output_str.append(f"\nCorrect Predictions: {correct_count} ({acc:.2f}%)")
    output_str.append(
        f"Incorrect Predictions: {total_samples - correct_count} ({100-acc:.2f}%)"
    )
    output_str.append(f"\nConfusion Matrix saved to: confusion_matrix.png")
    output_str.extend(lines)

    log_path = os.path.join(save_dir, 'prediction_log.txt')
    with open(log_path, 'w') as f:
        f.write("\n".join(output_str))
    print(f"Prediction log saved to: {log_path}")

    print("\nCollecting ROC data (y_true & y_score) ...")

    y_true_np = np.array(all_y_true)
    y_score_np = np.array(all_y_score)
    y_pred_np = np.array(all_y_pred)

    roc_json_path = os.path.join(save_dir, "roc_data_test.json")
    roc_data_json = {
        "metadata": {
            "seed": args.seed,
            "dataset": args.dataset,
            "num_samples": int(total_samples),
            "positive_count": int(np.sum(y_true_np)),
            "negative_count": int(total_samples - np.sum(y_true_np)),
            "model": "fuzzy_hesitant_ensemble",
        },
        "y_true": y_true_np.tolist(),
        "y_score": y_score_np.tolist(),
        "y_pred": y_pred_np.tolist(),
    }
    with open(roc_json_path, 'w', encoding='utf-8') as f:
        json.dump(roc_data_json, f, indent=2, ensure_ascii=False)
    print(f"ROC data saved (JSON): {roc_json_path}")

    roc_txt_path = os.path.join(save_dir, "roc_data_test.txt")
    with open(roc_txt_path, 'w', encoding='utf-8') as f:
        f.write("y_true\ty_score\ty_pred\n")
        for t, s, p in zip(y_true_np, y_score_np, y_pred_np):
            f.write(f"{int(t)}\t{s:.6f}\t{int(p)}\n")
    print(f"ROC data saved (TXT):  {roc_txt_path}")

    vis_dir = os.path.join(save_dir, 'visualizations')
    generate_visualizations(
        model,
        test_loader,
        device,
        vis_dir,
        model_names,
        args.num_grad_cam_samples,
        args.num_lime_samples,
        args.dataset,
        is_main=True,
    )

    return acc, y_true_np, y_score_np
