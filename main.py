
import argparse
import json
import os
import warnings

import torch

from utils.seed_device import set_seed, setup_device
from models.fuzzy_ensemble import FuzzyHesitantEnsemble, load_pruned_models
from evaluation import (
    evaluate_single_model_exact,
    final_evaluation_unified,
)
from training import train_hesitant_fuzzy
from dataset_utils import create_dataloaders
from efficiency_utils import Stopwatch, summarize_training_times, benchmark_efficiency, print_headline

warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser(description="Optimized Fuzzy Hesitant Ensemble Training")
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--dataset', type=str, required=True, choices=['wild','artiface', 'hard_fake_real', 'deepfake_lab', 'uadfV', 'custom_genai', 'custom_genai_v2', 'real_fake_dataset', '20k'])
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--model_paths', type=str, nargs='+', required=True)
    parser.add_argument('--model_names', type=str, nargs='+', required=True)
    parser.add_argument('--save_dir', type=str, default='./output')
    parser.add_argument('--seed', type=int, default=42)
    
    parser.add_argument('--num_memberships', type=int, default=3)
    parser.add_argument('--cum_weight_threshold', type=float, default=0.9)
    parser.add_argument('--hesitancy_threshold', type=float, default=0.2)
    
    parser.add_argument('--num_grad_cam_samples', type=int, default=5)
    parser.add_argument('--num_lime_samples', type=int, default=5)
    
    args = parser.parse_args()

    if len(args.model_names) != len(args.model_paths):
        raise ValueError("Number of model_names must match model_paths")

    set_seed(args.seed)
    
    device = setup_device()
    pipeline_sw = Stopwatch(device).start()

    print("="*70)
    print(f"OPTIMIZED FUZZY HESITANT ENSEMBLE TRAINING")
    print(f"Single GPU Mode | Seed: {args.seed}")
    print("="*70)
    print(f"Dataset: {args.dataset}")
    print(f"Data directory: {args.data_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Models: {len(args.model_paths)}")
    print("="*70 + "\n")

    DEFAULT_MEANS = [(0.5207, 0.4258, 0.3806), (0.4460, 0.3622, 0.3416), (0.4668, 0.3816, 0.3414)]
    DEFAULT_STDS = [(0.2490, 0.2239, 0.2212), (0.2057, 0.1849, 0.1761), (0.2410, 0.2161, 0.2081)]
    
    num_models = len(args.model_paths)
    if num_models > len(DEFAULT_MEANS):
        MEANS = DEFAULT_MEANS + [DEFAULT_MEANS[-1]] * (num_models - len(DEFAULT_MEANS))
        STDS = DEFAULT_STDS + [DEFAULT_STDS[-1]] * (num_models - len(DEFAULT_STDS))
    else:
        MEANS = DEFAULT_MEANS[:num_models]
        STDS = DEFAULT_STDS[:num_models]

    base_models = load_pruned_models(args.model_paths, device)
    MODEL_NAMES = args.model_names[:len(base_models)]

    ensemble = FuzzyHesitantEnsemble(
        base_models, MEANS, STDS,
        num_memberships=args.num_memberships, 
        freeze_models=True,
        cum_weight_threshold=args.cum_weight_threshold,
        hesitancy_threshold=args.hesitancy_threshold
    ).to(device)

    trainable = sum(p.numel() for p in ensemble.parameters() if p.requires_grad)
    total = sum(p.numel() for p in ensemble.parameters())
    print(f"Total params: {total:,} | Trainable: {trainable:,} | Frozen: {total-trainable:,}\n")

    train_loader, val_loader, test_loader = create_dataloaders(
        args.data_dir, args.batch_size, dataset_type=args.dataset,
        is_distributed=False, seed=args.seed, is_main=True)

    print("\n" + "="*70)
    print("INDIVIDUAL MODEL PERFORMANCE (Before Training)")
    print("="*70)

    individual_accs = []
    for i, model in enumerate(base_models):
        acc = evaluate_single_model_exact(model, test_loader, device, 
                                          f"Model {i+1} ({MODEL_NAMES[i]})", 
                                          MEANS[i], STDS[i])
        individual_accs.append(acc)

    best_single = max(individual_accs)
    best_idx = individual_accs.index(best_single)
    print(f"\nBest Single: Model {best_idx+1} ({MODEL_NAMES[best_idx]}) → {best_single:.2f}%")
    print("="*70)

    best_val_acc, train_timing = train_hesitant_fuzzy(
        ensemble, train_loader, val_loader,
        args.epochs, args.lr, device, args.save_dir, MODEL_NAMES
    )

    ckpt_path = os.path.join(args.save_dir, 'best_hesitant_fuzzy.pt')
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        ensemble.hesitant_fuzzy.load_state_dict(ckpt['hesitant_state_dict'])
        print("Best model loaded.\n")

        if 'val_routing_stats' in ckpt:
            vs = ckpt['val_routing_stats']
            print("="*70)
            print(f"BEST EPOCH VALIDATION STATS  (T={ckpt['T']}, H={ckpt['H']})")
            print(f"  Best epoch: {ckpt['best_epoch']} | Val Acc: {ckpt['best_val_acc']:.2f}% "
                  f"| Val Loss: {ckpt['best_val_loss']:.4f}")
            print(f"  Mean active models: {vs['mean_active']:.2f}")
            print(f"  Distribution of active count:")
            for k in sorted(vs['dist_counts'].keys()):
                print(f"    {k} model(s): {vs['dist_counts'][k]:6d} samples "
                      f"({vs['dist_pcts'][k]:6.2f}%)")
            print(f"  Only 1 active model: {vs['pct_1_model']:.2f}%")
            print("="*70)

    print("\n" + "="*70)
    print("FINAL ENSEMBLE EVALUATION")
    print("="*70)

    efficiency = benchmark_efficiency(ensemble, test_loader, device, MODEL_NAMES,
                                  latency_samples=min(300, len(test_loader.dataset)))
    efficiency['training_time'] = train_timing
    print_headline(efficiency)

    eval_sw = Stopwatch(device).start()
    final_acc, y_true, y_scores = final_evaluation_unified(
        ensemble, test_loader, device, args.save_dir, MODEL_NAMES, args
    )
    final_eval_time = eval_sw.stop()

    print("\n" + "="*70)
    print("FINAL COMPARISON")
    print("="*70)
    print(f"Best Single Model: {best_single:.2f}%")
    print(f"Ensemble Accuracy: {final_acc:.2f}%")
    print(f"Improvement: {final_acc - best_single:+.2f}%")
    print("="*70)

    final_results = {
        'seed': args.seed,
        'method': 'Fuzzy_Hesitant',
        'best_single_model': {'name': MODEL_NAMES[best_idx], 'accuracy': float(best_single)},
        'ensemble': {'test_accuracy': float(final_acc)},
        'improvement': float(final_acc - best_single)
    }
    with open(os.path.join(args.save_dir, 'final_results.json'), 'w') as f:
        json.dump(final_results, f, indent=4)

    torch.save({
        'ensemble_state_dict': ensemble.state_dict(),
        'model_names': MODEL_NAMES,
        'means': MEANS,
        'stds': STDS,
        'seed': args.seed
    }, os.path.join(args.save_dir, 'final_ensemble_model.pt'))

    efficiency['final_evaluation_time_s'] = final_eval_time
    efficiency['total_pipeline_time_s'] = pipeline_sw.stop()
    with open(os.path.join(args.save_dir, 'efficiency_report.json'), 'w') as f:
        json.dump(efficiency, f, indent=2)
    print(f"Efficiency report saved: {os.path.join(args.save_dir, 'efficiency_report.json')}")



if __name__ == "__main__":
    main()
