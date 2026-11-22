from typing import List, Tuple
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiModelNormalization(nn.Module):

    def __init__(self, means: List[Tuple[float]], stds: List[Tuple[float]]):
        super().__init__()
        for i, (m, s) in enumerate(zip(means, stds)):
            self.register_buffer(f'mean_{i}', torch.tensor(m).view(1, 3, 1, 1))
            self.register_buffer(f'std_{i}', torch.tensor(s).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        return (x - getattr(self, f'mean_{idx}')) / getattr(self, f'std_{idx}')


class HesitantFuzzyMembership(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_models: int,
        num_memberships: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_models = num_models
        self.num_memberships = num_memberships
        self.feature_net = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.membership_generator = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(128, num_models * num_memberships),
        )
        self.aggregation_weights = nn.Parameter(
            torch.ones(num_memberships) / num_memberships
        )

    def forward(self, x: torch.Tensor):
        features = self.feature_net(x).flatten(1)
        memberships = self.membership_generator(features)
        memberships = memberships.view(-1, self.num_models, self.num_memberships)
        memberships = torch.sigmoid(memberships)
        agg_weights = F.softmax(self.aggregation_weights, dim=0)
        final_weights = (memberships * agg_weights.view(1, 1, -1)).sum(dim=2)
        final_weights = F.softmax(final_weights, dim=1)
        return final_weights, memberships


class FuzzyHesitantEnsemble(nn.Module):


    def __init__(
        self,
        models: List[nn.Module],
        means: List[Tuple[float]],
        stds: List[Tuple[float]],
        num_memberships: int = 3,
        freeze_models: bool = True,
        cum_weight_threshold: float = 0.9,
        hesitancy_threshold: float = 0.2,
    ):
        super().__init__()
        self.num_models = len(models)
        self.models = nn.ModuleList(models)
        self.normalizations = MultiModelNormalization(means, stds)
        self.hesitant_fuzzy = HesitantFuzzyMembership(
            input_dim=128,
            num_models=self.num_models,
            num_memberships=num_memberships,
        )
        self.cum_weight_threshold = cum_weight_threshold
        self.hesitancy_threshold = hesitancy_threshold

        if freeze_models:
            for model in self.models:
                model.eval()
                for p in model.parameters():
                    p.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        # Force base models to remain in eval mode
        for model in self.models:
            model.eval()
        return self

    def _compute_mask_vectorized(
        self, final_weights: torch.Tensor, avg_hesitancy: torch.Tensor
    ):
        sorted_weights, sorted_indices = torch.sort(
            final_weights, dim=1, descending=True
        )
        cum_weights = torch.cumsum(sorted_weights, dim=1)
        mask = (cum_weights <= self.cum_weight_threshold).float()
        mask[:, 0] = 1.0
        high_hesitancy_mask = (avg_hesitancy > self.hesitancy_threshold).unsqueeze(1)
        mask = torch.where(high_hesitancy_mask, torch.ones_like(mask), mask)
        final_mask = torch.zeros_like(final_weights)
        final_mask.scatter_(1, sorted_indices, mask)
        return final_mask

    def forward(self, x: torch.Tensor, return_details: bool = False):
        final_weights, all_memberships = self.hesitant_fuzzy(x)
        hesitancy = all_memberships.var(dim=2, unbiased=False)
        avg_hesitancy = hesitancy.mean(dim=1)
        mask = self._compute_mask_vectorized(final_weights, avg_hesitancy)
        final_weights = final_weights * mask
        final_weights = final_weights / (final_weights.sum(dim=1, keepdim=True) + 1e-8)

        outputs = torch.zeros(x.size(0), self.num_models, 1, device=x.device)

        for i in range(self.num_models):
            idx = final_weights[:, i] > 0
            if not idx.any():
                continue
            x_sub = x[idx]
            x_n = self.normalizations(x_sub, i)
            with torch.no_grad():
                out_i = self.models[i](x_n)
                if isinstance(out_i, (tuple, list)):
                    out_i = out_i[0]
            outputs[idx, i] = out_i

        final_output = (outputs * final_weights.unsqueeze(-1)).sum(dim=1)
        if return_details:
            return final_output, final_weights, all_memberships, outputs
        return final_output, final_weights


def load_pruned_models(
    model_paths: List[str], device: torch.device
) -> List[nn.Module]:
    """Load pruned ResNet-50 models from checkpoint paths."""
    try:
        # Prefer the local models package layout
        from models.ResNet_pruned import ResNet_50_pruned_hardfakevsreal
    except ImportError:
        try:
            from model.ResNet_pruned import ResNet_50_pruned_hardfakevsreal
        except ImportError:
            raise ImportError(
                "Cannot import ResNet_50_pruned_hardfakevsreal. "
                "Make sure models/ResNet_pruned.py (or model/) is on PYTHONPATH."
            )

    models = []
    print(f"Loading {len(model_paths)} pruned models...")
    for i, path in enumerate(model_paths):
        if not os.path.exists(path):
            print(f" [WARNING] File not found: {path}")
            continue
        print(f" [{i+1}/{len(model_paths)}] Loading: {os.path.basename(path)}")
        try:
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
            model = ResNet_50_pruned_hardfakevsreal(masks=ckpt['masks'])
            model.load_state_dict(ckpt['model_state_dict'])
            model = model.to(device).eval()
            models.append(model)
        except Exception as e:
            print(f" [ERROR] Failed to load {path}: {e}")
    if len(models) == 0:
        raise ValueError("No models loaded!")
    return models
