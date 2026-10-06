from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskAwareScoreHead(nn.Module):
    """Image-level score head that is sensitive to small forged regions."""

    def __init__(self, feature_channels: int = 256, hidden_dim: int = 128, topk_ratio: float = 0.02):
        super().__init__()
        self.topk_ratio = topk_ratio
        in_dim = 5 + 2 * feature_channels
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def _mask_stats(self, mask_logits: torch.Tensor) -> torch.Tensor:
        prob = torch.sigmoid(mask_logits).flatten(1)
        logits = mask_logits.flatten(1)
        k = max(1, int(prob.shape[1] * self.topk_ratio))
        top_prob = torch.topk(prob, k=k, dim=1).values
        top_logits = torch.topk(logits, k=k, dim=1).values
        return torch.stack(
            [
                prob.mean(dim=1),
                prob.amax(dim=1),
                top_prob.mean(dim=1),
                top_prob.std(dim=1, unbiased=False),
                top_logits.mean(dim=1),
            ],
            dim=1,
        )

    def _feature_stats(self, feature: torch.Tensor) -> torch.Tensor:
        avg = F.adaptive_avg_pool2d(feature, 1).flatten(1)
        mx = F.adaptive_max_pool2d(feature, 1).flatten(1)
        return torch.cat([avg, mx], dim=1)

    def forward(self, mask_logits: torch.Tensor, feature: torch.Tensor) -> torch.Tensor:
        stats = torch.cat([self._mask_stats(mask_logits), self._feature_stats(feature)], dim=1)
        return self.classifier(stats).flatten(1).squeeze(1)
