from __future__ import annotations

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        prob = torch.sigmoid(logits)
        target = target.float()
        if valid_mask is None:
            valid_mask = torch.ones_like(target)
        else:
            valid_mask = valid_mask.float()

        prob = prob * valid_mask
        target = target * valid_mask
        dims = tuple(range(1, prob.ndim))
        inter = torch.sum(prob * target, dim=dims)
        den = torch.sum(prob * prob + target * target, dim=dims)
        dice = (2.0 * inter + self.smooth) / (den + self.smooth)
        return 1.0 - dice.mean()
