from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScoreLoss(nn.Module):
    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(logits.view(-1), target.float().view(-1))
