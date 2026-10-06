from __future__ import annotations

import torch
import torch.nn as nn

from .dice_loss import DiceLoss
from .focal_loss import BinaryFocalLoss
from .score_loss import ScoreLoss


class DocLiteLoss(nn.Module):
    def __init__(
        self,
        mask_focal_weight: float = 1.0,
        mask_dice_weight: float = 1.0,
        score_weight: float = 0.5,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.mask_focal_weight = mask_focal_weight
        self.mask_dice_weight = mask_dice_weight
        self.score_weight = score_weight
        self.focal = BinaryFocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice = DiceLoss()
        self.score = ScoreLoss()

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        mask: torch.Tensor | None = None,
        label: torch.Tensor | None = None,
        has_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        total = outputs["score_logits"].new_tensor(0.0)
        loss_dict: dict[str, torch.Tensor] = {}

        if mask is not None and "mask_logits" in outputs:
            mask_logits = outputs["mask_logits"]
            if has_mask is not None:
                valid = has_mask.view(-1, 1, 1, 1).to(mask_logits.device).float()
            else:
                valid = torch.ones(mask_logits.shape[0], 1, 1, 1, device=mask_logits.device)

            focal_loss = self.focal(mask_logits, mask, valid_mask=valid.expand_as(mask_logits))
            dice_loss = self.dice(mask_logits, mask, valid_mask=valid.expand_as(mask_logits))
            total = total + self.mask_focal_weight * focal_loss + self.mask_dice_weight * dice_loss
            loss_dict["loss_mask_focal"] = focal_loss.detach()
            loss_dict["loss_mask_dice"] = dice_loss.detach()

        if label is not None:
            score_loss = self.score(outputs["score_logits"], label)
            total = total + self.score_weight * score_loss
            loss_dict["loss_score"] = score_loss.detach()

        loss_dict["loss_total"] = total
        return loss_dict
