from __future__ import annotations

from typing import Any

import torch


def forgery_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "image": torch.stack([item["image"] for item in batch]),
        "mask": torch.stack([item["mask"] for item in batch]),
        "label": torch.stack([item["label"] for item in batch]),
        "has_mask": torch.stack([item["has_mask"] for item in batch]),
        "path": [item["path"] for item in batch],
    }
