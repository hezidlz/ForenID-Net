from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .annotations import load_records
from .transforms import DocLiteTransform


def _to_image_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.asarray(image).astype(np.float32) / 255.0
    return torch.from_numpy(arr.transpose(2, 0, 1)).contiguous()


def _to_mask_tensor(mask: Image.Image | None, size: tuple[int, int]) -> tuple[torch.Tensor, bool]:
    if mask is None:
        arr = np.zeros(size, dtype=np.float32)
        return torch.from_numpy(arr).unsqueeze(0), False
    arr = np.asarray(mask.convert("L"))
    arr = (arr > 127).astype(np.float32)
    return torch.from_numpy(arr).unsqueeze(0), True


class JsonForgeryDataset(Dataset):
    def __init__(
        self,
        annotation_path: str | Path,
        transform: DocLiteTransform,
        root: str | Path | None = None,
    ):
        self.annotation_path = Path(annotation_path)
        self.root = Path(root) if root else self.annotation_path.parent
        self.records = load_records(self.annotation_path)
        self.transform = transform

    def _resolve(self, path: str | Path | None) -> Path | None:
        if path is None:
            return None
        path = Path(path)
        if path.is_absolute():
            return path
        return self.root / path

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        record = self.records[index]
        image_path = self._resolve(record["image"])
        mask_path = self._resolve(record["mask"])
        assert image_path is not None
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L") if mask_path is not None else None
        if record["label"] is None:
            label_hint = 1.0 if mask_path is not None else 0.0
        else:
            label_hint = float(record["label"])
        image, mask = self.transform(image, mask, label=label_hint)
        image_tensor = _to_image_tensor(image)
        mask_tensor, has_mask = _to_mask_tensor(mask, size=(image.height, image.width))

        if record["label"] is None:
            label = float(mask_tensor.max().item() > 0.5)
        else:
            label = float(record["label"])

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "label": torch.tensor(label, dtype=torch.float32),
            "has_mask": torch.tensor(has_mask, dtype=torch.bool),
            "path": str(image_path),
        }
