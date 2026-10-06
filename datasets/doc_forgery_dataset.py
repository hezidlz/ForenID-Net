from __future__ import annotations

from pathlib import Path

from .base_dataset import JsonForgeryDataset
from .transforms import DocLiteTransform


class DocForgeryDataset(JsonForgeryDataset):
    """Document/ID finetuning dataset.

    Mask is optional. When mask is missing, the sample contributes only to the
    image-level score loss.
    """

    def __init__(self, annotation_path: str | Path, transform: DocLiteTransform, root: str | Path | None = None):
        super().__init__(annotation_path=annotation_path, transform=transform, root=root)
