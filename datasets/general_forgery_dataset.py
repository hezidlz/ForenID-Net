from __future__ import annotations

from pathlib import Path

from .base_dataset import JsonForgeryDataset
from .transforms import DocLiteTransform


class GeneralForgeryDataset(JsonForgeryDataset):
    """General forgery pretraining dataset.

    Annotation formats supported:
    - TruFor-style txt: image_path,mask_path
    - TruFor-style txt: image_path,None
    - [["image.jpg", "mask.png"], ["real.jpg", "Negative"]]
    - [{"image": "...", "mask": "...", "label": 1}]
    """

    def __init__(self, annotation_path: str | Path, transform: DocLiteTransform, root: str | Path | None = None):
        super().__init__(annotation_path=annotation_path, transform=transform, root=root)
