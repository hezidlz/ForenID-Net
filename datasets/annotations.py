from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def is_empty_mask(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.lower() in {"negative", "none", "null", ""})


def is_numeric_label(value: str) -> bool:
    try:
        label = float(value)
    except ValueError:
        return False
    return label in {0.0, 1.0}


def load_json_records(path: str | Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    records: list[dict[str, Any]] = []
    if isinstance(data, dict) and "samples" in data:
        data = data["samples"]
    if not isinstance(data, list):
        raise ValueError("Dataset JSON must be a list or a dict with key 'samples'")

    for item in data:
        if isinstance(item, dict):
            image = item.get("image") or item.get("img") or item.get("path")
            mask = item.get("mask")
            label = item.get("label")
        elif isinstance(item, (list, tuple)):
            image = item[0]
            mask = item[1] if len(item) > 1 else None
            label = item[2] if len(item) > 2 else None
        else:
            raise ValueError(f"Unsupported record type: {type(item)!r}")
        if image is None:
            raise ValueError(f"Record has no image path: {item!r}")
        if is_empty_mask(mask):
            mask = None
        records.append({"image": image, "mask": mask, "label": label})
    return records


def load_txt_records(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            if "," in line:
                parts = [part.strip() for part in line.split(",")]
            else:
                parts = line.split()

            if not parts or not parts[0]:
                continue

            image = parts[0]
            mask: str | None = None
            label: float | None = None

            if len(parts) == 1:
                pass
            elif len(parts) == 2:
                second = parts[1]
                if is_empty_mask(second):
                    mask = None
                    label = 0.0
                elif is_numeric_label(second):
                    label = float(second)
                else:
                    mask = second
            else:
                mask = None if is_empty_mask(parts[1]) else parts[1]
                label = float(parts[2]) if parts[2] != "" else None

            records.append({"image": image, "mask": mask, "label": label, "line_no": line_no})
    return records


def load_records(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.suffix.lower() == ".json":
        return load_json_records(path)
    if path.suffix.lower() in {".txt", ".lst", ".csv"}:
        return load_txt_records(path)
    raise ValueError(f"Unsupported annotation file extension: {path.suffix}")
