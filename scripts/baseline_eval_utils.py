from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

np: Any = None
Image: Any = None
tqdm: Any = None


def load_numpy() -> Any:
    global np
    if np is None:
        try:
            import numpy as _np
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("Missing dependency: pip install numpy") from exc
        np = _np
    return np


def load_image_module() -> Any:
    global Image
    if Image is None:
        try:
            from PIL import Image as _Image
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("Missing dependency: pip install pillow") from exc
        Image = _Image
    return Image


def load_tqdm() -> Any:
    global tqdm
    if tqdm is None:
        try:
            from tqdm import tqdm as _tqdm
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("Missing dependency: pip install tqdm") from exc
        tqdm = _tqdm
    return tqdm


def parse_list_line(line: str) -> tuple[str, str | None, int]:
    parts = [part.strip() for part in line.strip().split(",")]
    if len(parts) < 3:
        parts = [part.strip() for part in line.strip().split()]
    if len(parts) < 3:
        raise ValueError(f"Expected image,mask,label row, got: {line!r}")
    image = parts[0].replace("\\", "/")
    mask_raw = parts[1].replace("\\", "/")
    mask = None if mask_raw.lower() in {"", "none", "null", "negative"} else mask_raw
    label = int(float(parts[2]))
    return image, mask, label


def read_eval_list(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            image, mask, label = parse_list_line(line)
            rows.append({"image": image, "mask": mask, "label": label})
    return rows


def resolve_path(root: Path, rel_or_abs: str | None) -> Path | None:
    if rel_or_abs is None:
        return None
    path = Path(rel_or_abs)
    if path.is_absolute():
        return path
    return root / path


def load_rgb_array(path: Path) -> Any:
    np_mod = load_numpy()
    image_mod = load_image_module()
    with image_mod.open(path) as im:
        return np_mod.asarray(im.convert("RGB"))


def resize_long_side(image: Any, max_side: int) -> Any:
    np_mod = load_numpy()
    image_mod = load_image_module()
    if max_side is None or max_side <= 0:
        return image
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return image
    scale = float(max_side) / float(longest)
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return np_mod.asarray(image_mod.fromarray(image).resize(new_size, resample=image_mod.BILINEAR))


def load_mask(path: Path | None, size_hw: tuple[int, int]) -> Any:
    np_mod = load_numpy()
    image_mod = load_image_module()
    height, width = size_hw
    if path is None:
        return np_mod.zeros((height, width), dtype=np_mod.uint8)
    mask = image_mod.open(path).convert("L")
    if mask.size != (width, height):
        mask = mask.resize((width, height), resample=image_mod.NEAREST)
    return (np_mod.asarray(mask) > 127).astype(np_mod.uint8)


def resize_score_map(score: Any, size_hw: tuple[int, int]) -> Any:
    np_mod = load_numpy()
    image_mod = load_image_module()
    height, width = size_hw
    if score.shape[:2] == (height, width):
        return score.astype(np_mod.float32)
    score_uint8 = np_mod.clip(score * 255.0, 0, 255).astype(np_mod.uint8)
    resized = image_mod.fromarray(score_uint8).resize((width, height), resample=image_mod.BILINEAR)
    return (np_mod.asarray(resized).astype(np_mod.float32) / 255.0)


def binary_at_threshold(score: Any, target: Any, threshold: float) -> dict[str, float]:
    np_mod = load_numpy()
    pred = score > threshold
    target = target > 0.5
    tp = float(np_mod.logical_and(pred, target).sum())
    tn = float(np_mod.logical_and(~pred, ~target).sum())
    fp = float(np_mod.logical_and(pred, ~target).sum())
    fn = float(np_mod.logical_and(~pred, target).sum())
    eps = 1e-8
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    acc = (tp + tn) / (tp + tn + fp + fn + eps)
    return {"precision": precision, "recall": recall, "f1": f1, "iou": iou, "acc": acc}


def nan_ranking_metrics() -> dict[str, float]:
    return {
        "auc": math.nan,
        "ap": math.nan,
        "best_threshold": math.nan,
        "best_precision": math.nan,
        "best_recall": math.nan,
        "best_f1": math.nan,
        "best_iou": math.nan,
        "best_accuracy": math.nan,
    }


def ranking_metrics(score: Any, target: Any) -> dict[str, float]:
    np_mod = load_numpy()
    score = np_mod.asarray(score, dtype=np_mod.float64).reshape(-1)
    target = (np_mod.asarray(target).reshape(-1) > 0.5).astype(np_mod.float64)
    valid = np_mod.isfinite(score)
    score = score[valid]
    target = target[valid]
    if score.size == 0:
        return nan_ranking_metrics()
    positives = float(target.sum())
    negatives = float(target.size - positives)
    if positives <= 0 or negatives <= 0:
        return nan_ranking_metrics()

    order = np_mod.argsort(-score)
    sorted_score = score[order]
    sorted_target = target[order]
    is_last = np_mod.ones(sorted_score.shape[0], dtype=bool)
    is_last[:-1] = sorted_score[:-1] != sorted_score[1:]

    tp = np_mod.cumsum(sorted_target)[is_last]
    fp = np_mod.cumsum(1.0 - sorted_target)[is_last]
    thresholds = sorted_score[is_last]

    eps = 1e-8
    precision = tp / (tp + fp + eps)
    recall = tp / (positives + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    iou = tp / (positives + fp + eps)
    acc = (tp + (negatives - fp)) / (positives + negatives + eps)

    fpr = fp / (negatives + eps)
    auc = np_mod.trapz(np_mod.concatenate([[0.0], recall]), np_mod.concatenate([[0.0], fpr]))
    prev_recall = np_mod.concatenate([[0.0], recall[:-1]])
    ap = float(np_mod.sum((recall - prev_recall) * precision))

    best_idx = int(np_mod.argmax(f1))
    return {
        "auc": float(auc),
        "ap": ap,
        "best_threshold": float(thresholds[best_idx]),
        "best_precision": float(precision[best_idx]),
        "best_recall": float(recall[best_idx]),
        "best_f1": float(f1[best_idx]),
        "best_iou": float(iou[best_idx]),
        "best_accuracy": float(acc[best_idx]),
    }


def sample_pixels(score: Any, target: Any, limit: int | None) -> tuple[Any, Any]:
    np_mod = load_numpy()
    score = score.reshape(-1)
    target = target.reshape(-1)
    if limit is not None and limit > 0 and score.size > limit:
        idx = np_mod.linspace(0, score.size - 1, num=limit).astype(np_mod.int64)
        score = score[idx]
        target = target[idx]
    return score, target


def finalize_metrics(
    *,
    rows: list[dict[str, Any]],
    image_scores: list[float],
    image_labels: list[int],
    fixed_pixel_stats: list[dict[str, float]],
    pixel_scores: list[Any],
    pixel_labels: list[Any],
    checkpoint: str,
    threshold: float,
) -> dict[str, Any]:
    np_mod = load_numpy()
    img_rank = ranking_metrics(np_mod.asarray(image_scores), np_mod.asarray(image_labels))
    img_fixed = binary_at_threshold(np_mod.asarray(image_scores), np_mod.asarray(image_labels), threshold)
    stats: dict[str, Any] = {
        "samples": len(rows),
        "fake": int(np_mod.sum(np_mod.asarray(image_labels) == 1)),
        "real": int(np_mod.sum(np_mod.asarray(image_labels) == 0)),
        "checkpoint": checkpoint,
        "image_acc": img_fixed["acc"],
        "image_precision": img_fixed["precision"],
        "image_recall": img_fixed["recall"],
        "image_f1": img_fixed["f1"],
        "image_auc": img_rank["auc"],
        "image_ap": img_rank["ap"],
        "best_image_threshold": img_rank["best_threshold"],
        "image_precision_best": img_rank["best_precision"],
        "image_recall_best": img_rank["best_recall"],
        "image_f1_best": img_rank["best_f1"],
        "image_acc_best": img_rank["best_accuracy"],
    }

    if fixed_pixel_stats:
        for key in ("precision", "recall", "f1", "iou"):
            stats[f"pixel_{key}"] = float(np_mod.mean([item[key] for item in fixed_pixel_stats]))

    if pixel_scores:
        all_pixel_scores = np_mod.concatenate(pixel_scores)
        all_pixel_labels = np_mod.concatenate(pixel_labels)
        pixel_rank = ranking_metrics(all_pixel_scores, all_pixel_labels)
        stats.update(
            {
                "pixel_auc": pixel_rank["auc"],
                "pixel_ap": pixel_rank["ap"],
                "best_mask_threshold": pixel_rank["best_threshold"],
                "pixel_precision_best": pixel_rank["best_precision"],
                "pixel_recall_best": pixel_rank["best_recall"],
                "pixel_f1_best": pixel_rank["best_f1"],
                "pixel_iou_best": pixel_rank["best_iou"],
                "pixel_metric_pixels": int(all_pixel_scores.size),
            }
        )
    return stats


def write_outputs(stats: dict[str, Any], per_image: list[dict[str, Any]], out_json: Path, per_image_csv: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8")

    per_image_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for item in per_image for key in item.keys()})
    with open(per_image_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_image)
