from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader
from tqdm import tqdm


def repository_imports(repository: Path):
    sys.path.insert(0, str(repository.resolve()))
    from datasets.collate import forgery_collate
    from datasets.doc_forgery_dataset import DocForgeryDataset
    from datasets.transforms import build_transform
    from models import build_model_from_config
    from utils.checkpoint import load_checkpoint
    from utils.config import load_config

    return forgery_collate, DocForgeryDataset, build_transform, build_model_from_config, load_checkpoint, load_config


def resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable. Revision experiments must not silently fall back to CPU.")
    return torch.device(requested)


def safe(function, *args) -> float:
    try:
        value = float(function(*args))
    except (ValueError, ZeroDivisionError):
        return math.nan
    return value if math.isfinite(value) else math.nan


def calibrate_image_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    if thresholds.size == 0:
        return 0.5
    f1 = 2.0 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    best = np.flatnonzero(np.isclose(f1, np.nanmax(f1), rtol=0.0, atol=1e-12))
    # Conservative tie-break for a financial screening setting: use the largest
    # threshold among equally good development-set F1 values.
    return float(np.max(thresholds[best]))


def image_metrics(frame: pd.DataFrame, threshold: float) -> dict[str, float]:
    labels = frame["label"].to_numpy(dtype=int)
    scores = frame["score"].to_numpy(dtype=float)
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else math.nan
    sensitivity = tp / (tp + fn) if (tp + fn) else math.nan
    if np.unique(labels).size < 2:
        auc = ap = balanced = mcc = tpr_1pct = math.nan
    else:
        auc = safe(roc_auc_score, labels, scores)
        ap = safe(average_precision_score, labels, scores)
        balanced = safe(balanced_accuracy_score, labels, predictions)
        mcc = safe(matthews_corrcoef, labels, predictions)
        fpr, tpr, _ = roc_curve(labels, scores)
        tpr_1pct = float(np.max(tpr[fpr <= 0.01 + 1e-12]))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    accuracy = (tp + tn) / max(1, len(labels))
    return {
        "image_auc": auc,
        "image_ap": ap,
        "image_accuracy": accuracy,
        "image_precision": precision,
        "image_recall": recall,
        "image_f1": f1,
        "image_specificity": specificity,
        "image_balanced_accuracy": balanced,
        "image_mcc": mcc,
        "image_tpr_at_fpr_1pct": tpr_1pct,
    }


def parse_metadata(path: str) -> dict[str, str]:
    normalized = path.replace("\\", "/")
    parts = normalized.split("/")
    stem = Path(normalized).stem
    attack = "none"
    if "attack" in parts:
        index = parts.index("attack")
        if index + 1 < len(parts):
            attack = parts[index + 1]
    return {
        "path": normalized,
        "group_key": stem,
        "template": stem.split("-", 1)[0].lower(),
        "device": parts[-2] if len(parts) >= 2 else "unknown",
        "attack_type": attack,
    }


def bbox(binary: np.ndarray) -> tuple[int, int, int, int] | None:
    rows, cols = np.where(binary)
    if rows.size == 0:
        return None
    return int(cols.min()), int(rows.min()), int(cols.max()) + 1, int(rows.max()) + 1


def box_iou(left: tuple[int, int, int, int] | None, right: tuple[int, int, int, int] | None) -> float:
    if left is None or right is None:
        return 0.0
    lx0, ly0, lx1, ly1 = left
    rx0, ry0, rx1, ry1 = right
    width = max(0, min(lx1, rx1) - max(lx0, rx0))
    height = max(0, min(ly1, ry1) - max(ly0, ry0))
    intersection = width * height
    union = (lx1 - lx0) * (ly1 - ly0) + (rx1 - rx0) * (ry1 - ry0) - intersection
    return intersection / max(1, union)


def confusion_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    tp = int(np.logical_and(prediction, target).sum())
    fp = int(np.logical_and(prediction, ~target).sum())
    fn = int(np.logical_and(~prediction, target).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    iou = tp / max(1, tp + fp + fn)
    return {"precision": precision, "recall": recall, "f1": f1, "iou": iou}


AREA_BINS = (
    ("le_2pct", -math.inf, 0.02),
    ("2_to_5pct", 0.02, 0.05),
    ("5_to_10pct", 0.05, 0.10),
    ("gt_10pct", 0.10, math.inf),
)


def localization_metrics(frame: pd.DataFrame) -> dict[str, float]:
    attacked = frame.loc[frame["label"] == 1]
    result: dict[str, float] = {}
    for column in (
        "boxmask_precision",
        "boxmask_recall",
        "boxmask_f1",
        "boxmask_iou",
        "enclosing_box_iou",
        "pointing_game",
    ):
        result[column] = float(attacked[column].mean()) if column in attacked else math.nan

    if "mask_area_ratio" not in attacked:
        for name, _, _ in AREA_BINS:
            result[f"area_{name}_n"] = 0.0
            for metric in ("boxmask_recall", "pointing_game", "enclosing_box_iou"):
                result[f"area_{name}_{metric}"] = math.nan
        result["small_region_n"] = 0.0
        result["small_region_recall"] = math.nan
        return result

    for name, lower, upper in AREA_BINS:
        if math.isinf(lower):
            selected = attacked.loc[attacked["mask_area_ratio"] <= upper]
        elif math.isinf(upper):
            selected = attacked.loc[attacked["mask_area_ratio"] > lower]
        else:
            selected = attacked.loc[
                (attacked["mask_area_ratio"] > lower) & (attacked["mask_area_ratio"] <= upper)
            ]
        result[f"area_{name}_n"] = float(len(selected))
        for metric in ("boxmask_recall", "pointing_game", "enclosing_box_iou"):
            result[f"area_{name}_{metric}"] = float(selected[metric].mean()) if len(selected) else math.nan

    result["small_region_n"] = result["area_le_2pct_n"]
    result["small_region_recall"] = result["area_le_2pct_boxmask_recall"]
    return result


class MaskThresholdCalibrator:
    def __init__(self, bins: int = 1000):
        self.edges = np.linspace(0.0, 1.0, bins + 1)
        self.rows: list[tuple[np.ndarray, np.ndarray, int]] = []

    def add(self, score: np.ndarray, target: np.ndarray) -> None:
        target = target.astype(bool).ravel()
        score = score.astype(np.float32).ravel()
        positive, _ = np.histogram(score[target], bins=self.edges)
        negative, _ = np.histogram(score[~target], bins=self.edges)
        self.rows.append((positive, negative, int(target.sum())))

    def select(self) -> tuple[float, float]:
        if not self.rows:
            return 0.5, math.nan
        macro_f1: list[float] = []
        for index in range(1, len(self.edges) - 1):
            sample_f1 = []
            for positive, negative, total_positive in self.rows:
                tp = int(positive[index:].sum())
                fp = int(negative[index:].sum())
                fn = total_positive - tp
                precision = tp / max(1, tp + fp)
                recall = tp / max(1, tp + fn)
                sample_f1.append(2.0 * precision * recall / max(1e-12, precision + recall))
            macro_f1.append(float(np.mean(sample_f1)))
        values = np.asarray(macro_f1)
        best = np.flatnonzero(np.isclose(values, np.nanmax(values), rtol=0.0, atol=1e-12))
        selected_index = int(best[-1]) + 1
        return float(self.edges[selected_index]), float(values[selected_index - 1])


def build_loader(list_path: Path, root: Path, cfg: dict, constructors) -> DataLoader:
    forgery_collate, DocForgeryDataset, build_transform = constructors
    transform = build_transform(cfg.get("data", {}), train=False)
    dataset = DocForgeryDataset(list_path, transform=transform, root=root)
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=forgery_collate)


@torch.no_grad()
def calibrate(model, loader: DataLoader, device: torch.device) -> tuple[dict[str, float], pd.DataFrame]:
    model.eval()
    rows: list[dict[str, object]] = []
    mask_calibrator = MaskThresholdCalibrator()
    for batch in tqdm(loader, desc="calibrate-dev"):
        image = batch["image"].to(device, non_blocking=True)
        outputs = model(image)
        score = float(outputs["pred_score"].detach().cpu().item())
        label = int(batch["label"].item())
        metadata = parse_metadata(batch["path"][0])
        rows.append({**metadata, "label": label, "score": score})
        if bool(batch["has_mask"].item()):
            prediction = outputs["pred_mask"].detach().cpu().numpy()[0, 0]
            target = batch["mask"].numpy()[0, 0] > 0.5
            mask_calibrator.add(prediction, target)
    frame = pd.DataFrame(rows)
    image_threshold = calibrate_image_threshold(frame["label"].to_numpy(), frame["score"].to_numpy())
    mask_threshold, dev_mask_macro_f1 = mask_calibrator.select()
    calibration = {
        "image_threshold": image_threshold,
        "mask_threshold": mask_threshold,
        "dev_mask_macro_f1_at_selected_threshold": dev_mask_macro_f1,
        **image_metrics(frame, image_threshold),
    }
    return calibration, frame


@torch.no_grad()
def evaluate_once(
    model,
    loader: DataLoader,
    device: torch.device,
    image_threshold: float,
    mask_threshold: float,
    max_metric_pixels: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    model.eval()
    rows: list[dict[str, object]] = []
    sampled_scores: list[np.ndarray] = []
    sampled_targets: list[np.ndarray] = []
    per_image_budget = max(1, max_metric_pixels // max(1, len(loader)))
    for batch in tqdm(loader, desc="evaluate-once"):
        image = batch["image"].to(device, non_blocking=True)
        outputs = model(image)
        score = float(outputs["pred_score"].detach().cpu().item())
        label = int(batch["label"].item())
        row: dict[str, object] = {**parse_metadata(batch["path"][0]), "label": label, "score": score}
        if bool(batch["has_mask"].item()):
            probability = outputs["pred_mask"].detach().cpu().numpy()[0, 0]
            target = batch["mask"].numpy()[0, 0] > 0.5
            prediction = probability >= mask_threshold
            metrics = confusion_metrics(prediction, target)
            row.update(
                {
                    "mask_area_ratio": float(target.mean()),
                    "boxmask_precision": metrics["precision"],
                    "boxmask_recall": metrics["recall"],
                    "boxmask_f1": metrics["f1"],
                    "boxmask_iou": metrics["iou"],
                    "enclosing_box_iou": box_iou(bbox(prediction), bbox(target)),
                    "pointing_game": float(target[np.unravel_index(np.argmax(probability), probability.shape)]),
                }
            )
            flat_score = probability.ravel()
            flat_target = target.ravel().astype(np.uint8)
            if flat_score.size > per_image_budget:
                indices = np.linspace(0, flat_score.size - 1, num=per_image_budget, dtype=np.int64)
                flat_score = flat_score[indices]
                flat_target = flat_target[indices]
            sampled_scores.append(flat_score.astype(np.float32))
            sampled_targets.append(flat_target)
        rows.append(row)
    frame = pd.DataFrame(rows)
    metrics = image_metrics(frame, image_threshold)
    metrics.update(localization_metrics(frame))
    if sampled_scores:
        pixel_score = np.concatenate(sampled_scores)
        pixel_target = np.concatenate(sampled_targets)
        metrics["boxmask_pixel_auc_sampled"] = safe(roc_auc_score, pixel_target, pixel_score)
        metrics["boxmask_pixel_ap_sampled"] = safe(average_precision_score, pixel_target, pixel_score)
        metrics["boxmask_metric_pixels"] = float(pixel_score.size)
    return frame, metrics


def bootstrap_ci(
    frame: pd.DataFrame,
    image_threshold: float,
    iterations: int,
    seed: int,
) -> dict[str, list[float]]:
    groups = {key: value for key, value in frame.groupby("group_key", sort=True)}
    names = np.array(sorted(groups), dtype=object)
    rng = np.random.default_rng(seed)
    image_draws: dict[str, list[float]] = {}
    location_columns = [
        "boxmask_precision",
        "boxmask_recall",
        "boxmask_f1",
        "boxmask_iou",
        "enclosing_box_iou",
        "pointing_game",
    ]
    location_draws = {column: [] for column in location_columns if column in frame}
    area_draws: dict[str, list[float]] = {}
    for _ in range(iterations):
        selected = rng.choice(names, size=len(names), replace=True)
        sample = pd.concat([groups[name] for name in selected], ignore_index=True)
        values = image_metrics(sample, image_threshold)
        for name, value in values.items():
            if math.isfinite(value):
                image_draws.setdefault(name, []).append(value)
        attacked = sample.loc[sample["label"] == 1]
        for column in location_draws:
            value = float(attacked[column].mean())
            if math.isfinite(value):
                location_draws[column].append(value)
        if "mask_area_ratio" in attacked:
            for name, lower, upper in AREA_BINS:
                if math.isinf(lower):
                    selected_area = attacked.loc[attacked["mask_area_ratio"] <= upper]
                elif math.isinf(upper):
                    selected_area = attacked.loc[attacked["mask_area_ratio"] > lower]
                else:
                    selected_area = attacked.loc[
                        (attacked["mask_area_ratio"] > lower) & (attacked["mask_area_ratio"] <= upper)
                    ]
                for column in ("boxmask_recall", "pointing_game", "enclosing_box_iou"):
                    if len(selected_area) and column in selected_area:
                        value = float(selected_area[column].mean())
                        if math.isfinite(value):
                            area_draws.setdefault(f"area_{name}_{column}", []).append(value)
    intervals: dict[str, list[float]] = {}
    for name, draws in {**image_draws, **location_draws, **area_draws}.items():
        intervals[name] = np.quantile(draws, [0.025, 0.975]).tolist() if draws else [math.nan, math.nan]
    return intervals


def summarize_subgroups(
    frame: pd.DataFrame,
    image_threshold: float,
    iterations: int,
    seed: int,
) -> dict[str, dict[str, object]]:
    output: dict[str, dict[str, object]] = {}
    for field in ("attack_type", "device", "template"):
        field_output: dict[str, object] = {}
        if field == "attack_type":
            categories = sorted(set(frame.loc[frame["label"] == 1, field].dropna()) - {"none"})
        else:
            categories = sorted(frame[field].dropna().unique())
        for category_index, category in enumerate(categories):
            if field == "attack_type":
                subset = frame.loc[(frame["label"] == 0) | (frame[field] == category)].copy()
            else:
                subset = frame.loc[frame[field] == category].copy()
            labels = subset["label"].to_numpy(dtype=int)
            point = {**image_metrics(subset, image_threshold), **localization_metrics(subset)}
            field_output[str(category)] = {
                "n": int(len(subset)),
                "groups": int(subset["group_key"].nunique()),
                "real": int((labels == 0).sum()),
                "fake": int((labels == 1).sum()),
                "metrics": point,
                "cluster_bootstrap_95ci": bootstrap_ci(
                    subset,
                    image_threshold,
                    iterations,
                    seed + category_index * 1000,
                ),
            }
        output[field] = field_output
    return output


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected NAME=PATH")
    name, path = value.split("=", 1)
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Development-calibrated, one-pass revision evaluation.")
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dev-list", type=Path, required=True)
    parser.add_argument("--test", action="append", type=parse_named_path, required=True, help="NAME=PATH; repeatable")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260829)
    parser.add_argument("--max-metric-pixels", type=int, default=2_000_000)
    args = parser.parse_args()

    imports = repository_imports(args.repository)
    forgery_collate, DocForgeryDataset, build_transform, build_model_from_config, load_checkpoint, load_config = imports
    cfg = load_config(args.config)
    device = resolve_device(args.device)
    model = build_model_from_config(cfg).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device, strict=False)
    constructors = (forgery_collate, DocForgeryDataset, build_transform)

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    dev_loader = build_loader(args.dev_list, args.root, cfg, constructors)
    calibration, dev_frame = calibrate(model, dev_loader, device)
    (output / "calibration.json").write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    dev_frame.to_csv(output / "dev_scores.csv", index=False)

    summary: dict[str, object] = {
        "checkpoint": str(args.checkpoint.resolve()),
        "config": str(args.config.resolve()),
        "calibration": calibration,
        "tests": {},
    }
    for index, (name, list_path) in enumerate(args.test):
        loader = build_loader(list_path, args.root, cfg, constructors)
        frame, metrics = evaluate_once(
            model,
            loader,
            device,
            calibration["image_threshold"],
            calibration["mask_threshold"],
            args.max_metric_pixels,
        )
        frame.to_csv(output / f"{name}_per_image.csv", index=False)
        intervals = bootstrap_ci(
            frame,
            calibration["image_threshold"],
            args.bootstrap,
            args.bootstrap_seed + index * 10000,
        )
        result = {
            "list": str(list_path.resolve()),
            "n": len(frame),
            "groups": int(frame["group_key"].nunique()),
            "metrics": metrics,
            "cluster_bootstrap_95ci": intervals,
            "subgroups": summarize_subgroups(
                frame,
                calibration["image_threshold"],
                args.bootstrap,
                args.bootstrap_seed + index * 10000 + 500000,
            ),
        }
        summary["tests"][name] = result
        (output / f"{name}_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (output / "evaluation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
