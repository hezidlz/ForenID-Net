from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
)


METHOD_FILES = {
    "CAT-Net": "catnet_fantasyid_per_image.csv",
    "MVSS-Net": "mvssnet_fantasyid_per_image.csv",
    "PSCC-Net": "psccnet_fantasyid_per_image.csv",
}

IMAGE_METRICS = (
    "image_auc",
    "image_ap",
    "balanced_accuracy_at_0_5",
    "specificity_at_0_5",
    "sensitivity_at_0_5",
    "mcc_at_0_5",
    "tpr_at_fpr_1pct",
)

LOCALIZATION_METRICS = (
    "boxmask_precision_at_0_5",
    "boxmask_recall_at_0_5",
    "boxmask_f1_at_0_5",
    "boxmask_iou_at_0_5",
)


def normalize_path(value: str) -> str:
    return str(value).replace("\\", "/").lstrip("./")


def safe_metric(function, *args) -> float:
    try:
        value = float(function(*args))
    except (ValueError, ZeroDivisionError):
        return math.nan
    return value if math.isfinite(value) else math.nan


def tpr_at_fpr(labels: np.ndarray, scores: np.ndarray, target_fpr: float = 0.01) -> float:
    if np.unique(labels).size < 2:
        return math.nan
    fpr, tpr, _ = roc_curve(labels, scores)
    valid = fpr <= target_fpr + 1e-12
    return float(np.max(tpr[valid])) if np.any(valid) else 0.0


def image_metrics(frame: pd.DataFrame) -> dict[str, float]:
    labels = frame["label"].to_numpy(dtype=int)
    scores = frame["score"].to_numpy(dtype=float)
    predictions = (scores >= 0.5).astype(int)
    if np.unique(labels).size == 2:
        tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
        specificity = tn / max(1, tn + fp)
        sensitivity = tp / max(1, tp + fn)
    else:
        specificity = math.nan
        sensitivity = math.nan
    return {
        "image_auc": safe_metric(roc_auc_score, labels, scores),
        "image_ap": safe_metric(average_precision_score, labels, scores),
        "balanced_accuracy_at_0_5": safe_metric(balanced_accuracy_score, labels, predictions),
        "specificity_at_0_5": float(specificity),
        "sensitivity_at_0_5": float(sensitivity),
        "mcc_at_0_5": safe_metric(matthews_corrcoef, labels, predictions),
        "tpr_at_fpr_1pct": tpr_at_fpr(labels, scores),
    }


def localization_metrics(frame: pd.DataFrame) -> dict[str, float]:
    attacked = frame.loc[frame["label"] == 1]
    return {
        "boxmask_precision_at_0_5": float(attacked["pixel_precision"].mean()),
        "boxmask_recall_at_0_5": float(attacked["pixel_recall"].mean()),
        "boxmask_f1_at_0_5": float(attacked["pixel_f1"].mean()),
        "boxmask_iou_at_0_5": float(attacked["pixel_iou"].mean()),
    }


def cluster_bootstrap(
    frame: pd.DataFrame,
    metric_function,
    metric_names: tuple[str, ...],
    iterations: int,
    seed: int,
) -> dict[str, tuple[float, float]]:
    rng = np.random.default_rng(seed)
    grouped = {group: rows for group, rows in frame.groupby("group_key", sort=True)}
    groups = np.array(sorted(grouped), dtype=object)
    draws: dict[str, list[float]] = {name: [] for name in metric_names}
    for _ in range(iterations):
        selected = rng.choice(groups, size=len(groups), replace=True)
        sampled = pd.concat([grouped[group] for group in selected], ignore_index=True)
        values = metric_function(sampled)
        for name in metric_names:
            value = values[name]
            if math.isfinite(value):
                draws[name].append(value)
    intervals: dict[str, tuple[float, float]] = {}
    for name, values in draws.items():
        if not values:
            intervals[name] = (math.nan, math.nan)
        else:
            intervals[name] = tuple(np.quantile(values, [0.025, 0.975]).tolist())
    return intervals


def summarize_method(
    method: str,
    frame: pd.DataFrame,
    iterations: int,
    seed: int,
) -> dict[str, object]:
    image = image_metrics(frame)
    location = localization_metrics(frame)
    image_ci = cluster_bootstrap(frame, image_metrics, IMAGE_METRICS, iterations, seed)
    location_ci = cluster_bootstrap(
        frame.loc[frame["label"] == 1],
        localization_metrics,
        LOCALIZATION_METRICS,
        iterations,
        seed + 1000,
    )
    estimates = {**image, **location}
    intervals = {**image_ci, **location_ci}
    result: dict[str, object] = {
        "method": method,
        "n": int(len(frame)),
        "groups": int(frame["group_key"].nunique()),
        "real": int((frame["label"] == 0).sum()),
        "fake": int((frame["label"] == 1).sum()),
    }
    for name, value in estimates.items():
        lower, upper = intervals[name]
        result[name] = value
        result[f"{name}_ci_low"] = lower
        result[f"{name}_ci_high"] = upper
    return result


def subgroup_counts(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dimension in ("attack_type", "device", "template_stratum"):
        for value, group in frame.groupby(dimension, dropna=False, sort=True):
            rows.append(
                {
                    "dimension": dimension,
                    "value": value,
                    "samples": len(group),
                    "groups": group["group_key"].nunique(),
                    "real": int((group["label"] == 0).sum()),
                    "fake": int((group["label"] == 1).sum()),
                }
            )
    return pd.DataFrame(rows)


def subgroup_summaries(
    method: str,
    frame: pd.DataFrame,
    iterations: int,
    seed: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    offset = 0
    for attack in sorted(set(frame.loc[frame["label"] == 1, "attack_type"]) - {"none"}):
        subset = frame.loc[(frame["label"] == 0) | (frame["attack_type"] == attack)].copy()
        result = summarize_method(method, subset, iterations, seed + offset)
        result.update({"dimension": "attack_type", "value": attack})
        rows.append(result)
        offset += 1000
    for dimension in ("device", "template_stratum"):
        for value, subset in frame.groupby(dimension, dropna=False, sort=True):
            result = summarize_method(method, subset.copy(), iterations, seed + offset)
            result.update({"dimension": dimension, "value": value})
            rows.append(result)
            offset += 1000
    return rows


def markdown_table(results: pd.DataFrame) -> str:
    columns = [
        "method",
        "n",
        "groups",
        "image_auc",
        "image_ap",
        "balanced_accuracy_at_0_5",
        "specificity_at_0_5",
        "mcc_at_0_5",
        "tpr_at_fpr_1pct",
        "boxmask_f1_at_0_5",
        "boxmask_iou_at_0_5",
    ]
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, divider]
    for _, row in results[columns].iterrows():
        values = []
        for column in columns:
            value = row[column]
            values.append(f"{value:.4f}" if isinstance(value, (float, np.floating)) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-analyse existing transfer-only baseline predictions.")
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260829)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    manifest["path"] = manifest["path"].map(normalize_path)
    expected = set(manifest["path"])
    results: list[dict[str, object]] = []
    subgroup_results: list[dict[str, object]] = []
    merged_for_counts: pd.DataFrame | None = None
    for index, (method, filename) in enumerate(METHOD_FILES.items()):
        predictions = pd.read_csv(args.workspace / filename)
        predictions["image"] = predictions["image"].map(normalize_path)
        available = set(predictions["image"])
        missing = sorted(expected - available)
        if missing:
            raise RuntimeError(f"{method}: {len(missing)} manifest images lack predictions; first={missing[0]}")
        frame = manifest.merge(predictions, left_on="path", right_on="image", how="left", validate="one_to_one")
        if not np.array_equal(frame["label_x"].to_numpy(dtype=int), frame["label_y"].to_numpy(dtype=int)):
            raise RuntimeError(f"{method}: label mismatch between manifest and prediction CSV")
        frame = frame.rename(columns={"label_x": "label"}).drop(columns=["label_y"])
        results.append(summarize_method(method, frame, args.bootstrap, args.seed + index * 10000))
        subgroup_results.extend(
            subgroup_summaries(method, frame, args.bootstrap, args.seed + index * 100000 + 50000)
        )
        if merged_for_counts is None:
            merged_for_counts = frame

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    result_frame = pd.DataFrame(results)
    result_frame.to_csv(output / "transfer_only_external_clean_metrics.csv", index=False)
    subgroup_result_frame = pd.DataFrame(subgroup_results)
    subgroup_result_frame.to_csv(output / "transfer_only_subgroup_metrics.csv", index=False)
    assert merged_for_counts is not None
    subgroup_counts(merged_for_counts).to_csv(output / "external_clean_subgroup_counts.csv", index=False)
    payload = {
        "scope": "official checkpoints, transfer-only, FantasyID external clean subset",
        "mask_note": "Localization metrics are computed against box-derived masks at threshold 0.5.",
        "bootstrap": {
            "unit": "base-document group",
            "iterations": args.bootstrap,
            "confidence": 0.95,
            "seed": args.seed,
        },
        "results": results,
        "subgroup_results": subgroup_results,
    }
    (output / "transfer_only_external_clean_metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "transfer_only_external_clean_metrics.md").write_text(
        "# Transfer-only baseline audit\n\n"
        "These values are reference results from official checkpoints; they are not fair fine-tuned comparisons. "
        "All confidence intervals use base-document-group bootstrap resampling. Localization is measured against "
        "box-derived masks at a fixed threshold of 0.5.\n\n"
        + markdown_table(result_frame)
        + "\n",
        encoding="utf-8",
    )
    print(result_frame[["method", "n", "groups", *IMAGE_METRICS, *LOCALIZATION_METRICS]].to_string(index=False))


if __name__ == "__main__":
    main()
