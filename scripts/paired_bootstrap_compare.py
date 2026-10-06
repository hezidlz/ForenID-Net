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


def named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected NAME=CSV")
    name, path = value.split("=", 1)
    return name, Path(path)


def normalize(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    frame = frame.copy()
    if "path" not in frame and "image" in frame:
        frame = frame.rename(columns={"image": "path"})
    required = {"path", "label", "score"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Missing columns in {prefix}: {sorted(missing)}")
    frame["path"] = frame["path"].astype(str).str.replace("\\", "/", regex=False)
    if "group_key" not in frame:
        frame["group_key"] = frame["path"].map(lambda value: Path(value).stem)
    if frame["path"].duplicated().any():
        raise ValueError(f"Duplicate path rows in {prefix}")
    keep = ["path", "group_key", "label", "score"]
    for column in ("boxmask_f1", "boxmask_recall", "enclosing_box_iou", "pointing_game"):
        if column in frame:
            keep.append(column)
    renamed = {column: f"{prefix}_{column}" for column in keep if column not in {"path", "group_key", "label"}}
    return frame[keep].rename(columns=renamed)


def finite_metric(function, *args) -> float:
    try:
        value = float(function(*args))
    except ValueError:
        return math.nan
    return value if math.isfinite(value) else math.nan


def metrics(labels: np.ndarray, scores: np.ndarray, threshold: float | None) -> dict[str, float]:
    result: dict[str, float] = {
        "image_auc": finite_metric(roc_auc_score, labels, scores),
        "image_ap": finite_metric(average_precision_score, labels, scores),
    }
    if np.unique(labels).size == 2:
        fpr, tpr, _ = roc_curve(labels, scores)
        result["image_tpr_at_fpr_1pct"] = float(np.max(tpr[fpr <= 0.01 + 1e-12]))
    else:
        result["image_tpr_at_fpr_1pct"] = math.nan
    if threshold is None:
        return result
    prediction = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, prediction, labels=[0, 1]).ravel()
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    result.update(
        {
            "image_accuracy": (tp + tn) / max(1, len(labels)),
            "image_f1": 2 * precision * recall / max(1e-12, precision + recall),
            "image_specificity": tn / max(1, tn + fp),
            "image_balanced_accuracy": finite_metric(balanced_accuracy_score, labels, prediction),
            "image_mcc": finite_metric(matthews_corrcoef, labels, prediction),
        }
    )
    return result


def all_differences(
    frame: pd.DataFrame,
    left_prefix: str,
    right_prefix: str,
    left_threshold: float | None,
    right_threshold: float | None,
) -> dict[str, float]:
    labels = frame["label"].to_numpy(dtype=int)
    left_metrics = metrics(labels, frame[f"{left_prefix}_score"].to_numpy(dtype=float), left_threshold)
    right_metrics = metrics(labels, frame[f"{right_prefix}_score"].to_numpy(dtype=float), right_threshold)
    differences = {
        name: left_metrics[name] - right_metrics[name]
        for name in left_metrics.keys() & right_metrics.keys()
        if math.isfinite(left_metrics[name]) and math.isfinite(right_metrics[name])
    }
    attacked = frame.loc[frame["label"] == 1]
    for metric_name in ("boxmask_f1", "boxmask_recall", "enclosing_box_iou", "pointing_game"):
        left_column = f"{left_prefix}_{metric_name}"
        right_column = f"{right_prefix}_{metric_name}"
        if left_column in attacked and right_column in attacked:
            valid = attacked[[left_column, right_column]].dropna()
            if len(valid):
                differences[metric_name] = float((valid[left_column] - valid[right_column]).mean())
    return differences


def markdown_table(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for values in frame.itertuples(index=False, name=None):
        cells = []
        for value in values:
            if isinstance(value, (float, np.floating)):
                cells.append("nan" if not math.isfinite(float(value)) else f"{float(value):.6f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired base-document-group bootstrap model comparison.")
    parser.add_argument("--left", type=named_path, required=True)
    parser.add_argument("--right", type=named_path, required=True)
    parser.add_argument("--left-threshold", type=float)
    parser.add_argument("--right-threshold", type=float)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    left_name, left_path = args.left
    right_name, right_path = args.right
    left_prefix, right_prefix = "left", "right"
    left = normalize(pd.read_csv(left_path), left_prefix)
    right = normalize(pd.read_csv(right_path), right_prefix)
    paired = left.merge(right, on="path", how="inner", suffixes=("_left_meta", "_right_meta"), validate="one_to_one")
    if len(paired) != len(left) or len(paired) != len(right):
        raise ValueError(f"Models do not cover the same paths: left={len(left)}, right={len(right)}, paired={len(paired)}")
    if not np.array_equal(paired["label_left_meta"].to_numpy(), paired["label_right_meta"].to_numpy()):
        raise ValueError("Label mismatch between paired files")
    if not np.array_equal(paired["group_key_left_meta"].to_numpy(), paired["group_key_right_meta"].to_numpy()):
        raise ValueError("group_key mismatch between paired files")
    paired = paired.rename(
        columns={"label_left_meta": "label", "group_key_left_meta": "group_key"}
    ).drop(columns=["label_right_meta", "group_key_right_meta"])

    point = all_differences(
        paired,
        left_prefix,
        right_prefix,
        args.left_threshold,
        args.right_threshold,
    )
    grouped = {name: group for name, group in paired.groupby("group_key", sort=True)}
    group_names = np.array(sorted(grouped), dtype=object)
    rng = np.random.default_rng(args.seed)
    draws: dict[str, list[float]] = {name: [] for name in point}
    for _ in range(args.iterations):
        selected = rng.choice(group_names, size=len(group_names), replace=True)
        sample = pd.concat([grouped[name] for name in selected], ignore_index=True)
        values = all_differences(
            sample,
            left_prefix,
            right_prefix,
            args.left_threshold,
            args.right_threshold,
        )
        for name, value in values.items():
            if math.isfinite(value):
                draws.setdefault(name, []).append(value)

    rows = []
    for metric_name, difference in point.items():
        values = np.asarray(draws.get(metric_name, []), dtype=float)
        lower, upper = np.quantile(values, [0.025, 0.975]) if values.size else (math.nan, math.nan)
        if values.size:
            lower_tail = (float(np.sum(values <= 0)) + 1.0) / (values.size + 1.0)
            upper_tail = (float(np.sum(values >= 0)) + 1.0) / (values.size + 1.0)
            p_value = min(1.0, 2.0 * min(lower_tail, upper_tail))
        else:
            p_value = math.nan
        rows.append(
            {
                "metric": metric_name,
                "difference_left_minus_right": difference,
                "ci95_lower": lower,
                "ci95_upper": upper,
                "bootstrap_two_sided_p": p_value,
            }
        )

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(output / "paired_differences.csv", index=False)
    result = {
        "left": {"name": left_name, "csv": str(left_path.resolve()), "threshold": args.left_threshold},
        "right": {"name": right_name, "csv": str(right_path.resolve()), "threshold": args.right_threshold},
        "n": int(len(paired)),
        "groups": int(len(group_names)),
        "iterations": args.iterations,
        "seed": args.seed,
        "differences": rows,
    }
    (output / "paired_differences.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (output / "paired_differences.md").write_text(
        f"# Paired bootstrap: {left_name} minus {right_name}\n\n" + markdown_table(table) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
