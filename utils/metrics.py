from __future__ import annotations

import math

import torch

EPS = 1e-8


def _as_binary_target(target: torch.Tensor) -> torch.Tensor:
    return (target.detach().flatten().float() > 0.5).float()


def _as_score(score: torch.Tensor) -> torch.Tensor:
    return score.detach().flatten().float()


@torch.no_grad()
def binary_metrics_from_confusion(
    tp: torch.Tensor,
    tn: torch.Tensor,
    fp: torch.Tensor,
    fn: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return threshold metrics from binary confusion counts.

    Precision controls false alarms, recall controls missed forgeries, F1
    balances both, and IoU measures overlap between predicted and true regions.
    """
    tp = tp.float()
    tn = tn.float()
    fp = fp.float()
    fn = fn.float()
    precision = tp / (tp + fp + EPS)
    recall = tp / (tp + fn + EPS)
    f1 = 2.0 * precision * recall / (precision + recall + EPS)
    iou = tp / (tp + fp + fn + EPS)
    accuracy = (tp + tn) / (tp + tn + fp + fn + EPS)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "accuracy": accuracy,
    }


@torch.no_grad()
def binary_metrics_at_threshold(
    score: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Compute binary metrics after applying one operating threshold."""
    score = _as_score(score)
    target = _as_binary_target(target)
    pred = (score > threshold).float()
    tp = torch.sum(pred * target)
    tn = torch.sum((1.0 - pred) * (1.0 - target))
    fp = torch.sum(pred * (1.0 - target))
    fn = torch.sum((1.0 - pred) * target)
    return binary_metrics_from_confusion(tp, tn, fp, fn)


def _nan_metrics() -> dict[str, float]:
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


@torch.no_grad()
def binary_ranking_metrics(score: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """Compute ranking metrics and the best F1 threshold.

    AUC/AP use the full score ranking instead of a fixed 0.5 threshold. This is
    important when validation scores are well ordered but need calibration.
    """
    score = _as_score(score).cpu()
    target = _as_binary_target(target).cpu()
    valid = torch.isfinite(score)
    score = score[valid]
    target = target[valid]
    if score.numel() == 0:
        return _nan_metrics()

    positives = target.sum()
    negatives = target.numel() - positives
    if positives <= 0 or negatives <= 0:
        return _nan_metrics()

    order = torch.argsort(score, descending=True)
    sorted_score = score[order]
    sorted_target = target[order]

    # Collapse equal scores so ROC/PR curves move only at real threshold changes.
    is_last_in_tie = torch.ones(sorted_score.numel(), dtype=torch.bool)
    is_last_in_tie[:-1] = sorted_score[:-1] != sorted_score[1:]

    tp = torch.cumsum(sorted_target, dim=0)[is_last_in_tie]
    fp = torch.cumsum(1.0 - sorted_target, dim=0)[is_last_in_tie]
    thresholds = sorted_score[is_last_in_tie]

    precision = tp / (tp + fp + EPS)
    recall = tp / (positives + EPS)
    f1 = 2.0 * precision * recall / (precision + recall + EPS)
    iou = tp / (positives + fp + EPS)
    accuracy = (tp + (negatives - fp)) / (positives + negatives + EPS)

    fpr = fp / (negatives + EPS)
    tpr = recall
    auc = torch.trapz(
        torch.cat([torch.zeros(1), tpr]),
        torch.cat([torch.zeros(1), fpr]),
    )

    previous_recall = torch.cat([torch.zeros(1), recall[:-1]])
    ap = torch.sum((recall - previous_recall) * precision)

    best_idx = torch.argmax(f1)
    return {
        "auc": float(auc),
        "ap": float(ap),
        "best_threshold": float(thresholds[best_idx]),
        "best_precision": float(precision[best_idx]),
        "best_recall": float(recall[best_idx]),
        "best_f1": float(f1[best_idx]),
        "best_iou": float(iou[best_idx]),
        "best_accuracy": float(accuracy[best_idx]),
    }


@torch.no_grad()
def pixel_confusion(pred_mask: torch.Tensor, target_mask: torch.Tensor, threshold: float = 0.5) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pred = (pred_mask > threshold).float()
    target = (target_mask > 0.5).float()
    dims = tuple(range(1, pred.ndim))
    tp = torch.sum(pred * target, dim=dims)
    tn = torch.sum((1.0 - pred) * (1.0 - target), dim=dims)
    fp = torch.sum(pred * (1.0 - target), dim=dims)
    fn = torch.sum((1.0 - pred) * target, dim=dims)
    return tp, tn, fp, fn


@torch.no_grad()
def pixel_f1(pred_mask: torch.Tensor, target_mask: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    tp, _, fp, fn = pixel_confusion(pred_mask, target_mask, threshold)
    return binary_metrics_from_confusion(tp, torch.zeros_like(tp), fp, fn)["f1"]


@torch.no_grad()
def image_accuracy(pred_score: torch.Tensor, label: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    pred = (pred_score > threshold).float()
    label = label.float()
    return (pred == label).float().mean()
