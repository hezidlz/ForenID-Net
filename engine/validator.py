from __future__ import annotations

from collections import defaultdict

import torch
from tqdm import tqdm

from utils.metrics import binary_metrics_at_threshold, binary_metrics_from_confusion, binary_ranking_metrics, image_accuracy, pixel_confusion


def _to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for key, value in batch.items():
        out[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return out


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader,
    loss_fn: torch.nn.Module,
    device: torch.device,
    max_metric_pixels: int | None = 2_000_000,
    small_region_ratio: float = 0.02,
) -> dict[str, float]:
    model.eval()
    meters: dict[str, list[float]] = defaultdict(list)
    image_scores: list[torch.Tensor] = []
    image_labels: list[torch.Tensor] = []
    pixel_scores: list[torch.Tensor] = []
    pixel_labels: list[torch.Tensor] = []
    pixel_sample_limit = None
    if max_metric_pixels is not None and max_metric_pixels > 0:
        pixel_sample_limit = max(1, max_metric_pixels // max(1, len(loader)))

    for batch in tqdm(loader, desc="valid", leave=False):
        batch = _to_device(batch, device)
        outputs = model(batch["image"])
        losses = loss_fn(
            outputs,
            mask=batch.get("mask"),
            label=batch.get("label"),
            has_mask=batch.get("has_mask"),
        )
        for key, value in losses.items():
            meters[key].append(float(value.detach().cpu()))

        pred_mask = outputs["pred_mask"]
        has_mask = batch["has_mask"].bool()
        if has_mask.any():
            valid_pred = pred_mask[has_mask]
            valid_target = batch["mask"][has_mask]
            tp, tn, fp, fn = pixel_confusion(valid_pred, valid_target, threshold=0.5)
            pixel_stats = binary_metrics_from_confusion(tp, tn, fp, fn)
            for name in ("precision", "recall", "f1", "iou"):
                meters[f"pixel_{name}"].extend([float(x.cpu()) for x in pixel_stats[name]])

            # Small edited regions are common in documents; track their recall separately.
            area_ratio = valid_target.float().flatten(1).mean(dim=1)
            small = area_ratio <= small_region_ratio
            if small.any():
                meters["pixel_small_region_recall"].extend(
                    [float(x.cpu()) for x in pixel_stats["recall"][small]]
                )

            flat_pred = valid_pred.detach().flatten().cpu()
            flat_target = valid_target.detach().flatten().cpu()
            if pixel_sample_limit is not None and flat_pred.numel() > pixel_sample_limit:
                sample_idx = torch.linspace(0, flat_pred.numel() - 1, steps=pixel_sample_limit).long()
                flat_pred = flat_pred[sample_idx]
                flat_target = flat_target[sample_idx]
            pixel_scores.append(flat_pred)
            pixel_labels.append(flat_target)

        score = outputs["pred_score"].detach().flatten().cpu()
        label = batch["label"].detach().flatten().cpu()
        image_scores.append(score)
        image_labels.append(label)
        image_acc = image_accuracy(outputs["pred_score"], batch["label"])
        meters["image_acc"].append(float(image_acc.cpu()))
        image_stats = binary_metrics_at_threshold(score, label, threshold=0.5)
        for name in ("precision", "recall", "f1"):
            meters[f"image_{name}"].append(float(image_stats[name].cpu()))

    stats = {key: sum(values) / max(1, len(values)) for key, values in meters.items()}

    if image_scores:
        image_rank = binary_ranking_metrics(torch.cat(image_scores), torch.cat(image_labels))
        stats.update(
            {
                "image_auc": image_rank["auc"],
                "image_ap": image_rank["ap"],
                "best_image_threshold": image_rank["best_threshold"],
                "image_precision_best": image_rank["best_precision"],
                "image_recall_best": image_rank["best_recall"],
                "image_f1_best": image_rank["best_f1"],
                "image_acc_best": image_rank["best_accuracy"],
            }
        )

    if pixel_scores:
        pixel_score = torch.cat(pixel_scores)
        pixel_label = torch.cat(pixel_labels)
        pixel_rank = binary_ranking_metrics(pixel_score, pixel_label)
        stats.update(
            {
                "pixel_auc": pixel_rank["auc"],
                "pixel_ap": pixel_rank["ap"],
                "best_mask_threshold": pixel_rank["best_threshold"],
                "pixel_precision_best": pixel_rank["best_precision"],
                "pixel_recall_best": pixel_rank["best_recall"],
                "pixel_f1_best": pixel_rank["best_f1"],
                "pixel_iou_best": pixel_rank["best_iou"],
                "pixel_metric_pixels": float(pixel_score.numel()),
            }
        )

    return stats
