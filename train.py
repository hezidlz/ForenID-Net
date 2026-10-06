from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from datasets.collate import forgery_collate
from datasets.doc_forgery_dataset import DocForgeryDataset
from datasets.general_forgery_dataset import GeneralForgeryDataset
from datasets.transforms import build_transform
from engine.scheduler import build_optimizer, cosine_lr, set_lr
from engine.trainer import train_one_epoch
from engine.validator import validate
from losses import DocLiteLoss
from models import build_model_from_config
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.config import load_config, merge_cli_overrides
from utils.seed import seed_everything


def resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def build_dataset(cfg: dict, split: str):
    data_cfg = cfg.get("data", {})
    split_cfg = data_cfg.get(split, {})
    annotation_path = split_cfg.get("list") or split_cfg.get("txt") or split_cfg.get("json")
    if not annotation_path:
        return None
    dataset_type = split_cfg.get("type", data_cfg.get("type", "general"))
    transform = build_transform({**data_cfg, **split_cfg}, train=(split == "train"))
    root = split_cfg.get("root", data_cfg.get("root"))
    if dataset_type == "doc":
        return DocForgeryDataset(annotation_path=annotation_path, transform=transform, root=root)
    if dataset_type == "general":
        return GeneralForgeryDataset(annotation_path=annotation_path, transform=transform, root=root)
    raise ValueError(f"Unknown dataset type: {dataset_type}")


def build_loader(dataset, cfg: dict, train: bool):
    train_cfg = cfg.get("train", {})
    batch_size = int(train_cfg.get("batch_size", 4 if train else 1))
    num_workers = int(train_cfg.get("num_workers", 4))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=train,
        collate_fn=forgery_collate,
    )


def main() -> None:
    parser = argparse.ArgumentParser("Train ForenID-Net")
    parser.add_argument("--config", default="configs/forenid_sparsevit.yaml")
    parser.add_argument("--resume", default="")
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg = merge_cli_overrides(load_config(args.config), args.opts)
    seed_everything(int(cfg.get("seed", 42)))

    device = resolve_device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    train_dataset = build_dataset(cfg, "train")
    valid_dataset = build_dataset(cfg, "valid")
    if train_dataset is None:
        raise ValueError("data.train.list/txt/json is required")
    print(f"train samples: {len(train_dataset)}")
    if valid_dataset is not None:
        print(f"valid samples: {len(valid_dataset)}")

    train_loader = build_loader(train_dataset, cfg, train=True)
    valid_loader = build_loader(valid_dataset, cfg, train=False) if valid_dataset is not None else None

    model = build_model_from_config(cfg).to(device)
    freeze_backbone_stages = int(cfg.get("train", {}).get("freeze_backbone_stages", 0))
    if freeze_backbone_stages:
        model.freeze_backbone_stages(freeze_backbone_stages)

    optimizer = build_optimizer(model, cfg)
    if args.resume:
        load_checkpoint(args.resume, model, optimizer=optimizer, map_location=device, strict=False)

    loss_cfg = cfg.get("loss", {})
    loss_fn = DocLiteLoss(
        mask_focal_weight=float(loss_cfg.get("mask_focal_weight", 1.0)),
        mask_dice_weight=float(loss_cfg.get("mask_dice_weight", 1.0)),
        score_weight=float(loss_cfg.get("score_weight", 0.5)),
        focal_alpha=float(loss_cfg.get("focal_alpha", 0.25)),
        focal_gamma=float(loss_cfg.get("focal_gamma", 2.0)),
    ).to(device)

    train_cfg = cfg.get("train", {})
    eval_cfg = cfg.get("eval", {})
    epochs = int(train_cfg.get("epochs", 60))
    base_lr = float(train_cfg.get("lr", 2e-4))
    min_lr = float(train_cfg.get("min_lr", 1e-6))
    warmup_epochs = int(train_cfg.get("warmup_epochs", 2))
    best_metric_name = str(train_cfg.get("best_metric", "pixel_f1"))
    output_dir = Path(train_cfg.get("output_dir", "outputs/default"))
    output_dir.mkdir(parents=True, exist_ok=True)

    best_metric = -1.0
    for epoch in range(epochs):
        lr = cosine_lr(base_lr, min_lr, epoch, epochs, warmup_epochs)
        set_lr(optimizer, lr)
        train_stats = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
            epoch,
            amp=bool(train_cfg.get("amp", True)),
        )
        print(f"epoch {epoch} train:", train_stats)

        metric = -train_stats.get("loss_total", 0.0)
        if valid_loader is not None:
            valid_stats = validate(
                model,
                valid_loader,
                loss_fn,
                device,
                max_metric_pixels=eval_cfg.get("max_metric_pixels", 2_000_000),
                small_region_ratio=float(eval_cfg.get("small_region_ratio", 0.02)),
            )
            print(f"epoch {epoch} valid:", valid_stats)
            metric = valid_stats.get(best_metric_name, valid_stats.get("pixel_f1", valid_stats.get("image_acc", metric)))
        else:
            valid_stats = {}

        save_checkpoint(output_dir / "last.pth", model, optimizer=optimizer, epoch=epoch, extra={"cfg": cfg})
        if metric > best_metric:
            best_metric = metric
            save_checkpoint(
                output_dir / "best.pth",
                model,
                optimizer=optimizer,
                epoch=epoch,
                extra={"cfg": cfg, "best_metric": best_metric, "best_metric_name": best_metric_name, "valid_stats": valid_stats},
            )


if __name__ == "__main__":
    main()
