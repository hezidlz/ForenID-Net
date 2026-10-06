from __future__ import annotations

import argparse
import json

import torch
from torch.utils.data import DataLoader

from datasets.collate import forgery_collate
from datasets.doc_forgery_dataset import DocForgeryDataset
from datasets.general_forgery_dataset import GeneralForgeryDataset
from datasets.transforms import build_transform
from engine.validator import validate
from losses import DocLiteLoss
from models import build_model_from_config
from utils.checkpoint import load_checkpoint
from utils.config import load_config, merge_cli_overrides


def resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def main() -> None:
    parser = argparse.ArgumentParser(
        "Development-set diagnostics for ForenID-Net",
        description=(
            "Compute ranking metrics and select diagnostic thresholds on a development "
            "set. For manuscript test results, use "
            "scripts/evaluate_revision_protocol.py so thresholds are frozen before "
            "test-set evaluation."
        ),
    )
    parser.add_argument("--config", default="configs/infer.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--list", required=True, help="txt/json annotation list")
    parser.add_argument("--type", choices=["general", "doc"], default="general")
    parser.add_argument("--root", default=None)
    parser.add_argument("--max-metric-pixels", type=int, default=None)
    parser.add_argument("--small-region-ratio", type=float, default=None)
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg = merge_cli_overrides(load_config(args.config), args.opts)
    device = resolve_device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    data_cfg = cfg.get("data", {})
    transform = build_transform(data_cfg, train=False)
    dataset_cls = GeneralForgeryDataset if args.type == "general" else DocForgeryDataset
    dataset = dataset_cls(args.list, transform=transform, root=args.root)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=forgery_collate)

    model = build_model_from_config(cfg).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device, strict=False)
    loss_fn = DocLiteLoss().to(device)
    eval_cfg = cfg.get("eval", {})
    max_metric_pixels = (
        args.max_metric_pixels
        if args.max_metric_pixels is not None
        else eval_cfg.get("max_metric_pixels", 2_000_000)
    )
    small_region_ratio = (
        args.small_region_ratio
        if args.small_region_ratio is not None
        else float(eval_cfg.get("small_region_ratio", 0.02))
    )
    stats = validate(
        model,
        loader,
        loss_fn,
        device,
        max_metric_pixels=max_metric_pixels,
        small_region_ratio=small_region_ratio,
    )
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
