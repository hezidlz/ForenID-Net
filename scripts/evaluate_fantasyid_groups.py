from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.collate import forgery_collate
from datasets.doc_forgery_dataset import DocForgeryDataset
from datasets.transforms import build_transform
from engine.validator import validate
from losses import DocLiteLoss
from models import build_model_from_config
from utils.checkpoint import load_checkpoint
from utils.config import load_config, merge_cli_overrides


SUMMARY_KEYS = [
    "loss_total",
    "image_auc",
    "image_ap",
    "image_acc",
    "image_acc_best",
    "image_f1",
    "image_f1_best",
    "best_image_threshold",
    "pixel_auc",
    "pixel_ap",
    "pixel_f1",
    "pixel_f1_best",
    "pixel_iou",
    "pixel_iou_best",
    "best_mask_threshold",
    "pixel_small_region_recall",
]


def resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def normalize_rel(path: str) -> str:
    return Path(path.replace("\\", "/")).as_posix()


def bool_from_csv(value: str) -> bool:
    return value.strip().lower() == "true"


def safe_name(value: str) -> str:
    value = value.strip().lower() or "unknown"
    value = re.sub(r"[^a-z0-9_.-]+", "_", value)
    return value.strip("_") or "unknown"


def parse_txt_line(line: str) -> tuple[str, str | None, int]:
    parts = [part.strip() for part in line.strip().split(",")]
    if len(parts) < 3:
        raise ValueError(f"Expected image,mask,label row, got: {line!r}")
    image = normalize_rel(parts[0])
    mask = None if parts[1].lower() in {"", "none", "null", "negative"} else normalize_rel(parts[1])
    label = int(float(parts[2]))
    return image, mask, label


def read_base_list(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            image, _, _ = parse_txt_line(line)
            mapping[image] = line
    return mapping


def read_fantasyid_csv(path: Path, base_map: dict[str, str]) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    missing = 0
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            image = normalize_rel(row["path"])
            list_line = base_map.get(image)
            if list_line is None:
                missing += 1
                continue
            is_attack = bool_from_csv(row.get("is_attack", "False"))
            attack_type = row.get("attack_type", "").strip().lower() or "bonafide"
            rows.append(
                {
                    "image": image,
                    "line": list_line,
                    "is_attack": is_attack,
                    "attack_type": attack_type if is_attack else "bonafide",
                    "device": device_from_path(image),
                }
            )
    return rows, missing


def device_from_path(image_rel: str) -> str:
    parts = Path(image_rel).parts
    if len(parts) >= 2:
        return parts[-2].lower()
    return "unknown"


def label_from_line(line: str) -> int:
    _, _, label = parse_txt_line(line)
    return label


def write_group_list(path: Path, rows: list[dict[str, Any]]) -> tuple[int, int, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    kept_lines: list[str] = []
    for row in rows:
        image = str(row["image"])
        if image in seen:
            continue
        seen.add(image)
        kept_lines.append(str(row["line"]))

    fake = 0
    real = 0
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for line in kept_lines:
            label = label_from_line(line)
            fake += int(label == 1)
            real += int(label == 0)
            f.write(line + "\n")
    return len(kept_lines), fake, real


def build_groups(rows: list[dict[str, Any]], include_attack_device: bool) -> list[tuple[str, str, list[dict[str, Any]]]]:
    groups: list[tuple[str, str, list[dict[str, Any]]]] = []
    all_real = [row for row in rows if not row["is_attack"]]
    groups.append(("overall", "all", rows))

    attack_types = sorted({str(row["attack_type"]) for row in rows if row["is_attack"]})
    for attack_type in attack_types:
        attack_rows = [row for row in rows if row["is_attack"] and row["attack_type"] == attack_type]
        groups.append(("attack_type", attack_type, attack_rows + all_real))

    devices = sorted({str(row["device"]) for row in rows})
    for device in devices:
        groups.append(("device", device, [row for row in rows if row["device"] == device]))

    if include_attack_device:
        for attack_type in attack_types:
            for device in devices:
                attack_rows = [
                    row
                    for row in rows
                    if row["is_attack"] and row["attack_type"] == attack_type and row["device"] == device
                ]
                real_rows = [row for row in all_real if row["device"] == device]
                if attack_rows and real_rows:
                    groups.append(("attack_device", f"{attack_type}_{device}", attack_rows + real_rows))
    return groups


def evaluate_list(
    list_path: Path,
    root: Path,
    cfg: dict[str, Any],
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    max_metric_pixels: int | None,
    small_region_ratio: float,
) -> dict[str, float]:
    data_cfg = cfg.get("data", {})
    transform = build_transform(data_cfg, train=False)
    dataset = DocForgeryDataset(list_path, transform=transform, root=root)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=forgery_collate,
    )
    return validate(
        model,
        loader,
        loss_fn,
        device,
        max_metric_pixels=max_metric_pixels,
        small_region_ratio=small_region_ratio,
    )


def number_or_blank(value: Any) -> str:
    if value is None:
        return ""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(value):
        return ""
    return f"{value:.6g}"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["group_type", "group", "samples", "fake", "real"] + SUMMARY_KEYS
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def print_markdown(rows: list[dict[str, Any]]) -> None:
    cols = [
        "group_type",
        "group",
        "samples",
        "fake",
        "real",
        "image_auc",
        "image_ap",
        "image_f1_best",
        "pixel_auc",
        "pixel_ap",
        "pixel_f1_best",
    ]
    print("| " + " | ".join(cols) + " |")
    print("| " + " | ".join(["---"] * len(cols)) + " |")
    for row in rows:
        print("| " + " | ".join(number_or_blank(row.get(col)) for col in cols) + " |")


def main() -> None:
    parser = argparse.ArgumentParser("Evaluate FantasyID by attack type and device.")
    parser.add_argument("--config", default="configs/forenid_sparsevit.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--root", default="datasets/FantasyID/FantasyID")
    parser.add_argument("--csv", default="test.csv", help="FantasyID csv under --root.")
    parser.add_argument("--base-list", default="data_lists/fantasyid_valid.txt")
    parser.add_argument("--out-dir", default="outputs/fantasyid_group_eval")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-metric-pixels", type=int, default=None)
    parser.add_argument("--small-region-ratio", type=float, default=None)
    parser.add_argument("--include-attack-device", action="store_true")
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg = merge_cli_overrides(load_config(args.config), args.opts)
    root = Path(args.root)
    out_dir = Path(args.out_dir)
    list_dir = out_dir / "lists"
    csv_path = root / args.csv
    base_list = Path(args.base_list)

    base_map = read_base_list(base_list)
    rows, missing = read_fantasyid_csv(csv_path, base_map)
    if missing:
        print(f"warning: {missing} csv rows were not present in {base_list}")
    if not rows:
        raise ValueError("No FantasyID rows matched the base list.")

    device = resolve_device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model_from_config(cfg).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device, strict=False)
    loss_fn = DocLiteLoss().to(device)
    eval_cfg = cfg.get("eval", {})
    max_metric_pixels = args.max_metric_pixels
    if max_metric_pixels is None:
        max_metric_pixels = eval_cfg.get("max_metric_pixels", 2_000_000)
    small_region_ratio = (
        args.small_region_ratio
        if args.small_region_ratio is not None
        else float(eval_cfg.get("small_region_ratio", 0.02))
    )

    results: list[dict[str, Any]] = []
    groups = build_groups(rows, include_attack_device=args.include_attack_device)
    for group_type, group_name, group_rows in groups:
        list_name = f"{safe_name(group_type)}__{safe_name(group_name)}.txt"
        group_list = list_dir / list_name
        samples, fake, real = write_group_list(group_list, group_rows)
        if samples == 0:
            continue
        print(f"evaluating {group_type}/{group_name}: samples={samples}, fake={fake}, real={real}")
        stats = evaluate_list(
            group_list,
            root=root,
            cfg=cfg,
            model=model,
            loss_fn=loss_fn,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_metric_pixels=max_metric_pixels,
            small_region_ratio=small_region_ratio,
        )
        row: dict[str, Any] = {
            "group_type": group_type,
            "group": group_name,
            "samples": samples,
            "fake": fake,
            "real": real,
        }
        row.update(stats)
        results.append(row)

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "fantasyid_group_metrics.json"
    csv_path_out = out_dir / "fantasyid_group_metrics.csv"
    json_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    write_csv(csv_path_out, results)
    print(f"wrote: {json_path}")
    print(f"wrote: {csv_path_out}")
    print_markdown(results)


if __name__ == "__main__":
    main()
