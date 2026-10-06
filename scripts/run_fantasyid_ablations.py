from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.config import load_config, set_by_dotted_key


SUMMARY_KEYS = [
    "image_auc",
    "image_ap",
    "image_f1_best",
    "image_acc_best",
    "pixel_auc",
    "pixel_ap",
    "pixel_f1_best",
    "best_image_threshold",
    "best_mask_threshold",
]


def format_command(command: list[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(command)
    return shlex.join(command)


def run_stream(command: list[str], cwd: Path, dry_run: bool) -> None:
    print("\n> " + format_command(command), flush=True)
    if dry_run:
        return
    proc = subprocess.Popen(command, cwd=str(cwd))
    code = proc.wait()
    if code != 0:
        raise RuntimeError(f"Command failed with exit code {code}: {format_command(command)}")


def extract_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for idx, char in enumerate(text):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    raise ValueError("Could not find a JSON object in evaluation output.")


def run_capture_json(command: list[str], cwd: Path, log_path: Path, dry_run: bool) -> dict[str, Any]:
    print("\n> " + format_command(command), flush=True)
    if dry_run:
        return {}
    proc = subprocess.run(
        command,
        cwd=str(cwd),
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(proc.stdout, encoding="utf-8")
    print(proc.stdout)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {proc.returncode}: {format_command(command)}")
    return extract_json_object(proc.stdout)


def ensure_model_only_checkpoint(checkpoint_path: Path, output_root: Path, dry_run: bool) -> Path:
    if checkpoint_path.name.endswith("_model_only.pth"):
        return checkpoint_path
    if dry_run:
        return output_root / f"{checkpoint_path.stem}_model_only.pth"

    import torch

    payload = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(payload, dict) and set(payload.keys()) == {"model"}:
        return checkpoint_path

    if isinstance(payload, dict):
        state_dict = payload.get("model", payload.get("state_dict", payload))
    else:
        state_dict = payload

    output_root.mkdir(parents=True, exist_ok=True)
    dst = output_root / f"{checkpoint_path.stem}_model_only.pth"
    torch.save({"model": state_dict}, dst)
    print(f"wrote model-only checkpoint: {dst}")
    return dst


def write_config(base_config: Path, output_path: Path, overrides: dict[str, Any]) -> None:
    cfg = load_config(base_config)
    for key, value in overrides.items():
        set_by_dotted_key(cfg, key, value)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="\n") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)


def variant_overrides(args: argparse.Namespace, name: str, output_dir: Path) -> tuple[dict[str, Any], bool]:
    common: dict[str, Any] = {
        "train.output_dir": output_dir.as_posix(),
        "train.epochs": args.epochs,
        "train.lr": args.lr,
        "train.best_metric": args.best_metric,
        "data.heavy_degrade": args.full_heavy_degrade,
        "model.use_noiseprint": True,
        "model.fusion_mode": "learned_gate",
    }
    if args.min_lr is not None:
        common["train.min_lr"] = args.min_lr
    if args.batch_size is not None:
        common["train.batch_size"] = args.batch_size
    if args.num_workers is not None:
        common["train.num_workers"] = args.num_workers

    train_required = True
    if name == "full":
        pass
    elif name == "scratch":
        pass
    elif name == "pretrain_only":
        train_required = False
    elif name == "rgb_only":
        common["model.use_noiseprint"] = False
        common["model.fusion_mode"] = "rgb_only"
    elif name == "noise_only":
        common["model.fusion_mode"] = "noise_only"
    elif name in {"simple_add", "simple_concat", "fixed_gate", "additive_gate", "learned_gate"}:
        common["model.fusion_mode"] = name
    elif name == "with_degrade":
        common["data.heavy_degrade"] = True
    elif name == "no_degrade":
        common["data.heavy_degrade"] = False
    else:
        raise ValueError(f"Unknown ablation variant: {name}")
    return common, train_required


def build_train_command(
    python_bin: str,
    config_path: Path,
    resume_path: Path | None,
) -> list[str]:
    command = [python_bin, "train.py", "--config", config_path.as_posix()]
    if resume_path is not None:
        command += ["--resume", resume_path.as_posix()]
    return command


def build_eval_command(
    python_bin: str,
    config_path: Path,
    checkpoint_path: Path,
    args: argparse.Namespace,
) -> list[str]:
    command = [
        python_bin,
        "evaluate.py",
        "--config",
        config_path.as_posix(),
        "--checkpoint",
        checkpoint_path.as_posix(),
        "--list",
        args.valid_list,
        "--type",
        "doc",
        "--root",
        args.root,
    ]
    if args.max_metric_pixels is not None:
        command += ["--max-metric-pixels", str(args.max_metric_pixels)]
    if args.small_region_ratio is not None:
        command += ["--small-region-ratio", str(args.small_region_ratio)]
    return command


def build_group_command(
    python_bin: str,
    config_path: Path,
    checkpoint_path: Path,
    out_dir: Path,
    args: argparse.Namespace,
) -> list[str]:
    command = [
        python_bin,
        "scripts/evaluate_fantasyid_groups.py",
        "--config",
        config_path.as_posix(),
        "--checkpoint",
        checkpoint_path.as_posix(),
        "--root",
        args.root,
        "--csv",
        args.fantasyid_csv,
        "--base-list",
        args.valid_list,
        "--out-dir",
        out_dir.as_posix(),
    ]
    if args.max_metric_pixels is not None:
        command += ["--max-metric-pixels", str(args.max_metric_pixels)]
    if args.small_region_ratio is not None:
        command += ["--small-region-ratio", str(args.small_region_ratio)]
    return command


def write_summary(rows: list[dict[str, Any]], output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "ablation_summary.csv"
    fieldnames = ["variant", "checkpoint", "config", "trained", "note"] + SUMMARY_KEYS
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    md_path = output_root / "ablation_summary.md"
    columns = ["variant"] + SUMMARY_KEYS
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        values = [str(row.get("variant", ""))]
        for key in SUMMARY_KEYS:
            value = row.get(key, "")
            values.append("" if value == "" else f"{float(value):.4f}")
        lines.append("| " + " | ".join(values) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote: {csv_path}")
    print(f"wrote: {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser("Run FantasyID ablation experiments for ForenID-Net.")
    parser.add_argument("--base-config", default="configs/forenid_mit_b2.yaml")
    parser.add_argument("--pretrained", default="outputs/pretrain_general_all/best_model_only.pth")
    parser.add_argument("--output-root", default="outputs/ablations_fantasyid")
    parser.add_argument("--root", default="data/FantasyID/FantasyID")
    parser.add_argument("--valid-list", default="splits/fantasyid_v1/lists/fantasyid_dev.txt")
    parser.add_argument("--fantasyid-csv", default="test.csv")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["full", "rgb_only", "noise_only", "simple_add", "simple_concat", "fixed_gate", "additive_gate"],
        choices=["full", "scratch", "pretrain_only", "rgb_only", "noise_only", "simple_add", "simple_concat", "fixed_gate", "additive_gate", "learned_gate", "with_degrade", "no_degrade"],
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--min-lr", type=float, default=None)
    parser.add_argument("--best-metric", default="image_auc")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--full-heavy-degrade", action="store_true")
    parser.add_argument("--max-metric-pixels", type=int, default=None)
    parser.add_argument("--small-region-ratio", type=float, default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--run-groups", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--force", action="store_true", help="Retrain even when best.pth already exists.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    args = parser.parse_args()

    base_config = Path(args.base_config)
    output_root = Path(args.output_root)
    pretrained = Path(args.pretrained)
    if not pretrained.exists() and pretrained.name.endswith("_model_only.pth"):
        fallback = pretrained.with_name(pretrained.name.replace("_model_only.pth", ".pth"))
        if fallback.exists():
            print(f"pretrained model-only checkpoint not found; using {fallback} and converting it.")
            pretrained = fallback
    pretrained_model_only = ensure_model_only_checkpoint(pretrained, output_root, dry_run=args.dry_run)

    rows: list[dict[str, Any]] = []
    for variant in args.variants:
        variant_dir = output_root / variant
        config_path = output_root / "configs" / f"{variant}.yaml"
        overrides, train_required = variant_overrides(args, variant, variant_dir)
        write_config(base_config, config_path, overrides)

        checkpoint_path = variant_dir / "best.pth"
        trained = False
        note = ""
        try:
            if train_required and not args.eval_only:
                if checkpoint_path.exists() and not args.force:
                    print(f"skip training {variant}: found {checkpoint_path}")
                else:
                    resume = None if variant == "scratch" else pretrained_model_only
                    run_stream(build_train_command(args.python, config_path, resume), REPO_ROOT, args.dry_run)
                    trained = True
            elif not train_required:
                checkpoint_path = pretrained_model_only
                note = "evaluated the general pretraining checkpoint without FantasyID finetuning"

            if train_required and not checkpoint_path.exists() and not args.dry_run:
                raise FileNotFoundError(f"Missing checkpoint for {variant}: {checkpoint_path}")

            eval_log = variant_dir / "eval_stdout.txt"
            metrics = run_capture_json(
                build_eval_command(args.python, config_path, checkpoint_path, args),
                REPO_ROOT,
                eval_log,
                args.dry_run,
            )
            metrics_path = variant_dir / "eval_metrics.json"
            if not args.dry_run:
                metrics_path.parent.mkdir(parents=True, exist_ok=True)
                metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
                print(f"wrote: {metrics_path}")

            if args.run_groups:
                run_stream(
                    build_group_command(args.python, config_path, checkpoint_path, variant_dir / "group_eval", args),
                    REPO_ROOT,
                    args.dry_run,
                )

            row: dict[str, Any] = {
                "variant": variant,
                "checkpoint": checkpoint_path.as_posix(),
                "config": config_path.as_posix(),
                "trained": int(trained),
                "note": note,
            }
            row.update({key: metrics.get(key, "") for key in SUMMARY_KEYS})
            rows.append(row)
        except Exception as exc:
            if not args.keep_going:
                raise
            print(f"warning: {variant} failed: {exc}", file=sys.stderr)
            rows.append(
                {
                    "variant": variant,
                    "checkpoint": checkpoint_path.as_posix(),
                    "config": config_path.as_posix(),
                    "trained": int(trained),
                    "note": f"failed: {exc}",
                }
            )

    write_summary(rows, output_root)


if __name__ == "__main__":
    main()
