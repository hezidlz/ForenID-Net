from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter
from tqdm import tqdm

from evaluate_revision_protocol import (
    bbox,
    box_iou,
    build_loader,
    confusion_metrics,
    image_metrics,
    localization_metrics,
    parse_metadata,
    parse_named_path,
    repository_imports,
    resolve_device,
)


STRESS_GRID: dict[str, list[float]] = {
    "jpeg": [100, 90, 70, 50, 30, 10],
    "gaussian_blur": [0.0, 0.5, 1.0, 2.0, 3.0],
    "downsample": [1.0, 0.75, 0.50, 0.35, 0.25],
    "gaussian_noise": [0.0, 0.01, 0.02, 0.04, 0.08],
    "gamma": [1.0, 0.8, 0.6, 1.2, 1.5],
    "moire_print_scan": [0.0, 1.0, 2.0, 3.0],
    "noise_consistency": [0.0, 0.35, 0.65, 0.90],
}


def stable_seed(path: str, perturbation: str, strength: float) -> int:
    digest = hashlib.sha256(f"{path}|{perturbation}|{strength}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def to_pil(image: torch.Tensor) -> Image.Image:
    array = image.detach().cpu().clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).numpy()
    return Image.fromarray(array, mode="RGB")


def from_pil(image: Image.Image, reference: torch.Tensor) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array.transpose(2, 0, 1)).to(device=reference.device, dtype=reference.dtype)


def jpeg_roundtrip(image: Image.Image, quality: int) -> Image.Image:
    stream = io.BytesIO()
    image.save(stream, format="JPEG", quality=int(quality), subsampling=2, optimize=False)
    stream.seek(0)
    with Image.open(stream) as decoded:
        return decoded.convert("RGB").copy()


def perturb_one(image: torch.Tensor, kind: str, strength: float, path: str) -> torch.Tensor:
    pil = to_pil(image)
    width, height = pil.size
    rng = np.random.default_rng(stable_seed(path, kind, strength))

    if kind == "jpeg":
        transformed = jpeg_roundtrip(pil, int(strength))
    elif kind == "gaussian_blur":
        transformed = pil.filter(ImageFilter.GaussianBlur(radius=float(strength)))
    elif kind == "downsample":
        if strength >= 0.999:
            transformed = pil
        else:
            small = pil.resize(
                (max(1, round(width * strength)), max(1, round(height * strength))),
                Image.Resampling.BICUBIC,
            )
            transformed = small.resize((width, height), Image.Resampling.BICUBIC)
    elif kind == "gaussian_noise":
        array = np.asarray(pil, dtype=np.float32) / 255.0
        array = np.clip(array + rng.normal(0.0, float(strength), size=array.shape), 0.0, 1.0)
        transformed = Image.fromarray(np.round(array * 255).astype(np.uint8), mode="RGB")
    elif kind == "gamma":
        array = np.asarray(pil, dtype=np.float32) / 255.0
        array = np.clip(array, 0.0, 1.0) ** float(strength)
        transformed = Image.fromarray(np.round(array * 255).astype(np.uint8), mode="RGB")
    elif kind == "moire_print_scan":
        level = int(round(strength))
        if level == 0:
            transformed = pil
        else:
            array = np.asarray(pil, dtype=np.float32) / 255.0
            yy, xx = np.mgrid[0:height, 0:width]
            period = {1: 28.0, 2: 18.0, 3: 12.0}[level]
            amplitude = {1: 0.018, 2: 0.035, 3: 0.060}[level]
            phase = rng.uniform(0, 2 * np.pi)
            pattern = np.sin(2 * np.pi * (xx / period + yy / (period * 1.7)) + phase)[..., None]
            array = np.clip(array + amplitude * pattern, 0.0, 1.0)
            transformed = Image.fromarray(np.round(array * 255).astype(np.uint8), mode="RGB")
            transformed = transformed.filter(ImageFilter.GaussianBlur(radius={1: 0.3, 2: 0.7, 3: 1.2}[level]))
            transformed = jpeg_roundtrip(transformed, {1: 90, 2: 75, 3: 55}[level])
    elif kind == "noise_consistency":
        # Transparent proxy for residual-rematching/evasion: attenuate the
        # high-frequency residual while preserving the low-frequency image.
        if strength <= 0:
            transformed = pil
        else:
            array = np.asarray(pil, dtype=np.float32) / 255.0
            smooth = np.asarray(pil.filter(ImageFilter.GaussianBlur(radius=1.0)), dtype=np.float32) / 255.0
            array = np.clip(smooth + (array - smooth) * (1.0 - float(strength)), 0.0, 1.0)
            transformed = Image.fromarray(np.round(array * 255).astype(np.uint8), mode="RGB")
    else:
        raise ValueError(f"Unknown perturbation: {kind}")
    return from_pil(transformed, image)


@torch.no_grad()
def evaluate_condition(model, loader, device, image_threshold: float, mask_threshold: float, kind: str, strength: float):
    model.eval()
    rows: list[dict[str, object]] = []
    for batch in tqdm(loader, desc=f"{kind}={strength:g}"):
        path = batch["path"][0]
        image = batch["image"].to(device, non_blocking=True)
        image[0] = perturb_one(image[0], kind, strength, path)
        outputs = model(image)
        score = float(outputs["pred_score"].detach().cpu().item())
        label = int(batch["label"].item())
        row: dict[str, object] = {
            **parse_metadata(path),
            "label": label,
            "score": score,
            "perturbation": kind,
            "strength": strength,
        }
        if bool(batch["has_mask"].item()):
            probability = outputs["pred_mask"].detach().cpu().numpy()[0, 0]
            target = batch["mask"].numpy()[0, 0] > 0.5
            prediction = probability >= mask_threshold
            pixel = confusion_metrics(prediction, target)
            row.update(
                {
                    "mask_area_ratio": float(target.mean()),
                    "boxmask_precision": pixel["precision"],
                    "boxmask_recall": pixel["recall"],
                    "boxmask_f1": pixel["f1"],
                    "boxmask_iou": pixel["iou"],
                    "enclosing_box_iou": box_iou(bbox(prediction), bbox(target)),
                    "pointing_game": float(target[np.unravel_index(np.argmax(probability), probability.shape)]),
                }
            )
        rows.append(row)
    frame = pd.DataFrame(rows)
    point = stress_curve_metrics(frame, image_threshold)
    return frame, point


def stress_curve_metrics(frame: pd.DataFrame, image_threshold: float) -> dict[str, float]:
    point = {**image_metrics(frame, image_threshold), **localization_metrics(frame)}
    real = frame.loc[frame["label"] == 0]
    point["real_specificity"] = image_metrics(real, image_threshold)["image_specificity"] if len(real) else math.nan
    for attack in sorted(set(frame.loc[frame["label"] == 1, "attack_type"]) - {"none"}):
        attacked = frame.loc[(frame["label"] == 1) & (frame["attack_type"] == attack)]
        predictions = (attacked["score"].to_numpy(dtype=float) >= image_threshold).astype(int)
        point[f"attack_{attack}_recall"] = float(predictions.mean()) if len(predictions) else math.nan
        localized = localization_metrics(attacked)
        point[f"attack_{attack}_pointing_game"] = localized["pointing_game"]
        point[f"attack_{attack}_enclosing_box_iou"] = localized["enclosing_box_iou"]
    return point


def stress_bootstrap_ci(frame: pd.DataFrame, image_threshold: float, iterations: int, seed: int):
    groups = {key: value for key, value in frame.groupby("group_key", sort=True)}
    names = np.array(sorted(groups), dtype=object)
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {}
    for _ in range(iterations):
        selected = rng.choice(names, size=len(names), replace=True)
        sample = pd.concat([groups[name] for name in selected], ignore_index=True)
        for metric, value in stress_curve_metrics(sample, image_threshold).items():
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                draws.setdefault(metric, []).append(float(value))
    return {
        metric: np.quantile(values, [0.025, 0.975]).tolist() if values else [math.nan, math.nan]
        for metric, values in draws.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-strength degradation and evasion stress test.")
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--test", action="append", type=parse_named_path, required=True, help="NAME=PATH")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--only", action="append", choices=sorted(STRESS_GRID))
    args = parser.parse_args()

    imports = repository_imports(args.repository)
    forgery_collate, DocForgeryDataset, build_transform, build_model_from_config, load_checkpoint, load_config = imports
    cfg = load_config(args.config)
    device = resolve_device(args.device)
    model = build_model_from_config(cfg).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device, strict=False)
    constructors = (forgery_collate, DocForgeryDataset, build_transform)
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    image_threshold = float(calibration["image_threshold"])
    mask_threshold = float(calibration["mask_threshold"])

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    kinds = args.only or list(STRESS_GRID)
    summary: dict[str, object] = {
        "checkpoint": str(args.checkpoint.resolve()),
        "calibration": str(args.calibration.resolve()),
        "grid": {kind: STRESS_GRID[kind] for kind in kinds},
        "tests": {},
        "notes": {
            "moire_print_scan": "Deterministic sinusoidal interference + blur + JPEG proxy, not a physical scanner benchmark.",
            "noise_consistency": "Deterministic high-frequency residual attenuation proxy, not an optimized white-box attack.",
        },
    }
    curve_rows: list[dict[str, object]] = []
    for test_index, (test_name, list_path) in enumerate(args.test):
        test_output: dict[str, object] = {}
        for kind_index, kind in enumerate(kinds):
            kind_output: dict[str, object] = {}
            loader = build_loader(list_path, args.root, cfg, constructors)
            for strength_index, strength in enumerate(STRESS_GRID[kind]):
                frame, point = evaluate_condition(
                    model,
                    loader,
                    device,
                    image_threshold,
                    mask_threshold,
                    kind,
                    strength,
                )
                safe_strength = str(strength).replace(".", "p")
                frame.to_csv(output / f"{test_name}_{kind}_{safe_strength}_per_image.csv", index=False)
                intervals = stress_bootstrap_ci(
                    frame,
                    image_threshold,
                    args.bootstrap,
                    args.seed + test_index * 1_000_000 + kind_index * 100_000 + strength_index * 1000,
                )
                kind_output[str(strength)] = {"metrics": point, "cluster_bootstrap_95ci": intervals}
                curve_rows.extend(
                    {
                        "test": test_name,
                        "perturbation": kind,
                        "strength": strength,
                        "metric": metric,
                        "value": value,
                    }
                    for metric, value in point.items()
                    if isinstance(value, (int, float))
                )
            test_output[kind] = kind_output
        summary["tests"][test_name] = test_output

    pd.DataFrame(curve_rows).to_csv(output / "stress_curves_long.csv", index=False)
    (output / "stress_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
