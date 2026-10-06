from __future__ import annotations

import argparse
from glob import glob
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from datasets.transforms import DocLiteTransform, TransformConfig, get_letterbox_params
from models import build_model_from_config
from utils.checkpoint import load_checkpoint
from utils.config import load_config, merge_cli_overrides


def resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def list_images(input_path: str) -> list[Path]:
    if "*" in input_path:
        return [Path(p) for p in glob(input_path, recursive=True) if Path(p).is_file()]
    path = Path(input_path)
    if path.is_file():
        return [path]
    if path.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
        return [p for p in path.rglob("*") if p.suffix.lower() in exts]
    raise ValueError(f"Input not found: {input_path}")


def image_to_tensor(
    image: Image.Image,
    image_size: int,
    keep_aspect_ratio: bool = True,
    pad_value: int = 255,
) -> tuple[torch.Tensor, dict[str, int | float] | None]:
    meta = get_letterbox_params(image.size, image_size) if keep_aspect_ratio else None
    transform = DocLiteTransform(
        TransformConfig(
            image_size=image_size,
            train=False,
            keep_aspect_ratio=keep_aspect_ratio,
            pad_value=pad_value,
        )
    )
    image, _ = transform(image, None)
    arr = np.asarray(image).astype(np.float32) / 255.0
    return torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0), meta


def remove_letterbox(mask: torch.Tensor, meta: dict[str, int | float] | None) -> torch.Tensor:
    if meta is None:
        return mask
    pad_top = int(meta["pad_top"])
    pad_left = int(meta["pad_left"])
    resized_height = int(meta["resized_height"])
    resized_width = int(meta["resized_width"])
    return mask[..., pad_top : pad_top + resized_height, pad_left : pad_left + resized_width]


def image_to_tensor_no_resize(image: Image.Image) -> torch.Tensor:
    image = image.convert("RGB")
    arr = np.asarray(image).astype(np.float32) / 255.0
    return torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).contiguous()


def tile_starts(length: int, tile_size: int, overlap: int) -> list[int]:
    if length <= 0:
        raise ValueError(f"Invalid length: {length}")
    if tile_size <= 0:
        raise ValueError(f"tile_size must be positive, got {tile_size}")
    if overlap < 0 or overlap >= tile_size:
        raise ValueError(f"tile_overlap must be in [0, tile_size), got {overlap}")
    if length <= tile_size:
        return [0]

    step = tile_size - overlap
    starts = list(range(0, length - tile_size + 1, step))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def tile_weight(height: int, width: int, device: torch.device) -> torch.Tensor:
    if height <= 1:
        y = torch.ones(height, device=device)
    else:
        y = torch.hann_window(height, periodic=False, device=device)
    if width <= 1:
        x = torch.ones(width, device=device)
    else:
        x = torch.hann_window(width, periodic=False, device=device)
    weight = torch.outer(y, x).clamp_min(0.05)
    return weight.view(1, 1, height, width)


@torch.no_grad()
def infer_resized(
    model: torch.nn.Module,
    image: Image.Image,
    device: torch.device,
    image_size: int,
    keep_aspect_ratio: bool,
    pad_value: int,
) -> tuple[np.ndarray, float, dict[str, object]]:
    original_size = image.size[::-1]
    tensor, letterbox_meta = image_to_tensor(
        image,
        image_size,
        keep_aspect_ratio=keep_aspect_ratio,
        pad_value=pad_value,
    )
    outputs = model(tensor.to(device))
    mask = remove_letterbox(outputs["pred_mask"], letterbox_meta)
    mask = F.interpolate(mask, size=original_size, mode="bilinear", align_corners=False)
    mask_np = mask[0, 0].cpu().numpy().astype(np.float32)
    score = float(outputs["pred_score"][0].cpu())
    return mask_np, score, {"mode": "resized", "letterbox_meta": letterbox_meta}


@torch.no_grad()
def infer_sliding_window(
    model: torch.nn.Module,
    image: Image.Image,
    device: torch.device,
    tile_size: int,
    tile_overlap: int,
) -> tuple[np.ndarray, float, dict[str, object]]:
    width, height = image.size
    xs = tile_starts(width, tile_size, tile_overlap)
    ys = tile_starts(height, tile_size, tile_overlap)
    mask_acc = torch.zeros(1, 1, height, width, dtype=torch.float32)
    weight_acc = torch.zeros_like(mask_acc)
    tile_scores: list[float] = []

    for top in ys:
        for left in xs:
            right = min(left + tile_size, width)
            bottom = min(top + tile_size, height)
            tile = image.crop((left, top, right, bottom))
            tensor = image_to_tensor_no_resize(tile).to(device)
            outputs = model(tensor)
            pred_mask = outputs["pred_mask"]
            tile_h, tile_w = bottom - top, right - left
            if pred_mask.shape[-2:] != (tile_h, tile_w):
                pred_mask = F.interpolate(pred_mask, size=(tile_h, tile_w), mode="bilinear", align_corners=False)

            weight = tile_weight(tile_h, tile_w, pred_mask.device)
            mask_acc[..., top:bottom, left:right] += (pred_mask * weight).cpu()
            weight_acc[..., top:bottom, left:right] += weight.cpu()
            tile_scores.append(float(outputs["pred_score"][0].cpu()))

    mask_np = (mask_acc / weight_acc.clamp_min(1e-6))[0, 0].numpy().astype(np.float32)
    score = max(tile_scores) if tile_scores else 0.0
    meta = {
        "mode": "sliding_window",
        "tile_size": tile_size,
        "tile_overlap": tile_overlap,
        "tiles_x": len(xs),
        "tiles_y": len(ys),
        "tile_scores": tile_scores,
    }
    return mask_np, score, meta


def colorize_mask(mask: np.ndarray) -> Image.Image:
    mask = np.clip(mask.astype(np.float32), 0.0, 1.0)
    colors = np.array(
        [
            [0, 0, 80],
            [0, 128, 255],
            [0, 255, 255],
            [255, 255, 0],
            [255, 0, 0],
        ],
        dtype=np.float32,
    )
    scaled = mask * (len(colors) - 1)
    lower = np.floor(scaled).astype(np.int32)
    lower = np.clip(lower, 0, len(colors) - 2)
    upper = lower + 1
    frac = (scaled - lower)[..., None]
    rgb = colors[lower] * (1.0 - frac) + colors[upper] * frac
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")


def make_overlay(image: Image.Image, heatmap: Image.Image, mask: np.ndarray, alpha: float) -> Image.Image:
    image_arr = np.asarray(image.convert("RGB")).astype(np.float32)
    heatmap_arr = np.asarray(heatmap.convert("RGB")).astype(np.float32)
    alpha_map = np.clip(mask.astype(np.float32), 0.0, 1.0)[..., None] * alpha
    overlay = image_arr * (1.0 - alpha_map) + heatmap_arr * alpha_map
    return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8), mode="RGB")


def save_visual_outputs(
    image: Image.Image,
    mask: np.ndarray,
    output_root: Path,
    image_name: str,
    overlay_alpha: float,
) -> dict[str, Path]:
    mask = np.clip(mask, 0.0, 1.0)
    mask_image = Image.fromarray((mask * 255.0).round().astype(np.uint8), mode="L")
    heatmap = colorize_mask(mask)
    overlay = make_overlay(image, heatmap, mask, overlay_alpha)

    paths = {
        "mask_png": output_root / f"{image_name}.mask.png",
        "heatmap_png": output_root / f"{image_name}.heatmap.png",
        "overlay_png": output_root / f"{image_name}.overlay.png",
    }
    mask_image.save(paths["mask_png"])
    heatmap.save(paths["heatmap_png"])
    overlay.save(paths["overlay_png"])
    return paths


def main() -> None:
    parser = argparse.ArgumentParser("Infer with ForenID-Net")
    parser.add_argument("--config", default="configs/infer.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="outputs/infer")
    parser.add_argument("--overlay-alpha", type=float, default=0.45)
    parser.add_argument("--sliding-window", action="store_true", help="Run high-resolution tiled inference.")
    parser.add_argument("--no-sliding-window", action="store_true", help="Disable sliding-window mode from config.")
    parser.add_argument("--tile-size", type=int, default=None, help="Tile size for sliding-window inference.")
    parser.add_argument("--tile-overlap", type=int, default=None, help="Tile overlap in pixels.")
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg = merge_cli_overrides(load_config(args.config), args.opts)
    device = resolve_device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    data_cfg = cfg.get("data", {})
    infer_cfg = cfg.get("infer", {})
    image_size = int(data_cfg.get("image_size", 512))
    keep_aspect_ratio = bool(data_cfg.get("keep_aspect_ratio", True))
    pad_value = int(data_cfg.get("pad_value", 255))
    sliding_window = bool(infer_cfg.get("sliding_window", False))
    if args.sliding_window:
        sliding_window = True
    if args.no_sliding_window:
        sliding_window = False
    tile_size = args.tile_size if args.tile_size is not None else int(infer_cfg.get("tile_size", image_size))
    tile_overlap = args.tile_overlap if args.tile_overlap is not None else int(infer_cfg.get("tile_overlap", tile_size // 4))

    model = build_model_from_config(cfg).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device, strict=False)
    model.eval()

    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    images = list_images(args.input)

    with torch.no_grad():
        for image_path in images:
            pil = Image.open(image_path).convert("RGB")
            if sliding_window:
                mask_np, score, infer_meta = infer_sliding_window(
                    model,
                    pil,
                    device,
                    tile_size=tile_size,
                    tile_overlap=tile_overlap,
                )
            else:
                mask_np, score, infer_meta = infer_resized(
                    model,
                    pil,
                    device,
                    image_size=image_size,
                    keep_aspect_ratio=keep_aspect_ratio,
                    pad_value=pad_value,
                )
            visual_paths = save_visual_outputs(
                pil,
                mask_np,
                output_root,
                image_path.name,
                overlay_alpha=float(np.clip(args.overlay_alpha, 0.0, 1.0)),
            )

            out_path = output_root / (image_path.name + ".npz")
            np.savez(
                out_path,
                mask=mask_np,
                score=score,
                imgsize=pil.size[::-1],
                image=str(image_path),
                infer_meta=str(infer_meta),
                tile_scores=np.asarray(infer_meta.get("tile_scores", []), dtype=np.float32),
                mask_png=str(visual_paths["mask_png"]),
                heatmap_png=str(visual_paths["heatmap_png"]),
                overlay_png=str(visual_paths["overlay_png"]),
            )
            print(
                f"{image_path} -> score={score:.4f}, "
                f"saved={out_path}, overlay={visual_paths['overlay_png']}"
            )


if __name__ == "__main__":
    main()
