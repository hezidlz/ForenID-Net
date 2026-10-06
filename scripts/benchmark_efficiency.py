from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import statistics
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def bytes_to_mb(value: int | float | None) -> float | None:
    if value is None:
        return None
    return float(value) / (1024.0 * 1024.0)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * q
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return float(ordered[low])
    weight = position - low
    return float(ordered[low] * (1.0 - weight) + ordered[high] * weight)


def summarize_ms(values: list[float], batch_size: int = 1) -> dict[str, float]:
    if not values:
        return {
            "samples": 0.0,
            "mean_ms_per_batch": math.nan,
            "median_ms_per_batch": math.nan,
            "p90_ms_per_batch": math.nan,
            "p95_ms_per_batch": math.nan,
            "min_ms_per_batch": math.nan,
            "max_ms_per_batch": math.nan,
            "mean_ms_per_image": math.nan,
            "fps": math.nan,
        }
    mean_ms = statistics.fmean(values)
    return {
        "samples": float(len(values)),
        "mean_ms_per_batch": float(mean_ms),
        "median_ms_per_batch": float(statistics.median(values)),
        "p90_ms_per_batch": percentile(values, 0.90),
        "p95_ms_per_batch": percentile(values, 0.95),
        "min_ms_per_batch": float(min(values)),
        "max_ms_per_batch": float(max(values)),
        "mean_ms_per_image": float(mean_ms / max(1, batch_size)),
        "fps": float(max(1, batch_size) * 1000.0 / mean_ms) if mean_ms > 0 else math.nan,
    }


def flatten_dict(value: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, item in value.items():
        next_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, dict):
            flat.update(flatten_dict(item, next_key))
        else:
            flat[next_key] = item
    return flat


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def format_value(value: Any) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.6g}"
    if value is None:
        return ""
    return str(value)


def write_outputs(result: dict[str, Any], out_json: str | None, out_csv: str | None, out_md: str | None) -> None:
    if out_json:
        path = Path(out_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, sort_keys=True, default=json_default)
        print(f"wrote: {path}")

    if out_csv:
        path = Path(out_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        flat = flatten_dict(result)
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(flat.keys()))
            writer.writeheader()
            writer.writerow(flat)
        print(f"wrote: {path}")

    if out_md:
        path = Path(out_md)
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Efficiency Benchmark",
            "",
            "| Metric | Value |",
            "| --- | --- |",
        ]
        for key, item in flatten_dict(result).items():
            lines.append(f"| {key} | {format_value(item)} |")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"wrote: {path}")


def resolve_device(torch_mod: Any, requested: str) -> Any:
    if requested.startswith("cuda") and not torch_mod.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        return torch_mod.device("cpu")
    return torch_mod.device(requested)


def autocast_context(torch_mod: Any, device: Any, enabled: bool) -> Any:
    if not enabled or device.type != "cuda":
        return nullcontext()
    if hasattr(torch_mod, "amp") and hasattr(torch_mod.amp, "autocast"):
        return torch_mod.amp.autocast("cuda", enabled=True)
    return torch_mod.cuda.amp.autocast(enabled=True)


def cuda_sync(torch_mod: Any, device: Any) -> None:
    if device.type == "cuda":
        torch_mod.cuda.synchronize(device)


def reset_cuda_peak(torch_mod: Any, device: Any) -> None:
    if device.type == "cuda":
        torch_mod.cuda.empty_cache()
        torch_mod.cuda.reset_peak_memory_stats(device)


def cuda_memory_stats(torch_mod: Any, device: Any) -> dict[str, float | None]:
    if device.type != "cuda":
        return {
            "allocated_mb": None,
            "reserved_mb": None,
            "peak_allocated_mb": None,
            "peak_reserved_mb": None,
        }
    cuda_sync(torch_mod, device)
    return {
        "allocated_mb": bytes_to_mb(torch_mod.cuda.memory_allocated(device)),
        "reserved_mb": bytes_to_mb(torch_mod.cuda.memory_reserved(device)),
        "peak_allocated_mb": bytes_to_mb(torch_mod.cuda.max_memory_allocated(device)),
        "peak_reserved_mb": bytes_to_mb(torch_mod.cuda.max_memory_reserved(device)),
    }


def timed_forward(model: Any, tensor: Any, torch_mod: Any, device: Any, amp: bool) -> float:
    if device.type == "cuda":
        start = torch_mod.cuda.Event(enable_timing=True)
        end = torch_mod.cuda.Event(enable_timing=True)
        start.record()
        with autocast_context(torch_mod, device, amp):
            _ = model(tensor)
        end.record()
        torch_mod.cuda.synchronize(device)
        return float(start.elapsed_time(end))

    start_time = time.perf_counter()
    with autocast_context(torch_mod, device, amp):
        _ = model(tensor)
    return float((time.perf_counter() - start_time) * 1000.0)


def benchmark_synthetic(
    model: Any,
    torch_mod: Any,
    device: Any,
    image_size: int,
    batch_size: int,
    warmup: int,
    iters: int,
    amp: bool,
) -> dict[str, Any]:
    tensor = torch_mod.rand(batch_size, 3, image_size, image_size, device=device)
    model.eval()

    with torch_mod.inference_mode():
        for _ in range(max(0, warmup)):
            with autocast_context(torch_mod, device, amp):
                _ = model(tensor)
        cuda_sync(torch_mod, device)

        reset_cuda_peak(torch_mod, device)
        timings_ms = [
            timed_forward(model, tensor, torch_mod=torch_mod, device=device, amp=amp)
            for _ in range(max(1, iters))
        ]

    stats = summarize_ms(timings_ms, batch_size=batch_size)
    stats.update(
        {
            "image_size": image_size,
            "batch_size": batch_size,
            "warmup": warmup,
            "iters": iters,
            "amp": bool(amp),
            "cuda_memory": cuda_memory_stats(torch_mod, device),
        }
    )
    return stats


def resolve_sample_path(root: Path, image_path: str | Path) -> Path:
    path = Path(image_path)
    if path.is_absolute():
        return path
    return root / path


def image_to_tensor(image_path: Path, transform: Any, torch_mod: Any) -> Any:
    import numpy as np
    from PIL import Image

    with Image.open(image_path) as image:
        image, _ = transform(image.convert("RGB"), None)
    array = np.asarray(image).astype(np.float32) / 255.0
    return torch_mod.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0).contiguous()


def benchmark_real_images(
    model: Any,
    cfg: dict[str, Any],
    torch_mod: Any,
    device: Any,
    list_path: str,
    root: str | None,
    num_images: int,
    real_warmup: int,
    amp: bool,
) -> dict[str, Any]:
    from datasets.annotations import load_records
    from datasets.transforms import build_transform

    data_cfg = cfg.get("data", {})
    transform = build_transform(data_cfg, train=False)
    records = load_records(list_path)
    root_path = Path(root or data_cfg.get("root") or Path(list_path).parent)
    timed_records = records[: max(0, num_images)]
    warmup_records = records[: min(max(0, real_warmup), len(records))]

    model.eval()
    with torch_mod.inference_mode():
        for record in warmup_records:
            tensor = image_to_tensor(resolve_sample_path(root_path, record["image"]), transform, torch_mod).to(device)
            with autocast_context(torch_mod, device, amp):
                _ = model(tensor)
        cuda_sync(torch_mod, device)

        reset_cuda_peak(torch_mod, device)
        preprocess_ms: list[float] = []
        forward_ms: list[float] = []
        e2e_ms: list[float] = []

        for record in timed_records:
            image_path = resolve_sample_path(root_path, record["image"])
            start_total = time.perf_counter()
            start_preprocess = time.perf_counter()
            tensor = image_to_tensor(image_path, transform, torch_mod)
            preprocess_ms.append(float((time.perf_counter() - start_preprocess) * 1000.0))
            tensor = tensor.to(device, non_blocking=True)
            forward_ms.append(timed_forward(model, tensor, torch_mod=torch_mod, device=device, amp=amp))
            e2e_ms.append(float((time.perf_counter() - start_total) * 1000.0))

    return {
        "list": list_path,
        "root": str(root_path),
        "requested_images": num_images,
        "measured_images": len(timed_records),
        "warmup_images": len(warmup_records),
        "amp": bool(amp),
        "preprocess": summarize_ms(preprocess_ms, batch_size=1),
        "forward": summarize_ms(forward_ms, batch_size=1),
        "end_to_end": summarize_ms(e2e_ms, batch_size=1),
        "cuda_memory": cuda_memory_stats(torch_mod, device),
    }


def model_size_stats(model: Any, checkpoint: str | None) -> dict[str, Any]:
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    parameter_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
    buffer_bytes = sum(buffer.numel() * buffer.element_size() for buffer in model.buffers())

    checkpoint_size_mb = None
    if checkpoint and Path(checkpoint).exists():
        checkpoint_size_mb = bytes_to_mb(Path(checkpoint).stat().st_size)

    return {
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "total_params_m": float(total_params / 1_000_000.0),
        "trainable_params_m": float(trainable_params / 1_000_000.0),
        "parameter_memory_mb": bytes_to_mb(parameter_bytes),
        "buffer_memory_mb": bytes_to_mb(buffer_bytes),
        "checkpoint_file_mb": checkpoint_size_mb,
    }


def environment_stats(torch_mod: Any, device: Any) -> dict[str, Any]:
    env = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch_mod.__version__,
        "cuda_available": bool(torch_mod.cuda.is_available()),
        "cuda": getattr(torch_mod.version, "cuda", None),
        "cudnn": torch_mod.backends.cudnn.version() if torch_mod.backends.cudnn.is_available() else None,
        "device": str(device),
    }
    if device.type == "cuda":
        props = torch_mod.cuda.get_device_properties(device)
        env.update(
            {
                "gpu_name": torch_mod.cuda.get_device_name(device),
                "gpu_total_memory_mb": bytes_to_mb(props.total_memory),
                "gpu_multiprocessor_count": int(props.multi_processor_count),
            }
        )
    return env


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Benchmark ForenID-Net efficiency")
    parser.add_argument("--config", default="configs/forenid_sparsevit.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default=None, help="Default: config device, then cuda if available.")
    parser.add_argument("--image-size", type=int, default=None, help="Synthetic input size. Default: data.image_size.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--amp", action="store_true", help="Use CUDA autocast during inference.")
    parser.add_argument("--compile", action="store_true", dest="compile_model", help="Use torch.compile when available.")
    parser.add_argument("--num-threads", type=int, default=None, help="Set torch CPU intra-op thread count.")
    parser.add_argument("--list", default=None, help="Optional annotation list for real-image end-to-end timing.")
    parser.add_argument("--root", default=None, help="Dataset root for --list. Default: data.root from config.")
    parser.add_argument("--num-real-images", type=int, default=100)
    parser.add_argument("--real-warmup", type=int, default=5)
    parser.add_argument("--out-json", default="outputs/efficiency/forenid_net_efficiency.json")
    parser.add_argument("--out-csv", default="outputs/efficiency/forenid_net_efficiency.csv")
    parser.add_argument("--out-md", default="outputs/efficiency/forenid_net_efficiency.md")
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    return parser


def main() -> None:
    args = build_argparser().parse_args()

    import torch

    from models import build_model_from_config
    from utils.checkpoint import load_checkpoint
    from utils.config import load_config, merge_cli_overrides

    cfg = merge_cli_overrides(load_config(args.config), args.opts)
    if args.num_threads is not None:
        torch.set_num_threads(args.num_threads)
    requested_device = args.device or cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    device = resolve_device(torch, str(requested_device))

    data_cfg = cfg.get("data", {})
    image_size = int(args.image_size or data_cfg.get("image_size", 512))

    model = build_model_from_config(cfg).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device, strict=False)
    model.eval()

    if args.compile_model:
        if hasattr(torch, "compile"):
            model = torch.compile(model)
        else:
            print("torch.compile is unavailable in this PyTorch version; continuing without compile.")

    cuda_sync(torch, device)
    static_cuda_memory = cuda_memory_stats(torch, device)

    result: dict[str, Any] = {
        "run": {
            "config": args.config,
            "checkpoint": args.checkpoint,
            "amp": bool(args.amp),
            "compiled": bool(args.compile_model),
        },
        "environment": environment_stats(torch, device),
        "model": model_size_stats(model, args.checkpoint),
        "static_cuda_memory": static_cuda_memory,
    }

    result["synthetic_forward"] = benchmark_synthetic(
        model=model,
        torch_mod=torch,
        device=device,
        image_size=image_size,
        batch_size=args.batch_size,
        warmup=args.warmup,
        iters=args.iters,
        amp=args.amp,
    )

    if args.list:
        result["real_image_end_to_end"] = benchmark_real_images(
            model=model,
            cfg=cfg,
            torch_mod=torch,
            device=device,
            list_path=args.list,
            root=args.root,
            num_images=args.num_real_images,
            real_warmup=args.real_warmup,
            amp=args.amp,
        )

    print(json.dumps(result, indent=2, sort_keys=True, default=json_default))
    write_outputs(result, args.out_json, args.out_csv, args.out_md)


if __name__ == "__main__":
    main()
