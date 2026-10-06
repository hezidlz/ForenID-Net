from __future__ import annotations

from collections import defaultdict

import torch
from tqdm import tqdm


def _to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for key, value in batch.items():
        out[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return out


def train_one_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    device: torch.device,
    epoch: int,
    amp: bool = True,
) -> dict[str, float]:
    model.train()
    meters: dict[str, list[float]] = defaultdict(list)
    scaler = torch.cuda.amp.GradScaler(enabled=amp and device.type == "cuda")

    for batch in tqdm(loader, desc=f"train {epoch}", leave=False):
        batch = _to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
            outputs = model(batch["image"])
            losses = loss_fn(
                outputs,
                mask=batch.get("mask"),
                label=batch.get("label"),
                has_mask=batch.get("has_mask"),
            )
            loss = losses["loss_total"]
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        for key, value in losses.items():
            meters[key].append(float(value.detach().cpu()))

    return {key: sum(values) / max(1, len(values)) for key, values in meters.items()}
