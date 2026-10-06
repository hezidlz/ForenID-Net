"""Standalone MiT-B2 feature backbone used by ForenID-Net.

The module follows the SegFormer MiT design and exposes four feature maps at
1/4, 1/8, 1/16 and 1/32 input resolution. The implementation is adapted from
the MIT-licensed CMX/SegFormer encoder distributed with TruFor; see
``third_party/licenses/CMX-MIT-LICENSE.txt``.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn


def _drop_path(x: torch.Tensor, drop_prob: float, training: bool) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _drop_path(x, self.drop_prob, self.training)


def _init_module(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
        fan_out //= module.groups
        nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(2.0 / fan_out))
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class DWConv(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=True)

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        batch, _, channels = x.shape
        x = x.transpose(1, 2).reshape(batch, channels, height, width)
        x = self.dwconv(x)
        return x.flatten(2).transpose(1, 2)


class MixFFN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.dwconv = DWConv(hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)
        self.apply(_init_module)

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        x = self.fc1(x)
        x = self.dwconv(x, height, width)
        x = self.drop(self.act(x))
        return self.drop(self.fc2(x))


class EfficientAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        sr_ratio: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        self.apply(_init_module)

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        batch, tokens, channels = x.shape
        q = self.q(x).reshape(batch, tokens, self.num_heads, channels // self.num_heads)
        q = q.permute(0, 2, 1, 3)

        source = x
        if self.sr_ratio > 1:
            source = x.transpose(1, 2).reshape(batch, channels, height, width)
            source = self.sr(source).reshape(batch, channels, -1).transpose(1, 2)
            source = self.norm(source)
        kv = self.kv(source).reshape(
            batch, -1, 2, self.num_heads, channels // self.num_heads
        )
        kv = kv.permute(2, 0, 3, 1, 4)
        key, value = kv[0], kv[1]

        attention = (q @ key.transpose(-2, -1)) * self.scale
        attention = self.attn_drop(attention.softmax(dim=-1))
        x = (attention @ value).transpose(1, 2).reshape(batch, tokens, channels)
        return self.proj_drop(self.proj(x))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        sr_ratio: int,
        drop: float,
        attn_drop: float,
        drop_path: float,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = EfficientAttention(
            dim=dim,
            num_heads=num_heads,
            sr_ratio=sr_ratio,
            qkv_bias=True,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = MixFFN(dim, int(dim * mlp_ratio), drop=drop)
        self.apply(_init_module)

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), height, width))
        x = x + self.drop_path(self.mlp(self.norm2(x), height, width))
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        stride: int,
        in_channels: int,
        embed_dim: int,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=patch_size // 2,
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.apply(_init_module)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        x = self.proj(x)
        height, width = x.shape[-2:]
        x = self.norm(x.flatten(2).transpose(1, 2))
        return x, height, width


class MiTB2Backbone(nn.Module):
    """SegFormer MiT-B2 encoder returning four spatial feature maps."""

    feature_channels = (64, 128, 320, 512)
    depths = (3, 4, 6, 3)
    num_heads = (1, 2, 5, 8)
    sr_ratios = (8, 4, 2, 1)

    def __init__(
        self,
        image_size: int = 512,
        pretrained_path: str | Path | None = None,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        channels = self.feature_channels
        self.patch_embed1 = OverlapPatchEmbed(image_size, 7, 4, 3, channels[0])
        self.patch_embed2 = OverlapPatchEmbed(image_size // 4, 3, 2, channels[0], channels[1])
        self.patch_embed3 = OverlapPatchEmbed(image_size // 8, 3, 2, channels[1], channels[2])
        self.patch_embed4 = OverlapPatchEmbed(image_size // 16, 3, 2, channels[2], channels[3])

        path_rates = torch.linspace(0, drop_path_rate, sum(self.depths)).tolist()
        cursor = 0
        for stage_index, (dim, depth, heads, sr_ratio) in enumerate(
            zip(channels, self.depths, self.num_heads, self.sr_ratios), start=1
        ):
            blocks = nn.ModuleList(
                TransformerBlock(
                    dim=dim,
                    num_heads=heads,
                    mlp_ratio=4.0,
                    sr_ratio=sr_ratio,
                    drop=drop_rate,
                    attn_drop=0.0,
                    drop_path=path_rates[cursor + block_index],
                )
                for block_index in range(depth)
            )
            setattr(self, f"block{stage_index}", blocks)
            setattr(self, f"norm{stage_index}", nn.LayerNorm(dim, eps=1e-6))
            cursor += depth

        self._frozen_stages = 0
        self.apply(_init_module)
        if pretrained_path is not None:
            self.load_pretrained(pretrained_path)

    @staticmethod
    def _unwrap_state_dict(payload: object) -> dict[str, torch.Tensor]:
        if not isinstance(payload, dict):
            raise TypeError("MiT-B2 checkpoint must contain a state dictionary")
        for key in ("state_dict", "model", "backbone"):
            value = payload.get(key)
            if isinstance(value, dict):
                payload = value
                break
        state: dict[str, torch.Tensor] = {}
        for raw_key, value in payload.items():
            if not isinstance(raw_key, str) or not torch.is_tensor(value):
                continue
            key = raw_key
            for prefix in ("module.", "backbone.", "encoder."):
                if key.startswith(prefix):
                    key = key[len(prefix) :]
            state[key] = value
        return state

    def load_pretrained(self, checkpoint: str | Path) -> tuple[list[str], list[str]]:
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"MiT-B2 checkpoint not found: {checkpoint}")
        payload = torch.load(checkpoint, map_location="cpu")
        state = self._unwrap_state_dict(payload)
        model_state = self.state_dict()
        compatible = {
            key: value
            for key, value in state.items()
            if key in model_state and model_state[key].shape == value.shape
        }
        if not compatible:
            raise RuntimeError(f"No compatible MiT-B2 parameters found in {checkpoint}")
        incompatible = self.load_state_dict(compatible, strict=False)
        return list(incompatible.missing_keys), list(incompatible.unexpected_keys)

    def freeze_stages(self, count: int) -> None:
        self._frozen_stages = max(0, min(int(count), 4))
        for stage in range(1, self._frozen_stages + 1):
            modules = (
                getattr(self, f"patch_embed{stage}"),
                getattr(self, f"block{stage}"),
                getattr(self, f"norm{stage}"),
            )
            for module in modules:
                module.eval()
                for parameter in module.parameters():
                    parameter.requires_grad = False

    def train(self, mode: bool = True) -> "MiTB2Backbone":
        super().train(mode)
        if self._frozen_stages:
            self.freeze_stages(self._frozen_stages)
        return self

    @staticmethod
    def _to_spatial(
        x: torch.Tensor, batch: int, height: int, width: int, norm: nn.LayerNorm
    ) -> torch.Tensor:
        x = norm(x)
        return x.reshape(batch, height, width, -1).permute(0, 3, 1, 2).contiguous()

    def forward(self, image: torch.Tensor) -> list[torch.Tensor]:
        batch = image.shape[0]
        outputs: list[torch.Tensor] = []
        x = image
        for stage in range(1, 5):
            patch_embed = getattr(self, f"patch_embed{stage}")
            blocks = getattr(self, f"block{stage}")
            norm = getattr(self, f"norm{stage}")
            x, height, width = patch_embed(x)
            for block in blocks:
                x = block(x, height, width)
            x = self._to_spatial(x, batch, height, width, norm)
            outputs.append(x)
        return outputs
