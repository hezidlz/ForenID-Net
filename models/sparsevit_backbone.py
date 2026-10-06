from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
        fan_out //= module.groups
        module.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if module.bias is not None:
            module.bias.data.zero_()


class PatchEmbed(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
        )
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(x))


class ConvMlp(nn.Module):
    def __init__(self, channels: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden = int(channels * mlp_ratio)
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvBlock(nn.Module):
    def __init__(self, channels: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.pos = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.norm1 = nn.BatchNorm2d(channels)
        self.dw = nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=channels)
        self.pw = nn.Conv2d(channels, channels, kernel_size=1)
        self.norm2 = nn.BatchNorm2d(channels)
        self.mlp = ConvMlp(channels, mlp_ratio=mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pos(x)
        x = x + self.pw(self.dw(self.norm1(x)))
        x = x + self.mlp(self.norm2(x))
        return x


def _window_partition(x: torch.Tensor, window_size: int) -> tuple[torch.Tensor, tuple[int, int]]:
    b, c, h, w = x.shape
    pad_h = (window_size - h % window_size) % window_size
    pad_w = (window_size - w % window_size) % window_size
    x = F.pad(x, (0, pad_w, 0, pad_h))
    hp, wp = h + pad_h, w + pad_w
    x = x.view(b, c, hp // window_size, window_size, wp // window_size, window_size)
    x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
    return x.view(-1, window_size * window_size, c), (hp, wp)


def _window_reverse(windows: torch.Tensor, original_hw: tuple[int, int], padded_hw: tuple[int, int], window_size: int, batch_size: int) -> torch.Tensor:
    h, w = original_hw
    hp, wp = padded_hw
    c = windows.shape[-1]
    x = windows.view(batch_size, hp // window_size, wp // window_size, window_size, window_size, c)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
    x = x.view(batch_size, c, hp, wp)
    return x[:, :, :h, :w].contiguous()


class WindowAttention(nn.Module):
    def __init__(self, channels: int, num_heads: int, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f"channels={channels} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(channels, channels * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(channels, channels)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).view(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj_drop(self.proj(out))


class SparseAttentionBlock(nn.Module):
    """SparseViT-style non-semantic block using local window attention."""

    def __init__(self, channels: int, num_heads: int, window_size: int = 8, mlp_ratio: float = 4.0):
        super().__init__()
        self.window_size = window_size
        self.pos = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.norm1 = nn.LayerNorm(channels)
        self.attn = WindowAttention(channels, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, int(channels * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(channels * mlp_ratio), channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x = x + self.pos(x)
        windows, padded_hw = _window_partition(x, self.window_size)
        shortcut = windows
        windows = shortcut + self.attn(self.norm1(windows))
        windows = windows + self.mlp(self.norm2(windows))
        return _window_reverse(windows, (h, w), padded_hw, self.window_size, b)


class SparseViTBackbone(nn.Module):
    """A compact SparseViT backbone that returns four feature scales."""

    def __init__(
        self,
        in_channels: int = 3,
        embed_dims: tuple[int, int, int, int] = (32, 64, 160, 256),
        depths: tuple[int, int, int, int] = (2, 2, 4, 2),
        window_sizes: tuple[int, int] = (8, 4),
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self._frozen_stages = 0
        self.out_channels = embed_dims
        heads = tuple(max(1, dim // 32) for dim in embed_dims)

        self.patch1 = PatchEmbed(in_channels, embed_dims[0], kernel_size=7, stride=4)
        self.stage1 = nn.Sequential(*[ConvBlock(embed_dims[0], mlp_ratio=mlp_ratio) for _ in range(depths[0])])

        self.patch2 = PatchEmbed(embed_dims[0], embed_dims[1], kernel_size=3, stride=2)
        self.stage2 = nn.Sequential(*[ConvBlock(embed_dims[1], mlp_ratio=mlp_ratio) for _ in range(depths[1])])

        self.patch3 = PatchEmbed(embed_dims[1], embed_dims[2], kernel_size=3, stride=2)
        self.stage3 = nn.Sequential(*[
            SparseAttentionBlock(embed_dims[2], heads[2], window_size=window_sizes[0], mlp_ratio=mlp_ratio)
            for _ in range(depths[2])
        ])

        self.patch4 = PatchEmbed(embed_dims[2], embed_dims[3], kernel_size=3, stride=2)
        self.stage4 = nn.Sequential(*[
            SparseAttentionBlock(embed_dims[3], heads[3], window_size=window_sizes[1], mlp_ratio=mlp_ratio)
            for _ in range(depths[3])
        ])

        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        f1 = self.stage1(self.patch1(x))
        f2 = self.stage2(self.patch2(f1))
        f3 = self.stage3(self.patch3(f2))
        f4 = self.stage4(self.patch4(f3))
        return [f1, f2, f3, f4]

    def freeze_stages(self, num_stages: int) -> None:
        self._frozen_stages = max(self._frozen_stages, num_stages)
        stages = [
            [self.patch1, self.stage1],
            [self.patch2, self.stage2],
            [self.patch3, self.stage3],
            [self.patch4, self.stage4],
        ]
        for stage in stages[:num_stages]:
            for module in stage:
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False

    def train(self, mode: bool = True) -> "SparseViTBackbone":
        super().train(mode)
        if self._frozen_stages:
            stages = [
                [self.patch1, self.stage1],
                [self.patch2, self.stage2],
                [self.patch3, self.stage3],
                [self.patch4, self.stage4],
            ]
            for stage in stages[: self._frozen_stages]:
                for module in stage:
                    module.eval()
        return self
