from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearProjection(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_channels, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x.transpose(1, 2).reshape(b, -1, h, w)


class MLPDecoder(nn.Module):
    """TruFor/SegFormer-style decoder for four feature scales."""

    def __init__(
        self,
        in_channels: tuple[int, int, int, int],
        embed_dim: int = 256,
        out_channels: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.proj = nn.ModuleList([LinearProjection(ch, embed_dim) for ch in in_channels])
        self.fuse = nn.Sequential(
            nn.Conv2d(embed_dim * 4, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )
        self.pred = nn.Conv2d(embed_dim, out_channels, kernel_size=1)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        if len(features) != 4:
            raise ValueError(f"MLPDecoder expects 4 feature maps, got {len(features)}")
        target_size = features[0].shape[-2:]
        projected = []
        for feat, proj in zip(features, self.proj):
            x = proj(feat)
            if x.shape[-2:] != target_size:
                x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
            projected.append(x)
        x = torch.cat(projected, dim=1)
        return self.pred(self.fuse(x))
