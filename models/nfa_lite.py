from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class NFALiteGate(nn.Module):
    """Noise-guided feature amplification.

    Noiseprint++ produces a residual prior. This module converts it to
    scale-specific gates and multiplicatively amplifies SparseViT features.
    """

    def __init__(
        self,
        feature_channels: tuple[int, int, int, int],
        noise_channels: int = 16,
        init_alpha: float = 0.1,
        gated_scales: tuple[int, ...] = (1, 2, 3),
    ):
        super().__init__()
        self.gated_scales = set(gated_scales)
        self.adapters = nn.ModuleList()
        self.alpha = nn.ParameterList()
        for channels in feature_channels:
            self.adapters.append(
                nn.Sequential(
                    nn.Conv2d(noise_channels, channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels, channels, kernel_size=1),
                    nn.Sigmoid(),
                )
            )
            self.alpha.append(nn.Parameter(torch.tensor(float(init_alpha))))

    def forward(self, features: list[torch.Tensor], noise_feature: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        fused: list[torch.Tensor] = []
        gates: list[torch.Tensor] = []
        for idx, feat in enumerate(features):
            resized_noise = F.interpolate(noise_feature, size=feat.shape[-2:], mode="bilinear", align_corners=False)
            gate = self.adapters[idx](resized_noise)
            gates.append(gate)
            if idx in self.gated_scales:
                fused.append(feat * (1.0 + self.alpha[idx] * gate))
            else:
                fused.append(feat)
        return fused, gates
