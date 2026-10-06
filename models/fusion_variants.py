from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _resize(noise: torch.Tensor, feature: torch.Tensor) -> torch.Tensor:
    return F.interpolate(noise, size=feature.shape[-2:], mode="bilinear", align_corners=False)


def _gate_adapter(noise_channels: int, feature_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(noise_channels, feature_channels, kernel_size=1, bias=False),
        nn.BatchNorm2d(feature_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(feature_channels, feature_channels, kernel_size=1),
        nn.Sigmoid(),
    )


def _noise_projection(noise_channels: int, feature_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(noise_channels, feature_channels, kernel_size=1, bias=False),
        nn.BatchNorm2d(feature_channels),
        nn.ReLU(inplace=True),
    )


def _freeze_if_unused(module: nn.Module, index: int, gated_scales: set[int]) -> nn.Module:
    """Retain a stable four-scale state dict without trainable dead parameters."""
    if index not in gated_scales:
        module.requires_grad_(False)
    return module


class FixedGateFusion(nn.Module):
    """The proposed multiplicative gate with a fixed, non-learnable alpha."""

    def __init__(
        self,
        feature_channels: tuple[int, int, int, int],
        noise_channels: int = 16,
        alpha: float = 0.1,
        gated_scales: tuple[int, ...] = (1, 2, 3),
    ):
        super().__init__()
        self.gated_scales = set(gated_scales)
        self.adapters = nn.ModuleList(
            [
                _freeze_if_unused(_gate_adapter(noise_channels, channels), index, self.gated_scales)
                for index, channels in enumerate(feature_channels)
            ]
        )
        self.register_buffer("fixed_alpha", torch.tensor(float(alpha)), persistent=True)

    def forward(self, features: list[torch.Tensor], noise_feature: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        fused: list[torch.Tensor] = []
        gates: list[torch.Tensor] = []
        for index, (feature, adapter) in enumerate(zip(features, self.adapters)):
            if index in self.gated_scales:
                gate = adapter(_resize(noise_feature, feature))
                gates.append(gate)
                fused.append(feature * (1.0 + self.fixed_alpha * gate))
            else:
                gates.append(torch.zeros_like(feature))
                fused.append(feature)
        return fused, gates


class AdditiveGateFusion(nn.Module):
    """A learnable additive-gate control for the proposed multiplicative gate."""

    def __init__(
        self,
        feature_channels: tuple[int, int, int, int],
        noise_channels: int = 16,
        init_alpha: float = 0.1,
        gated_scales: tuple[int, ...] = (1, 2, 3),
    ):
        super().__init__()
        self.gated_scales = set(gated_scales)
        self.adapters = nn.ModuleList(
            [
                _freeze_if_unused(_gate_adapter(noise_channels, channels), index, self.gated_scales)
                for index, channels in enumerate(feature_channels)
            ]
        )
        self.alpha = nn.ParameterList(
            [
                nn.Parameter(
                    torch.tensor(float(init_alpha)),
                    requires_grad=index in self.gated_scales,
                )
                for index, _ in enumerate(feature_channels)
            ]
        )

    def forward(self, features: list[torch.Tensor], noise_feature: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        fused: list[torch.Tensor] = []
        gates: list[torch.Tensor] = []
        for index, (feature, adapter) in enumerate(zip(features, self.adapters)):
            if index in self.gated_scales:
                gate = adapter(_resize(noise_feature, feature))
                gates.append(gate)
                fused.append(feature + self.alpha[index] * gate)
            else:
                gates.append(torch.zeros_like(feature))
                fused.append(feature)
        return fused, gates


class SimpleAddFusion(nn.Module):
    """Project and add Noiseprint++ features without a gate."""

    def __init__(
        self,
        feature_channels: tuple[int, int, int, int],
        noise_channels: int = 16,
        gated_scales: tuple[int, ...] = (1, 2, 3),
    ):
        super().__init__()
        self.gated_scales = set(gated_scales)
        self.projections = nn.ModuleList(
            [
                _freeze_if_unused(_noise_projection(noise_channels, channels), index, self.gated_scales)
                for index, channels in enumerate(feature_channels)
            ]
        )

    def forward(self, features: list[torch.Tensor], noise_feature: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        fused: list[torch.Tensor] = []
        projected: list[torch.Tensor] = []
        for index, (feature, projection) in enumerate(zip(features, self.projections)):
            if index in self.gated_scales:
                noise = projection(_resize(noise_feature, feature))
                projected.append(noise)
                fused.append(feature + noise)
            else:
                projected.append(torch.zeros_like(feature))
                fused.append(feature)
        return fused, projected


class SimpleConcatFusion(nn.Module):
    """Project, concatenate, and reduce RGB/noise features without a gate."""

    def __init__(
        self,
        feature_channels: tuple[int, int, int, int],
        noise_channels: int = 16,
        gated_scales: tuple[int, ...] = (1, 2, 3),
    ):
        super().__init__()
        self.gated_scales = set(gated_scales)
        self.projections = nn.ModuleList(
            [
                _freeze_if_unused(_noise_projection(noise_channels, channels), index, self.gated_scales)
                for index, channels in enumerate(feature_channels)
            ]
        )
        self.reducers = nn.ModuleList(
            [
                _freeze_if_unused(
                    nn.Sequential(
                        nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
                        nn.BatchNorm2d(channels),
                        nn.ReLU(inplace=True),
                    ),
                    index,
                    self.gated_scales,
                )
                for index, channels in enumerate(feature_channels)
            ]
        )

    def forward(self, features: list[torch.Tensor], noise_feature: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        projected: list[torch.Tensor] = []
        fused: list[torch.Tensor] = []
        for index, (feature, projection, reducer) in enumerate(zip(features, self.projections, self.reducers)):
            if index in self.gated_scales:
                noise = projection(_resize(noise_feature, feature))
                projected.append(noise)
                fused.append(reducer(torch.cat([feature, noise], dim=1)))
            else:
                projected.append(torch.zeros_like(feature))
                fused.append(feature)
        return fused, projected


class NoiseOnlyPyramid(nn.Module):
    """Noise-only evidence control with the same four output shapes as the RGB branch."""

    def __init__(self, noise_channels: int, feature_channels: tuple[int, int, int, int]):
        super().__init__()
        stages: list[nn.Module] = []
        in_channels = noise_channels
        for index, channels in enumerate(feature_channels):
            stride = 4 if index == 0 else 2
            kernel = 7 if index == 0 else 3
            stages.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, channels, kernel_size=kernel, stride=stride, padding=kernel // 2, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                )
            )
            in_channels = channels
        self.stages = nn.ModuleList(stages)

    def forward(self, noise_feature: torch.Tensor) -> list[torch.Tensor]:
        outputs: list[torch.Tensor] = []
        feature = noise_feature
        for stage in self.stages:
            feature = stage(feature)
            outputs.append(feature)
        return outputs


def build_fusion_variant(
    mode: str,
    feature_channels: tuple[int, int, int, int],
    noise_channels: int,
    init_alpha: float,
    gated_scales: tuple[int, ...],
) -> nn.Module | None:
    if mode in {"rgb_only", "noise_only", "learned_gate"}:
        return None
    if mode == "fixed_gate":
        return FixedGateFusion(feature_channels, noise_channels, init_alpha, gated_scales)
    if mode == "additive_gate":
        return AdditiveGateFusion(feature_channels, noise_channels, init_alpha, gated_scales)
    if mode == "simple_add":
        return SimpleAddFusion(feature_channels, noise_channels, gated_scales)
    if mode == "simple_concat":
        return SimpleConcatFusion(feature_channels, noise_channels, gated_scales)
    raise ValueError(f"Unknown fusion mode: {mode}")
