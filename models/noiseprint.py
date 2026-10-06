from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn


def conv_with_padding(
    in_planes: int,
    out_planes: int,
    kernel_size: int,
    stride: int = 1,
    dilation: int = 1,
    bias: bool = False,
    padding: int | None = None,
) -> nn.Conv2d:
    if padding is None:
        padding = kernel_size // 2
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=kernel_size,
        stride=stride,
        dilation=dilation,
        padding=padding,
        bias=bias,
    )


def conv_init(conv: nn.Conv2d) -> None:
    n = conv.kernel_size[0] * conv.kernel_size[1] * conv.out_channels
    conv.weight.data.normal_(0, math.sqrt(2.0 / n))


def batchnorm_init(module: nn.BatchNorm2d, kernel_size: int = 3) -> None:
    n = kernel_size**2 * module.num_features
    module.weight.data.normal_(0, math.sqrt(2.0 / n))
    module.bias.data.zero_()


def make_activation(name: str | None) -> nn.Module | None:
    if name in {None, "linear"}:
        return None
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "tanh":
        return nn.Tanh()
    if name == "leaky_relu":
        return nn.LeakyReLU(inplace=True)
    raise ValueError(f"Unsupported activation: {name}")


def make_dncnn_layers(
    in_channels: int = 3,
    out_channels: int = 1,
    features: int = 64,
    depth: int = 17,
    kernel_size: int = 3,
    bn_momentum: float = 0.1,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    channel_plan = [features] * (depth - 1) + [out_channels]
    for idx, out_ch in enumerate(channel_plan):
        in_ch = in_channels if idx == 0 else channel_plan[idx - 1]
        use_bn = 0 < idx < depth - 1
        conv = conv_with_padding(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=not use_bn,
        )
        conv_init(conv)
        layers.append(conv)
        if use_bn:
            bn = nn.BatchNorm2d(out_ch, momentum=bn_momentum)
            batchnorm_init(bn, kernel_size=kernel_size)
            layers.append(bn)
        act = make_activation("relu" if idx < depth - 1 else "linear")
        if act is not None:
            layers.append(act)
    return nn.Sequential(*layers)


class NoiseprintExtractor(nn.Module):
    """TruFor Noiseprint++ wrapper.

    The official checkpoint stores weights under the ``network`` key and maps to
    the DnCNN-style sequential module below. This module is frozen by default and
    should receive RGB tensors in [0, 1].
    """

    def __init__(
        self,
        weights_path: str | Path | None = None,
        out_channels: int = 1,
        frozen: bool = True,
        strict: bool = True,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.frozen = frozen
        self.network = make_dncnn_layers(out_channels=out_channels)
        if weights_path:
            self.load_weights(weights_path, strict=strict)
        if frozen:
            self.freeze()

    def load_weights(self, weights_path: str | Path, strict: bool = True) -> None:
        payload = torch.load(weights_path, map_location="cpu")
        state_dict = payload.get("network", payload.get("model", payload))
        self.network.load_state_dict(state_dict, strict=strict)

    def freeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        self.frozen = True

    def train(self, mode: bool = True) -> "NoiseprintExtractor":
        super().train(mode)
        if self.frozen:
            super().train(False)
        return self

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if self.frozen:
            with torch.no_grad():
                return self.network(image)
        return self.network(image)


class NoiseprintAdapter(nn.Module):
    """Small trainable adapter that converts a 1-channel residual map to gates."""

    def __init__(self, hidden_channels: int = 16):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, noise_map: torch.Tensor) -> torch.Tensor:
        return self.stem(noise_map)
