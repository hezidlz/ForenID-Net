from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .decoder import MLPDecoder
from .fusion_variants import NoiseOnlyPyramid, build_fusion_variant
from .mit_b2 import MiTB2Backbone
from .nfa_lite import NFALiteGate
from .noiseprint import NoiseprintAdapter, NoiseprintExtractor
from .score_head import MaskAwareScoreHead


class ForenIDNet(nn.Module):
    """Dual-branch document forgery detector with a MiT-B2 RGB backbone."""

    feature_channels = MiTB2Backbone.feature_channels

    def __init__(
        self,
        noiseprint_weights: str | Path | None = None,
        mit_b2_weights: str | Path | None = None,
        image_size: int = 512,
        freeze_noiseprint: bool = True,
        decoder_embed_dim: int = 256,
        noise_channels: int = 16,
        nfa_init_alpha: float = 0.1,
        fusion_mode: str = "learned_gate",
        gated_scales: tuple[int, ...] = (1, 2, 3),
        use_noiseprint: bool = True,
    ):
        super().__init__()
        self.fusion_mode = str(fusion_mode).lower()
        self.use_noiseprint = bool(use_noiseprint) and self.fusion_mode != "rgb_only"
        self.noise_channels = int(noise_channels)

        self.backbone = MiTB2Backbone(
            image_size=image_size,
            pretrained_path=mit_b2_weights,
        )

        if self.use_noiseprint:
            self.noiseprint = NoiseprintExtractor(
                weights_path=noiseprint_weights,
                frozen=freeze_noiseprint,
                strict=True,
            )
            self.noise_adapter = NoiseprintAdapter(hidden_channels=self.noise_channels)
        else:
            self.noiseprint = None
            self.noise_adapter = None

        self.noise_only_pyramid = (
            NoiseOnlyPyramid(self.noise_channels, self.feature_channels)
            if self.fusion_mode == "noise_only"
            else None
        )
        self.learned_gate = (
            NFALiteGate(
                feature_channels=self.feature_channels,
                noise_channels=self.noise_channels,
                init_alpha=nfa_init_alpha,
                gated_scales=gated_scales,
            )
            if self.fusion_mode == "learned_gate"
            else None
        )
        self.fusion = build_fusion_variant(
            mode=self.fusion_mode,
            feature_channels=self.feature_channels,
            noise_channels=self.noise_channels,
            init_alpha=nfa_init_alpha,
            gated_scales=gated_scales,
        )

        self.decoder = MLPDecoder(
            in_channels=self.feature_channels,
            embed_dim=decoder_embed_dim,
            out_channels=1,
        )
        self.score_head = MaskAwareScoreHead(feature_channels=self.feature_channels[-1])

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("rgb_mean", mean, persistent=False)
        self.register_buffer("rgb_std", std, persistent=False)

    def normalize_rgb(self, image: torch.Tensor) -> torch.Tensor:
        return (image - self.rgb_mean) / self.rgb_std

    def freeze_backbone_stages(self, count: int) -> None:
        self.backbone.freeze_stages(count)

    def _noise_features(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.noiseprint is None or self.noise_adapter is None:
            batch, _, height, width = image.shape
            noise_map = image.new_zeros((batch, 1, height, width))
            noise_feature = image.new_zeros(
                (batch, self.noise_channels, height, width)
            )
            return noise_map, noise_feature
        noise_map = self.noiseprint(image)
        return noise_map, self.noise_adapter(noise_map)

    def forward(
        self, image: torch.Tensor, return_aux: bool = False
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f"Expected [B,3,H,W] input, got {tuple(image.shape)}")

        original_size = image.shape[-2:]
        noise_map, noise_feature = self._noise_features(image)
        rgb_features = self.backbone(self.normalize_rgb(image))

        if self.fusion_mode == "rgb_only":
            fused_features = rgb_features
            gates: list[torch.Tensor] = []
        elif self.fusion_mode == "noise_only":
            if self.noise_only_pyramid is None:
                raise RuntimeError("noise_only fusion is not initialized")
            fused_features = self.noise_only_pyramid(noise_feature)
            gates = []
        elif self.fusion_mode == "learned_gate":
            if self.learned_gate is None:
                raise RuntimeError("learned_gate fusion is not initialized")
            fused_features, gates = self.learned_gate(rgb_features, noise_feature)
        else:
            if self.fusion is None:
                raise RuntimeError(f"Fusion mode {self.fusion_mode!r} is not initialized")
            fused_features, gates = self.fusion(rgb_features, noise_feature)

        mask_logits_low = self.decoder(fused_features)
        mask_logits = F.interpolate(
            mask_logits_low,
            size=original_size,
            mode="bilinear",
            align_corners=False,
        )
        score_logits = self.score_head(mask_logits, fused_features[-1])

        output: dict[str, torch.Tensor | list[torch.Tensor]] = {
            "mask_logits": mask_logits,
            "pred_mask": torch.sigmoid(mask_logits),
            "score_logits": score_logits,
            "pred_score": torch.sigmoid(score_logits),
        }
        if return_aux:
            output.update(
                {
                    "noise_map": noise_map,
                    "rgb_features": rgb_features,
                    "fused_features": fused_features,
                    "nfa_gates": gates,
                }
            )
        return output


def build_model_from_config(cfg: dict) -> ForenIDNet:
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    return ForenIDNet(
        noiseprint_weights=model_cfg.get("noiseprint_weights"),
        mit_b2_weights=model_cfg.get("mit_b2_weights"),
        image_size=int(data_cfg.get("image_size", 512)),
        freeze_noiseprint=bool(model_cfg.get("freeze_noiseprint", True)),
        decoder_embed_dim=int(model_cfg.get("decoder_embed_dim", 256)),
        noise_channels=int(model_cfg.get("noise_channels", 16)),
        nfa_init_alpha=float(model_cfg.get("nfa_init_alpha", 0.1)),
        fusion_mode=str(model_cfg.get("fusion_mode", "learned_gate")),
        gated_scales=tuple(model_cfg.get("nfa_gated_scales", [1, 2, 3])),
        use_noiseprint=bool(model_cfg.get("use_noiseprint", True)),
    )
