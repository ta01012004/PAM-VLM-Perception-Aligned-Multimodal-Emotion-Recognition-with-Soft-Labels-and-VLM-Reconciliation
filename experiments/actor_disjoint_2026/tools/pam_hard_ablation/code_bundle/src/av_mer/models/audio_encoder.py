"""Configurable speech encoder wrapper with an offline tiny test backend."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from av_mer.models.config import AudioEncoderConfig


class TinyAudioBackbone(nn.Module):
    """Small strided convolution used only for offline CPU tests."""

    def __init__(self, hidden_dim: int, stride: int) -> None:
        super().__init__()
        if hidden_dim <= 0 or stride <= 0:
            raise ValueError("Tiny audio hidden_dim and stride must be positive.")
        self.stride = stride
        self.projection = nn.Conv1d(1, hidden_dim, kernel_size=stride, stride=stride)

    def forward(self, values: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        if values.shape[1] < self.stride:
            raise ValueError(
                f"Tiny audio backend needs at least {self.stride} samples, "
                f"received {values.shape[1]}."
            )
        masked_values = values.masked_fill(~mask, 0.0)
        tokens = self.projection(masked_values.unsqueeze(1)).transpose(1, 2)
        token_mask = F.max_pool1d(
            mask.to(dtype=torch.float32).unsqueeze(1),
            kernel_size=self.stride,
            stride=self.stride,
        ).squeeze(1).to(dtype=torch.bool)
        return tokens, token_mask


class AudioEncoder(nn.Module):
    """Encode ``audio_values[B,T]`` into masked tokens ``[B,Ta,H]``."""

    def __init__(self, config: AudioEncoderConfig, backbone: nn.Module | None = None) -> None:
        super().__init__()
        self.config = config
        self.backbone = backbone or self._create_backbone(config)
        self._unfrozen_layers: list[nn.Module] = []
        self._configure_backbone_trainability()
        self.projector = nn.Sequential(
            nn.Linear(config.backbone_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.Dropout(config.dropout),
        )

    def _configure_backbone_trainability(self) -> None:
        count = self.config.unfreeze_last_n_layers
        if count < 0:
            raise ValueError("unfreeze_last_n_layers must be non-negative.")
        if self.config.freeze_backbone:
            if count:
                raise ValueError("A frozen audio backbone cannot unfreeze transformer layers.")
            self.backbone.requires_grad_(False)
            self.backbone.eval()
            return
        if count == 0:
            return
        encoder = getattr(self.backbone, "encoder", None)
        layers = list(getattr(encoder, "layers", []))
        if count > len(layers):
            raise ValueError(
                f"Requested {count} trainable audio layers, backbone exposes {len(layers)}."
            )
        self.backbone.requires_grad_(False)
        self._unfrozen_layers = layers[-count:]
        for layer in self._unfrozen_layers:
            layer.requires_grad_(True)

    @property
    def backbone_is_trainable(self) -> bool:
        return any(parameter.requires_grad for parameter in self.backbone.parameters())

    @staticmethod
    def _create_backbone(config: AudioEncoderConfig) -> nn.Module:
        if config.backend == "tiny":
            return TinyAudioBackbone(config.backbone_dim, config.tiny_stride)
        if config.backend == "transformers":
            from transformers import AutoModel

            return AutoModel.from_pretrained(config.encoder_id)
        raise ValueError(
            f"Unknown audio backend {config.backend!r} for encoder {config.encoder_id!r}."
        )

    def train(self, mode: bool = True) -> "AudioEncoder":
        super().train(mode)
        if not self.backbone_is_trainable:
            self.backbone.eval()
        elif self._unfrozen_layers:
            self.backbone.eval()
            for layer in self._unfrozen_layers:
                layer.train(mode)
        return self

    def _run_transformers_backbone(self, values: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        output: Any = self.backbone(input_values=values, attention_mask=mask.long())
        tokens = output.last_hidden_state
        mask_builder = getattr(self.backbone, "_get_feature_vector_attention_mask", None)
        if mask_builder is not None:
            token_mask = mask_builder(tokens.shape[1], mask.long()).to(dtype=torch.bool)
        else:
            token_mask = F.interpolate(
                mask.to(dtype=torch.float32).unsqueeze(1),
                size=tokens.shape[1],
                mode="nearest",
            ).squeeze(1).to(dtype=torch.bool)
        return tokens, token_mask

    def forward(self, audio_values: Tensor, audio_mask: Tensor) -> tuple[Tensor, Tensor]:
        if audio_values.ndim != 2 or audio_mask.shape != audio_values.shape:
            raise ValueError(
                "Expected audio_values and audio_mask with shape [B,T], received "
                f"{tuple(audio_values.shape)} and {tuple(audio_mask.shape)}."
            )
        audio_mask = audio_mask.to(dtype=torch.bool)
        if (~audio_mask.any(dim=1)).any():
            raise ValueError("Every audio sample must contain at least one valid position.")
        values = audio_values.masked_fill(~audio_mask, 0.0)
        context = torch.no_grad() if not self.backbone_is_trainable else nullcontext()
        with context:
            if self.config.backend == "tiny":
                tokens, token_mask = self.backbone(values, audio_mask)
            else:
                tokens, token_mask = self._run_transformers_backbone(values, audio_mask)
        projected = self.projector(tokens)
        projected = projected.masked_fill(~token_mask.unsqueeze(-1), 0.0)
        return projected, token_mask
