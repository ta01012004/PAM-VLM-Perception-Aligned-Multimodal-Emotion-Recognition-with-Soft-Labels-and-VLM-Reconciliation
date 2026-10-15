"""Per-frame visual backbone followed by masked temporal attention."""

from __future__ import annotations

from contextlib import nullcontext

import torch
from torch import Tensor, nn

from av_mer.models.config import VideoEncoderConfig


class TinyFrameBackbone(nn.Module):
    """Small per-frame CNN used only for offline CPU tests."""

    def __init__(self, output_dim: int) -> None:
        super().__init__()
        if output_dim <= 0:
            raise ValueError("Tiny video output_dim must be positive.")
        self.network = nn.Sequential(
            nn.Conv2d(3, output_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

    def forward(self, frames: Tensor) -> Tensor:
        return self.network(frames)


def _create_torchvision_backbone(config: VideoEncoderConfig) -> nn.Module:
    from torchvision.models import get_model

    weights = "DEFAULT" if config.pretrained else None
    backbone = get_model(config.backbone_id, weights=weights)
    if hasattr(backbone, "classifier"):
        backbone.classifier = nn.Identity()
    elif hasattr(backbone, "fc"):
        backbone.fc = nn.Identity()
    else:
        raise ValueError(
            f"Torchvision backbone {config.backbone_id!r} has no supported classifier head."
        )
    return backbone


class VideoEncoder(nn.Module):
    """Encode ``video_values[B,F,C,H,W]`` into tokens ``[B,F,H]``."""

    def __init__(self, config: VideoEncoderConfig, backbone: nn.Module | None = None) -> None:
        super().__init__()
        self.config = config
        if backbone is not None:
            self.backbone = backbone
        elif config.backend == "tiny":
            self.backbone = TinyFrameBackbone(config.backbone_dim)
        elif config.backend == "torchvision":
            self.backbone = _create_torchvision_backbone(config)
        else:
            raise ValueError(
                f"Unknown video backend {config.backend!r} for {config.backbone_id!r}."
            )
        self._unfrozen_stages: list[nn.Module] = []
        self._configure_backbone_trainability()
        self.projector = nn.Sequential(
            nn.Linear(config.backbone_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
        )
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            temporal_layer,
            num_layers=config.temporal_layers,
            enable_nested_tensor=False,
        )

    def _configure_backbone_trainability(self) -> None:
        count = self.config.unfreeze_last_n_stages
        if count < 0:
            raise ValueError("unfreeze_last_n_stages must be non-negative.")
        if self.config.freeze_backbone:
            if count:
                raise ValueError("A frozen video backbone cannot unfreeze feature stages.")
            self.backbone.requires_grad_(False)
            self.backbone.eval()
            return
        if count == 0:
            return
        features = getattr(self.backbone, "features", None)
        stages = list(features.children()) if features is not None else []
        if count > len(stages):
            raise ValueError(
                f"Requested {count} trainable video stages, backbone exposes {len(stages)}."
            )
        self.backbone.requires_grad_(False)
        self._unfrozen_stages = stages[-count:]
        for stage in self._unfrozen_stages:
            stage.requires_grad_(True)

    @property
    def backbone_is_trainable(self) -> bool:
        return any(parameter.requires_grad for parameter in self.backbone.parameters())

    def train(self, mode: bool = True) -> "VideoEncoder":
        super().train(mode)
        if not self.backbone_is_trainable:
            self.backbone.eval()
        elif self._unfrozen_stages:
            self.backbone.eval()
            for stage in self._unfrozen_stages:
                stage.train(mode)
        return self

    def forward(self, video_values: Tensor, video_mask: Tensor) -> tuple[Tensor, Tensor]:
        if video_values.ndim != 5:
            raise ValueError(
                f"Expected video_values with shape [B,F,C,H,W], got {tuple(video_values.shape)}."
            )
        batch_size, frame_count = video_values.shape[:2]
        if video_mask.shape != (batch_size, frame_count):
            raise ValueError(
                f"Expected video_mask shape {(batch_size, frame_count)}, "
                f"got {tuple(video_mask.shape)}."
            )
        video_mask = video_mask.to(dtype=torch.bool)
        if (~video_mask.any(dim=1)).any():
            raise ValueError("Every video sample must contain at least one valid frame.")

        masked_video = video_values.masked_fill(
            ~video_mask[:, :, None, None, None],
            0.0,
        )
        flat_frames = masked_video.flatten(0, 1)
        context = torch.no_grad() if not self.backbone_is_trainable else nullcontext()
        with context:
            frame_features = self.backbone(flat_frames)
        if frame_features.ndim != 2 or frame_features.shape[1] != self.config.backbone_dim:
            raise ValueError(
                f"Video backbone {self.config.backbone_id!r} must return "
                f"[B*F,{self.config.backbone_dim}], received {tuple(frame_features.shape)}."
            )
        tokens = self.projector(frame_features).reshape(batch_size, frame_count, -1)
        tokens = self.temporal_encoder(tokens, src_key_padding_mask=~video_mask)
        tokens = tokens.masked_fill(~video_mask.unsqueeze(-1), 0.0)
        return tokens, video_mask
