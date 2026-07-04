import torch

from av_mer.models.config import (
    AudioEncoderConfig,
    FusionConfig,
    PamVlmConfig,
    VideoEncoderConfig,
)


def tiny_model_config(*, freeze_backbones: bool = True) -> PamVlmConfig:
    return PamVlmConfig(
        num_classes=6,
        head_dropout=0.0,
        audio=AudioEncoderConfig(
            encoder_id="local-tiny-audio",
            backend="tiny",
            backbone_dim=8,
            hidden_dim=8,
            freeze_backbone=freeze_backbones,
            dropout=0.0,
            tiny_stride=4,
        ),
        video=VideoEncoderConfig(
            backbone_id="local-tiny-video",
            backend="tiny",
            backbone_dim=8,
            hidden_dim=8,
            temporal_layers=2,
            num_heads=2,
            dropout=0.0,
            freeze_backbone=freeze_backbones,
            pretrained=False,
        ),
        fusion=FusionConfig(hidden_dim=8, num_heads=2, dropout=0.0),
    )


def tiny_batch() -> dict[str, torch.Tensor]:
    return {
        "audio_values": torch.randn(2, 16),
        "audio_mask": torch.tensor(
            [[True] * 16, [True] * 12 + [False] * 4],
            dtype=torch.bool,
        ),
        "video_values": torch.randn(2, 5, 3, 12, 12),
        "video_mask": torch.tensor(
            [[True] * 5, [True] * 3 + [False] * 2],
            dtype=torch.bool,
        ),
    }

