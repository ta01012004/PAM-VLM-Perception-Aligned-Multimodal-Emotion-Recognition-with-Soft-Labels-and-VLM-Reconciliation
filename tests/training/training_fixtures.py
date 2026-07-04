import torch
from torch.utils.data import DataLoader, Dataset

from av_mer.data.collate import collate_av_batch
from av_mer.models.config import (
    AudioEncoderConfig,
    FusionConfig,
    PamVlmConfig,
    VideoEncoderConfig,
)


class TinyAvDataset(Dataset[dict]):
    def __init__(self, size: int) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict:
        generator = torch.Generator().manual_seed(100 + index)
        label = index % 6
        soft_label = torch.nn.functional.one_hot(torch.tensor(label), 6).float()
        return {
            "clip_id": f"tiny-{index}",
            "actor_id": 1001 + index,
            "label": label,
            "audio": torch.randn(16, generator=generator),
            "audio_sample_rate": 16000,
            "video": torch.randn(4, 3, 8, 8, generator=generator),
            "soft_voice": soft_label.clone(),
            "soft_face": soft_label.clone(),
            "soft_av": soft_label.clone(),
        }


def tiny_loader(size: int = 4, batch_size: int = 2) -> DataLoader:
    return DataLoader(
        TinyAvDataset(size),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_av_batch,
    )


def tiny_model_config() -> PamVlmConfig:
    return PamVlmConfig(
        num_classes=6,
        head_dropout=0.0,
        audio=AudioEncoderConfig(
            encoder_id="offline-tiny-audio",
            backend="tiny",
            backbone_dim=8,
            hidden_dim=8,
            freeze_backbone=True,
            dropout=0.0,
            tiny_stride=4,
        ),
        video=VideoEncoderConfig(
            backbone_id="offline-tiny-video",
            backend="tiny",
            backbone_dim=8,
            hidden_dim=8,
            temporal_layers=1,
            num_heads=2,
            dropout=0.0,
            freeze_backbone=True,
            pretrained=False,
        ),
        fusion=FusionConfig(hidden_dim=8, num_heads=2, dropout=0.0),
    )

