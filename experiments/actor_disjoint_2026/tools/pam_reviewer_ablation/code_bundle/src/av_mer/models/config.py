"""Typed model configuration loaded from tracked JSON files."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class AudioEncoderConfig:
    encoder_id: str
    backend: str
    backbone_dim: int
    hidden_dim: int
    freeze_backbone: bool
    dropout: float
    tiny_stride: int
    unfreeze_last_n_layers: int = 0
    normalize_waveform: bool = False
    pass_attention_mask: bool = True


@dataclass(frozen=True, slots=True)
class VideoEncoderConfig:
    backbone_id: str
    backend: str
    backbone_dim: int
    hidden_dim: int
    temporal_layers: int
    num_heads: int
    dropout: float
    freeze_backbone: bool
    pretrained: bool
    unfreeze_last_n_stages: int = 0
    weights_path: str | None = None
    normalize_frames: bool = False


@dataclass(frozen=True, slots=True)
class FusionConfig:
    hidden_dim: int
    num_heads: int
    dropout: float
    audio_gate_bias: float = 0.0
    modality_dropout: float = 0.0
    cross_attention_enabled: bool = True
    gate_mode: str = "learned"


@dataclass(frozen=True, slots=True)
class PamVlmConfig:
    num_classes: int
    head_dropout: float
    audio: AudioEncoderConfig
    video: VideoEncoderConfig
    fusion: FusionConfig

    @classmethod
    def from_json(cls, path: str | Path) -> "PamVlmConfig":
        config_path = Path(path)
        with config_path.open(encoding="utf-8") as handle:
            values = json.load(handle)
        if not isinstance(values, Mapping):
            raise ValueError(f"Model config {config_path} must contain a JSON object.")
        try:
            return cls(
                num_classes=int(values["num_classes"]),
                head_dropout=float(values["head_dropout"]),
                audio=AudioEncoderConfig(**_typed_section(values, "audio")),
                video=VideoEncoderConfig(**_typed_section(values, "video")),
                fusion=FusionConfig(**_typed_section(values, "fusion")),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid model config {config_path}: {error}.") from error


def _typed_section(values: Mapping[str, Any], key: str) -> dict[str, Any]:
    section = values[key]
    if not isinstance(section, Mapping):
        raise TypeError(f"Config section {key!r} must be an object.")
    return dict(section)
