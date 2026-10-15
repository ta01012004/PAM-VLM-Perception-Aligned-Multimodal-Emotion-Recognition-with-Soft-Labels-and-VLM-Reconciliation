"""PAM-VLM scoring backbone without the future language-model branch."""

from __future__ import annotations

from typing import Optional, TypedDict

from torch import Tensor, nn

from av_mer.data.filename_parser import LABELS
from av_mer.models.audio_encoder import AudioEncoder
from av_mer.models.config import PamVlmConfig
from av_mer.models.fusion import BidirectionalCrossAttentionFusion, masked_mean
from av_mer.models.heads import ClassificationHead
from av_mer.models.video_encoder import VideoEncoder


class PamVlmOutput(TypedDict):
    loss: Optional[Tensor]
    logits: Tensor
    audio_logits: Tensor
    video_logits: Tensor
    fusion_embedding: Tensor
    generated_text: None


class PamVlmModel(nn.Module):
    """Pure tensor-in/tensor-out audio-visual emotion classifier skeleton."""

    def __init__(
        self,
        config: PamVlmConfig,
        *,
        audio_encoder: AudioEncoder | None = None,
        video_encoder: VideoEncoder | None = None,
    ) -> None:
        super().__init__()
        if config.num_classes != len(LABELS):
            raise ValueError(
                f"PAM-VLM requires {len(LABELS)} canonical classes, "
                f"received {config.num_classes}."
            )
        hidden_dims = {
            config.audio.hidden_dim,
            config.video.hidden_dim,
            config.fusion.hidden_dim,
        }
        if len(hidden_dims) != 1:
            raise ValueError(
                "Audio, video, and fusion hidden dimensions must match; "
                f"received {sorted(hidden_dims)}."
            )
        self.config = config
        self.audio_encoder = audio_encoder or AudioEncoder(config.audio)
        self.video_encoder = video_encoder or VideoEncoder(config.video)
        self.fusion = BidirectionalCrossAttentionFusion(config.fusion)
        self.audio_head = ClassificationHead(
            config.audio.hidden_dim,
            config.num_classes,
            config.head_dropout,
        )
        self.video_head = ClassificationHead(
            config.video.hidden_dim,
            config.num_classes,
            config.head_dropout,
        )
        self.fusion_head = ClassificationHead(
            config.fusion.hidden_dim,
            config.num_classes,
            config.head_dropout,
        )

    def parameter_counts(self) -> dict[str, int]:
        """Expose total and trainable parameter counts."""

        return {
            "total": sum(parameter.numel() for parameter in self.parameters()),
            "trainable": sum(
                parameter.numel() for parameter in self.parameters() if parameter.requires_grad
            ),
        }

    def forward(
        self,
        audio_values: Tensor,
        audio_mask: Tensor,
        video_values: Tensor,
        video_mask: Tensor,
    ) -> PamVlmOutput:
        audio_tokens, audio_token_mask = self.audio_encoder(audio_values, audio_mask)
        video_tokens, video_token_mask = self.video_encoder(video_values, video_mask)
        audio_embedding = masked_mean(audio_tokens, audio_token_mask)
        video_embedding = masked_mean(video_tokens, video_token_mask)
        fusion_embedding = self.fusion(
            audio_tokens,
            audio_token_mask,
            video_tokens,
            video_token_mask,
        )
        return {
            "loss": None,
            "logits": self.fusion_head(fusion_embedding),
            "audio_logits": self.audio_head(audio_embedding),
            "video_logits": self.video_head(video_embedding),
            "fusion_embedding": fusion_embedding,
            "generated_text": None,
        }

