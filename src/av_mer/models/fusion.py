"""Bidirectional cross-attention and gated masked pooling."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from av_mer.models.config import FusionConfig


def masked_mean(tokens: Tensor, mask: Tensor) -> Tensor:
    """Mean temporal tokens over valid positions only."""

    if tokens.ndim != 3 or mask.shape != tokens.shape[:2]:
        raise ValueError(
            f"Expected tokens [B,T,H] and mask [B,T], received "
            f"{tuple(tokens.shape)} and {tuple(mask.shape)}."
        )
    mask = mask.to(dtype=torch.bool)
    if (~mask.any(dim=1)).any():
        raise ValueError("Cannot pool a token sequence with all positions padded.")
    weights = mask.unsqueeze(-1).to(dtype=tokens.dtype)
    return (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


class BidirectionalCrossAttentionFusion(nn.Module):
    """Fuse independent audio/video timelines without equating token indices."""

    def __init__(self, config: FusionConfig) -> None:
        super().__init__()
        if config.hidden_dim % config.num_heads != 0:
            raise ValueError(
                f"Fusion hidden_dim {config.hidden_dim} must be divisible by "
                f"num_heads {config.num_heads}."
            )
        self.config = config
        self.audio_queries_video = nn.MultiheadAttention(
            config.hidden_dim,
            config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.video_queries_audio = nn.MultiheadAttention(
            config.hidden_dim,
            config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.audio_norm = nn.LayerNorm(config.hidden_dim)
        self.video_norm = nn.LayerNorm(config.hidden_dim)
        self.gate = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        self.output_norm = nn.LayerNorm(config.hidden_dim)

    def forward(
        self,
        audio_tokens: Tensor,
        audio_mask: Tensor,
        video_tokens: Tensor,
        video_mask: Tensor,
    ) -> Tensor:
        """Return one fusion embedding with shape ``[B,H]``."""

        if audio_tokens.shape[0] != video_tokens.shape[0]:
            raise ValueError("Audio and video token batches must have the same size.")
        if audio_tokens.shape[2] != self.config.hidden_dim:
            raise ValueError(
                f"Expected audio hidden dim {self.config.hidden_dim}, "
                f"received {audio_tokens.shape[2]}."
            )
        if video_tokens.shape[2] != self.config.hidden_dim:
            raise ValueError(
                f"Expected video hidden dim {self.config.hidden_dim}, "
                f"received {video_tokens.shape[2]}."
            )
        audio_mask = audio_mask.to(dtype=torch.bool)
        video_mask = video_mask.to(dtype=torch.bool)
        if (~audio_mask.any(dim=1)).any() or (~video_mask.any(dim=1)).any():
            raise ValueError("Cross-attention requires at least one valid token per modality.")

        audio_context, _ = self.audio_queries_video(
            query=audio_tokens,
            key=video_tokens,
            value=video_tokens,
            key_padding_mask=~video_mask,
            need_weights=False,
        )
        video_context, _ = self.video_queries_audio(
            query=video_tokens,
            key=audio_tokens,
            value=audio_tokens,
            key_padding_mask=~audio_mask,
            need_weights=False,
        )
        audio_fused = self.audio_norm(audio_tokens + audio_context)
        video_fused = self.video_norm(video_tokens + video_context)
        audio_fused = audio_fused.masked_fill(~audio_mask.unsqueeze(-1), 0.0)
        video_fused = video_fused.masked_fill(~video_mask.unsqueeze(-1), 0.0)

        audio_pooled = masked_mean(audio_fused, audio_mask)
        video_pooled = masked_mean(video_fused, video_mask)
        gate = torch.sigmoid(self.gate(torch.cat([audio_pooled, video_pooled], dim=-1)))
        return self.output_norm(gate * audio_pooled + (1.0 - gate) * video_pooled)

