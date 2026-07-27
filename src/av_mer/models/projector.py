"""Project fusion features + branch signals into LLM prefix embeddings."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class FusionToLLMProjector(nn.Module):
    """Map the fusion embedding and per-modality evidence to soft LLM tokens.

    Input is the concatenation of ``fusion_embedding[B,H]``, ``p_audio[B,6]``,
    ``p_video[B,6]`` and the scalar disagreement signal ``[B,1]``. Output is a
    prefix of ``num_prefix_tokens`` embeddings of dimension ``llm_hidden`` that is
    prepended to the LLM token embeddings (soft-prompt style).
    """

    def __init__(
        self,
        fusion_dim: int,
        llm_hidden: int,
        *,
        num_classes: int = 6,
        num_prefix_tokens: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if fusion_dim <= 0 or llm_hidden <= 0 or num_prefix_tokens <= 0:
            raise ValueError("projector dims and prefix length must be positive.")
        self.num_prefix_tokens = num_prefix_tokens
        self.llm_hidden = llm_hidden
        # fusion embedding + p_audio + p_video + disagreement scalar
        self.input_dim = fusion_dim + 2 * num_classes + 1
        self.network = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, llm_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(llm_hidden, llm_hidden * num_prefix_tokens),
        )

    def forward(
        self,
        fusion_embedding: Tensor,
        p_audio: Tensor,
        p_video: Tensor,
        disagreement: Tensor,
    ) -> Tensor:
        if fusion_embedding.ndim != 2:
            raise ValueError(
                f"fusion_embedding must be [B,H], got {tuple(fusion_embedding.shape)}."
            )
        batch_size = fusion_embedding.shape[0]
        if disagreement.ndim == 1:
            disagreement = disagreement.unsqueeze(-1)
        features = torch.cat([fusion_embedding, p_audio, p_video, disagreement], dim=-1)
        if features.shape[1] != self.input_dim:
            raise ValueError(
                f"Concatenated projector input has width {features.shape[1]}, "
                f"expected {self.input_dim}."
            )
        prefix = self.network(features)
        return prefix.view(batch_size, self.num_prefix_tokens, self.llm_hidden)
