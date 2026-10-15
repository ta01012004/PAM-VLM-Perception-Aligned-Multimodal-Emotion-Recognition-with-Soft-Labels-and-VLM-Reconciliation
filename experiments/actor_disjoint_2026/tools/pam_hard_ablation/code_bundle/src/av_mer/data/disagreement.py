"""Audio/video disagreement signal for routing clips to the VLM reconciler."""

from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


def audio_video_kl(audio_logits: Tensor, video_logits: Tensor) -> Tensor:
    """Symmetric KL between the audio and video class distributions, per sample.

    Returns a ``[B]`` tensor. Higher = the two modalities disagree more about the
    emotion. Symmetric (Jensen-style average of both KL directions) so neither
    modality is privileged.
    """

    if audio_logits.shape != video_logits.shape or audio_logits.ndim != 2:
        raise ValueError(
            f"Expected matching [B,C] logits, got {tuple(audio_logits.shape)} "
            f"and {tuple(video_logits.shape)}."
        )
    log_pa = F.log_softmax(audio_logits, dim=-1)
    log_pv = F.log_softmax(video_logits, dim=-1)
    pa = log_pa.exp()
    pv = log_pv.exp()
    kl_av = (pa * (log_pa - log_pv)).sum(dim=-1)
    kl_va = (pv * (log_pv - log_pa)).sum(dim=-1)
    return 0.5 * (kl_av + kl_va)


def high_disagreement_mask(
    audio_logits: Tensor, video_logits: Tensor, *, threshold: float
) -> Tensor:
    """Boolean ``[B]`` mask selecting samples whose A/V KL exceeds ``threshold``."""

    if threshold < 0:
        raise ValueError(f"threshold must be non-negative, got {threshold}.")
    return audio_video_kl(audio_logits, video_logits) > threshold
