"""Audio/video disagreement signal tests."""

from __future__ import annotations

import torch

from av_mer.data.disagreement import audio_video_kl, high_disagreement_mask


def test_kl_zero_when_distributions_match() -> None:
    logits = torch.randn(4, 6)
    kl = audio_video_kl(logits, logits.clone())
    assert torch.allclose(kl, torch.zeros(4), atol=1e-6)


def test_kl_positive_and_symmetric() -> None:
    a = torch.tensor([[3.0, 0, 0, 0, 0, 0]])
    v = torch.tensor([[0.0, 0, 0, 0, 0, 3.0]])
    kl_av = audio_video_kl(a, v)
    kl_va = audio_video_kl(v, a)
    assert kl_av.item() > 0
    assert torch.allclose(kl_av, kl_va, atol=1e-6)  # symmetric


def test_high_disagreement_mask() -> None:
    a = torch.tensor([[5.0, 0, 0, 0, 0, 0], [1.0, 1, 1, 1, 1, 1]])
    v = torch.tensor([[0.0, 0, 0, 0, 0, 5.0], [1.0, 1, 1, 1, 1, 1]])
    mask = high_disagreement_mask(a, v, threshold=0.5)
    assert mask[0].item() is True
    assert mask[1].item() is False  # identical uniform -> zero KL
