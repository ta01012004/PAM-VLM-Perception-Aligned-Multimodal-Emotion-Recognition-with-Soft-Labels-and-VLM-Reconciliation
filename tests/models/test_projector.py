"""Projector shape tests (no LLM download)."""

from __future__ import annotations

import pytest
import torch

from av_mer.models.projector import FusionToLLMProjector


def test_projector_output_shape() -> None:
    proj = FusionToLLMProjector(fusion_dim=256, llm_hidden=896, num_prefix_tokens=4)
    b = 3
    prefix = proj(
        torch.randn(b, 256),
        torch.softmax(torch.randn(b, 6), dim=-1),
        torch.softmax(torch.randn(b, 6), dim=-1),
        torch.rand(b),
    )
    assert prefix.shape == (b, 4, 896)
    assert torch.isfinite(prefix).all()


def test_projector_accepts_2d_disagreement() -> None:
    proj = FusionToLLMProjector(fusion_dim=256, llm_hidden=128, num_prefix_tokens=2)
    b = 2
    out = proj(
        torch.randn(b, 256),
        torch.softmax(torch.randn(b, 6), dim=-1),
        torch.softmax(torch.randn(b, 6), dim=-1),
        torch.rand(b, 1),
    )
    assert out.shape == (b, 2, 128)


def test_projector_rejects_bad_fusion_shape() -> None:
    proj = FusionToLLMProjector(fusion_dim=256, llm_hidden=128)
    with pytest.raises(ValueError):
        proj(torch.randn(256), torch.rand(1, 6), torch.rand(1, 6), torch.rand(1))
