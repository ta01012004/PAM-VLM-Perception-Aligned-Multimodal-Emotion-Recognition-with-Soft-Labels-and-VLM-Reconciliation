import torch

from av_mer.models.pam_vlm import PamVlmModel
from helpers import tiny_batch, tiny_model_config


def test_frozen_backbones_are_excluded_from_trainable_count() -> None:
    model = PamVlmModel(tiny_model_config(freeze_backbones=True))
    counts = model.parameter_counts()
    expected_trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )

    assert counts["total"] > counts["trainable"] > 0
    assert counts["trainable"] == expected_trainable
    assert not any(parameter.requires_grad for parameter in model.audio_encoder.backbone.parameters())
    assert not any(parameter.requires_grad for parameter in model.video_encoder.backbone.parameters())


def test_gradients_reach_heads_fusion_and_projectors_only() -> None:
    model = PamVlmModel(tiny_model_config(freeze_backbones=True))
    output = model(**tiny_batch())

    output["logits"].sum().backward()

    assert model.fusion_head.network[-1].weight.grad is not None
    assert model.fusion.gate.weight.grad is not None
    assert model.audio_encoder.projector[0].weight.grad is not None
    assert model.video_encoder.projector[0].weight.grad is not None
    assert all(parameter.grad is None for parameter in model.audio_encoder.backbone.parameters())
    assert all(parameter.grad is None for parameter in model.video_encoder.backbone.parameters())
