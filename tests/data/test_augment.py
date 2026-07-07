"""Unit tests for train-time augmentation."""

from __future__ import annotations

import torch

from av_mer.data.augment import (
    AudioAugment,
    AudioAugmentConfig,
    VideoAugment,
    VideoAugmentConfig,
    build_augmenters,
)


def test_audio_augment_preserves_shape_and_finiteness() -> None:
    aug = AudioAugment(AudioAugmentConfig(apply_prob=1.0), seed=0)
    wav = torch.randn(16000)
    out = aug(wav)
    assert out.shape == wav.shape
    assert out.dtype == wav.dtype
    assert torch.isfinite(out).all()
    # With apply_prob=1 and noise/gain, output should differ from input.
    assert not torch.allclose(out, wav)


def test_audio_augment_is_reproducible_under_seed() -> None:
    wav = torch.randn(8000)
    a = AudioAugment(AudioAugmentConfig(apply_prob=1.0), seed=123)
    b = AudioAugment(AudioAugmentConfig(apply_prob=1.0), seed=123)
    assert torch.allclose(a(wav.clone()), b(wav.clone()))


def test_video_augment_preserves_shape_and_range() -> None:
    aug = VideoAugment(VideoAugmentConfig(apply_prob=1.0), seed=0)
    frames = torch.rand(16, 3, 64, 64)
    out = aug(frames)
    assert out.shape == frames.shape
    assert out.dtype == frames.dtype
    assert torch.isfinite(out).all()
    assert out.min() >= 0.0 and out.max() <= 1.0  # clamped to valid image range


def test_video_flip_applied_consistently_across_frames() -> None:
    # flip_prob=1, no jitter -> output must be an exact horizontal flip.
    aug = VideoAugment(
        VideoAugmentConfig(flip_prob=1.0, brightness=0.0, contrast=0.0, apply_prob=1.0),
        seed=0,
    )
    frames = torch.rand(4, 3, 8, 8)
    out = aug(frames)
    assert torch.allclose(out, torch.flip(frames, dims=[-1]))


def test_build_augmenters_disabled_returns_none() -> None:
    assert build_augmenters(None, seed=42) == (None, None)
    assert build_augmenters({"enabled": False}, seed=42) == (None, None)


def test_build_augmenters_enabled_returns_callables() -> None:
    audio_aug, video_aug = build_augmenters(
        {"enabled": True, "audio": {"apply_prob": 1.0}, "video": {"apply_prob": 1.0}},
        seed=42,
    )
    assert isinstance(audio_aug, AudioAugment)
    assert isinstance(video_aug, VideoAugment)
    assert torch.isfinite(audio_aug(torch.randn(4000))).all()
    assert torch.isfinite(video_aug(torch.rand(4, 3, 16, 16))).all()
