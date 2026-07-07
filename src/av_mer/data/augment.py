"""Train-time data augmentation for audio waveforms and video frame stacks.

Augmentations are intentionally light and label-preserving (they must not change
the emotion). They exist to curb the severe train-set memorisation observed in
the fusion/video baselines (train Macro-F1 ~1.0 vs val ~0.5-0.73) caused by
training with no augmentation at all.

Each augmenter is a callable ``Tensor -> Tensor`` matching the
``TensorAugment`` protocol consumed by :class:`av_mer.data.dataset.CremaDataset`.
Randomness is drawn from a per-instance ``torch.Generator`` so runs stay
reproducible under a fixed seed.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class AudioAugmentConfig:
    noise_std: float = 0.005          # additive Gaussian noise (relative to signal)
    gain_min: float = 0.9             # random amplitude scaling range
    gain_max: float = 1.1
    max_shift_frac: float = 0.1       # circular time-shift up to this fraction of length
    polarity_prob: float = 0.5        # random waveform polarity inversion
    apply_prob: float = 0.8           # probability of applying the whole pipeline


@dataclass(frozen=True, slots=True)
class VideoAugmentConfig:
    flip_prob: float = 0.5            # horizontal flip
    brightness: float = 0.2           # +/- brightness jitter
    contrast: float = 0.2             # +/- contrast jitter
    apply_prob: float = 0.8


class AudioAugment:
    """Waveform augmentation on a 1-D tensor ``[T]`` (values roughly in [-1, 1])."""

    def __init__(self, config: AudioAugmentConfig, *, seed: int = 42) -> None:
        self.config = config
        self._generator = torch.Generator().manual_seed(seed)

    def _rand(self) -> float:
        return float(torch.rand(1, generator=self._generator).item())

    def __call__(self, waveform: Tensor) -> Tensor:
        if waveform.ndim != 1:
            raise ValueError(f"AudioAugment expects a 1-D waveform, got {tuple(waveform.shape)}.")
        if self._rand() > self.config.apply_prob:
            return waveform
        out = waveform
        length = out.shape[0]

        # Circular time shift.
        max_shift = int(length * self.config.max_shift_frac)
        if max_shift > 0:
            shift = int(torch.randint(-max_shift, max_shift + 1, (1,), generator=self._generator).item())
            if shift != 0:
                out = torch.roll(out, shifts=shift, dims=0)

        # Random gain.
        gain = self.config.gain_min + self._rand() * (self.config.gain_max - self.config.gain_min)
        out = out * gain

        # Additive Gaussian noise scaled by the signal RMS.
        if self.config.noise_std > 0:
            rms = out.pow(2).mean().clamp_min(1e-8).sqrt()
            noise = torch.randn(out.shape, generator=self._generator) * (self.config.noise_std * rms)
            out = out + noise

        # Polarity inversion (sign flip is emotion-preserving for speech).
        if self._rand() < self.config.polarity_prob:
            out = -out

        return out.to(dtype=waveform.dtype)


class VideoAugment:
    """Frame-stack augmentation on ``[F, C, H, W]`` tensors in [0, 1]."""

    def __init__(self, config: VideoAugmentConfig, *, seed: int = 42) -> None:
        self.config = config
        self._generator = torch.Generator().manual_seed(seed)

    def _rand(self) -> float:
        return float(torch.rand(1, generator=self._generator).item())

    def __call__(self, frames: Tensor) -> Tensor:
        if frames.ndim != 4:
            raise ValueError(f"VideoAugment expects [F,C,H,W], got {tuple(frames.shape)}.")
        if self._rand() > self.config.apply_prob:
            return frames
        out = frames

        # Horizontal flip applied consistently across all frames in the clip.
        if self._rand() < self.config.flip_prob:
            out = torch.flip(out, dims=[-1])

        # Brightness jitter (additive).
        if self.config.brightness > 0:
            delta = (self._rand() * 2.0 - 1.0) * self.config.brightness
            out = out + delta

        # Contrast jitter (scale around per-clip mean).
        if self.config.contrast > 0:
            factor = 1.0 + (self._rand() * 2.0 - 1.0) * self.config.contrast
            mean = out.mean()
            out = (out - mean) * factor + mean

        return out.clamp(0.0, 1.0).to(dtype=frames.dtype)


def build_augmenters(
    augmentation: dict | None,
    *,
    seed: int,
) -> tuple[AudioAugment | None, VideoAugment | None]:
    """Build (audio, video) augmenters from a loader-config ``augmentation`` block.

    Returns ``(None, None)`` when augmentation is absent or disabled, so callers
    can pass the results straight through to the dataset for the train split only.
    """

    if not augmentation or not augmentation.get("enabled", False):
        return None, None
    audio_cfg = AudioAugmentConfig(**(augmentation.get("audio", {})))
    video_cfg = VideoAugmentConfig(**(augmentation.get("video", {})))
    return (
        AudioAugment(audio_cfg, seed=seed),
        VideoAugment(video_cfg, seed=seed + 1),
    )
