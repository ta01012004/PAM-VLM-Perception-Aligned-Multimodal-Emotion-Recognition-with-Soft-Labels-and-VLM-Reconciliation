"""CPU audio decoding and resampling for CREMA-D."""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly
import soundfile as sf
import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class AudioLoadResult:
    """Decoded mono audio and its post-resampling length."""

    waveform: Tensor
    sample_rate: int
    length: int


def load_audio(path: str | Path, sample_rate: int) -> AudioLoadResult:
    """Load a WAV file as mono float32 and resample to ``sample_rate``.

    The returned waveform has shape ``[num_samples]``.
    """

    media_path = Path(path)
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, received {sample_rate}.")
    try:
        samples, source_rate = sf.read(media_path, dtype="float32", always_2d=True)
    except (OSError, RuntimeError) as error:
        raise RuntimeError(f"Could not decode audio file {media_path}: {error}.") from error
    if samples.shape[0] == 0:
        raise ValueError(f"Decoded audio file is empty: {media_path}.")

    mono = samples.mean(axis=1, dtype=np.float32)
    if source_rate != sample_rate:
        common_divisor = gcd(int(source_rate), sample_rate)
        mono = resample_poly(
            mono,
            up=sample_rate // common_divisor,
            down=int(source_rate) // common_divisor,
        ).astype(np.float32, copy=False)
    waveform = torch.from_numpy(np.ascontiguousarray(mono, dtype=np.float32))
    return AudioLoadResult(
        waveform=waveform,
        sample_rate=sample_rate,
        length=int(waveform.numel()),
    )

