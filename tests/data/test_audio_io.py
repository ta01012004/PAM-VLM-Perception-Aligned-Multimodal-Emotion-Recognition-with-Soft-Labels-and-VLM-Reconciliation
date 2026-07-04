from pathlib import Path

import pytest
import torch

from av_mer.data.audio_io import load_audio
from media_fixtures import write_test_wav


def test_audio_loader_mixes_to_mono_and_resamples(tmp_path: Path) -> None:
    path = tmp_path / "stereo.wav"
    write_test_wav(path, sample_rate=8000, stereo=True)

    result = load_audio(path, sample_rate=16000)

    assert result.waveform.dtype == torch.float32
    assert result.waveform.ndim == 1
    assert result.sample_rate == 16000
    assert result.length == 800
    assert torch.isfinite(result.waveform).all()


def test_audio_loader_reports_missing_file(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="Could not decode audio"):
        load_audio(tmp_path / "missing.wav", sample_rate=16000)

