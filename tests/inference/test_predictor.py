from __future__ import annotations

from pathlib import Path

import pytest

from av_mer.inference.predictor import PamVlmPredictor


def test_infer_audio_path_finds_sidecar_wav(tmp_path) -> None:
    video = tmp_path / "clip.mp4"
    audio = tmp_path / "clip.wav"
    video.write_bytes(b"video")
    audio.write_bytes(b"audio")

    assert PamVlmPredictor.infer_audio_path(video) == audio


def test_infer_audio_path_requires_sidecar_or_explicit_audio(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="No sidecar audio"):
        PamVlmPredictor.infer_audio_path(Path(tmp_path / "clip.mp4"))
