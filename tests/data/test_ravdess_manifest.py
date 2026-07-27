from __future__ import annotations

from scripts.build_zeroshot_ravdess_manifest import build_rows, parse_ravdess_stem


def test_parse_supported_ravdess_stem() -> None:
    assert parse_ravdess_stem("01-01-05-01-01-01-12") == ("angry", 12)


def test_parse_unsupported_ravdess_stem_returns_none() -> None:
    assert parse_ravdess_stem("01-01-08-01-01-01-01") is None
    assert parse_ravdess_stem("03-01-05-01-01-01-12") is None


def test_build_rows_pairs_audio_video_and_skips_unsupported(tmp_path) -> None:
    (tmp_path / "Actor_01").mkdir()
    video = tmp_path / "Actor_01" / "01-01-03-01-01-01-01.mp4"
    audio = tmp_path / "Actor_01" / "01-01-03-01-01-01-01.wav"
    unsupported_video = tmp_path / "Actor_01" / "01-01-08-01-01-01-01.mp4"
    unsupported_audio = tmp_path / "Actor_01" / "01-01-08-01-01-01-01.wav"
    for path in (video, audio, unsupported_video, unsupported_audio):
        path.write_bytes(b"x")

    rows = build_rows(tmp_path)
    assert len(rows) == 1
    assert rows[0]["emotion"] == "happy"
    assert rows[0]["emotion_id"] == 3
    assert rows[0]["video_path"] == "Actor_01/01-01-03-01-01-01-01.mp4"
