#!/usr/bin/env python3
"""Build a six-class RAVDESS zero-shot manifest for PAM-VLM.

Expected raw tree can be either flat or nested. The script searches recursively
for RAVDESS audio/video filenames and pairs clips by extension-free stem.
Unsupported RAVDESS emotions (calm, surprised) are skipped because the current
PAM-VLM parser and classifier use the six CREMA-D labels.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

RAVDESS_TO_LABEL = {
    "01": "neutral",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fear",
    "07": "disgust",
}
LABEL_TO_ID = {
    "angry": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "neutral": 4,
    "sad": 5,
}
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv"}
AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".ogg"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/ravdess"))
    parser.add_argument("--output", type=Path, default=Path("data/splits/zeroshot_ravdess.csv"))
    return parser.parse_args()


def _repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def parse_ravdess_stem(stem: str) -> tuple[str, int] | None:
    """Return label/actor for supported full-AV speech, otherwise ``None``."""

    parts = stem.split("-")
    if len(parts) != 7:
        return None
    if parts[0] != "01" or parts[1] != "01":
        return None
    emotion_code = parts[2]
    label = RAVDESS_TO_LABEL.get(emotion_code)
    if label is None:
        return None
    return label, int(parts[6])


def build_rows(raw_root: Path) -> list[dict[str, str | int]]:
    videos: dict[str, Path] = {}
    audios: dict[str, Path] = {}
    for path in raw_root.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in VIDEO_SUFFIXES:
            videos[path.stem] = path
        elif suffix in AUDIO_SUFFIXES:
            audios[path.stem] = path

    rows: list[dict[str, str | int]] = []
    for stem, video_path in sorted(videos.items()):
        parsed = parse_ravdess_stem(stem)
        audio_path = audios.get(stem)
        if parsed is None or audio_path is None:
            continue
        label, actor_id = parsed
        rows.append({
            "clip_id": stem,
            "actor_id": actor_id,
            "emotion": label,
            "emotion_id": LABEL_TO_ID[label],
            "video_path": video_path.relative_to(raw_root).as_posix(),
            "audio_path": audio_path.relative_to(raw_root).as_posix(),
            "split": "test",
        })
    return rows


def main() -> int:
    args = _parse_args()
    raw_root = _repo_path(args.raw_root)
    if not raw_root.exists():
        raise FileNotFoundError(f"RAVDESS raw root not found: {raw_root}")
    rows = build_rows(raw_root)
    if not rows:
        raise RuntimeError(
            f"No paired six-class RAVDESS audio/video clips found under {raw_root}."
        )
    output = _repo_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
