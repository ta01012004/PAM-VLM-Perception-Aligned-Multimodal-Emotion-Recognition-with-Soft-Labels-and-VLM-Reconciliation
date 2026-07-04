#!/usr/bin/env python3
"""Decode three real CREMA-D clips; run only through the SLURM wrapper."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from av_mer.data.audio_io import load_audio  # noqa: E402
from av_mer.data.dataset import DatasetConfig  # noqa: E402
from av_mer.data.video_io import DecodeCounters, VideoConfig, create_face_detector, load_video  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=REPOSITORY_ROOT / "data/raw/crema_d")
    parser.add_argument("--config", type=Path, default=REPOSITORY_ROOT / "configs/data/loader.json")
    args = parser.parse_args()

    config = DatasetConfig.from_json(args.config)
    detector = create_face_detector(config.face_detector)
    counters = DecodeCounters()
    video_paths = sorted((args.raw / "VideoFlash").glob("*.flv"))[:3]
    if len(video_paths) != 3:
        raise RuntimeError(f"Expected at least 3 FLV clips under {args.raw}, found {len(video_paths)}.")
    for video_path in video_paths:
        audio_path = args.raw / "AudioWAV" / f"{video_path.stem}.wav"
        audio = load_audio(audio_path, config.sample_rate)
        video = load_video(
            video_path,
            VideoConfig(config.num_frames, config.frame_size, config.face_margin),
            detector,
            counters=counters,
        )
        print(
            f"clip={video_path.stem} audio={tuple(audio.waveform.shape)} "
            f"video={tuple(video.frames.shape)} face_failures={video.face_failure_count}"
        )
    print(
        f"decoded=3 face_detection_failures={counters.face_detection_failures} "
        f"undecodable_media={counters.undecodable_media}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

