from pathlib import Path

import numpy as np
import torch

from av_mer.data.video_io import DecodeCounters, VideoConfig, load_video
from media_fixtures import write_test_mp4


class AlwaysFailDetector:
    def __call__(self, frame_rgb: np.ndarray) -> None:
        return None


class FullFrameDetector:
    def __call__(self, frame_rgb: np.ndarray) -> tuple[int, int, int, int]:
        height, width = frame_rgb.shape[:2]
        return 0, 0, width, height


def test_video_loader_center_crop_fallback_is_counted(tmp_path: Path) -> None:
    path = tmp_path / "fixture.mp4"
    write_test_mp4(path, frame_count=5)
    counters = DecodeCounters()

    result = load_video(
        path,
        VideoConfig(num_frames=3, frame_size=16),
        AlwaysFailDetector(),
        counters=counters,
    )

    assert result.frames.shape == (3, 3, 16, 16)
    assert result.frames.dtype == torch.float32
    assert result.face_failure_count == 3
    assert counters.face_detection_failures == 3
    assert counters.undecodable_media == 0


def test_validation_sampling_is_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "fixture.mp4"
    write_test_mp4(path, frame_count=7)
    config = VideoConfig(num_frames=4, frame_size=16)

    first = load_video(path, config, FullFrameDetector(), training=False)
    second = load_video(path, config, FullFrameDetector(), training=False)

    assert first.sampled_indices == second.sampled_indices
    assert torch.equal(first.frames, second.frames)

