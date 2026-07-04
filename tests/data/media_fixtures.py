from pathlib import Path

import cv2
import numpy as np
import soundfile as sf


def write_test_wav(path: Path, *, sample_rate: int = 8000, stereo: bool = False) -> None:
    time = np.arange(sample_rate // 20, dtype=np.float32) / sample_rate
    signal = 0.25 * np.sin(2.0 * np.pi * 220.0 * time)
    samples = np.column_stack([signal, signal * 0.5]) if stereo else signal
    sf.write(path, samples, sample_rate)


def write_test_mp4(path: Path, *, frame_count: int = 5, size: int = 32) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        8.0,
        (size, size),
    )
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV could not create synthetic MP4 fixture at {path}.")
    try:
        for index in range(frame_count):
            frame = np.zeros((size, size, 3), dtype=np.uint8)
            frame[:, :, index % 3] = 40 + index * 30
            cv2.rectangle(frame, (index, index), (size - 1, size - 1), (255, 255, 255), 1)
            writer.write(frame)
    finally:
        writer.release()

