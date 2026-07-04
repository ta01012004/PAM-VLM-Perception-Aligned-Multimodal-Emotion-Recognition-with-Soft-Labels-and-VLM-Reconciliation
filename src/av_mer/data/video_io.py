"""CPU video decoding, temporal sampling, and face cropping."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Protocol

import cv2
import numpy as np
import torch
from torch import Tensor


FaceBox = tuple[int, int, int, int]


class FaceDetector(Protocol):
    """Return an ``(x, y, width, height)`` box or ``None`` on failure."""

    def __call__(self, frame_rgb: np.ndarray) -> FaceBox | None: ...


@dataclass(slots=True)
class DecodeCounters:
    """Mutable decoding diagnostics owned by one loader or dataset."""

    face_detection_failures: int = 0
    undecodable_media: int = 0


@dataclass(frozen=True, slots=True)
class VideoConfig:
    """Configuration for deterministic video preprocessing."""

    num_frames: int
    frame_size: int
    face_margin: float = 0.15


@dataclass(frozen=True, slots=True)
class VideoLoadResult:
    """Sampled RGB frames and per-clip diagnostics."""

    frames: Tensor
    sampled_indices: tuple[int, ...]
    source_frame_count: int
    face_failure_count: int


class OpenCvHaarFaceDetector:
    """Offline OpenCV Haar face detector with no model download."""

    def __init__(self) -> None:
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(str(cascade_path))
        if self._cascade.empty():
            raise RuntimeError(f"Could not load OpenCV Haar cascade from {cascade_path}.")

    def __call__(self, frame_rgb: np.ndarray) -> FaceBox | None:
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        boxes = self._cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
        if len(boxes) == 0:
            return None
        x, y, width, height = max(boxes, key=lambda box: int(box[2]) * int(box[3]))
        return int(x), int(y), int(width), int(height)


class NoFaceDetector:
    """Detector that explicitly selects the center-crop fallback policy."""

    def __call__(self, frame_rgb: np.ndarray) -> None:
        del frame_rgb
        return None


def create_face_detector(name: str) -> FaceDetector:
    """Create a configured offline face detector."""

    if name == "opencv_haar":
        return OpenCvHaarFaceDetector()
    if name == "none":
        return NoFaceDetector()
    raise ValueError(
        f"Unknown face_detector {name!r}; expected 'opencv_haar' or 'none'."
    )


def _decode_all_frames(path: Path) -> list[np.ndarray]:
    capture = cv2.VideoCapture(
        str(path),
        cv2.CAP_FFMPEG,
        [cv2.CAP_PROP_N_THREADS, 1],
    )
    if not capture.isOpened():
        capture.release()
        capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video file {path} with OpenCV.")
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if not frames:
        raise RuntimeError(f"Video file {path} decoded zero frames.")
    return frames


def _sample_indices(
    frame_count: int,
    num_frames: int,
    *,
    training: bool,
    rng: random.Random | None,
) -> tuple[int, ...]:
    if frame_count <= 0 or num_frames <= 0:
        raise ValueError(
            f"frame_count and num_frames must be positive; received "
            f"{frame_count} and {num_frames}."
        )
    if training and frame_count >= num_frames:
        generator = rng if rng is not None else random
        return tuple(sorted(generator.sample(range(frame_count), num_frames)))
    return tuple(np.rint(np.linspace(0, frame_count - 1, num_frames)).astype(int).tolist())


def _center_square_crop(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    side = min(height, width)
    y_start = (height - side) // 2
    x_start = (width - side) // 2
    return frame[y_start : y_start + side, x_start : x_start + side]


def _aligned_square_face_crop(
    frame: np.ndarray,
    box: FaceBox,
    margin: float,
) -> np.ndarray:
    x, y, width, height = box
    frame_height, frame_width = frame.shape[:2]
    if width <= 0 or height <= 0:
        return _center_square_crop(frame)
    side = max(width, height) * (1.0 + 2.0 * margin)
    center_x = x + width / 2.0
    center_y = y + height / 2.0
    x_start = max(0, int(round(center_x - side / 2.0)))
    y_start = max(0, int(round(center_y - side / 2.0)))
    x_end = min(frame_width, int(round(center_x + side / 2.0)))
    y_end = min(frame_height, int(round(center_y + side / 2.0)))
    crop = frame[y_start:y_end, x_start:x_end]
    return _center_square_crop(crop) if crop.size else _center_square_crop(frame)


def load_video(
    path: str | Path,
    config: VideoConfig,
    face_detector: FaceDetector,
    *,
    counters: DecodeCounters | None = None,
    training: bool = False,
    rng: random.Random | None = None,
) -> VideoLoadResult:
    """Decode and return float RGB frames with shape ``[F,C,H,W]``."""

    if config.frame_size <= 0 or config.num_frames <= 0:
        raise ValueError(
            f"Video frame_size and num_frames must be positive, received {config}."
        )
    if config.face_margin < 0:
        raise ValueError(f"face_margin must be non-negative, received {config.face_margin}.")
    media_path = Path(path)
    try:
        decoded_frames = _decode_all_frames(media_path)
    except RuntimeError:
        if counters is not None:
            counters.undecodable_media += 1
        raise

    sampled_indices = _sample_indices(
        len(decoded_frames),
        config.num_frames,
        training=training,
        rng=rng,
    )
    processed_frames: list[np.ndarray] = []
    face_failure_count = 0
    for index in sampled_indices:
        frame = decoded_frames[index]
        box = face_detector(frame)
        if box is None:
            face_failure_count += 1
            crop = _center_square_crop(frame)
        else:
            crop = _aligned_square_face_crop(frame, box, config.face_margin)
        resized = cv2.resize(
            crop,
            (config.frame_size, config.frame_size),
            interpolation=cv2.INTER_LINEAR,
        )
        processed_frames.append(resized)

    if counters is not None:
        counters.face_detection_failures += face_failure_count
    frame_array = np.stack(processed_frames).astype(np.float32) / 255.0
    frame_tensor = torch.from_numpy(frame_array).permute(0, 3, 1, 2).contiguous()
    return VideoLoadResult(
        frames=frame_tensor,
        sampled_indices=sampled_indices,
        source_frame_count=len(decoded_frames),
        face_failure_count=face_failure_count,
    )
