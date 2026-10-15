"""Manifest-backed CREMA-D PyTorch dataset."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any, Callable, Mapping

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from av_mer.data.audio_io import load_audio
from av_mer.data.filename_parser import LABELS
from av_mer.data.video_io import (
    DecodeCounters,
    FaceDetector,
    VideoConfig,
    create_face_detector,
    load_video,
)


TensorAugment = Callable[[Tensor], Tensor]
SOFT_CONDITIONS: tuple[str, ...] = ("voice", "face", "av")
REQUIRED_MANIFEST_COLUMNS = {
    "clip_id",
    "actor_id",
    "emotion",
    "emotion_id",
    "video_path",
    "audio_path",
    "split",
}


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    """Media and deterministic sampling configuration."""

    sample_rate: int
    num_frames: int
    frame_size: int
    face_detector: str
    face_margin: float
    seed: int = 42

    @classmethod
    def from_json(cls, path: str | Path, *, seed: int = 42) -> "DatasetConfig":
        config_path = Path(path)
        with config_path.open(encoding="utf-8") as handle:
            values = json.load(handle)
        try:
            return cls(
                sample_rate=int(values["sample_rate"]),
                num_frames=int(values["num_frames"]),
                frame_size=int(values["frame_size"]),
                face_detector=str(values["face_detector"]),
                face_margin=float(values["face_margin"]),
                seed=seed,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid dataset config {config_path}: {error}.") from error


class CremaDataset(Dataset[dict[str, Any]]):
    """Decode paired media and attach hard and per-modality soft labels."""

    def __init__(
        self,
        manifest_path: str | Path,
        soft_label_paths: Mapping[str, str | Path],
        raw_root: str | Path,
        config: DatasetConfig,
        *,
        split: str,
        face_detector: FaceDetector | None = None,
        audio_augment: TensorAugment | None = None,
        video_augment: TensorAugment | None = None,
        decode_error_report: str | Path | None = None,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"split must be train, val, or test; received {split!r}.")
        self._manifest = pd.read_csv(manifest_path)
        missing_columns = REQUIRED_MANIFEST_COLUMNS - set(self._manifest.columns)
        if missing_columns:
            raise ValueError(
                f"Manifest {manifest_path} is missing columns {sorted(missing_columns)}."
            )
        manifest_splits = set(self._manifest["split"].astype(str))
        if manifest_splits != {split}:
            raise ValueError(
                f"Manifest {manifest_path} contains splits {sorted(manifest_splits)}, "
                f"expected only {split!r}."
            )

        missing_conditions = set(SOFT_CONDITIONS) - set(soft_label_paths)
        if missing_conditions:
            raise ValueError(f"Missing soft-label tables for {sorted(missing_conditions)}.")
        self._soft_labels: dict[str, pd.DataFrame] = {}
        for condition in SOFT_CONDITIONS:
            table = pd.read_parquet(
                soft_label_paths[condition],
                engine="pyarrow",
                use_threads=False,
            ).set_index("clip_id")
            missing_label_columns = set(LABELS) - set(table.columns)
            if missing_label_columns:
                raise ValueError(
                    f"Soft-label table for {condition} is missing "
                    f"{sorted(missing_label_columns)}."
                )
            self._soft_labels[condition] = table

        manifest_clip_ids = set(self._manifest["clip_id"].astype(str))
        for condition, table in self._soft_labels.items():
            missing_clips = sorted(manifest_clip_ids - set(table.index.astype(str)))
            if missing_clips:
                raise ValueError(
                    f"Soft-label table for {condition} is missing clips {missing_clips[:5]}."
                )

        self.raw_root = Path(raw_root)
        self.config = config
        self.split = split
        self.face_detector = face_detector or create_face_detector(config.face_detector)
        self.audio_augment = audio_augment
        self.video_augment = video_augment
        self.decode_error_report = Path(decode_error_report) if decode_error_report else None
        self.counters = DecodeCounters()

    def __len__(self) -> int:
        return len(self._manifest)

    def _record_decode_error(self, clip_id: str, media_path: Path, error: Exception) -> None:
        if self.decode_error_report is None:
            return
        self.decode_error_report.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self.decode_error_report.exists()
        with self.decode_error_report.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            if write_header:
                writer.writerow(["clip_id", "media_path", "error_type", "message"])
            writer.writerow([clip_id, media_path.as_posix(), type(error).__name__, str(error)])

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self._manifest.iloc[index]
        clip_id = str(row["clip_id"])
        audio_path = self.raw_root / str(row["audio_path"])
        video_path = self.raw_root / str(row["video_path"])
        try:
            audio_result = load_audio(audio_path, self.config.sample_rate)
        except Exception as error:
            self._record_decode_error(clip_id, audio_path, error)
            raise
        try:
            video_result = load_video(
                video_path,
                VideoConfig(
                    num_frames=self.config.num_frames,
                    frame_size=self.config.frame_size,
                    face_margin=self.config.face_margin,
                ),
                self.face_detector,
                counters=self.counters,
                training=self.split == "train",
                rng=random.Random(self.config.seed + index) if self.split == "train" else None,
            )
        except Exception as error:
            self._record_decode_error(clip_id, video_path, error)
            raise

        audio = audio_result.waveform
        video = video_result.frames
        if self.split == "train":
            if self.audio_augment is not None:
                audio = self.audio_augment(audio)
            if self.video_augment is not None:
                video = self.video_augment(video)

        item: dict[str, Any] = {
            "clip_id": clip_id,
            "actor_id": int(row["actor_id"]),
            "label": int(row["emotion_id"]),
            "audio": audio.to(dtype=torch.float32),
            "audio_sample_rate": audio_result.sample_rate,
            "video": video.to(dtype=torch.float32),
            "face_failure_count": video_result.face_failure_count,
        }
        for condition in SOFT_CONDITIONS:
            values = self._soft_labels[condition].loc[clip_id, list(LABELS)].to_numpy(
                dtype="float32"
            )
            item[f"soft_{condition}"] = torch.from_numpy(values)
        return item
