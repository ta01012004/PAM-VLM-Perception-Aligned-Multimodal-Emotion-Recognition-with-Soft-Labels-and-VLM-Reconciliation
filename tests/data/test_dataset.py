from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from av_mer.data.collate import collate_av_batch
from av_mer.data.dataset import CremaDataset, DatasetConfig
from av_mer.data.filename_parser import LABELS
from media_fixtures import write_test_mp4, write_test_wav


class FullFrameDetector:
    def __call__(self, frame_rgb: np.ndarray) -> tuple[int, int, int, int]:
        height, width = frame_rgb.shape[:2]
        return 0, 0, width, height


def build_fixture(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    rows = []
    soft_rows = []
    for index in range(4):
        clip_id = f"100{index + 1}_DFA_ANG_XX"
        audio_path = tmp_path / f"{clip_id}.wav"
        video_path = tmp_path / f"{clip_id}.mp4"
        write_test_wav(audio_path)
        write_test_mp4(video_path, frame_count=4 + index)
        rows.append(
            {
                "clip_id": clip_id,
                "actor_id": 1001 + index,
                "emotion": "angry",
                "emotion_id": 0,
                "video_path": video_path.name,
                "audio_path": audio_path.name,
                "split": "val",
            }
        )
        soft_row = {"clip_id": clip_id, **{label: 0.0 for label in LABELS}}
        soft_row["angry"] = 1.0
        soft_rows.append(soft_row)

    manifest_path = tmp_path / "manifest_val.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    soft_paths = {}
    for condition in ("voice", "face", "av"):
        path = tmp_path / f"soft_{condition}.parquet"
        pd.DataFrame(soft_rows).to_parquet(path, index=False)
        soft_paths[condition] = path
    return manifest_path, soft_paths


def test_dataset_is_deterministic_and_iterates_two_batches(tmp_path: Path) -> None:
    manifest_path, soft_paths = build_fixture(tmp_path)
    dataset = CremaDataset(
        manifest_path,
        soft_paths,
        tmp_path,
        DatasetConfig(
            sample_rate=8000,
            num_frames=3,
            frame_size=16,
            face_detector="none",
            face_margin=0.0,
        ),
        split="val",
        face_detector=FullFrameDetector(),
    )

    first = dataset[0]
    second = dataset[0]
    assert first["audio"].shape == (400,)
    assert first["video"].shape == (3, 3, 16, 16)
    assert torch.equal(first["video"], second["video"])
    assert first["soft_voice"].shape == (6,)

    batches = list(DataLoader(dataset, batch_size=2, collate_fn=collate_av_batch))
    assert len(batches) == 2
    assert all(batch["audio_values"].shape == (2, 400) for batch in batches)
    assert all(batch["video_values"].shape == (2, 3, 3, 16, 16) for batch in batches)


def test_augmentation_hook_is_disabled_for_validation(tmp_path: Path) -> None:
    manifest_path, soft_paths = build_fixture(tmp_path)
    calls = {"audio": 0, "video": 0}

    def audio_augment(tensor: torch.Tensor) -> torch.Tensor:
        calls["audio"] += 1
        return tensor

    def video_augment(tensor: torch.Tensor) -> torch.Tensor:
        calls["video"] += 1
        return tensor

    dataset = CremaDataset(
        manifest_path,
        soft_paths,
        tmp_path,
        DatasetConfig(8000, 3, 16, "none", 0.0),
        split="val",
        face_detector=FullFrameDetector(),
        audio_augment=audio_augment,
        video_augment=video_augment,
    )

    _ = dataset[0]
    assert calls == {"audio": 0, "video": 0}


def test_audio_decode_error_records_the_audio_path(tmp_path: Path) -> None:
    manifest_path, soft_paths = build_fixture(tmp_path)
    manifest = pd.read_csv(manifest_path)
    missing_name = str(manifest.loc[0, "audio_path"])
    (tmp_path / missing_name).unlink()
    report_path = tmp_path / "decode_errors.csv"
    dataset = CremaDataset(
        manifest_path,
        soft_paths,
        tmp_path,
        DatasetConfig(8000, 3, 16, "none", 0.0),
        split="val",
        face_detector=FullFrameDetector(),
        decode_error_report=report_path,
    )

    with pytest.raises(RuntimeError, match="Could not decode audio"):
        _ = dataset[0]

    report = pd.read_csv(report_path)
    assert report.loc[0, "media_path"].endswith(missing_name)
    assert report.loc[0, "error_type"] == "RuntimeError"
