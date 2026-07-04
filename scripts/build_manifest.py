#!/usr/bin/env python3
"""Build paired CREMA-D manifests, actor splits, and vote soft labels."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
import sys
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from av_mer.data.filename_parser import LABELS, parse_cremad_filename  # noqa: E402
from av_mer.data.split import (  # noqa: E402
    ActorSplitConfig,
    build_actor_to_split,
    compute_manifest_checksum,
    split_actors,
)
from av_mer.data.votes import aggregate_votes  # noqa: E402


LOGGER = logging.getLogger(__name__)
MANIFEST_FIELDS = (
    "clip_id",
    "actor_id",
    "emotion",
    "emotion_id",
    "video_path",
    "audio_path",
    "split",
    "sentence_id",
    "intensity",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True, help="CREMA-D raw-data root")
    parser.add_argument("--out", type=Path, required=True, help="Output split directory")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs/data/cremad.json",
        help="Data split and smoothing configuration",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override configured split seed")
    return parser.parse_args()


def load_config(path: Path, seed_override: int | None) -> tuple[ActorSplitConfig, float]:
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    try:
        actor_counts = config["actor_counts"]
        split_config = ActorSplitConfig(
            train_count=int(actor_counts["train"]),
            val_count=int(actor_counts["val"]),
            test_count=int(actor_counts["test"]),
            seed=int(config["seed"] if seed_override is None else seed_override),
        )
        alpha = float(config["laplace_alpha"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid CREMA-D config {path}: {error}.") from error
    return split_config, alpha


def index_media(directory: Path, suffix: str) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Required media directory does not exist: {directory}.")
    indexed: dict[str, Path] = {}
    duplicates: list[str] = []
    for path in sorted(directory.glob(f"*{suffix}")):
        if path.stem in indexed:
            duplicates.append(path.stem)
        indexed[path.stem] = path
    if duplicates:
        raise ValueError(f"Duplicate clip IDs in {directory}: {sorted(duplicates)[:5]}.")
    return indexed


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def build_manifest(raw_root: Path, output_dir: Path, config_path: Path, seed: int | None) -> None:
    split_config, alpha = load_config(config_path, seed)
    audio_by_clip = index_media(raw_root / "AudioWAV", ".wav")
    video_by_clip = index_media(raw_root / "VideoFlash", ".flv")
    paired_clip_ids = sorted(set(audio_by_clip) & set(video_by_clip))
    audio_only = sorted(set(audio_by_clip) - set(video_by_clip))
    video_only = sorted(set(video_by_clip) - set(audio_by_clip))
    LOGGER.info(
        "Media pairing: paired=%d audio_only=%d video_only=%d",
        len(paired_clip_ids),
        len(audio_only),
        len(video_only),
    )
    if audio_only:
        LOGGER.warning("Audio clips without video (%d): %s", len(audio_only), audio_only[:5])
    if video_only:
        LOGGER.warning("Video clips without audio (%d): %s", len(video_only), video_only[:5])
    if not paired_clip_ids:
        raise ValueError(f"No paired CREMA-D clips found under {raw_root}.")

    parsed_clips = [parse_cremad_filename(clip_id) for clip_id in paired_clip_ids]
    actor_splits = split_actors((clip.actor_id for clip in parsed_clips), split_config)
    actor_to_split = build_actor_to_split(actor_splits)
    rows: list[dict[str, Any]] = []
    for clip in parsed_clips:
        rows.append(
            {
                "clip_id": clip.clip_id,
                "actor_id": clip.actor_id,
                "emotion": clip.emotion,
                "emotion_id": clip.emotion_id,
                "video_path": video_by_clip[clip.clip_id].relative_to(raw_root).as_posix(),
                "audio_path": audio_by_clip[clip.clip_id].relative_to(raw_root).as_posix(),
                "split": actor_to_split[clip.actor_id],
                "sentence_id": clip.sentence_id,
                "intensity": clip.intensity,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, actor_ids in actor_splits.as_dict().items():
        split_rows = [row for row in rows if row["split"] == split_name]
        write_manifest(output_dir / f"manifest_{split_name}.csv", split_rows)
        with (output_dir / f"actors_{split_name}.json").open("w", encoding="utf-8") as handle:
            json.dump(list(actor_ids), handle, indent=2)
            handle.write("\n")
        print(f"split={split_name} clips={len(split_rows)} actors={len(actor_ids)}")

    checksum = compute_manifest_checksum(rows)
    (output_dir / "manifest_checksum.txt").write_text(f"{checksum}\n", encoding="ascii")
    label_mapping = {label: index for index, label in enumerate(LABELS)}
    with (output_dir / "label_mapping.json").open("w", encoding="utf-8") as handle:
        json.dump(label_mapping, handle, indent=2)
        handle.write("\n")

    hard_emotions = {clip.clip_id: clip.emotion for clip in parsed_clips}
    vote_result = aggregate_votes(
        raw_root / "finishedResponses.csv",
        paired_clip_ids,
        hard_emotions,
        alpha=alpha,
    )
    for condition, table in vote_result.tables.items():
        table.to_parquet(output_dir / f"soft_labels_{condition}.parquet", index=False)
        print(
            f"votes={condition} coverage={vote_result.coverage_percent[condition]:.2f}% "
            f"fallback={vote_result.fallback_counts[condition]}"
        )
    print(
        f"paired_clips={len(paired_clip_ids)} audio_only={len(audio_only)} "
        f"video_only={len(video_only)} ignored_vote_rows={vote_result.ignored_row_count}"
    )
    print(f"manifest_checksum={checksum}")


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    build_manifest(args.raw, args.out, args.config, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

