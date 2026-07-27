#!/usr/bin/env python3
"""Build a balanced Mini-CREMA-D subset from the locked train/validation splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data/splits/clip_801010_s42"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/splits/mini_crema_vlm_s42"),
    )
    parser.add_argument("--train-per-class", type=int, default=130)
    parser.add_argument("--val-per-class", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def balanced_sample(
    table: pd.DataFrame,
    *,
    per_class: int,
    seed: int,
) -> pd.DataFrame:
    if per_class <= 0:
        raise ValueError("per_class must be positive.")
    samples = []
    for offset, (_, group) in enumerate(sorted(table.groupby("emotion"))):
        if len(group) < per_class:
            raise ValueError(
                f"Class {group.iloc[0]['emotion']} has {len(group)} rows, "
                f"fewer than requested {per_class}."
            )
        samples.append(group.sample(n=per_class, random_state=seed + offset))
    result = pd.concat(samples, ignore_index=True)
    return result.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _checksum(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> int:
    args = _parse_args()
    source = _repo_path(args.source_dir)
    output = _repo_path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    counts: dict[str, dict[str, int]] = {}
    for split, per_class in (
        ("train", args.train_per_class),
        ("val", args.val_per_class),
    ):
        table = pd.read_csv(source / f"manifest_{split}.csv")
        sample = balanced_sample(table, per_class=per_class, seed=args.seed)
        destination = output / f"manifest_{split}.csv"
        sample.to_csv(destination, index=False)
        written.append(destination)
        counts[split] = {
            str(key): int(value)
            for key, value in sample["emotion"].value_counts().sort_index().items()
        }
    label_mapping_source = source / "label_mapping.json"
    label_mapping_destination = output / "label_mapping.json"
    label_mapping_destination.write_bytes(label_mapping_source.read_bytes())
    written.append(label_mapping_destination)
    checksum = _checksum(written)
    (output / "manifest_checksum.txt").write_text(checksum + "\n", encoding="ascii")
    metadata = {
        "protocol_id": "mini-cremad-vlm-s42",
        "source_protocol": "cremad-6class-clip-801010-s42-v2",
        "seed": args.seed,
        "counts": counts,
        "total": sum(sum(split.values()) for split in counts.values()),
        "test_used": False,
        "checksum": checksum,
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
