#!/usr/bin/env python3
"""Cache frozen PAM evidence for fast projector/Qwen training."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from av_mer.data.collate import collate_av_batch  # noqa: E402
from av_mer.data.dataset import CremaDataset, DatasetConfig  # noqa: E402
from av_mer.data.disagreement import audio_video_kl  # noqa: E402
from av_mer.data.filename_parser import LABEL_TO_ID  # noqa: E402
from av_mer.data.vlm_cache import save_vlm_cache  # noqa: E402
from av_mer.models.config import PamVlmConfig  # noqa: E402
from av_mer.models.pam_vlm import PamVlmModel  # noqa: E402
from av_mer.training.checkpoint import load_checkpoint  # noqa: E402

LOGGER = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "val", "test"),
        default=("train", "val"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        values = yaml.safe_load(handle)
    if not isinstance(values, dict):
        raise ValueError(f"Config {path} must contain a mapping.")
    return values


def _soft_label_paths(data: Mapping[str, Any]) -> dict[str, Path]:
    values = data["soft_labels"]
    return {
        key: _repo_path(str(values[key]))
        for key in ("voice", "face", "av")
    }


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("Feature caching must run inside a SLURM allocation.")
    if not torch.cuda.is_available():
        raise RuntimeError("Feature caching requires CUDA.")
    config = _load_yaml(args.config)
    data = config["data"]
    cache = config["cache"]
    device = torch.device("cuda", torch.cuda.current_device())
    model = PamVlmModel(PamVlmConfig.from_json(_repo_path(config["model_config"])))
    checksum = _repo_path(data["split_checksum"]).read_text(encoding="ascii").strip()
    label_mapping = json.loads(_repo_path(data["label_mapping"]).read_text(encoding="utf-8"))
    load_checkpoint(
        _repo_path(config["backbone_checkpoint"]),
        model=model,
        expected_label_mapping=LABEL_TO_ID,
        map_location=device,
        restore_rng=False,
    )
    if label_mapping != LABEL_TO_ID:
        raise ValueError("Configured label mapping does not match canonical labels.")
    model.to(device).eval()
    loader_config = DatasetConfig.from_json(
        _repo_path(data["loader_config"]),
        seed=int(config["seed"]),
    )
    output_dir = _repo_path(cache["output_dir"])

    for split in args.splits:
        manifest_key = f"{split}_manifest"
        if manifest_key not in data:
            raise ValueError(f"Config has no data.{manifest_key}.")
        destination = output_dir / f"{split}.pt"
        if destination.exists() and not args.overwrite:
            LOGGER.info("Keeping existing cache %s", destination)
            continue
        dataset = CremaDataset(
            _repo_path(data[manifest_key]),
            _soft_label_paths(data),
            _repo_path(data["raw_root"]),
            loader_config,
            split=split,
            decode_error_report=_repo_path(data["decode_error_report"]),
        )
        loader = DataLoader(
            dataset,
            batch_size=int(cache.get("batch_size", 4)),
            shuffle=False,
            num_workers=int(cache.get("num_workers", 8)),
            pin_memory=True,
            persistent_workers=int(cache.get("num_workers", 8)) > 0,
            collate_fn=collate_av_batch,
        )
        values: dict[str, list[torch.Tensor]] = {
            key: []
            for key in (
                "actor_ids",
                "fusion_embedding",
                "audio_probs",
                "video_probs",
                "fusion_probs",
                "disagreement",
                "labels",
            )
        }
        clip_ids: list[str] = []
        with torch.inference_mode():
            for batch in loader:
                output = model(
                    batch["audio_values"].to(device, non_blocking=True),
                    batch["audio_mask"].to(device, non_blocking=True),
                    batch["video_values"].to(device, non_blocking=True),
                    batch["video_mask"].to(device, non_blocking=True),
                )
                audio_probs = torch.softmax(output["audio_logits"].float(), dim=-1)
                video_probs = torch.softmax(output["video_logits"].float(), dim=-1)
                fusion_probs = torch.softmax(output["logits"].float(), dim=-1)
                disagreement = audio_video_kl(
                    output["audio_logits"].float(),
                    output["video_logits"].float(),
                )
                clip_ids.extend(batch["clip_ids"])
                values["actor_ids"].append(batch["actor_ids"])
                values["labels"].append(batch["labels"])
                values["fusion_embedding"].append(
                    output["fusion_embedding"].detach().cpu().to(torch.float16)
                )
                values["audio_probs"].append(audio_probs.cpu().to(torch.float16))
                values["video_probs"].append(video_probs.cpu().to(torch.float16))
                values["fusion_probs"].append(fusion_probs.cpu().to(torch.float16))
                values["disagreement"].append(disagreement.cpu().to(torch.float32))
        stacked = {key: torch.cat(parts) for key, parts in values.items()}
        save_vlm_cache(
            destination,
            clip_ids=clip_ids,
            metadata={
                "split": split,
                "protocol_id": data["protocol_id"],
                "split_checksum": checksum,
                "backbone_checkpoint": str(_repo_path(config["backbone_checkpoint"])),
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            **stacked,
        )
        LOGGER.info("Cached %d %s samples to %s", len(dataset), split, destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
