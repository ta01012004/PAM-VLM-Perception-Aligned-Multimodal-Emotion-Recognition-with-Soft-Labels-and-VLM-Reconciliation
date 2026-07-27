#!/usr/bin/env python3
"""Evaluate a trained VLM reconciler with its validation-selected routing threshold."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any

import torch
from torch.utils.data import DataLoader
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from av_mer.data.filename_parser import LABELS  # noqa: E402
from av_mer.data.vlm_cache import CachedVlmDataset  # noqa: E402
from av_mer.evaluation.metrics import compute_classification_metrics  # noqa: E402
from av_mer.evaluation.vlm_cached import predict_cached  # noqa: E402
from av_mer.evaluation.vlm_routing import routed_predictions  # noqa: E402
from av_mer.models.llm_adapter import LLMAdapterConfig, LLMReconciler  # noqa: E402
from av_mer.models.projector import FusionToLLMProjector  # noqa: E402
from av_mer.models.vlm_text import EmotionTextCodec  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def main() -> int:
    args = _parse_args()
    if "SLURM_JOB_ID" not in os.environ or not torch.cuda.is_available():
        raise RuntimeError("VLM evaluation requires a GPU SLURM allocation.")
    with args.config.open(encoding="utf-8") as handle:
        config: dict[str, Any] = yaml.safe_load(handle)
    checkpoint = torch.load(_repo_path(args.checkpoint), map_location="cpu", weights_only=False)
    vlm = config["vlm"]
    device = torch.device("cuda", torch.cuda.current_device())
    dataset = CachedVlmDataset(
        _repo_path(config["cache"]["output_dir"]) / f"{args.split}.pt"
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"].get("eval_batch_size", 8)),
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    reconciler = LLMReconciler(
        LLMAdapterConfig(
            model_id=str(vlm["model_id"]),
            lora_rank=int(vlm["lora_rank"]),
            lora_alpha=int(vlm["lora_alpha"]),
            lora_dropout=float(vlm["lora_dropout"]),
            num_prefix_tokens=int(vlm["num_prefix_tokens"]),
            max_new_tokens=int(vlm.get("max_new_tokens", 24)),
            freeze_base=True,
        )
    ).to(device)
    reconciler.load_adapter_state_dict(checkpoint["adapter_state"])
    projector = FusionToLLMProjector(
        fusion_dim=int(dataset.tensors["fusion_embedding"].shape[1]),
        llm_hidden=reconciler.hidden_size,
        num_prefix_tokens=int(vlm["num_prefix_tokens"]),
        dropout=float(vlm.get("projector_dropout", 0.1)),
    ).to(device)
    projector.load_state_dict(checkpoint["projector_state"])
    codec = EmotionTextCodec(
        reconciler.tokenizer,
        _repo_path(vlm["prompt_file"]).read_text(encoding="utf-8"),
    )
    result = predict_cached(
        reconciler=reconciler,
        projector=projector,
        codec=codec,
        loader=loader,
        device=device,
        amp=bool(config["training"].get("amp", True)),
    )
    threshold = float(checkpoint["routing_threshold"])
    routed = routed_predictions(
        result["classifier_predictions"],
        result["vlm_predictions"],
        result["disagreement"],
        threshold,
        classifier_confidence=result["classifier_confidence"],
        classifier_confidence_max=float(
            checkpoint.get("classifier_confidence_max", float("inf"))
        ),
    )
    metrics = {
        "split": args.split,
        "routing_threshold_from_validation": threshold,
        "classifier_confidence_max_from_validation": float(
            checkpoint.get("classifier_confidence_max", float("inf"))
        ),
        "routed_fraction": float((result["disagreement"] >= threshold).mean()),
        "classifier": result["classifier_metrics"],
        "vlm": result["vlm_metrics"],
        "routed": compute_classification_metrics(result["targets"], routed),
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }
    output_dir = _repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "clip_id",
                "true",
                "classifier",
                "vlm",
                "routed",
                "disagreement",
                "classifier_confidence",
            ]
        )
        for index, clip_id in enumerate(result["clip_ids"]):
            writer.writerow(
                [
                    clip_id,
                    LABELS[int(result["targets"][index])],
                    LABELS[int(result["classifier_predictions"][index])],
                    LABELS[int(result["vlm_predictions"][index])],
                    LABELS[int(routed[index])],
                    float(result["disagreement"][index]),
                    float(result["classifier_confidence"][index]),
                ]
            )
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
