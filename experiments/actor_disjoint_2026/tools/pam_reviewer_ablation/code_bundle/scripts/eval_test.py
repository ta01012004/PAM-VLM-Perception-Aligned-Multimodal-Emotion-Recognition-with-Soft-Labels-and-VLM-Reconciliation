#!/usr/bin/env python3
"""Evaluate a trained PAM-VLM baseline on the frozen actor-independent TEST set.

Loads a run's ``best.pt``, runs the immutable test manifest once, writes
``test_metrics.json`` + per-sample predictions, and renders test-set figures
(confusion matrix, per-class F1) under ``<run_dir>/figures_test/``.

Run only inside a GPU SLURM allocation (same rule as training).
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from av_mer.data.collate import collate_av_batch  # noqa: E402
from av_mer.data.dataset import CremaDataset, DatasetConfig  # noqa: E402
from av_mer.data.filename_parser import LABELS  # noqa: E402
from av_mer.evaluation.calibration import compute_ece  # noqa: E402
from av_mer.evaluation.metrics import compute_classification_metrics  # noqa: E402
from av_mer.models.config import PamVlmConfig  # noqa: E402
from av_mer.models.pam_vlm import PamVlmModel  # noqa: E402
from av_mer.training.checkpoint import load_checkpoint  # noqa: E402
from av_mer.training.loop import BASELINE_TO_OUTPUT  # noqa: E402
from av_mer.training.reproducibility import write_json  # noqa: E402

LOGGER = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Training run directory.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=12)
    return parser.parse_args()


def _repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("Test evaluation must run via sbatch on a GPU node, not the login node.")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA device is required for test evaluation.")

    run_dir = args.run_dir.resolve()
    config = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))
    baseline = str(config["baseline"])
    data_config = config["data"]

    loader_config = DatasetConfig.from_json(
        _repo_path(str(data_config["loader_config"])), seed=int(config["seed"])
    )
    soft_paths = {
        cond: _repo_path(str(data_config["soft_labels"][cond])) for cond in ("voice", "face", "av")
    }
    dataset = CremaDataset(
        _repo_path(str(data_config["test_manifest"])),
        soft_paths,
        _repo_path(str(data_config["raw_root"])),
        loader_config,
        split="test",
        decode_error_report=run_dir / "decode_errors_test.csv",
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_av_batch,
    )

    model_config = PamVlmConfig.from_json(_repo_path(str(config["model_config"])))
    model = PamVlmModel(model_config)
    device = torch.device("cuda", torch.cuda.current_device())
    best_ckpt = run_dir / "checkpoints" / "best.pt"
    load_checkpoint(best_ckpt, model=model, map_location=device, restore_rng=False)
    model.to(device)
    model.eval()

    output_key = BASELINE_TO_OUTPUT[baseline]
    targets: list[int] = []
    predictions: list[int] = []
    prob_rows: list[list[float]] = []
    rows: list[dict[str, object]] = []
    with torch.inference_mode():
        for batch in loader:
            audio = batch["audio_values"].to(device)
            audio_mask = batch["audio_mask"].to(device)
            video = batch["video_values"].to(device)
            video_mask = batch["video_mask"].to(device)
            labels = batch["labels"]
            output = model(audio, audio_mask, video, video_mask)
            logits = output[output_key]
            probs = torch.softmax(logits.float(), dim=-1).cpu()
            preds = probs.argmax(dim=-1)
            targets.extend(labels.tolist())
            predictions.extend(preds.tolist())
            prob_rows.extend(probs.tolist())
            for i, clip_id in enumerate(batch["clip_ids"]):
                rows.append({
                    "clip_id": clip_id,
                    "actor_id": int(batch["actor_ids"][i]),
                    "true": LABELS[int(labels[i])],
                    "pred": LABELS[int(preds[i])],
                    **{f"p_{lab}": float(probs[i, j]) for j, lab in enumerate(LABELS)},
                })

    metrics = compute_classification_metrics(targets, predictions)
    metrics["ece"] = compute_ece(prob_rows, targets, num_bins=15)
    metrics["split"] = "test"
    metrics["checkpoint"] = str(best_ckpt)
    metrics["evaluated_at"] = datetime.now(timezone.utc).isoformat()
    write_json(run_dir / "test_metrics.json", metrics)

    predictions_path = run_dir / "test_predictions.csv"
    with predictions_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    LOGGER.info(
        "TEST baseline=%s acc=%.4f macro_f1=%.4f uar=%.4f ece=%.4f (n=%d)",
        baseline, metrics["accuracy"], metrics["macro_f1"], metrics["uar"],
        metrics["ece"]["ece"], metrics["num_samples"],
    )

    try:
        from plot_results import _plot_confusion, plot_per_class_f1  # noqa: E402
        import numpy as np

        figures_dir = run_dir / "figures_test"
        figures_dir.mkdir(parents=True, exist_ok=True)
        cm = np.asarray(metrics["confusion_matrix"], dtype=int)
        _plot_confusion(cm, list(LABELS), f"TEST confusion matrix ({baseline})",
                        figures_dir / "test_confusion_matrix.png", 150, normalise=False)
        _plot_confusion(cm, list(LABELS), f"TEST row-normalised confusion ({baseline})",
                        figures_dir / "test_confusion_matrix_norm.png", 150, normalise=True)
        plot_per_class_f1({"epoch": "test", "val_metrics": metrics},
                          figures_dir / "test_per_class_f1.png", 150)
        LOGGER.info("Test figures written to %s", figures_dir)
    except Exception:  # noqa: BLE001
        LOGGER.exception("Test figure generation failed; metrics are intact.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
