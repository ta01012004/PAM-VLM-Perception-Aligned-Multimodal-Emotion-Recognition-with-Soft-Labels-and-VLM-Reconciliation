#!/usr/bin/env python3
"""Train one hard-label PAM-VLM baseline inside a GPU SLURM allocation."""

from __future__ import annotations

import argparse
from copy import deepcopy
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
from av_mer.data.filename_parser import LABEL_TO_ID  # noqa: E402
from av_mer.models.config import PamVlmConfig  # noqa: E402
from av_mer.models.pam_vlm import PamVlmModel  # noqa: E402
from av_mer.training.checkpoint import load_checkpoint  # noqa: E402
from av_mer.training.loop import LoopConfig, Trainer, seed_everything  # noqa: E402
from av_mer.training.losses import LossConfig, MultiBranchClassificationLoss  # noqa: E402
from av_mer.training.reproducibility import (  # noqa: E402
    build_reproducibility_record,
    write_json,
)


LOGGER = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path, default=None)
    return parser.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        values = yaml.safe_load(handle)
    if not isinstance(values, dict):
        raise ValueError(f"Training config {path} must contain a mapping.")
    return values


def _repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _soft_label_paths(data_config: Mapping[str, Any]) -> dict[str, Path]:
    values = data_config["soft_labels"]
    if not isinstance(values, Mapping):
        raise ValueError("data.soft_labels must map voice, face, and av to Parquet files.")
    return {condition: _repo_path(str(values[condition])) for condition in ("voice", "face", "av")}


def _make_loader(
    *,
    manifest: Path,
    split: str,
    data_config: Mapping[str, Any],
    loader_config: DatasetConfig,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    dataset = CremaDataset(
        manifest,
        _soft_label_paths(data_config),
        _repo_path(str(data_config["raw_root"])),
        loader_config,
        split=split,
        decode_error_report=_repo_path(str(data_config["decode_error_report"])),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=split == "train",
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        collate_fn=collate_av_batch,
    )


def _loop_config(config: Mapping[str, Any], seed: int) -> LoopConfig:
    training = config["training"]
    return LoopConfig(
        epochs=int(training["epochs"]),
        gradient_accumulation_steps=int(training["gradient_accumulation_steps"]),
        amp=bool(training["amp"]),
        max_grad_norm=(
            float(training["max_grad_norm"])
            if training.get("max_grad_norm") is not None
            else None
        ),
        early_stopping_patience=int(training["early_stopping_patience"]),
        primary_metric=str(training["primary_metric"]),
        baseline=str(config["baseline"]),
        seed=seed,
    )


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError(
            "Real training is forbidden on the login node. Submit a scripts/train_*.sbatch job."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA device is required for real baseline training.")

    config = _load_yaml(args.config)
    seed = int(config["seed"])
    seed_everything(seed)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_id = f"{timestamp}_{config['run_name']}"
    output_root = _repo_path(str(config["output_root"]))
    run_dir = args.resume.parent.parent if args.resume is not None else output_root / run_id
    checkpoint_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)

    resolved_config = deepcopy(config)
    resolved_config["run_id"] = run_id
    resolved_config["slurm_job_id"] = os.environ["SLURM_JOB_ID"]
    write_json(run_dir / "resolved_config.json", resolved_config)

    data_config = config["data"]
    loader_config = DatasetConfig.from_json(
        _repo_path(str(data_config["loader_config"])),
        seed=seed,
    )
    training_config = config["training"]
    train_loader = _make_loader(
        manifest=_repo_path(str(data_config["train_manifest"])),
        split="train",
        data_config=data_config,
        loader_config=loader_config,
        batch_size=int(training_config["batch_size"]),
        num_workers=int(training_config["num_workers"]),
    )
    val_loader = _make_loader(
        manifest=_repo_path(str(data_config["val_manifest"])),
        split="val",
        data_config=data_config,
        loader_config=loader_config,
        batch_size=int(training_config["batch_size"]),
        num_workers=int(training_config["num_workers"]),
    )

    model_config = PamVlmConfig.from_json(_repo_path(str(config["model_config"])))
    model = PamVlmModel(model_config)
    device = torch.device("cuda", torch.cuda.current_device())
    model.to(device)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer_config = config["optimizer"]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(optimizer_config["learning_rate"]),
        weight_decay=float(optimizer_config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(training_config["epochs"]),
        eta_min=float(optimizer_config["minimum_learning_rate"]),
    )
    criterion = MultiBranchClassificationLoss(LossConfig.from_mapping(config["loss"]))
    split_checksum = _repo_path(str(data_config["split_checksum"])).read_text(
        encoding="ascii"
    ).strip()
    label_mapping = json.loads(
        _repo_path(str(data_config["label_mapping"])).read_text(encoding="utf-8")
    )
    if label_mapping != LABEL_TO_ID:
        raise ValueError(
            f"Configured label mapping {label_mapping} does not match canonical {LABEL_TO_ID}."
        )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        config=_loop_config(config, seed),
        device=device,
        checkpoint_dir=checkpoint_dir,
        label_mapping=label_mapping,
        resolved_config=resolved_config,
        split_checksum=split_checksum,
    )
    start_epoch = 0
    global_step = 0
    best_metric = float("-inf")
    if args.resume is not None:
        resume_state = load_checkpoint(
            args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=trainer.scaler,
            expected_label_mapping=label_mapping,
            expected_split_checksum=split_checksum,
            map_location=device,
        )
        start_epoch = resume_state.epoch
        global_step = resume_state.global_step
        best_metric = resume_state.best_metric

    fit_result = trainer.fit(
        train_loader,
        val_loader,
        start_epoch=start_epoch,
        global_step=global_step,
        best_metric=best_metric,
    )
    write_json(run_dir / "metrics.json", fit_result.best_validation_metrics)
    write_json(run_dir / "history.json", {"epochs": fit_result.history})
    result_row = {
        "run": run_id,
        "audio_encoder": model_config.audio.encoder_id,
        "visual_encoder": model_config.video.backbone_id,
        "fusion": "bidirectional_cross_attention_gated_pooling",
        "llm": False,
        "baseline": config["baseline"],
        "macro_f1": fit_result.best_validation_metrics["macro_f1"],
        "uar": fit_result.best_validation_metrics["uar"],
        "accuracy": fit_result.best_validation_metrics["accuracy"],
        "parameters": model.parameter_counts(),
        "notes": "hard-label actor-independent validation; test not evaluated",
    }
    write_json(run_dir / "result_row.json", result_row)
    reproducibility = build_reproducibility_record(
        run_id=run_id,
        repository_root=REPOSITORY_ROOT,
        resolved_config=resolved_config,
        seed=seed,
        split_checksum=split_checksum,
        label_mapping=label_mapping,
        model=model,
        best_checkpoint_path=fit_result.best_checkpoint_path,
        best_validation_metric=fit_result.best_metric,
        test_metrics=None,
    )
    write_json(run_dir / "reproducibility.json", reproducibility)
    try:
        from plot_results import (  # noqa: E402
            _best_record,
            plot_confusion_matrices,
            plot_loss_components,
            plot_loss_curves,
            plot_metric_curves,
            plot_per_class_f1,
        )

        figures_dir = run_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        history_records = fit_result.history
        plot_loss_curves(history_records, figures_dir / "loss_curves.png", 150)
        plot_metric_curves(history_records, figures_dir / "metric_curves.png", 150)
        plot_loss_components(history_records, figures_dir / "loss_components.png", 150)
        best_record = _best_record(history_records)
        plot_per_class_f1(best_record, figures_dir / "per_class_f1.png", 150)
        plot_confusion_matrices(best_record, figures_dir, 150)
        LOGGER.info("Figures written to %s", figures_dir)
    except Exception:  # noqa: BLE001 - plotting must never lose a trained model
        LOGGER.exception("Figure generation failed; training artifacts are intact.")
    LOGGER.info("Best validation %s=%.6f", training_config["primary_metric"], fit_result.best_metric)
    LOGGER.info("Run artifacts: %s", run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
