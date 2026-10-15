"""Atomic full-state checkpoints for exact training resume."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn


CHECKPOINT_FORMAT_VERSION = 1


@dataclass(frozen=True, slots=True)
class ResumeState:
    """Loop counters and selection state restored from a checkpoint."""

    epoch: int
    global_step: int
    best_metric: float


def capture_rng_state() -> dict[str, Any]:
    """Capture Python, NumPy, CPU Torch, and available CUDA RNG state."""

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """Restore all RNG streams captured by :func:`capture_rng_state`."""

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    cuda_state = state.get("torch_cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    scaler: Any | None,
    epoch: int,
    global_step: int,
    best_metric: float,
    label_mapping: Mapping[str, int],
    resolved_config: Mapping[str, Any],
    split_checksum: str,
) -> None:
    """Atomically persist model, optimization, metadata, counters, and RNG state."""

    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_metric": float(best_metric),
        "rng_state": capture_rng_state(),
        "label_mapping": dict(label_mapping),
        "resolved_config": dict(resolved_config),
        "split_checksum": str(split_checksum),
    }
    temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(checkpoint_path)


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    expected_label_mapping: Mapping[str, int] | None = None,
    expected_split_checksum: str | None = None,
    map_location: str | torch.device = "cpu",
    restore_rng: bool = True,
) -> ResumeState:
    """Restore checkpoint state and reject label/split protocol mismatches."""

    checkpoint_path = Path(path)
    payload = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    if payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint format {payload.get('format_version')} in "
            f"{checkpoint_path}; expected {CHECKPOINT_FORMAT_VERSION}."
        )
    if expected_label_mapping is not None and payload["label_mapping"] != dict(
        expected_label_mapping
    ):
        raise ValueError("Checkpoint label mapping does not match the current run.")
    if (
        expected_split_checksum is not None
        and payload["split_checksum"] != expected_split_checksum
    ):
        raise ValueError("Checkpoint split checksum does not match the current run.")

    model.load_state_dict(payload["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    if scheduler is not None:
        if payload["scheduler_state"] is None:
            raise ValueError("Checkpoint has no scheduler state to restore.")
        scheduler.load_state_dict(payload["scheduler_state"])
    if scaler is not None:
        if payload["scaler_state"] is None:
            raise ValueError("Checkpoint has no scaler state to restore.")
        scaler.load_state_dict(payload["scaler_state"])
    if restore_rng:
        restore_rng_state(payload["rng_state"])
    return ResumeState(
        epoch=int(payload["epoch"]),
        global_step=int(payload["global_step"]),
        best_metric=float(payload["best_metric"]),
    )

