"""Deterministic train/validation loop for the three hard-label baselines."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import logging
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from av_mer.evaluation.metrics import compute_classification_metrics
from av_mer.training.checkpoint import save_checkpoint
from av_mer.training.losses import LossResult, MultiBranchClassificationLoss


LOGGER = logging.getLogger(__name__)
BASELINE_TO_OUTPUT = {
    "audio": "audio_logits",
    "video": "video_logits",
    "fusion": "logits",
}


@dataclass(frozen=True, slots=True)
class LoopConfig:
    """Training behavior that must remain fixed across baseline comparisons."""

    epochs: int
    gradient_accumulation_steps: int
    amp: bool
    max_grad_norm: float | None
    early_stopping_patience: int
    primary_metric: str
    baseline: str
    seed: int

    def validate(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive.")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive.")
        if self.early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be positive.")
        if self.max_grad_norm is not None and self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive when configured.")
        if self.baseline not in BASELINE_TO_OUTPUT:
            raise ValueError(
                f"baseline must be one of {sorted(BASELINE_TO_OUTPUT)}, "
                f"received {self.baseline!r}."
            )
        if self.primary_metric not in {"macro_f1", "uar", "accuracy", "weighted_f1"}:
            raise ValueError(f"Unsupported primary metric {self.primary_metric!r}.")


@dataclass(frozen=True)
class EpochResult:
    loss: float
    loss_components: dict[str, float]
    metrics: dict[str, Any]
    global_step: int


@dataclass(frozen=True)
class FitResult:
    best_metric: float
    best_epoch: int
    best_validation_metrics: dict[str, Any]
    final_global_step: int
    history: list[dict[str, Any]]
    best_checkpoint_path: Path
    last_checkpoint_path: Path


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, CPU Torch, and all available CUDA devices."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }


def _model_inputs(batch: Mapping[str, Any]) -> dict[str, Tensor]:
    return {
        "audio_values": batch["audio_values"],
        "audio_mask": batch["audio_mask"],
        "video_values": batch["video_values"],
        "video_mask": batch["video_mask"],
    }


def _soft_targets(batch: Mapping[str, Any]) -> dict[str, Tensor]:
    return {
        "audio": batch["soft_voice"],
        "video": batch["soft_face"],
        "fusion": batch["soft_av"],
    }


def _select_logits(model_output: Mapping[str, Tensor | None], baseline: str) -> Tensor:
    output_key = BASELINE_TO_OUTPUT[baseline]
    logits = model_output.get(output_key)
    if logits is None:
        raise ValueError(f"Model output {output_key!r} is required for {baseline} baseline.")
    if logits.ndim != 2:
        raise ValueError(f"Expected {output_key} [B,C], received {tuple(logits.shape)}.")
    return logits


def _accumulate_components(
    totals: dict[str, float],
    result: LossResult,
    batch_size: int,
) -> None:
    for name, value in result.components.items():
        totals[name] = totals.get(name, 0.0) + float(value.detach().cpu()) * batch_size


def run_train_epoch(
    *,
    model: nn.Module,
    data_loader: Iterable[Mapping[str, Any]],
    optimizer: torch.optim.Optimizer,
    criterion: MultiBranchClassificationLoss,
    config: LoopConfig,
    device: torch.device,
    scaler: Any,
    global_step: int,
) -> EpochResult:
    """Run one epoch and step the optimizer after each accumulation group."""

    model.train()
    optimizer.zero_grad(set_to_none=True)
    num_batches = len(data_loader)  # type: ignore[arg-type]
    if num_batches == 0:
        raise ValueError("Training data loader is empty.")
    total_samples = 0
    component_totals: dict[str, float] = {}
    targets: list[int] = []
    predictions: list[int] = []
    amp_enabled = config.amp and device.type == "cuda"

    for batch_index, raw_batch in enumerate(data_loader):
        batch = _move_batch(raw_batch, device)
        labels = batch["labels"]
        group_start = (batch_index // config.gradient_accumulation_steps) * (
            config.gradient_accumulation_steps
        )
        group_size = min(
            config.gradient_accumulation_steps,
            num_batches - group_start,
        )
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            output = model(**_model_inputs(batch))
            result = criterion(output, labels, _soft_targets(batch))
            scaled_loss = result.total / group_size
        if not torch.isfinite(result.total):
            raise FloatingPointError(f"Non-finite training loss at batch {batch_index}.")
        scaler.scale(scaled_loss).backward()

        is_group_end = (
            (batch_index + 1) % config.gradient_accumulation_steps == 0
            or batch_index + 1 == num_batches
        )
        if is_group_end:
            if config.max_grad_norm is not None:
                scaler.unscale_(optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.max_grad_norm,
                )
                if not torch.isfinite(gradient_norm):
                    raise FloatingPointError(
                        f"Non-finite gradient norm at optimizer step {global_step}."
                    )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        logits = _select_logits(output, config.baseline)
        if logits.shape != (labels.shape[0], 6):
            raise ValueError(
                f"Expected baseline logits [B,6], received {tuple(logits.shape)}."
            )
        batch_size = int(labels.shape[0])
        _accumulate_components(component_totals, result, batch_size)
        total_samples += batch_size
        targets.extend(labels.detach().cpu().tolist())
        predictions.extend(logits.detach().argmax(dim=-1).cpu().tolist())

    metrics = compute_classification_metrics(targets, predictions)
    components = {name: value / total_samples for name, value in component_totals.items()}
    return EpochResult(
        loss=components["total"],
        loss_components=components,
        metrics=metrics,
        global_step=global_step,
    )


def run_validation_epoch(
    *,
    model: nn.Module,
    data_loader: Iterable[Mapping[str, Any]],
    criterion: MultiBranchClassificationLoss,
    config: LoopConfig,
    device: torch.device,
    global_step: int,
) -> EpochResult:
    """Run deterministic validation without gradients."""

    model.eval()
    total_samples = 0
    component_totals: dict[str, float] = {}
    targets: list[int] = []
    predictions: list[int] = []
    amp_enabled = config.amp and device.type == "cuda"
    with torch.inference_mode():
        for batch_index, raw_batch in enumerate(data_loader):
            batch = _move_batch(raw_batch, device)
            labels = batch["labels"]
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                output = model(**_model_inputs(batch))
                result = criterion(output, labels, _soft_targets(batch))
            if not torch.isfinite(result.total):
                raise FloatingPointError(f"Non-finite validation loss at batch {batch_index}.")
            logits = _select_logits(output, config.baseline)
            if logits.shape != (labels.shape[0], 6):
                raise ValueError(
                    f"Expected validation logits [B,6], received {tuple(logits.shape)}."
                )
            batch_size = int(labels.shape[0])
            _accumulate_components(component_totals, result, batch_size)
            total_samples += batch_size
            targets.extend(labels.cpu().tolist())
            predictions.extend(logits.argmax(dim=-1).cpu().tolist())
    if total_samples == 0:
        raise ValueError("Validation data loader is empty.")
    metrics = compute_classification_metrics(targets, predictions)
    components = {name: value / total_samples for name, value in component_totals.items()}
    return EpochResult(
        loss=components["total"],
        loss_components=components,
        metrics=metrics,
        global_step=global_step,
    )


class Trainer:
    """Coordinate hard-label training, validation, selection, and checkpointing."""

    def __init__(
        self,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any | None,
        criterion: MultiBranchClassificationLoss,
        config: LoopConfig,
        device: torch.device,
        checkpoint_dir: str | Path,
        label_mapping: Mapping[str, int],
        resolved_config: Mapping[str, Any],
        split_checksum: str,
    ) -> None:
        config.validate()
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.config = config
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.label_mapping = dict(label_mapping)
        self.resolved_config = dict(resolved_config)
        self.split_checksum = split_checksum
        self.scaler = torch.amp.GradScaler(
            device.type,
            enabled=config.amp and device.type == "cuda",
        )

    def fit(
        self,
        train_loader: Iterable[Mapping[str, Any]],
        val_loader: Iterable[Mapping[str, Any]],
        *,
        start_epoch: int = 0,
        global_step: int = 0,
        best_metric: float = float("-inf"),
    ) -> FitResult:
        batch_size = getattr(train_loader, "batch_size", None)
        effective_batch_size = (
            int(batch_size) * self.config.gradient_accumulation_steps
            if batch_size is not None
            else None
        )
        LOGGER.info(
            "Training baseline=%s batch_size=%s accumulation=%d effective_batch_size=%s",
            self.config.baseline,
            batch_size,
            self.config.gradient_accumulation_steps,
            effective_batch_size,
        )
        LOGGER.info(
            "Parameters total=%d trainable=%d",
            sum(parameter.numel() for parameter in self.model.parameters()),
            sum(
                parameter.numel()
                for parameter in self.model.parameters()
                if parameter.requires_grad
            ),
        )

        best_path = self.checkpoint_dir / "best.pt"
        last_path = self.checkpoint_dir / "last.pt"
        history: list[dict[str, Any]] = []
        bad_epochs = 0
        best_epoch = start_epoch - 1
        best_validation_metrics: dict[str, Any] = {}

        for epoch in range(start_epoch, self.config.epochs):
            train_result = run_train_epoch(
                model=self.model,
                data_loader=train_loader,
                optimizer=self.optimizer,
                criterion=self.criterion,
                config=self.config,
                device=self.device,
                scaler=self.scaler,
                global_step=global_step,
            )
            global_step = train_result.global_step
            validation_result = run_validation_epoch(
                model=self.model,
                data_loader=val_loader,
                criterion=self.criterion,
                config=self.config,
                device=self.device,
                global_step=global_step,
            )
            if self.scheduler is not None:
                self.scheduler.step()

            primary_value = float(validation_result.metrics[self.config.primary_metric])
            epoch_record = {
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": train_result.loss,
                "train_loss_components": train_result.loss_components,
                "train_metrics": train_result.metrics,
                "val_loss": validation_result.loss,
                "val_loss_components": validation_result.loss_components,
                "val_metrics": validation_result.metrics,
            }
            history.append(epoch_record)
            LOGGER.info(
                "epoch=%d step=%d train_loss=%.6f val_loss=%.6f %s=%.6f",
                epoch,
                global_step,
                train_result.loss,
                validation_result.loss,
                self.config.primary_metric,
                primary_value,
            )
            for name, value in train_result.loss_components.items():
                LOGGER.info("train_loss_component %s=%.6f", name, value)
            for name, value in validation_result.loss_components.items():
                LOGGER.info("val_loss_component %s=%.6f", name, value)

            improved = primary_value > best_metric
            if improved:
                best_metric = primary_value
                best_epoch = epoch
                best_validation_metrics = validation_result.metrics
                bad_epochs = 0
                save_checkpoint(
                    best_path,
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    scaler=self.scaler,
                    epoch=epoch + 1,
                    global_step=global_step,
                    best_metric=best_metric,
                    label_mapping=self.label_mapping,
                    resolved_config=self.resolved_config,
                    split_checksum=self.split_checksum,
                )
            else:
                bad_epochs += 1

            save_checkpoint(
                last_path,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                epoch=epoch + 1,
                global_step=global_step,
                best_metric=best_metric,
                label_mapping=self.label_mapping,
                resolved_config=self.resolved_config,
                split_checksum=self.split_checksum,
            )
            if bad_epochs >= self.config.early_stopping_patience:
                LOGGER.info("Early stopping after %d non-improving epochs.", bad_epochs)
                break

        if not best_path.exists():
            raise RuntimeError("Training completed without producing a best checkpoint.")
        return FitResult(
            best_metric=best_metric,
            best_epoch=best_epoch,
            best_validation_metrics=best_validation_metrics,
            final_global_step=global_step,
            history=history,
            best_checkpoint_path=best_path,
            last_checkpoint_path=last_path,
        )
