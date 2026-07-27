#!/usr/bin/env python3
"""Train a soft-prefix projector and optional Qwen LoRA on cached PAM evidence."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from av_mer.data.vlm_cache import CachedVlmDataset  # noqa: E402
from av_mer.evaluation.vlm_cached import (  # noqa: E402
    evidence_from_batch,
    predict_cached,
    projector_inputs,
)
from av_mer.evaluation.vlm_routing import select_routing_threshold  # noqa: E402
from av_mer.models.llm_adapter import LLMAdapterConfig, LLMReconciler  # noqa: E402
from av_mer.models.projector import FusionToLLMProjector  # noqa: E402
from av_mer.models.vlm_text import EmotionTextCodec  # noqa: E402

LOGGER = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
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


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _save_checkpoint(
    path: Path,
    *,
    reconciler: LLMReconciler,
    projector: FusionToLLMProjector,
    epoch: int,
    routing: dict[str, Any],
    config: dict[str, Any],
) -> None:
    payload = {
        "format_version": 1,
        "projector_state": {
            key: value.detach().cpu()
            for key, value in projector.state_dict().items()
        },
        "adapter_state": {
            key: value.detach().cpu()
            for key, value in reconciler.adapter_state_dict().items()
        },
        "epoch": int(epoch),
        "routing_threshold": float(routing["threshold"]),
        "classifier_confidence_max": float(routing["classifier_confidence_max"]),
        "validation_routing": _json_ready(routing),
        "config": deepcopy(config),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _make_loader(path: Path, batch_size: int, *, shuffle: bool) -> DataLoader:
    return DataLoader(
        CachedVlmDataset(path),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True,
    )


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("VLM training must run inside a SLURM allocation.")
    if not torch.cuda.is_available():
        raise RuntimeError("VLM training requires CUDA.")
    config = _load_yaml(args.config)
    seed = int(config["seed"])
    _seed_everything(seed)
    training = config["training"]
    vlm = config["vlm"]
    cache_dir = _repo_path(config["cache"]["output_dir"])
    train_loader = _make_loader(
        cache_dir / "train.pt",
        int(training["batch_size"]),
        shuffle=True,
    )
    val_loader = _make_loader(
        cache_dir / "val.pt",
        int(training.get("eval_batch_size", training["batch_size"])),
        shuffle=False,
    )
    train_dataset = train_loader.dataset
    assert isinstance(train_dataset, CachedVlmDataset)
    fusion_dim = int(train_dataset.tensors["fusion_embedding"].shape[1])
    device = torch.device("cuda", torch.cuda.current_device())
    reconciler = LLMReconciler(
        LLMAdapterConfig(
            model_id=str(vlm["model_id"]),
            lora_rank=int(vlm["lora_rank"]),
            lora_alpha=int(vlm["lora_alpha"]),
            lora_dropout=float(vlm["lora_dropout"]),
            num_prefix_tokens=int(vlm["num_prefix_tokens"]),
            max_new_tokens=int(vlm.get("max_new_tokens", 24)),
            freeze_base=bool(vlm.get("freeze_base", True)),
        )
    ).to(device)
    train_lora = bool(training.get("train_lora", True))
    reconciler.set_lora_trainable(train_lora)
    projector = FusionToLLMProjector(
        fusion_dim=fusion_dim,
        llm_hidden=reconciler.hidden_size,
        num_prefix_tokens=int(vlm["num_prefix_tokens"]),
        dropout=float(vlm.get("projector_dropout", 0.1)),
    ).to(device)
    prompt = _repo_path(vlm["prompt_file"]).read_text(encoding="utf-8")
    codec = EmotionTextCodec(reconciler.tokenizer, prompt)
    lora_parameters = [
        parameter for parameter in reconciler.parameters() if parameter.requires_grad
    ]
    projector_parameters = [
        parameter for parameter in projector.parameters() if parameter.requires_grad
    ]
    parameters = lora_parameters + projector_parameters
    learning_rate = float(training["learning_rate"])
    optimizer = torch.optim.AdamW(
        [
            {
                "params": lora_parameters,
                "lr": float(training.get("lora_learning_rate", learning_rate)),
            },
            {
                "params": projector_parameters,
                "lr": float(training.get("projector_learning_rate", learning_rate)),
            },
        ],
        weight_decay=float(training.get("weight_decay", 0.0)),
    )
    epochs = int(training["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=float(training.get("minimum_learning_rate", 1e-6)),
    )
    amp = bool(training.get("amp", True))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    accumulation = int(training.get("gradient_accumulation_steps", 1))
    max_grad_norm = float(training.get("max_grad_norm", 1.0))
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir = _repo_path(config["output_root"]) / f"{timestamp}_{config['run_name']}"
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved = deepcopy(config)
    resolved["run_id"] = run_dir.name
    resolved["slurm_job_id"] = os.environ["SLURM_JOB_ID"]
    _write_json(run_dir / "resolved_config.json", resolved)
    history: list[dict[str, Any]] = []
    best_ranking = (-1.0, -1.0)
    checkpoint_path = run_dir / "checkpoints" / "best.pt"

    for epoch in range(epochs):
        reconciler.train()
        projector.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        total_samples = 0
        for batch_index, batch in enumerate(train_loader):
            target_ids = [int(value) for value in batch["labels"].tolist()]
            text = codec.encode_targets(target_ids, evidence_from_batch(batch))
            with torch.autocast(device_type="cuda", enabled=amp):
                prefix = projector(*projector_inputs(batch, device))
                output = reconciler(
                    prefix,
                    text.input_ids.to(device),
                    text.attention_mask.to(device),
                    labels=text.labels.to(device),
                )
                loss = output.loss / accumulation
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite VLM loss at epoch {epoch}.")
            scaler.scale(loss).backward()
            should_step = (
                (batch_index + 1) % accumulation == 0
                or batch_index + 1 == len(train_loader)
            )
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            batch_size = len(target_ids)
            total_loss += float(loss.detach().cpu()) * accumulation * batch_size
            total_samples += batch_size
        scheduler.step()
        validation = predict_cached(
            reconciler=reconciler,
            projector=projector,
            codec=codec,
            loader=val_loader,
            device=device,
            amp=amp,
        )
        routing = select_routing_threshold(
            validation["targets"],
            validation["classifier_predictions"],
            validation["vlm_predictions"],
            validation["disagreement"],
            validation["classifier_confidence"],
        )
        row = {
            "epoch": epoch,
            "train_loss": total_loss / total_samples,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "classifier_metrics": validation["classifier_metrics"],
            "vlm_metrics": validation["vlm_metrics"],
            "routing": routing,
        }
        history.append(row)
        _write_json(run_dir / "history.json", {"epochs": history})
        macro_f1 = float(routing["metrics"]["macro_f1"])
        ranking = (macro_f1, float(validation["vlm_metrics"]["macro_f1"]))
        if ranking > best_ranking:
            best_ranking = ranking
            _save_checkpoint(
                checkpoint_path,
                reconciler=reconciler,
                projector=projector,
                epoch=epoch,
                routing=routing,
                config=resolved,
            )
        LOGGER.info(
            "epoch=%d loss=%.4f classifier_f1=%.4f vlm_f1=%.4f routed_f1=%.4f "
            "threshold=%.4f routed=%.1f%%",
            epoch,
            row["train_loss"],
            validation["classifier_metrics"]["macro_f1"],
            validation["vlm_metrics"]["macro_f1"],
            macro_f1,
            routing["threshold"],
            100.0 * routing["routed_fraction"],
        )
    _write_json(
        run_dir / "summary.json",
        {
            "best_validation_routed_macro_f1": best_ranking[0],
            "best_validation_vlm_macro_f1_at_tie": best_ranking[1],
            "checkpoint": str(checkpoint_path),
            "train_lora": train_lora,
            "trainable_parameters": sum(parameter.numel() for parameter in parameters),
        },
    )
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
