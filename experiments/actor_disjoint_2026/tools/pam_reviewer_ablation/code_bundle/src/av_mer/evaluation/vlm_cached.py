"""Constrained VLM prediction from cached PAM evidence."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import torch
from torch import Tensor

from av_mer.data.filename_parser import LABELS
from av_mer.evaluation.metrics import compute_classification_metrics
from av_mer.models.llm_adapter import LLMReconciler
from av_mer.models.projector import FusionToLLMProjector
from av_mer.models.vlm_text import EmotionTextCodec, format_probability_evidence


def projector_inputs(
    batch: Mapping[str, Any],
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return (
        batch["fusion_embedding"].to(device, dtype=torch.float32),
        batch["audio_probs"].to(device, dtype=torch.float32),
        batch["video_probs"].to(device, dtype=torch.float32),
        batch["disagreement"].to(device, dtype=torch.float32),
    )


def evidence_from_batch(batch: Mapping[str, Any]) -> list[str]:
    """Build one label-aware textual evidence block per cached sample."""

    audio = batch["audio_probs"].float().cpu()
    video = batch["video_probs"].float().cpu()
    fusion = batch["fusion_probs"].float().cpu()
    disagreement = batch["disagreement"].float().cpu()
    return [
        format_probability_evidence(
            audio[index].tolist(),
            video[index].tolist(),
            fusion[index].tolist(),
            float(disagreement[index]),
        )
        for index in range(audio.shape[0])
    ]


@torch.no_grad()
def predict_cached(
    *,
    reconciler: LLMReconciler,
    projector: FusionToLLMProjector,
    codec: EmotionTextCodec,
    loader: Iterable[Mapping[str, Any]],
    device: torch.device,
    amp: bool = True,
) -> dict[str, Any]:
    """Score all six valid labels and select the minimum-NLL candidate."""

    reconciler.eval()
    projector.eval()
    targets: list[int] = []
    classifier_predictions: list[int] = []
    classifier_confidence: list[float] = []
    vlm_predictions: list[int] = []
    disagreements: list[float] = []
    clip_ids: list[str] = []
    candidate_nll: list[list[float]] = []
    for batch in loader:
        batch_size = int(batch["labels"].shape[0])
        with torch.autocast(
            device_type=device.type,
            enabled=amp and device.type == "cuda",
        ):
            prefix = projector(*projector_inputs(batch, device))
            text = codec.encode_candidates(
                batch_size,
                evidence_from_batch(batch),
            )
            nll = reconciler.sequence_nll(
                prefix.repeat_interleave(len(LABELS), dim=0),
                text.input_ids.to(device),
                text.attention_mask.to(device),
                text.labels.to(device),
            ).view(batch_size, len(LABELS))
        vlm_predictions.extend(nll.argmin(dim=1).cpu().tolist())
        classifier_predictions.extend(batch["fusion_probs"].argmax(dim=1).tolist())
        classifier_confidence.extend(
            batch["fusion_probs"].float().max(dim=1).values.tolist()
        )
        targets.extend(batch["labels"].tolist())
        disagreements.extend(batch["disagreement"].float().tolist())
        clip_ids.extend(str(value) for value in batch["clip_id"])
        candidate_nll.extend(nll.float().cpu().tolist())
    if not targets:
        raise ValueError("Cannot evaluate an empty cached feature loader.")
    return {
        "clip_ids": clip_ids,
        "targets": np.asarray(targets, dtype=np.int64),
        "classifier_predictions": np.asarray(classifier_predictions, dtype=np.int64),
        "classifier_confidence": np.asarray(classifier_confidence, dtype=np.float64),
        "vlm_predictions": np.asarray(vlm_predictions, dtype=np.int64),
        "disagreement": np.asarray(disagreements, dtype=np.float64),
        "candidate_nll": np.asarray(candidate_nll, dtype=np.float64),
        "classifier_metrics": compute_classification_metrics(targets, classifier_predictions),
        "vlm_metrics": compute_classification_metrics(targets, vlm_predictions),
    }
