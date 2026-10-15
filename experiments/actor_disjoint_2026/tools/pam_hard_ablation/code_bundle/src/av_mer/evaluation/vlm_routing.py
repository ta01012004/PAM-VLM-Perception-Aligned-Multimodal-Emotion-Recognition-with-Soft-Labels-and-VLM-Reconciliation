"""Validation-only routing between PAM and the VLM reconciler."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from av_mer.evaluation.metrics import compute_classification_metrics


def routed_predictions(
    classifier_predictions: Sequence[int],
    vlm_predictions: Sequence[int],
    disagreement: Sequence[float],
    threshold: float,
    *,
    classifier_confidence: Sequence[float] | None = None,
    classifier_confidence_max: float = float("inf"),
) -> np.ndarray:
    classifier = np.asarray(classifier_predictions, dtype=np.int64)
    vlm = np.asarray(vlm_predictions, dtype=np.int64)
    scores = np.asarray(disagreement, dtype=np.float64)
    if classifier.shape != vlm.shape or classifier.shape != scores.shape:
        raise ValueError("Classifier, VLM, and disagreement arrays must align.")
    route = scores >= float(threshold)
    if classifier_confidence is not None:
        confidence = np.asarray(classifier_confidence, dtype=np.float64)
        if confidence.shape != classifier.shape:
            raise ValueError("classifier_confidence must align with predictions.")
        route &= confidence <= float(classifier_confidence_max)
    return np.where(route, vlm, classifier)


def select_routing_threshold(
    targets: Sequence[int],
    classifier_predictions: Sequence[int],
    vlm_predictions: Sequence[int],
    disagreement: Sequence[float],
    classifier_confidence: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Choose validation-only disagreement and confidence safety thresholds."""

    scores = np.asarray(disagreement, dtype=np.float64)
    if scores.ndim != 1 or scores.size == 0:
        raise ValueError("disagreement must be a non-empty vector.")
    candidates = np.unique(
        np.concatenate(
            [
                np.asarray([np.nextafter(scores.min(), -np.inf)]),
                np.quantile(scores, np.linspace(0.0, 1.0, 41)),
                np.asarray([np.nextafter(scores.max(), np.inf)]),
            ]
        )
    )
    confidence = (
        np.asarray(classifier_confidence, dtype=np.float64)
        if classifier_confidence is not None
        else np.ones_like(scores)
    )
    if confidence.shape != scores.shape:
        raise ValueError("classifier_confidence must align with disagreement.")
    confidence_candidates = (
        np.unique(
            np.concatenate(
                [
                    np.quantile(confidence, np.linspace(0.0, 1.0, 41)),
                    np.asarray([np.nextafter(confidence.max(), np.inf)]),
                ]
            )
        )
        if classifier_confidence is not None
        else np.asarray([float("inf")])
    )
    best: dict[str, Any] | None = None
    for threshold in candidates:
        for confidence_max in confidence_candidates:
            predictions = routed_predictions(
                classifier_predictions,
                vlm_predictions,
                scores,
                float(threshold),
                classifier_confidence=(
                    confidence if classifier_confidence is not None else None
                ),
                classifier_confidence_max=float(confidence_max),
            )
            route = (scores >= threshold) & (confidence <= confidence_max)
            metrics = compute_classification_metrics(targets, predictions)
            routed_fraction = float(np.mean(route))
            candidate = {
                "threshold": float(threshold),
                "classifier_confidence_max": float(confidence_max),
                "routed_fraction": routed_fraction,
                "metrics": metrics,
            }
            ranking = (
                metrics["macro_f1"],
                metrics["accuracy"],
                -routed_fraction,
            )
            if best is None or ranking > best["_ranking"]:
                candidate["_ranking"] = ranking
                best = candidate
    assert best is not None
    best.pop("_ranking")
    return best
