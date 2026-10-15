"""Calibration metrics: Expected Calibration Error and reliability data."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def compute_ece(
    probabilities: Sequence[Sequence[float]] | np.ndarray,
    targets: Sequence[int] | np.ndarray,
    *,
    num_bins: int = 15,
) -> dict[str, Any]:
    """Expected Calibration Error over equal-width confidence bins.

    Uses the top-1 (max-probability) prediction per sample. Returns the scalar
    ECE, the maximum per-bin gap (MCE), overall accuracy, mean confidence, and
    per-bin reliability-diagram data (confidence, accuracy, count).
    """

    probs = np.asarray(probabilities, dtype=np.float64)
    target_array = np.asarray(targets, dtype=np.int64)
    if probs.ndim != 2:
        raise ValueError(f"probabilities must be 2-D [N,C], received shape {probs.shape}.")
    if target_array.ndim != 1 or target_array.shape[0] != probs.shape[0]:
        raise ValueError("targets must be 1-D [N] aligned with probabilities.")
    if probs.shape[0] == 0:
        raise ValueError("Cannot compute ECE on an empty set.")
    if num_bins <= 0:
        raise ValueError(f"num_bins must be positive, received {num_bins}.")
    row_sums = probs.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-3):
        raise ValueError("Each probability row must sum to 1 (within tolerance).")

    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == target_array).astype(np.float64)
    total = probs.shape[0]

    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    mce = 0.0
    bins: list[dict[str, float]] = []
    for lower, upper in zip(bin_edges[:-1], bin_edges[1:]):
        # Last bin is closed on the right so confidence == 1.0 is counted.
        if upper >= 1.0:
            in_bin = (confidences > lower) & (confidences <= upper)
        else:
            in_bin = (confidences >= lower) & (confidences < upper)
        count = int(in_bin.sum())
        if count == 0:
            bins.append({"lower": float(lower), "upper": float(upper), "count": 0,
                         "confidence": 0.0, "accuracy": 0.0})
            continue
        bin_confidence = float(confidences[in_bin].mean())
        bin_accuracy = float(correct[in_bin].mean())
        gap = abs(bin_confidence - bin_accuracy)
        ece += (count / total) * gap
        mce = max(mce, gap)
        bins.append({"lower": float(lower), "upper": float(upper), "count": count,
                     "confidence": bin_confidence, "accuracy": bin_accuracy})

    return {
        "ece": float(ece),
        "mce": float(mce),
        "num_bins": num_bins,
        "accuracy": float(correct.mean()),
        "mean_confidence": float(confidences.mean()),
        "num_samples": total,
        "bins": bins,
    }
