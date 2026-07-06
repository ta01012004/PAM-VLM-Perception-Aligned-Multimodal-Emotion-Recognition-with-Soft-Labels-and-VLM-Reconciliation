"""Unit tests for Expected Calibration Error."""

from __future__ import annotations

import numpy as np
import pytest

from av_mer.evaluation.calibration import compute_ece


def test_perfectly_calibrated_confident_correct_has_zero_ece() -> None:
    # All samples predicted with confidence ~1.0 and all correct -> ECE ~ 0.
    probs = np.array([[0.99, 0.01], [0.02, 0.98], [0.995, 0.005]])
    targets = np.array([0, 1, 0])
    result = compute_ece(probs, targets, num_bins=10)
    assert result["ece"] == pytest.approx(0.0, abs=0.05)
    assert result["accuracy"] == pytest.approx(1.0)


def test_overconfident_wrong_predictions_have_high_ece() -> None:
    # High confidence (~0.95) but always wrong -> large confidence/accuracy gap.
    probs = np.array([[0.95, 0.05], [0.95, 0.05], [0.95, 0.05], [0.95, 0.05]])
    targets = np.array([1, 1, 1, 1])  # all wrong
    result = compute_ece(probs, targets, num_bins=10)
    assert result["accuracy"] == pytest.approx(0.0)
    assert result["ece"] > 0.9
    assert result["mce"] > 0.9


def test_ece_rejects_bad_shapes_and_unnormalised_rows() -> None:
    with pytest.raises(ValueError):
        compute_ece(np.array([0.5, 0.5]), np.array([0]))  # not 2-D
    with pytest.raises(ValueError):
        compute_ece(np.array([[0.3, 0.3]]), np.array([0]))  # rows do not sum to 1


def test_ece_bins_cover_all_samples() -> None:
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(200, 6))
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    targets = rng.integers(0, 6, size=200)
    result = compute_ece(probs, targets, num_bins=15)
    assert sum(b["count"] for b in result["bins"]) == 200
    assert 0.0 <= result["ece"] <= 1.0
