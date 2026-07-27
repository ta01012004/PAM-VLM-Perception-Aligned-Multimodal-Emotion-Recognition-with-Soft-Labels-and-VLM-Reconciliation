from __future__ import annotations

import numpy as np

from av_mer.evaluation.vlm_routing import (
    routed_predictions,
    select_routing_threshold,
)


def test_routed_predictions_use_vlm_only_above_threshold() -> None:
    result = routed_predictions(
        [0, 0, 1],
        [1, 1, 0],
        [0.1, 0.7, 0.8],
        0.75,
    )
    assert result.tolist() == [0, 0, 0]


def test_threshold_selection_can_improve_classifier_on_validation() -> None:
    targets = np.asarray([0, 1, 2, 3, 4, 5])
    classifier = np.asarray([0, 1, 2, 3, 4, 0])
    vlm = np.asarray([1, 0, 1, 2, 3, 5])
    disagreement = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 1.0])
    selected = select_routing_threshold(
        targets,
        classifier,
        vlm,
        disagreement,
    )
    assert selected["metrics"]["accuracy"] == 1.0
    assert selected["routed_fraction"] == 1 / 6


def test_confidence_guard_prevents_replacing_confident_classifier() -> None:
    result = routed_predictions(
        [0, 1],
        [1, 0],
        [1.0, 1.0],
        0.5,
        classifier_confidence=[0.95, 0.4],
        classifier_confidence_max=0.5,
    )
    assert result.tolist() == [0, 0]
