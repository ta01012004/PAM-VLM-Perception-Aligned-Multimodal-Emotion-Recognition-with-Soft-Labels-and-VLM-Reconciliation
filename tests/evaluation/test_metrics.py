import pytest

from av_mer.evaluation.metrics import compute_classification_metrics


def test_metrics_match_known_confusion_matrix() -> None:
    metrics = compute_classification_metrics(
        targets=[0, 0, 1, 1, 2, 2],
        predictions=[0, 1, 1, 1, 2, 0],
        label_names=("a", "b", "c"),
    )

    assert metrics["confusion_matrix"] == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]
    assert metrics["accuracy"] == pytest.approx(4.0 / 6.0)
    assert metrics["uar"] == pytest.approx((0.5 + 1.0 + 0.5) / 3.0)
    assert metrics["macro_f1"] == pytest.approx((0.5 + 0.8 + 2.0 / 3.0) / 3.0)
    assert metrics["weighted_f1"] == pytest.approx(metrics["macro_f1"])
    assert metrics["per_class"]["b"] == {
        "precision": pytest.approx(2.0 / 3.0),
        "recall": pytest.approx(1.0),
        "f1": pytest.approx(0.8),
        "support": 2,
    }


def test_metrics_reject_invalid_class_ids() -> None:
    with pytest.raises(ValueError, match="Invalid predictions"):
        compute_classification_metrics([0, 1], [0, 6])
