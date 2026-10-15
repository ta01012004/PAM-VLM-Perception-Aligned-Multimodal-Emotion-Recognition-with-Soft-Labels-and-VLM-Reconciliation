"""Classification metrics with a fixed, explicit label order."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from av_mer.data.filename_parser import LABELS


def compute_classification_metrics(
    targets: Sequence[int] | np.ndarray,
    predictions: Sequence[int] | np.ndarray,
    *,
    label_names: Sequence[str] = LABELS,
) -> dict[str, Any]:
    """Compute Macro-F1, UAR, accuracy, per-class metrics, and confusion matrix."""

    target_array = np.asarray(targets, dtype=np.int64)
    prediction_array = np.asarray(predictions, dtype=np.int64)
    if target_array.ndim != 1 or prediction_array.ndim != 1:
        raise ValueError("targets and predictions must be one-dimensional arrays.")
    if target_array.shape != prediction_array.shape or target_array.size == 0:
        raise ValueError(
            "targets and predictions must be non-empty and have identical shape; "
            f"received {target_array.shape} and {prediction_array.shape}."
        )
    class_ids = np.arange(len(label_names), dtype=np.int64)
    for name, values in (("targets", target_array), ("predictions", prediction_array)):
        invalid = values[(values < 0) | (values >= len(label_names))]
        if invalid.size:
            raise ValueError(
                f"Invalid {name} IDs {np.unique(invalid).tolist()}; expected "
                f"[0,{len(label_names) - 1}]."
            )

    precision, recall, f1, support = precision_recall_fscore_support(
        target_array,
        prediction_array,
        labels=class_ids,
        zero_division=0,
    )
    total_support = int(support.sum())
    weighted_f1 = float(np.dot(f1, support) / total_support)
    per_class = {
        label: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, label in enumerate(label_names)
    }
    return {
        "macro_f1": float(f1.mean()),
        "uar": float(recall.mean()),
        "accuracy": float(accuracy_score(target_array, prediction_array)),
        "weighted_f1": weighted_f1,
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(
            target_array,
            prediction_array,
            labels=class_ids,
        ).astype(int).tolist(),
        "num_samples": int(target_array.size),
    }

