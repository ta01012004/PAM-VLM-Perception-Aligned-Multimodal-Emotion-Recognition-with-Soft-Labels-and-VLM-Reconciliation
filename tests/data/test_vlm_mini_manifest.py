from __future__ import annotations

import pandas as pd

from scripts.build_vlm_mini_manifest import balanced_sample


def test_balanced_sample_has_exact_per_class_count_and_is_deterministic() -> None:
    rows = []
    for emotion_id, emotion in enumerate(
        ("angry", "disgust", "fear", "happy", "neutral", "sad")
    ):
        for index in range(8):
            rows.append(
                {
                    "clip_id": f"{emotion}-{index}",
                    "emotion": emotion,
                    "emotion_id": emotion_id,
                    "split": "train",
                }
            )
    table = pd.DataFrame(rows)
    first = balanced_sample(table, per_class=3, seed=42)
    second = balanced_sample(table, per_class=3, seed=42)
    assert first["emotion"].value_counts().to_dict() == {
        emotion: 3 for emotion in table["emotion"].unique()
    }
    assert first["clip_id"].tolist() == second["clip_id"].tolist()
