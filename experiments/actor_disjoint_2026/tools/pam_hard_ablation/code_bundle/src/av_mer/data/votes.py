"""Aggregate CREMA-D annotator responses into per-modality soft labels."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from av_mer.data.filename_parser import LABELS, LABEL_TO_ID


RESPONSE_TO_LABEL = {
    "A": "angry",
    "D": "disgust",
    "F": "fear",
    "H": "happy",
    "N": "neutral",
    "S": "sad",
}
QUERY_TYPE_TO_CONDITION = {"1": "voice", "2": "face", "3": "av"}
REQUIRED_VOTE_COLUMNS = {"clipName", "queryType", "respEmo"}


@dataclass(frozen=True)
class VoteAggregationResult:
    """Soft-label tables and coverage diagnostics for all modalities."""

    tables: dict[str, pd.DataFrame]
    coverage_percent: dict[str, float]
    fallback_counts: dict[str, int]
    ignored_row_count: int


def aggregate_votes(
    csv_path: str | Path,
    clip_ids: Sequence[str],
    hard_emotions: Mapping[str, str],
    *,
    alpha: float,
) -> VoteAggregationResult:
    """Build smoothed six-class distributions for voice, face, and AV votes.

    Missing clip-condition pairs receive an exact one-hot distribution based on
    the intended emotion encoded in the filename and are explicitly marked.
    """

    if alpha <= 0:
        raise ValueError(f"Laplace smoothing alpha must be positive, received {alpha}.")
    ordered_clip_ids = sorted(set(clip_ids))
    if len(ordered_clip_ids) != len(clip_ids):
        raise ValueError("clip_ids must be unique when aggregating votes.")
    missing_hard_labels = sorted(set(ordered_clip_ids) - set(hard_emotions))
    if missing_hard_labels:
        raise ValueError(
            "Missing hard-emotion fallback labels for clips: "
            f"{missing_hard_labels[:5]}."
        )
    unknown_hard_labels = sorted(
        {hard_emotions[clip_id] for clip_id in ordered_clip_ids} - set(LABELS)
    )
    if unknown_hard_labels:
        raise ValueError(f"Unknown hard-emotion labels: {unknown_hard_labels}.")

    votes = pd.read_csv(csv_path, dtype=str, usecols=lambda name: name in REQUIRED_VOTE_COLUMNS)
    missing_columns = REQUIRED_VOTE_COLUMNS - set(votes.columns)
    if missing_columns:
        raise ValueError(
            f"Vote CSV {Path(csv_path)} is missing required columns "
            f"{sorted(missing_columns)}."
        )
    for column in REQUIRED_VOTE_COLUMNS:
        votes[column] = votes[column].fillna("").str.strip()

    unknown_query_types = sorted(set(votes["queryType"]) - set(QUERY_TYPE_TO_CONDITION))
    if unknown_query_types:
        raise ValueError(f"Unknown CREMA-D queryType values: {unknown_query_types}.")
    unknown_responses = sorted(set(votes["respEmo"]) - set(RESPONSE_TO_LABEL))
    if unknown_responses:
        raise ValueError(f"Unknown CREMA-D respEmo values: {unknown_responses}.")

    valid_clip_set = set(ordered_clip_ids)
    in_manifest = votes["clipName"].isin(valid_clip_set)
    ignored_row_count = int((~in_manifest).sum())
    votes = votes.loc[in_manifest].copy()
    votes["condition"] = votes["queryType"].map(QUERY_TYPE_TO_CONDITION)
    votes["emotion"] = votes["respEmo"].map(RESPONSE_TO_LABEL)

    tables: dict[str, pd.DataFrame] = {}
    coverage_percent: dict[str, float] = {}
    fallback_counts: dict[str, int] = {}
    for condition in QUERY_TYPE_TO_CONDITION.values():
        condition_votes = votes.loc[votes["condition"] == condition]
        counts = pd.crosstab(condition_votes["clipName"], condition_votes["emotion"])
        counts = counts.reindex(index=ordered_clip_ids, columns=LABELS, fill_value=0)
        counts = counts.astype("int64")
        vote_count = counts.sum(axis=1)
        has_votes = vote_count > 0

        probabilities = (counts + alpha).div(vote_count + alpha * len(LABELS), axis=0)
        for clip_id in counts.index[~has_votes]:
            probabilities.loc[clip_id, :] = 0.0
            probabilities.loc[clip_id, hard_emotions[clip_id]] = 1.0

        table = probabilities.reset_index(names="clip_id")
        table["vote_count"] = vote_count.to_numpy(dtype="int64")
        table["used_fallback"] = (~has_votes).to_numpy(dtype="bool")
        if not table.loc[:, list(LABELS)].sum(axis=1).sub(1.0).abs().le(1e-6).all():
            raise RuntimeError(f"Soft-label rows for condition {condition!r} do not sum to 1.")

        fallback_count = int((~has_votes).sum())
        tables[condition] = table
        fallback_counts[condition] = fallback_count
        coverage_percent[condition] = (
            100.0 * int(has_votes.sum()) / len(ordered_clip_ids) if ordered_clip_ids else 0.0
        )

    return VoteAggregationResult(
        tables=tables,
        coverage_percent=coverage_percent,
        fallback_counts=fallback_counts,
        ignored_row_count=ignored_row_count,
    )

