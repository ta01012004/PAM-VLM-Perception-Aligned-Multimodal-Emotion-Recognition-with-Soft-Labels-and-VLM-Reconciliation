from pathlib import Path

import pandas as pd
import pytest

from av_mer.data.filename_parser import LABELS
from av_mer.data.votes import aggregate_votes


def write_votes(path: Path, rows: list[dict[str, str]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def test_vote_aggregation_smooths_and_falls_back(tmp_path: Path) -> None:
    vote_path = tmp_path / "votes.csv"
    write_votes(
        vote_path,
        [
            {"clipName": "1001_DFA_ANG_XX", "queryType": "1", "respEmo": "A"},
            {"clipName": "1001_DFA_ANG_XX", "queryType": "1", "respEmo": "A"},
            {"clipName": "1001_DFA_ANG_XX", "queryType": "1", "respEmo": "H"},
            {"clipName": "1001_DFA_ANG_XX", "queryType": "2", "respEmo": "D"},
            {"clipName": "1002_DFA_NEU_XX", "queryType": "2", "respEmo": "N"},
            {"clipName": "1002_DFA_NEU_XX", "queryType": "3", "respEmo": "S"},
            {"clipName": "practice", "queryType": "3", "respEmo": "H"},
        ],
    )
    clip_ids = ["1001_DFA_ANG_XX", "1002_DFA_NEU_XX"]
    hard_emotions = {"1001_DFA_ANG_XX": "angry", "1002_DFA_NEU_XX": "neutral"}

    result = aggregate_votes(vote_path, clip_ids, hard_emotions, alpha=1.0)

    voice = result.tables["voice"].set_index("clip_id")
    assert voice.loc["1001_DFA_ANG_XX", "angry"] == pytest.approx(3.0 / 9.0)
    assert voice.loc["1001_DFA_ANG_XX", "happy"] == pytest.approx(2.0 / 9.0)
    assert voice.loc["1001_DFA_ANG_XX", "neutral"] == pytest.approx(1.0 / 9.0)
    assert voice.loc["1002_DFA_NEU_XX", "neutral"] == 1.0
    assert bool(voice.loc["1002_DFA_NEU_XX", "used_fallback"])
    assert result.fallback_counts["voice"] == 1
    assert result.coverage_percent["voice"] == 50.0
    assert result.ignored_row_count == 1
    for table in result.tables.values():
        assert table.loc[:, list(LABELS)].sum(axis=1).to_numpy() == pytest.approx([1.0, 1.0])


def test_unknown_response_is_rejected(tmp_path: Path) -> None:
    vote_path = tmp_path / "votes.csv"
    write_votes(
        vote_path,
        [{"clipName": "1001_DFA_ANG_XX", "queryType": "1", "respEmo": "X"}],
    )

    with pytest.raises(ValueError, match="respEmo"):
        aggregate_votes(
            vote_path,
            ["1001_DFA_ANG_XX"],
            {"1001_DFA_ANG_XX": "angry"},
            alpha=1.0,
        )
