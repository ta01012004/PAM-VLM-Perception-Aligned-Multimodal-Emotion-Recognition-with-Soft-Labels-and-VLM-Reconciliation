from __future__ import annotations

import json

import pytest

from av_mer.evaluation.report import (
    collect_run,
    to_markdown,
    validate_comparable_runs,
    write_report,
)


def test_collect_run_flattens_saved_metrics(tmp_path) -> None:
    run_dir = tmp_path / "run-a"
    run_dir.mkdir()
    (run_dir / "resolved_config.json").write_text(
        json.dumps({"baseline": "fusion", "loss": {"soft_labels_enabled": True}}),
        encoding="utf-8",
    )
    (run_dir / "test_metrics.json").write_text(
        json.dumps({"macro_f1": 0.12345, "uar": 0.5, "accuracy": 0.7,
                    "weighted_f1": 0.2, "ece": {"ece": 0.33333}}),
        encoding="utf-8",
    )

    row = collect_run(run_dir)
    assert row["run"] == "run-a"
    assert row["baseline"] == "fusion"
    assert row["soft_labels"] is True
    assert row["test_macro_f1"] == 0.1235
    assert row["test_ece"] == 0.3333


def test_to_markdown_and_write_report(tmp_path) -> None:
    rows = [{"run": "a", "baseline": "audio", "test_accuracy": 0.25}]
    assert "| run | baseline | test_accuracy |" in to_markdown(rows)
    out = tmp_path / "summary.md"
    write_report([], out)
    assert out.read_text(encoding="utf-8").strip() == "(no runs)"


def test_validate_comparable_runs_rejects_mixed_checksums() -> None:
    rows = [
        {"run": "a", "protocol_id": "v2", "split_checksum": "aaa"},
        {"run": "b", "protocol_id": "v2", "split_checksum": "bbb"},
    ]

    with pytest.raises(ValueError, match="mixes split_checksum"):
        validate_comparable_runs(rows)


def test_validate_comparable_runs_accepts_one_protocol() -> None:
    rows = [
        {"run": "a", "protocol_id": "v2", "split_checksum": "aaa"},
        {"run": "b", "protocol_id": "v2", "split_checksum": "aaa"},
    ]

    validate_comparable_runs(rows)
