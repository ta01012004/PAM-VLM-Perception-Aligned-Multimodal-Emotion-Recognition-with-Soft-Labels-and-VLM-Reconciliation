"""Generative output parsing + accounting tests (protocol §3)."""

from __future__ import annotations

import pytest

from av_mer.evaluation.generative import parse_emotion, score_generations


def test_parse_valid_emotion() -> None:
    p = parse_emotion("The tone is tense. <emotion>angry</emotion> because ...")
    assert p.valid and p.label == "angry" and not p.empty


def test_parse_is_case_insensitive_and_canonicalises() -> None:
    assert parse_emotion("<EMOTION>Happy</EMOTION>").label == "happy"


def test_parse_unknown_label_is_invalid() -> None:
    p = parse_emotion("<emotion>excited</emotion>")  # not one of the 6 canonical
    assert not p.valid and p.label is None


def test_parse_missing_tag_is_invalid() -> None:
    p = parse_emotion("I think the person is sad.")
    assert not p.valid and p.label is None and not p.empty


def test_parse_empty_flagged() -> None:
    p = parse_emotion("   ")
    assert p.empty and not p.valid


def test_score_generations_accounting() -> None:
    texts = [
        "<emotion>angry</emotion>",   # correct (target 0)
        "<emotion>sad</emotion>",     # wrong (target 1)
        "no tag here",                # invalid
        "",                           # empty + invalid
    ]
    targets = [0, 0, 2, 3]
    r = score_generations(texts, targets, token_counts=[5, 6, 4, 0])
    assert r["num_samples"] == 4
    assert r["exact_valid_accuracy"] == pytest.approx(0.25)   # only first correct
    assert r["invalid_format_rate"] == pytest.approx(0.5)     # 2 of 4 invalid
    assert r["empty_rate"] == pytest.approx(0.25)
    assert r["avg_token_count"] == pytest.approx(15 / 4)
