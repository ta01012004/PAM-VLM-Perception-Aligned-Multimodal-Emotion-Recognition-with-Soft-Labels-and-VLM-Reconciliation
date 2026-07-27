import pytest

from av_mer.data.split import (
    ActorSplitConfig,
    ActorSplits,
    ClipSplitConfig,
    assert_actor_disjointness,
    compute_manifest_checksum,
    split_clips_by_label,
    split_actors,
)


def test_actor_split_is_seeded_disjoint_and_complete() -> None:
    actors = range(1001, 1092)
    config = ActorSplitConfig(train_count=70, val_count=10, test_count=11, seed=42)

    first = split_actors(actors, config)
    second = split_actors(reversed(list(actors)), config)

    assert first == second
    assert len(first.train) == 70
    assert len(first.val) == 10
    assert len(first.test) == 11
    assert set(first.train).isdisjoint(first.val)
    assert set(first.train).isdisjoint(first.test)
    assert set(first.val).isdisjoint(first.test)
    assert set(first.train) | set(first.val) | set(first.test) == set(actors)


def test_actor_count_mismatch_is_rejected() -> None:
    config = ActorSplitConfig(train_count=2, val_count=1, test_count=1, seed=42)

    with pytest.raises(ValueError, match="requests 4 actors"):
        split_actors([1001, 1002, 1003], config)


def test_actor_overlap_is_rejected() -> None:
    splits = ActorSplits(train=(1001, 1002), val=(1002,), test=(1003,))

    with pytest.raises(ValueError, match="Actor leakage"):
        assert_actor_disjointness(splits)


def test_manifest_checksum_is_independent_of_row_order() -> None:
    rows = [
        {"clip_id": "1002_DFA_HAP_XX", "split": "test"},
        {"clip_id": "1001_DFA_ANG_XX", "split": "train"},
    ]

    assert compute_manifest_checksum(rows) == compute_manifest_checksum(list(reversed(rows)))


def test_clip_split_is_seeded_stratified_and_complete() -> None:
    clips = [(f"angry-{i}", "angry") for i in range(10)] + [
        (f"sad-{i}", "sad") for i in range(10)
    ]
    config = ClipSplitConfig(0.8, 0.1, 0.1, seed=42)

    first = split_clips_by_label(clips, config)
    second = split_clips_by_label(reversed(clips), config)

    assert first == second
    assert len(first.train) == 16
    assert len(first.val) == 2
    assert len(first.test) == 2
    assert {clip.split("-")[0] for clip in first.val} == {"angry", "sad"}
    assert set(first.train).isdisjoint(first.val)
    assert set(first.train).isdisjoint(first.test)
    assert set(first.val).isdisjoint(first.test)
