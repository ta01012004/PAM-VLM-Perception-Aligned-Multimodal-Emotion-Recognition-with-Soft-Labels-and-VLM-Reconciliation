"""Actor-independent split helpers and manifest checksums."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import random
from typing import Any


SPLIT_NAMES: tuple[str, ...] = ("train", "val", "test")


@dataclass(frozen=True, slots=True)
class ActorSplitConfig:
    """Configuration for an exact-count actor-independent split."""

    train_count: int
    val_count: int
    test_count: int
    seed: int

    @property
    def total_count(self) -> int:
        return self.train_count + self.val_count + self.test_count


@dataclass(frozen=True, slots=True)
class ActorSplits:
    """Actor IDs allocated to train, validation, and test."""

    train: tuple[int, ...]
    val: tuple[int, ...]
    test: tuple[int, ...]

    def as_dict(self) -> dict[str, tuple[int, ...]]:
        return {"train": self.train, "val": self.val, "test": self.test}


@dataclass(frozen=True, slots=True)
class ClipSplitConfig:
    """Deterministic stratified clip split; actor overlap is allowed."""

    train_ratio: float
    val_ratio: float
    test_ratio: float
    seed: int


@dataclass(frozen=True, slots=True)
class ClipSplits:
    train: tuple[str, ...]
    val: tuple[str, ...]
    test: tuple[str, ...]

    def as_dict(self) -> dict[str, tuple[str, ...]]:
        return {"train": self.train, "val": self.val, "test": self.test}


def _apportion(total: int, weights: Sequence[float]) -> list[int]:
    raw = [total * weight for weight in weights]
    counts = [int(value) for value in raw]
    for index in sorted(
        range(len(weights)), key=lambda item: (raw[item] - counts[item], -item), reverse=True
    )[: total - sum(counts)]:
        counts[index] += 1
    return counts


def split_clips_by_label(
    clip_labels: Iterable[tuple[str, str]], config: ClipSplitConfig
) -> ClipSplits:
    """Split clips by class while matching global ratios as closely as possible."""

    ratios = (config.train_ratio, config.val_ratio, config.test_ratio)
    if any(ratio <= 0 for ratio in ratios) or abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError(f"Clip split ratios must be positive and sum to 1, got {ratios}.")
    by_label: dict[str, list[str]] = {}
    seen: set[str] = set()
    for clip_id, label in clip_labels:
        if clip_id in seen:
            raise ValueError(f"Duplicate clip ID in split input: {clip_id}.")
        seen.add(clip_id)
        by_label.setdefault(label, []).append(clip_id)
    if not seen:
        raise ValueError("Cannot split an empty clip collection.")

    rng = random.Random(config.seed)
    output = {name: [] for name in SPLIT_NAMES}
    for label in sorted(by_label):
        clips = sorted(by_label[label])
        rng.shuffle(clips)
        counts = _apportion(len(clips), ratios)
        start = 0
        for split_name, count in zip(SPLIT_NAMES, counts, strict=True):
            output[split_name].extend(clips[start : start + count])
            start += count

    splits = ClipSplits(**{name: tuple(sorted(output[name])) for name in SPLIT_NAMES})
    split_sets = {name: set(ids) for name, ids in splits.as_dict().items()}
    if any(split_sets[a] & split_sets[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise AssertionError("Clip leakage detected across clip-level splits.")
    if set().union(*split_sets.values()) != seen:
        raise AssertionError("Clip split is incomplete.")
    return splits


def split_actors(actor_ids: Iterable[int], config: ActorSplitConfig) -> ActorSplits:
    """Deterministically allocate unique actor IDs using exact configured counts."""

    actors = sorted(set(actor_ids))
    if any(count <= 0 for count in (config.train_count, config.val_count, config.test_count)):
        raise ValueError("Actor split counts must all be positive integers.")
    if len(actors) != config.total_count:
        raise ValueError(
            f"Actor split config requests {config.total_count} actors but "
            f"the manifest contains {len(actors)} unique actors."
        )

    random.Random(config.seed).shuffle(actors)
    train_end = config.train_count
    val_end = train_end + config.val_count
    splits = ActorSplits(
        train=tuple(sorted(actors[:train_end])),
        val=tuple(sorted(actors[train_end:val_end])),
        test=tuple(sorted(actors[val_end:])),
    )
    assert_actor_disjointness(splits)
    return splits


def assert_actor_disjointness(splits: ActorSplits) -> None:
    """Raise when any actor occurs in more than one split."""

    train_actors = set(splits.train)
    val_actors = set(splits.val)
    test_actors = set(splits.test)
    overlaps = {
        "train/val": sorted(train_actors & val_actors),
        "train/test": sorted(train_actors & test_actors),
        "val/test": sorted(val_actors & test_actors),
    }
    nonempty = {name: values for name, values in overlaps.items() if values}
    if nonempty:
        raise ValueError(f"Actor leakage detected across splits: {nonempty}.")


def build_actor_to_split(splits: ActorSplits) -> dict[int, str]:
    """Build an actor-to-split lookup after validating disjointness."""

    assert_actor_disjointness(splits)
    return {
        actor_id: split_name
        for split_name, actor_ids in splits.as_dict().items()
        for actor_id in actor_ids
    }


def compute_manifest_checksum(rows: Sequence[Mapping[str, Any]]) -> str:
    """Return a stable SHA-256 checksum independent of input row order."""

    ordered_rows = sorted((dict(row) for row in rows), key=lambda row: str(row["clip_id"]))
    payload = json.dumps(
        ordered_rows,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
