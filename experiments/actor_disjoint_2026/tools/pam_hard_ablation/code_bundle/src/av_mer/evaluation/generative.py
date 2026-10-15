"""Parse and score constrained generative emotion outputs from the VLM branch."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import re
from typing import Any

from av_mer.data.filename_parser import LABELS

_EMOTION_RE = re.compile(r"<emotion>\s*([a-zA-Z]+)\s*</emotion>", re.IGNORECASE)
_LABEL_SET = {label.lower() for label in LABELS}


@dataclass(frozen=True, slots=True)
class ParsedGeneration:
    """One parsed generative output."""

    label: str | None       # canonical emotion label, or None if unparseable
    valid: bool             # a well-formed <emotion>known-label</emotion> was found
    empty: bool             # generation was empty / whitespace only


def parse_emotion(text: str) -> ParsedGeneration:
    """Extract a canonical emotion label from ``<emotion>...</emotion>`` markup.

    Valid iff the tag is present AND its content is one of the six canonical
    labels. Returns label=None with valid=False for malformed or unknown output.
    """

    if text is None or text.strip() == "":
        return ParsedGeneration(label=None, valid=False, empty=True)
    match = _EMOTION_RE.search(text)
    if match is None:
        return ParsedGeneration(label=None, valid=False, empty=False)
    candidate = match.group(1).lower()
    if candidate not in _LABEL_SET:
        return ParsedGeneration(label=None, valid=False, empty=False)
    return ParsedGeneration(label=candidate, valid=True, empty=False)


def score_generations(
    texts: Sequence[str],
    targets: Sequence[int],
    *,
    token_counts: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Accounting for a batch of generative outputs (protocol §3).

    Reports exact valid-label accuracy (over ALL samples, invalid counts as
    wrong), invalid-format rate, empty/refusal rate, and average token count.
    """

    if len(texts) != len(targets):
        raise ValueError("texts and targets must have equal length.")
    if len(texts) == 0:
        raise ValueError("Cannot score an empty batch.")
    total = len(texts)
    correct = 0
    invalid = 0
    empty = 0
    for text, target in zip(texts, targets):
        parsed = parse_emotion(text)
        if parsed.empty:
            empty += 1
        if not parsed.valid:
            invalid += 1
            continue
        if not (0 <= int(target) < len(LABELS)):
            raise ValueError(f"Invalid target id {target}.")
        if parsed.label == LABELS[int(target)]:
            correct += 1
    result = {
        "exact_valid_accuracy": correct / total,
        "invalid_format_rate": invalid / total,
        "empty_rate": empty / total,
        "num_samples": total,
    }
    if token_counts is not None:
        if len(token_counts) != total:
            raise ValueError("token_counts must align with texts.")
        result["avg_token_count"] = float(sum(token_counts) / total)
    return result
