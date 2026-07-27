"""Text targets and constrained label candidates for the Qwen reconciler."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import Tensor

from av_mer.data.filename_parser import LABELS


def emotion_answer(label: str) -> str:
    if label not in LABELS:
        raise ValueError(f"Unknown emotion label {label!r}.")
    return f"<emotion>{label}</emotion>"


def format_probability_evidence(
    audio_probs: Sequence[float],
    video_probs: Sequence[float],
    fusion_probs: Sequence[float],
    disagreement: float,
) -> str:
    """Render classifier evidence with explicit label semantics."""

    rows = {
        "voice": audio_probs,
        "face": video_probs,
        "fusion": fusion_probs,
    }
    rendered: list[str] = ["Classifier evidence:"]
    for name, values in rows.items():
        if len(values) != len(LABELS):
            raise ValueError(f"{name} probabilities must contain {len(LABELS)} values.")
        rendered.append(
            f"{name}: "
            + ", ".join(
                f"{label}={float(value):.4f}"
                for label, value in zip(LABELS, values)
            )
        )
    rendered.append(f"voice-face disagreement={float(disagreement):.4f}")
    return "\n".join(rendered)


@dataclass(frozen=True, slots=True)
class TextBatch:
    input_ids: Tensor
    attention_mask: Tensor
    labels: Tensor


class EmotionTextCodec:
    """Build teacher-forced targets while masking prompt tokens from LM loss."""

    def __init__(self, tokenizer: Any, prompt: str) -> None:
        self.tokenizer = tokenizer
        self.prompt = prompt.rstrip()
        self.prompt_ids = self._tokenize_prompt(None)
        if not self.prompt_ids:
            raise ValueError("The reconciler prompt tokenized to an empty sequence.")
        eos_id = tokenizer.eos_token_id
        if eos_id is None:
            raise ValueError("The reconciler tokenizer must define eos_token_id.")
        self.eos_id = int(eos_id)
        self.pad_id = int(tokenizer.pad_token_id)

    def _tokenize_prompt(self, evidence: str | None) -> list[int]:
        text = self.prompt
        if evidence:
            text += "\n\n" + evidence.strip()
        text += "\nAnswer:\n"
        return list(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def _answer_ids_and_targets(self, label_id: int) -> tuple[list[int], list[int]]:
        if not 0 <= int(label_id) < len(LABELS):
            raise ValueError(f"Invalid label id {label_id}.")
        opening = list(
            self.tokenizer("<emotion>", add_special_tokens=False)["input_ids"]
        )
        label = list(
            self.tokenizer(LABELS[int(label_id)], add_special_tokens=False)["input_ids"]
        )
        closing = list(
            self.tokenizer("</emotion>", add_special_tokens=False)["input_ids"]
        ) + [self.eos_id]
        if not label:
            raise ValueError(f"Label {LABELS[int(label_id)]!r} tokenized to nothing.")
        answer_ids = opening + label + closing
        # Only emotion-name tokens distinguish the six candidates. Masking common
        # markup prevents their easy prediction from dominating the training loss.
        answer_targets = [-100] * len(opening) + label + [-100] * len(closing)
        return answer_ids, answer_targets

    def encode_targets(
        self,
        label_ids: Sequence[int],
        evidence_rows: Sequence[str] | None = None,
    ) -> TextBatch:
        if evidence_rows is not None and len(evidence_rows) != len(label_ids):
            raise ValueError("evidence_rows must align with label_ids.")
        rows: list[list[int]] = []
        label_rows: list[list[int]] = []
        for index, label_id in enumerate(label_ids):
            prompt_ids = self._tokenize_prompt(
                evidence_rows[index] if evidence_rows is not None else None
            )
            answer_ids, answer_targets = self._answer_ids_and_targets(int(label_id))
            rows.append(prompt_ids + answer_ids)
            label_rows.append([-100] * len(prompt_ids) + answer_targets)
        return self._pad(rows, label_rows)

    def encode_candidates(
        self,
        batch_size: int,
        evidence_rows: Sequence[str] | None = None,
    ) -> TextBatch:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if evidence_rows is not None and len(evidence_rows) != batch_size:
            raise ValueError("evidence_rows must align with batch_size.")
        candidate_ids = list(range(len(LABELS))) * batch_size
        expanded_evidence = None
        if evidence_rows is not None:
            expanded_evidence = [
                evidence
                for evidence in evidence_rows
                for _ in LABELS
            ]
        return self.encode_targets(candidate_ids, expanded_evidence)

    def _pad(self, rows: list[list[int]], labels: list[list[int]]) -> TextBatch:
        width = max(len(row) for row in rows)
        input_ids = torch.full((len(rows), width), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(rows), width), dtype=torch.long)
        target_ids = torch.full((len(rows), width), -100, dtype=torch.long)
        for index, (row, targets) in enumerate(zip(rows, labels)):
            length = len(row)
            input_ids[index, :length] = torch.tensor(row, dtype=torch.long)
            attention_mask[index, :length] = 1
            target_ids[index, :length] = torch.tensor(targets, dtype=torch.long)
        return TextBatch(input_ids, attention_mask, target_ids)
