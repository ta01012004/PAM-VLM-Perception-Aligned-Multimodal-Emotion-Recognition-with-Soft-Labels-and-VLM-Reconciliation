"""Extensible multi-branch classification objective."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F


BRANCH_TO_OUTPUT = {
    "fusion": "logits",
    "audio": "audio_logits",
    "video": "video_logits",
}


@dataclass(frozen=True, slots=True)
class LossConfig:
    """Weights for hard CE and the step-05 soft-label extension point."""

    hard_fusion_weight: float
    hard_audio_weight: float
    hard_video_weight: float
    soft_labels_enabled: bool = False
    soft_fusion_weight: float = 0.0
    soft_audio_weight: float = 0.0
    soft_video_weight: float = 0.0

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "LossConfig":
        hard = values.get("hard_weights")
        soft = values.get("soft_weights", {})
        if not isinstance(hard, Mapping) or not isinstance(soft, Mapping):
            raise ValueError("loss.hard_weights and loss.soft_weights must be mappings.")
        try:
            return cls(
                hard_fusion_weight=float(hard["fusion"]),
                hard_audio_weight=float(hard["audio"]),
                hard_video_weight=float(hard["video"]),
                soft_labels_enabled=bool(values.get("soft_labels_enabled", False)),
                soft_fusion_weight=float(soft.get("fusion", 0.0)),
                soft_audio_weight=float(soft.get("audio", 0.0)),
                soft_video_weight=float(soft.get("video", 0.0)),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid loss configuration: {error}.") from error


@dataclass(frozen=True)
class LossResult:
    """Differentiable total and named scalar components for logging."""

    total: Tensor
    components: dict[str, Tensor]


class MultiBranchClassificationLoss(nn.Module):
    """Weighted hard-label CE with reserved soft-label configuration.

    Step 04 uses CE only. Step 05 can activate the already-configured soft-label
    terms without changing the training-loop interface.
    """

    def __init__(self, config: LossConfig) -> None:
        super().__init__()
        self.config = config
        hard_weights = self._hard_weights()
        if any(weight < 0 for weight in hard_weights.values()):
            raise ValueError(f"Hard-label loss weights must be non-negative: {hard_weights}.")
        if sum(hard_weights.values()) <= 0:
            raise ValueError("At least one hard-label loss weight must be positive.")
        soft_weights = self._soft_weights()
        if any(weight < 0 for weight in soft_weights.values()):
            raise ValueError(f"Soft-label loss weights must be non-negative: {soft_weights}.")
        if config.soft_labels_enabled:
            raise NotImplementedError(
                "Soft-label terms are reserved for step 05; keep "
                "soft_labels_enabled=false for hard-label baselines."
            )

    def _hard_weights(self) -> dict[str, float]:
        return {
            "fusion": self.config.hard_fusion_weight,
            "audio": self.config.hard_audio_weight,
            "video": self.config.hard_video_weight,
        }

    def _soft_weights(self) -> dict[str, float]:
        return {
            "fusion": self.config.soft_fusion_weight,
            "audio": self.config.soft_audio_weight,
            "video": self.config.soft_video_weight,
        }

    def forward(
        self,
        model_output: Mapping[str, Tensor | None],
        labels: Tensor,
        soft_targets: Mapping[str, Tensor] | None = None,
    ) -> LossResult:
        del soft_targets
        if labels.ndim != 1 or labels.dtype != torch.long:
            raise ValueError(
                f"Expected labels as LongTensor[B], received shape={tuple(labels.shape)} "
                f"dtype={labels.dtype}."
            )

        components: dict[str, Tensor] = {}
        weighted_terms: list[Tensor] = []
        for branch, weight in self._hard_weights().items():
            if weight == 0:
                continue
            output_key = BRANCH_TO_OUTPUT[branch]
            logits = model_output.get(output_key)
            if logits is None:
                raise ValueError(f"Model output {output_key!r} is required for {branch} loss.")
            if logits.ndim != 2 or logits.shape[0] != labels.shape[0]:
                raise ValueError(
                    f"Expected {output_key} shape [B,C] matching labels, received "
                    f"{tuple(logits.shape)} for {tuple(labels.shape)}."
                )
            num_classes = logits.shape[1]
            if ((labels < 0) | (labels >= num_classes)).any():
                invalid = labels[(labels < 0) | (labels >= num_classes)].tolist()
                raise ValueError(
                    f"Invalid class labels {invalid}; expected IDs in [0,{num_classes - 1}]."
                )
            if not torch.isfinite(logits).all():
                raise FloatingPointError(f"Non-finite values detected in {output_key}.")
            cross_entropy = F.cross_entropy(logits, labels)
            components[f"hard_{branch}_ce"] = cross_entropy
            weighted_terms.append(cross_entropy * weight)

        total = torch.stack(weighted_terms).sum()
        if not torch.isfinite(total):
            raise FloatingPointError("Non-finite total classification loss detected.")
        components["total"] = total
        return LossResult(total=total, components=components)

