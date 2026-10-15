"""Validated cached PAM features for lightweight VLM reconciliation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset


REQUIRED_TENSORS = {
    "fusion_embedding",
    "audio_probs",
    "video_probs",
    "fusion_probs",
    "disagreement",
    "labels",
}


class CachedVlmDataset(Dataset[dict[str, Tensor | str | int]]):
    """Dataset backed by one immutable tensor cache."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        payload = torch.load(self.path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise ValueError(f"VLM cache {self.path} must contain a mapping.")
        missing = REQUIRED_TENSORS - set(payload)
        if missing:
            raise ValueError(f"VLM cache {self.path} is missing {sorted(missing)}.")
        self.metadata = dict(payload.get("metadata", {}))
        self.clip_ids = [str(value) for value in payload["clip_ids"]]
        self.actor_ids = torch.as_tensor(payload["actor_ids"], dtype=torch.long)
        self.tensors = {
            key: torch.as_tensor(payload[key])
            for key in REQUIRED_TENSORS
        }
        lengths = {len(self.clip_ids), len(self.actor_ids)}
        lengths.update(int(tensor.shape[0]) for tensor in self.tensors.values())
        if len(lengths) != 1 or not self.clip_ids:
            raise ValueError(
                f"VLM cache {self.path} has inconsistent or empty lengths: {sorted(lengths)}."
            )
        if self.tensors["fusion_embedding"].ndim != 2:
            raise ValueError("fusion_embedding must have shape [N,H].")
        for key in ("audio_probs", "video_probs", "fusion_probs"):
            if self.tensors[key].ndim != 2:
                raise ValueError(f"{key} must have shape [N,C].")
        if self.tensors["labels"].ndim != 1:
            raise ValueError("labels must have shape [N].")

    def __len__(self) -> int:
        return len(self.clip_ids)

    def __getitem__(self, index: int) -> dict[str, Tensor | str | int]:
        return {
            "clip_id": self.clip_ids[index],
            "actor_id": int(self.actor_ids[index]),
            **{key: value[index] for key, value in self.tensors.items()},
        }


def save_vlm_cache(
    path: str | Path,
    *,
    clip_ids: list[str],
    actor_ids: Tensor,
    fusion_embedding: Tensor,
    audio_probs: Tensor,
    video_probs: Tensor,
    fusion_probs: Tensor,
    disagreement: Tensor,
    labels: Tensor,
    metadata: dict[str, Any],
) -> None:
    """Atomically save one feature split."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "clip_ids": clip_ids,
        "actor_ids": actor_ids.cpu(),
        "fusion_embedding": fusion_embedding.cpu(),
        "audio_probs": audio_probs.cpu(),
        "video_probs": video_probs.cpu(),
        "fusion_probs": fusion_probs.cpu(),
        "disagreement": disagreement.cpu(),
        "labels": labels.cpu(),
        "metadata": metadata,
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(output)
