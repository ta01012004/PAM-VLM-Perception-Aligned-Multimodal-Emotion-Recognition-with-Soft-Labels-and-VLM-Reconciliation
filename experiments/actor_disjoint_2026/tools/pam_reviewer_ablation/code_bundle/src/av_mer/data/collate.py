"""Padding collator for variable-length audio-visual samples."""

from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence


SOFT_LABEL_KEYS: tuple[str, ...] = ("soft_voice", "soft_face", "soft_av")


def collate_av_batch(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Pad audio/video sequences and create boolean validity masks."""

    if not items:
        raise ValueError("Cannot collate an empty audio-visual batch.")

    audio_tensors = [item["audio"] for item in items]
    audio_lengths = torch.tensor([tensor.shape[0] for tensor in audio_tensors], dtype=torch.long)
    audio_values = pad_sequence(audio_tensors, batch_first=True)
    audio_mask = torch.arange(audio_values.shape[1]).unsqueeze(0) < audio_lengths.unsqueeze(1)

    video_tensors = [item["video"] for item in items]
    video_shapes = {tuple(tensor.shape[1:]) for tensor in video_tensors}
    if len(video_shapes) != 1:
        raise ValueError(
            "All videos must share [C,H,W] dimensions before collation; "
            f"received {sorted(video_shapes)}."
        )
    video_lengths = torch.tensor([tensor.shape[0] for tensor in video_tensors], dtype=torch.long)
    max_frames = int(video_lengths.max().item())
    channels, height, width = video_tensors[0].shape[1:]
    video_values = video_tensors[0].new_zeros(
        (len(items), max_frames, channels, height, width)
    )
    for batch_index, video in enumerate(video_tensors):
        video_values[batch_index, : video.shape[0]] = video
    video_mask = torch.arange(max_frames).unsqueeze(0) < video_lengths.unsqueeze(1)

    batch: dict[str, Any] = {
        "clip_ids": [str(item["clip_id"]) for item in items],
        "actor_ids": torch.tensor([int(item["actor_id"]) for item in items], dtype=torch.long),
        "labels": torch.tensor([int(item["label"]) for item in items], dtype=torch.long),
        "audio_values": audio_values,
        "audio_mask": audio_mask,
        "video_values": video_values,
        "video_mask": video_mask,
        "audio_sample_rates": torch.tensor(
            [int(item["audio_sample_rate"]) for item in items], dtype=torch.long
        ),
        "face_failure_counts": torch.tensor(
            [int(item.get("face_failure_count", 0)) for item in items], dtype=torch.long
        ),
    }
    for key in SOFT_LABEL_KEYS:
        batch[key] = torch.stack([item[key] for item in items]).to(dtype=torch.float32)
    return batch

