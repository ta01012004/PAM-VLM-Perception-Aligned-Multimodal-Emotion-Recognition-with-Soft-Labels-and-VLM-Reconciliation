from __future__ import annotations

import torch

from av_mer.data.vlm_cache import CachedVlmDataset, save_vlm_cache


def test_cached_vlm_dataset_round_trip(tmp_path) -> None:
    path = tmp_path / "train.pt"
    n = 3
    save_vlm_cache(
        path,
        clip_ids=["a", "b", "c"],
        actor_ids=torch.arange(n),
        fusion_embedding=torch.randn(n, 8),
        audio_probs=torch.softmax(torch.randn(n, 6), dim=-1),
        video_probs=torch.softmax(torch.randn(n, 6), dim=-1),
        fusion_probs=torch.softmax(torch.randn(n, 6), dim=-1),
        disagreement=torch.rand(n),
        labels=torch.tensor([0, 1, 2]),
        metadata={"split": "train"},
    )
    dataset = CachedVlmDataset(path)
    assert len(dataset) == n
    assert dataset.metadata["split"] == "train"
    assert dataset[1]["clip_id"] == "b"
    assert dataset[1]["fusion_embedding"].shape == (8,)
