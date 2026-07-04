import torch

from av_mer.data.collate import collate_av_batch


def make_item(clip_id: str, audio_length: int, video_length: int, label: int) -> dict:
    return {
        "clip_id": clip_id,
        "actor_id": 1000 + label,
        "label": label,
        "audio": torch.arange(audio_length, dtype=torch.float32),
        "audio_sample_rate": 16000,
        "video": torch.ones(video_length, 3, 8, 8),
        "soft_voice": torch.nn.functional.one_hot(torch.tensor(label), 6).float(),
        "soft_face": torch.full((6,), 1.0 / 6.0),
        "soft_av": torch.nn.functional.one_hot(torch.tensor(label), 6).float(),
    }


def test_collator_pads_audio_video_and_builds_masks() -> None:
    batch = collate_av_batch(
        [
            make_item("clip-a", audio_length=5, video_length=2, label=0),
            make_item("clip-b", audio_length=3, video_length=4, label=1),
        ]
    )

    assert batch["audio_values"].shape == (2, 5)
    assert batch["audio_mask"].dtype == torch.bool
    assert batch["audio_mask"].tolist() == [
        [True, True, True, True, True],
        [True, True, True, False, False],
    ]
    assert batch["video_values"].shape == (2, 4, 3, 8, 8)
    assert batch["video_mask"].tolist() == [
        [True, True, False, False],
        [True, True, True, True],
    ]
    assert batch["labels"].shape == (2,)
    assert batch["soft_av"].shape == (2, 6)

