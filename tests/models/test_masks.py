import pytest
import torch

from av_mer.models.pam_vlm import PamVlmModel
from helpers import tiny_batch, tiny_model_config


def test_padded_values_do_not_change_outputs() -> None:
    torch.manual_seed(7)
    model = PamVlmModel(tiny_model_config()).eval()
    original = tiny_batch()
    changed = {key: value.clone() for key, value in original.items()}
    changed["audio_values"][~changed["audio_mask"]] = 10000.0
    changed["video_values"][~changed["video_mask"]] = -10000.0

    with torch.inference_mode():
        first = model(**original)
        second = model(**changed)

    for key in ("logits", "audio_logits", "video_logits", "fusion_embedding"):
        assert torch.allclose(first[key], second[key], atol=1e-6)


@pytest.mark.parametrize("modality", ["audio", "video"])
def test_all_padding_is_rejected(modality: str) -> None:
    model = PamVlmModel(tiny_model_config()).eval()
    batch = tiny_batch()
    batch[f"{modality}_mask"][0] = False

    with pytest.raises(ValueError, match="at least one valid"):
        model(**batch)

