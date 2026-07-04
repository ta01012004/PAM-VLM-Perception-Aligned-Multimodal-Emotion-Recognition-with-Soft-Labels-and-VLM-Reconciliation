import torch

from av_mer.models.pam_vlm import PamVlmModel
from helpers import tiny_batch, tiny_model_config


def test_tiny_forward_returns_documented_shapes() -> None:
    model = PamVlmModel(tiny_model_config()).eval()

    with torch.inference_mode():
        output = model(**tiny_batch())

    assert output["logits"].shape == (2, 6)
    assert output["audio_logits"].shape == (2, 6)
    assert output["video_logits"].shape == (2, 6)
    assert output["fusion_embedding"].shape == (2, 8)
    assert output["loss"] is None
    assert output["generated_text"] is None
    assert torch.isfinite(output["fusion_embedding"]).all()

