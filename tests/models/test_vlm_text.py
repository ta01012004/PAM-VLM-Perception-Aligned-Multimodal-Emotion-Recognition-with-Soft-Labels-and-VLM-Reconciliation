from __future__ import annotations

from av_mer.models.vlm_text import (
    EmotionTextCodec,
    emotion_answer,
    format_probability_evidence,
)


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def __call__(self, text: str, *, add_special_tokens: bool = False):
        del add_special_tokens
        return {"input_ids": [3 + (ord(character) % 23) for character in text]}


def test_emotion_codec_masks_prompt_and_supervises_answer() -> None:
    codec = EmotionTextCodec(FakeTokenizer(), "choose")
    batch = codec.encode_targets([0, 4])
    prompt_length = len(codec.prompt_ids)
    assert batch.input_ids.shape == batch.labels.shape
    assert batch.labels[:, :prompt_length].eq(-100).all()
    assert batch.labels[:, prompt_length:].ne(-100).any(dim=1).all()
    assert batch.labels[:, prompt_length:].eq(-100).any(dim=1).all()
    assert batch.attention_mask.sum(dim=1).le(batch.input_ids.shape[1]).all()


def test_emotion_codec_accepts_per_sample_evidence() -> None:
    codec = EmotionTextCodec(FakeTokenizer(), "choose")
    batch = codec.encode_targets([0, 4], ["voice says angry", "face says neutral"])
    assert batch.input_ids.shape[0] == 2
    assert batch.attention_mask[1].sum() > len(codec.prompt_ids)


def test_candidate_order_repeats_all_canonical_labels_per_sample() -> None:
    codec = EmotionTextCodec(FakeTokenizer(), "choose")
    candidates = codec.encode_candidates(batch_size=2, evidence_rows=["a", "b"])
    assert candidates.input_ids.shape[0] == 12


def test_emotion_answer_is_constrained_markup() -> None:
    assert emotion_answer("happy") == "<emotion>happy</emotion>"


def test_probability_evidence_names_every_probability_dimension() -> None:
    evidence = format_probability_evidence(
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        0.25,
    )
    assert "voice: angry=1.0000" in evidence
    assert "face:" in evidence and "disgust=1.0000" in evidence
    assert "fusion:" in evidence and "fear=1.0000" in evidence
    assert "disagreement=0.2500" in evidence
