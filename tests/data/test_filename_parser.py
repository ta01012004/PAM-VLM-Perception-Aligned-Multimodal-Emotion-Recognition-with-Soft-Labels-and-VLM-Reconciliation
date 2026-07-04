import pytest

from av_mer.data.filename_parser import LABELS, parse_cremad_filename


def test_parse_extension_free_clip_id() -> None:
    clip = parse_cremad_filename("1010_MTI_SAD_XX")

    assert clip.clip_id == "1010_MTI_SAD_XX"
    assert clip.actor_id == 1010
    assert clip.sentence_id == "MTI"
    assert clip.emotion == "sad"
    assert clip.emotion_id == 5
    assert clip.intensity == "XX"


def test_parse_media_path() -> None:
    clip = parse_cremad_filename("/tmp/1001_DFA_ANG_HI.wav")

    assert clip.clip_id == "1001_DFA_ANG_HI"
    assert clip.emotion == "angry"
    assert clip.emotion_id == 0


@pytest.mark.parametrize(
    "filename",
    [
        "1010_MTI_UNKNOWN_XX.wav",
        "1010_MTI_SAD.wav",
        "1010_mti_SAD_XX.wav",
        "1010_MTI_SAD_XX.txt",
    ],
)
def test_malformed_names_raise_actionable_error(filename: str) -> None:
    with pytest.raises(ValueError, match="CREMA-D"):
        parse_cremad_filename(filename)


def test_label_order_is_canonical() -> None:
    assert list(LABELS) == ["angry", "disgust", "fear", "happy", "neutral", "sad"]

