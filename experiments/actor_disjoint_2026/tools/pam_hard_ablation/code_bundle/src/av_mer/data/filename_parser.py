"""Parse the canonical CREMA-D clip naming convention."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


LABELS: tuple[str, ...] = (
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
)
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
EMOTION_CODE_TO_LABEL = {
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fear",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad",
}

_SUPPORTED_SUFFIXES = {".flv", ".mp3", ".wav"}
_CLIP_PATTERN = re.compile(
    r"^(?P<actor_id>\d{4})_"
    r"(?P<sentence_id>[A-Z]{3})_"
    r"(?P<emotion_code>ANG|DIS|FEA|HAP|NEU|SAD)_"
    r"(?P<intensity>HI|LO|MD|XX)$"
)


@dataclass(frozen=True, slots=True)
class CremaDClip:
    """Metadata encoded in one CREMA-D clip name."""

    clip_id: str
    actor_id: int
    sentence_id: str
    emotion: str
    emotion_id: int
    intensity: str


def parse_cremad_filename(value: str | Path) -> CremaDClip:
    """Parse a CREMA-D filename or extension-free clip ID.

    Raises:
        ValueError: If the suffix or clip naming convention is invalid.
    """

    filename = Path(value).name
    suffix = Path(filename).suffix.lower()
    if suffix and suffix not in _SUPPORTED_SUFFIXES:
        raise ValueError(
            f"Unsupported CREMA-D media suffix {suffix!r} for {filename!r}; "
            f"expected one of {sorted(_SUPPORTED_SUFFIXES)} or no suffix."
        )
    clip_id = filename[: -len(suffix)] if suffix else filename
    match = _CLIP_PATTERN.fullmatch(clip_id)
    if match is None:
        raise ValueError(
            f"Malformed CREMA-D clip name {filename!r}; expected "
            "ACTOR_SENTENCE_EMOTION_INTENSITY, for example "
            "1010_MTI_SAD_XX.wav."
        )

    emotion = EMOTION_CODE_TO_LABEL[match.group("emotion_code")]
    return CremaDClip(
        clip_id=clip_id,
        actor_id=int(match.group("actor_id")),
        sentence_id=match.group("sentence_id"),
        emotion=emotion,
        emotion_id=LABEL_TO_ID[emotion],
        intensity=match.group("intensity"),
    )

