"""High-level single-clip predictor API."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from av_mer.data.audio_io import load_audio
from av_mer.data.filename_parser import LABELS
from av_mer.data.video_io import VideoConfig, create_face_detector, load_video
from av_mer.models.config import PamVlmConfig
from av_mer.models.pam_vlm import PamVlmModel
from av_mer.training.checkpoint import load_checkpoint


@dataclass(frozen=True, slots=True)
class PredictorConfig:
    """Runtime media preprocessing and checkpoint configuration."""

    checkpoint: Path
    model_config: Path
    sample_rate: int = 16000
    num_frames: int = 16
    frame_size: int = 224
    face_detector: str = "opencv_haar"
    face_margin: float = 0.15
    device: str = "auto"


class PamVlmPredictor:
    """Load a frozen checkpoint and predict one audio-visual clip."""

    def __init__(self, config: PredictorConfig) -> None:
        self.config = config
        self.device = self._select_device(config.device)
        self.model = PamVlmModel(PamVlmConfig.from_json(config.model_config))
        load_checkpoint(config.checkpoint, model=self.model, map_location=self.device,
                        restore_rng=False)
        self.model.to(self.device).eval()
        self.face_detector = create_face_detector(config.face_detector)

    @staticmethod
    def _select_device(value: str) -> torch.device:
        if value == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but unavailable.")
            return torch.device("cuda")
        if value == "cpu":
            return torch.device("cpu")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def infer_audio_path(video_path: str | Path) -> Path:
        """Find a sidecar audio file next to a video clip."""

        video = Path(video_path)
        for suffix in (".wav", ".mp3", ".flac", ".ogg"):
            candidate = video.with_suffix(suffix)
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"No sidecar audio found for {video}; pass audio_path explicitly."
        )

    @torch.inference_mode()
    def predict(self, video_path: str | Path, audio_path: str | Path | None = None) -> dict[str, Any]:
        """Return label, confidence, probabilities, and decode diagnostics."""

        video = Path(video_path)
        audio = Path(audio_path) if audio_path is not None else self.infer_audio_path(video)
        audio_result = load_audio(audio, self.config.sample_rate)
        video_result = load_video(
            video,
            VideoConfig(
                num_frames=self.config.num_frames,
                frame_size=self.config.frame_size,
                face_margin=self.config.face_margin,
            ),
            self.face_detector,
            training=False,
        )
        audio_values = audio_result.waveform.unsqueeze(0).to(self.device)
        audio_mask = torch.ones_like(audio_values, dtype=torch.bool)
        video_values = video_result.frames.unsqueeze(0).to(self.device)
        video_mask = torch.ones((1, video_values.shape[1]), dtype=torch.bool, device=self.device)
        output = self.model(audio_values, audio_mask, video_values, video_mask)
        probs = torch.softmax(output["logits"].float(), dim=-1)[0].detach().cpu()
        pred_id = int(probs.argmax().item())
        return {
            "label": LABELS[pred_id],
            "confidence": float(probs[pred_id]),
            "probs": {label: float(probs[index]) for index, label in enumerate(LABELS)},
            "generated_text": output.get("generated_text"),
            "diagnostics": {
                "audio_path": str(audio),
                "video_path": str(video),
                "audio_samples": int(audio_result.length),
                "sampled_frames": list(video_result.sampled_indices),
                "source_frame_count": int(video_result.source_frame_count),
                "face_failure_count": int(video_result.face_failure_count),
            },
        }
