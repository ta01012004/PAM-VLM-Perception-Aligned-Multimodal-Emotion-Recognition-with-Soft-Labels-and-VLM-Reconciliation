#!/usr/bin/env python3
"""Zero-shot RAVDESS evaluation with a CREMA-D-trained PAM+VLM reconciler.

The script performs no target-side training. It scores the six valid emotion
tags by conditional NLL and applies the routing threshold selected on CREMA-D
validation.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
from typing import Any

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from av_mer.data.collate import collate_av_batch  # noqa: E402
from av_mer.data.audio_io import load_audio  # noqa: E402
from av_mer.data.dataset import DatasetConfig  # noqa: E402
from av_mer.data.disagreement import audio_video_kl  # noqa: E402
from av_mer.data.filename_parser import LABELS  # noqa: E402
from av_mer.data.video_io import VideoConfig, create_face_detector, load_video  # noqa: E402
from av_mer.evaluation.metrics import compute_classification_metrics  # noqa: E402
from av_mer.evaluation.vlm_routing import routed_predictions  # noqa: E402
from av_mer.models.config import PamVlmConfig  # noqa: E402
from av_mer.models.llm_adapter import LLMAdapterConfig, LLMReconciler  # noqa: E402
from av_mer.models.pam_vlm import PamVlmModel  # noqa: E402
from av_mer.models.projector import FusionToLLMProjector  # noqa: E402
from av_mer.models.vlm_text import (  # noqa: E402
    EmotionTextCodec,
    format_probability_evidence,
)
from av_mer.training.checkpoint import load_checkpoint  # noqa: E402

LOGGER = logging.getLogger(__name__)
REQUIRED_TARGET_COLUMNS = {
    "clip_id",
    "actor_id",
    "emotion",
    "emotion_id",
    "video_path",
    "audio_path",
}


class ZeroShotAvDataset(Dataset[dict[str, Any]]):
    """Generic target-manifest dataset with no target-side soft labels."""

    def __init__(
        self,
        manifest_path: Path,
        raw_root: Path,
        config: DatasetConfig,
        *,
        decode_error_report: Path | None = None,
    ) -> None:
        import pandas as pd

        self._manifest = pd.read_csv(manifest_path)
        missing = REQUIRED_TARGET_COLUMNS - set(self._manifest.columns)
        if missing:
            raise ValueError(f"Target manifest {manifest_path} is missing {sorted(missing)}.")
        self.raw_root = raw_root
        self.config = config
        self.face_detector = create_face_detector(config.face_detector)
        self.decode_error_report = decode_error_report
        self._uniform_soft = torch.full((len(LABELS),), 1.0 / len(LABELS), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self._manifest)

    def _record_decode_error(self, clip_id: str, media_path: Path, error: Exception) -> None:
        if self.decode_error_report is None:
            return
        self.decode_error_report.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self.decode_error_report.exists()
        with self.decode_error_report.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            if write_header:
                writer.writerow(["clip_id", "media_path", "error_type", "message"])
            writer.writerow([clip_id, media_path.as_posix(), type(error).__name__, str(error)])

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self._manifest.iloc[index]
        clip_id = str(row["clip_id"])
        audio_path = self.raw_root / str(row["audio_path"])
        video_path = self.raw_root / str(row["video_path"])
        try:
            audio_result = load_audio(audio_path, self.config.sample_rate)
        except Exception as error:
            self._record_decode_error(clip_id, audio_path, error)
            raise
        try:
            video_result = load_video(
                video_path,
                VideoConfig(
                    num_frames=self.config.num_frames,
                    frame_size=self.config.frame_size,
                    face_margin=self.config.face_margin,
                ),
                self.face_detector,
                training=False,
            )
        except Exception as error:
            self._record_decode_error(clip_id, video_path, error)
            raise
        return {
            "clip_id": clip_id,
            "actor_id": int(row["actor_id"]),
            "label": int(row["emotion_id"]),
            "audio": audio_result.waveform.to(dtype=torch.float32),
            "audio_sample_rate": audio_result.sample_rate,
            "video": video_result.frames.to(dtype=torch.float32),
            "face_failure_count": video_result.face_failure_count,
            "soft_voice": self._uniform_soft.clone(),
            "soft_face": self._uniform_soft.clone(),
            "soft_av": self._uniform_soft.clone(),
        }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/zeroshot_ravdess.yaml"))
    parser.add_argument("--reconciler-checkpoint", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke-test sample cap.")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    return parser.parse_args()


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config {path} must contain a YAML object.")
    return data


def _select_device(value: str) -> torch.device:
    if value == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable.")
        return torch.device("cuda")
    if value == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _format_prompt(template: str, labels: list[str]) -> str:
    return template.replace("{LABELS}", ", ".join(labels))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = _load_yaml(args.config)
    data_cfg = config["data"]
    eval_cfg = config["evaluation"]
    vlm_cfg = config["vlm"]

    manifest_path = _repo_path(data_cfg["manifest"])
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Zero-shot manifest not found: {manifest_path}. "
            "Place the target dataset manifest there before running zero-shot eval."
        )
    raw_root = _repo_path(data_cfg["raw_root"])
    if not raw_root.exists():
        raise FileNotFoundError(f"Zero-shot raw_root not found: {raw_root}.")

    labels = [str(label) for label in vlm_cfg["labels"]]
    if labels != list(LABELS):
        raise ValueError(
            "Current parser supports the six canonical labels only: "
            f"{list(LABELS)}; received {labels}."
        )

    device = _select_device(args.device)
    loader_config = DatasetConfig.from_json(
        _repo_path(data_cfg["loader_config"]),
        seed=int(config.get("seed", 42)),
    )
    dataset: Dataset[dict[str, Any]] = ZeroShotAvDataset(
        manifest_path,
        raw_root,
        loader_config,
        decode_error_report=_repo_path(data_cfg["decode_error_report"]),
    )
    if args.limit is not None:
        dataset = torch.utils.data.Subset(dataset, range(min(args.limit, len(dataset))))
    loader = DataLoader(
        dataset,
        batch_size=int(eval_cfg.get("batch_size", 1)),
        shuffle=False,
        num_workers=int(eval_cfg.get("num_workers", 2)),
        pin_memory=device.type == "cuda",
        collate_fn=collate_av_batch,
    )

    model = PamVlmModel(PamVlmConfig.from_json(_repo_path(config["model_config"])))
    load_checkpoint(_repo_path(config["backbone_checkpoint"]), model=model, map_location=device,
                    restore_rng=False)
    model.to(device).eval()

    rec = LLMReconciler(LLMAdapterConfig(
        model_id=str(vlm_cfg["model_id"]),
        lora_rank=int(vlm_cfg["lora_rank"]),
        lora_alpha=int(vlm_cfg["lora_alpha"]),
        lora_dropout=float(vlm_cfg["lora_dropout"]),
        num_prefix_tokens=int(vlm_cfg["num_prefix_tokens"]),
        max_new_tokens=int(vlm_cfg["max_new_tokens"]),
        freeze_base=bool(vlm_cfg.get("freeze_base", True)),
    )).to(device).eval()
    projector = FusionToLLMProjector(
        fusion_dim=model.config.fusion.hidden_dim,
        llm_hidden=rec.hidden_size,
        num_prefix_tokens=int(vlm_cfg["num_prefix_tokens"]),
    ).to(device).eval()

    reconciler_checkpoint = torch.load(
        _repo_path(args.reconciler_checkpoint),
        map_location="cpu",
        weights_only=False,
    )
    rec.load_adapter_state_dict(reconciler_checkpoint["adapter_state"])
    projector.load_state_dict(reconciler_checkpoint["projector_state"])
    routing_threshold = float(reconciler_checkpoint["routing_threshold"])
    prompt_template = _repo_path(vlm_cfg["prompt_file"]).read_text(encoding="utf-8")
    prompt = _format_prompt(prompt_template, labels)
    codec = EmotionTextCodec(rec.tokenizer, prompt)

    targets: list[int] = []
    classifier_predictions: list[int] = []
    vlm_predictions: list[int] = []
    disagreements: list[float] = []
    classifier_confidence: list[float] = []
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for batch in loader:
            audio = batch["audio_values"].to(device)
            audio_mask = batch["audio_mask"].to(device)
            video = batch["video_values"].to(device)
            video_mask = batch["video_mask"].to(device)
            output = model(audio, audio_mask, video, video_mask)
            p_audio = torch.softmax(output["audio_logits"].float(), dim=-1)
            p_video = torch.softmax(output["video_logits"].float(), dim=-1)
            disagreement = audio_video_kl(
                output["audio_logits"].float(),
                output["video_logits"].float(),
            )
            prefix = projector(output["fusion_embedding"], p_audio, p_video, disagreement)
            p_fusion = torch.softmax(output["logits"].float(), dim=-1)
            evidence = [
                format_probability_evidence(
                    p_audio[index].cpu().tolist(),
                    p_video[index].cpu().tolist(),
                    p_fusion[index].cpu().tolist(),
                    float(disagreement[index].cpu()),
                )
                for index in range(audio.shape[0])
            ]
            candidate_text = codec.encode_candidates(audio.shape[0], evidence)
            candidate_nll = rec.sequence_nll(
                prefix.repeat_interleave(len(LABELS), dim=0),
                candidate_text.input_ids.to(device),
                candidate_text.attention_mask.to(device),
                candidate_text.labels.to(device),
            ).view(audio.shape[0], len(LABELS))
            batch_vlm_predictions = candidate_nll.argmin(dim=1).cpu()
            batch_classifier_predictions = p_fusion.argmax(dim=1).cpu()
            labels_tensor = batch["labels"].cpu()
            for index, clip_id in enumerate(batch["clip_ids"]):
                target_id = int(labels_tensor[index])
                targets.append(target_id)
                classifier_id = int(batch_classifier_predictions[index])
                vlm_id = int(batch_vlm_predictions[index])
                classifier_predictions.append(classifier_id)
                vlm_predictions.append(vlm_id)
                disagreements.append(float(disagreement[index].detach().cpu()))
                classifier_confidence.append(float(p_fusion[index].max().detach().cpu()))
                rows.append({
                    "clip_id": clip_id,
                    "actor_id": int(batch["actor_ids"][index]),
                    "true": LABELS[target_id],
                    "classifier": LABELS[classifier_id],
                    "vlm": LABELS[vlm_id],
                    "disagreement": float(disagreement[index].detach().cpu()),
                    "classifier_confidence": classifier_confidence[-1],
                })

    confidence_max = float(
        reconciler_checkpoint.get("classifier_confidence_max", float("inf"))
    )
    routed = routed_predictions(
        classifier_predictions,
        vlm_predictions,
        disagreements,
        routing_threshold,
        classifier_confidence=classifier_confidence,
        classifier_confidence_max=confidence_max,
    )
    for index, prediction in enumerate(routed):
        rows[index]["routed"] = LABELS[int(prediction)]
    metrics = {
        "target": "ravdess",
        "num_samples": len(targets),
        "checkpoint": str(_repo_path(config["backbone_checkpoint"])),
        "reconciler_checkpoint": str(_repo_path(args.reconciler_checkpoint)),
        "prompt_file": str(_repo_path(vlm_cfg["prompt_file"])),
        "routing_threshold_from_cremad_validation": routing_threshold,
        "classifier_confidence_max_from_cremad_validation": confidence_max,
        "routed_fraction": float(
            (
                torch.tensor(disagreements).ge(routing_threshold)
                & torch.tensor(classifier_confidence).le(confidence_max)
            ).float().mean()
        ),
        "classifier": compute_classification_metrics(targets, classifier_predictions),
        "vlm": compute_classification_metrics(targets, vlm_predictions),
        "routed": compute_classification_metrics(targets, routed),
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }

    output_dir = _repo_path(eval_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "metrics.json", metrics)
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    LOGGER.info(
        "Zero-shot %s classifier_f1=%.4f vlm_f1=%.4f routed_f1=%.4f n=%d",
        metrics["target"],
        metrics["classifier"]["macro_f1"],
        metrics["vlm"]["macro_f1"],
        metrics["routed"]["macro_f1"],
        metrics["num_samples"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
