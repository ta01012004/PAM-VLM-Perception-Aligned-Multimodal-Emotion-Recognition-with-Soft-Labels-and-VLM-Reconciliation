from pathlib import Path

import pytest
import torch

from av_mer.data.filename_parser import LABEL_TO_ID
from av_mer.models.pam_vlm import PamVlmModel
from av_mer.training.checkpoint import load_checkpoint
from av_mer.training.loop import LoopConfig, Trainer, run_validation_epoch, seed_everything
from av_mer.training.losses import LossConfig, MultiBranchClassificationLoss
from training_fixtures import tiny_loader, tiny_model_config


def hard_fusion_criterion() -> MultiBranchClassificationLoss:
    return MultiBranchClassificationLoss(
        LossConfig(
            hard_fusion_weight=0.6,
            hard_audio_weight=0.2,
            hard_video_weight=0.2,
        )
    )


def loop_config() -> LoopConfig:
    return LoopConfig(
        epochs=1,
        gradient_accumulation_steps=2,
        amp=False,
        max_grad_norm=1.0,
        early_stopping_patience=1,
        primary_metric="macro_f1",
        baseline="fusion",
        seed=42,
    )


def test_cpu_smoke_train_validate_checkpoint_and_reload(tmp_path: Path) -> None:
    seed_everything(42)
    model = PamVlmModel(tiny_model_config())
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-3,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=hard_fusion_criterion(),
        config=loop_config(),
        device=torch.device("cpu"),
        checkpoint_dir=tmp_path / "checkpoints",
        label_mapping=LABEL_TO_ID,
        resolved_config={"fixture": True, "test_split_used": False},
        split_checksum="fixture-split-checksum",
    )
    train_loader = tiny_loader()
    val_loader = tiny_loader()

    fit_result = trainer.fit(train_loader, val_loader)

    assert fit_result.final_global_step == 1
    assert fit_result.best_checkpoint_path.exists()
    assert fit_result.last_checkpoint_path.exists()
    assert torch.isfinite(torch.tensor(fit_result.history[0]["train_loss"]))
    assert torch.isfinite(torch.tensor(fit_result.history[0]["val_loss"]))
    assert fit_result.best_validation_metrics["num_samples"] == 4

    restored_model = PamVlmModel(tiny_model_config())
    restored_optimizer = torch.optim.AdamW(
        [parameter for parameter in restored_model.parameters() if parameter.requires_grad],
        lr=1e-3,
    )
    restored_scheduler = torch.optim.lr_scheduler.StepLR(restored_optimizer, step_size=1)
    restored_scaler = torch.amp.GradScaler("cpu", enabled=False)
    resume = load_checkpoint(
        fit_result.best_checkpoint_path,
        model=restored_model,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        scaler=restored_scaler,
        expected_label_mapping=LABEL_TO_ID,
        expected_split_checksum="fixture-split-checksum",
    )
    assert resume.epoch == 1
    assert resume.global_step == 1
    validation = run_validation_epoch(
        model=restored_model,
        data_loader=val_loader,
        criterion=hard_fusion_criterion(),
        config=loop_config(),
        device=torch.device("cpu"),
        global_step=resume.global_step,
    )
    assert validation.metrics["num_samples"] == 4
    assert validation.loss >= 0.0


def test_loss_rejects_invalid_labels_and_nonfinite_logits() -> None:
    criterion = hard_fusion_criterion()
    valid_output = {
        "logits": torch.randn(2, 6),
        "audio_logits": torch.randn(2, 6),
        "video_logits": torch.randn(2, 6),
    }
    with pytest.raises(ValueError, match="Invalid class labels"):
        criterion(valid_output, torch.tensor([0, 6], dtype=torch.long))

    invalid_output = dict(valid_output)
    invalid_output["logits"] = torch.full((2, 6), float("nan"))
    with pytest.raises(FloatingPointError, match="Non-finite"):
        criterion(invalid_output, torch.tensor([0, 1], dtype=torch.long))


def test_soft_label_kl_loss_zero_when_prediction_matches_and_positive_otherwise() -> None:
    criterion = MultiBranchClassificationLoss(
        LossConfig(
            hard_fusion_weight=0.0,
            hard_audio_weight=0.0,
            hard_video_weight=0.0,
            soft_labels_enabled=True,
            soft_fusion_weight=0.6,
            soft_audio_weight=0.2,
            soft_video_weight=0.2,
        )
    )
    # Target distributions; build logits whose softmax equals the target exactly,
    # so forward KL(target || softmax(logits)) must be ~0.
    target = torch.tensor(
        [[0.7, 0.1, 0.1, 0.05, 0.025, 0.025], [0.1, 0.1, 0.2, 0.3, 0.2, 0.1]]
    )
    matched_logits = torch.log(target)
    labels = torch.tensor([0, 3], dtype=torch.long)
    soft = {"fusion": target, "audio": target, "video": target}
    matched = criterion(
        {"logits": matched_logits, "audio_logits": matched_logits, "video_logits": matched_logits},
        labels,
        soft,
    )
    assert matched.total.item() == pytest.approx(0.0, abs=1e-5)
    for name in ("soft_fusion_kl", "soft_audio_kl", "soft_video_kl"):
        assert name in matched.components

    mismatched_logits = torch.zeros_like(target)  # uniform softmax != target
    mismatched = criterion(
        {
            "logits": mismatched_logits,
            "audio_logits": mismatched_logits,
            "video_logits": mismatched_logits,
        },
        labels,
        soft,
    )
    assert mismatched.total.item() > 0.0


def test_soft_label_requires_positive_soft_weight() -> None:
    with pytest.raises(ValueError, match="positive soft weight"):
        MultiBranchClassificationLoss(
            LossConfig(
                hard_fusion_weight=0.0,
                hard_audio_weight=0.0,
                hard_video_weight=0.0,
                soft_labels_enabled=True,
                soft_fusion_weight=0.0,
                soft_audio_weight=0.0,
                soft_video_weight=0.0,
            )
        )


def test_combined_hard_ce_and_soft_kl_sums_both_terms() -> None:
    # Distillation mode: positive hard AND soft weights -> total includes both
    # hard_*_ce and soft_*_kl components (CE anchors argmax, KL adds calibration).
    criterion = MultiBranchClassificationLoss(
        LossConfig(
            hard_fusion_weight=0.6,
            hard_audio_weight=0.2,
            hard_video_weight=0.2,
            soft_labels_enabled=True,
            soft_fusion_weight=0.3,
            soft_audio_weight=0.1,
            soft_video_weight=0.1,
        )
    )
    logits = torch.randn(4, 6)
    output = {"logits": logits, "audio_logits": logits.clone(), "video_logits": logits.clone()}
    labels = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    target = torch.softmax(torch.randn(4, 6), dim=-1)
    soft = {"fusion": target, "audio": target, "video": target}
    result = criterion(output, labels, soft)
    assert {"hard_fusion_ce", "hard_audio_ce", "hard_video_ce",
            "soft_fusion_kl", "soft_audio_kl", "soft_video_kl"}.issubset(result.components)
    assert torch.isfinite(result.total)
    # total must equal the weighted sum of all six terms.
    expected = (
        0.6 * result.components["hard_fusion_ce"]
        + 0.2 * result.components["hard_audio_ce"]
        + 0.2 * result.components["hard_video_ce"]
        + 0.3 * result.components["soft_fusion_kl"]
        + 0.1 * result.components["soft_audio_kl"]
        + 0.1 * result.components["soft_video_kl"]
    )
    assert result.total.item() == pytest.approx(expected.item(), rel=1e-5)

