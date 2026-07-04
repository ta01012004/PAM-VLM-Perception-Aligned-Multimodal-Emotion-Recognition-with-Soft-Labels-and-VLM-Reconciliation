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


def test_soft_label_flag_is_reserved_for_step_five() -> None:
    with pytest.raises(NotImplementedError, match="step 05"):
        MultiBranchClassificationLoss(
            LossConfig(
                hard_fusion_weight=1.0,
                hard_audio_weight=0.0,
                hard_video_weight=0.0,
                soft_labels_enabled=True,
                soft_fusion_weight=0.6,
                soft_audio_weight=0.2,
                soft_video_weight=0.2,
            )
        )

