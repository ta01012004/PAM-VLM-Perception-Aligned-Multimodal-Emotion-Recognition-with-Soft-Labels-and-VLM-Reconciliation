from pathlib import Path
import random

import numpy as np
import pytest
import torch

from av_mer.training.checkpoint import load_checkpoint, save_checkpoint
from av_mer.training.reproducibility import build_reproducibility_record, write_json


def make_training_objects() -> tuple[
    torch.nn.Module,
    torch.optim.Optimizer,
    torch.optim.lr_scheduler.StepLR,
    torch.amp.GradScaler,
]:
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    return model, optimizer, scheduler, scaler


def test_resume_restores_full_state_and_rng(tmp_path: Path) -> None:
    random.seed(11)
    np.random.seed(11)
    torch.manual_seed(11)
    model, optimizer, scheduler, scaler = make_training_objects()
    loss = model(torch.ones(2, 3)).sum()
    loss.backward()
    optimizer.step()
    scheduler.step()
    saved_parameters = [parameter.detach().clone() for parameter in model.parameters()]
    checkpoint_path = tmp_path / "resume.pt"
    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        epoch=4,
        global_step=17,
        best_metric=0.75,
        label_mapping={"angry": 0, "sad": 1},
        resolved_config={"seed": 11},
        split_checksum="split-v1",
    )
    expected_python = random.random()
    expected_numpy = float(np.random.rand())
    expected_torch = torch.rand(3)

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(100.0)
    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)

    state = load_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        expected_label_mapping={"angry": 0, "sad": 1},
        expected_split_checksum="split-v1",
    )

    assert state.epoch == 4
    assert state.global_step == 17
    assert state.best_metric == pytest.approx(0.75)
    assert scheduler.last_epoch == 1
    for actual, expected in zip(model.parameters(), saved_parameters, strict=True):
        assert torch.equal(actual, expected)
    assert random.random() == expected_python
    assert float(np.random.rand()) == expected_numpy
    assert torch.equal(torch.rand(3), expected_torch)


def test_resume_rejects_split_mismatch(tmp_path: Path) -> None:
    model, optimizer, scheduler, scaler = make_training_objects()
    checkpoint_path = tmp_path / "resume.pt"
    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        epoch=1,
        global_step=1,
        best_metric=0.1,
        label_mapping={"angry": 0},
        resolved_config={},
        split_checksum="split-v1",
    )

    with pytest.raises(ValueError, match="split checksum"):
        load_checkpoint(
            checkpoint_path,
            model=model,
            expected_split_checksum="split-v2",
        )


def test_reproducibility_record_has_protocol_fields(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 2)
    record = build_reproducibility_record(
        run_id="fixture-run",
        repository_root=tmp_path,
        resolved_config={"seed": 42},
        seed=42,
        split_checksum="fixture-checksum",
        label_mapping={"angry": 0, "sad": 1},
        model=model,
        best_checkpoint_path="best.pt",
        best_validation_metric=0.5,
        test_metrics=None,
    )
    required_fields = {
        "run_id",
        "utc_timestamp",
        "git_commit",
        "dirty_worktree",
        "resolved_config",
        "seed",
        "split_manifest_checksum",
        "label_mapping",
        "package_versions",
        "cuda_version",
        "gpu_names",
        "parameter_count",
        "trainable_parameter_count",
        "best_checkpoint_path",
        "best_validation_metric",
        "test_metrics",
    }
    assert required_fields == set(record)
    assert record["test_metrics"] is None
    output_path = tmp_path / "record.json"
    write_json(output_path, record)
    assert output_path.exists()

