"""Persist the experiment identity required by the evaluation protocol."""

from __future__ import annotations

from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

import torch
from torch import nn


TRACKED_PACKAGES = (
    "numpy",
    "pandas",
    "pyarrow",
    "scikit-learn",
    "torch",
    "torchvision",
    "transformers",
)


def _git_identity(repository_root: Path) -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repository_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return commit, dirty
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None, None


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in TRACKED_PACKAGES:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def build_reproducibility_record(
    *,
    run_id: str,
    repository_root: str | Path,
    resolved_config: Mapping[str, Any],
    seed: int,
    split_checksum: str,
    label_mapping: Mapping[str, int],
    model: nn.Module,
    best_checkpoint_path: str | Path,
    best_validation_metric: float,
    test_metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create the complete protocol section-4 reproducibility record."""

    repository_path = Path(repository_root)
    git_commit, dirty_worktree = _git_identity(repository_path)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    gpu_names = (
        [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
        if torch.cuda.is_available()
        else []
    )
    return {
        "run_id": run_id,
        "utc_timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "dirty_worktree": dirty_worktree,
        "resolved_config": dict(resolved_config),
        "seed": int(seed),
        "split_manifest_checksum": split_checksum,
        "label_mapping": dict(label_mapping),
        "package_versions": _package_versions(),
        "cuda_version": torch.version.cuda,
        "gpu_names": gpu_names,
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_parameter_count,
        "best_checkpoint_path": str(best_checkpoint_path),
        "best_validation_metric": float(best_validation_metric),
        "test_metrics": dict(test_metrics) if test_metrics is not None else None,
    }


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Write a JSON artifact with stable, human-readable formatting."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
