#!/usr/bin/env python3
"""Render publication-quality figures from a PAM-VLM training run.

Reads ``history.json`` (+ ``metrics.json``) written by ``scripts/train.py`` and
produces separate PNG figures under ``<run_dir>/figures/``:

  - loss_curves.png          train vs. val total loss per epoch
  - metric_curves.png        macro-F1 / UAR / accuracy / weighted-F1 per epoch
  - loss_components.png       per-branch CE components (train & val)
  - per_class_f1.png          best-epoch per-class precision/recall/F1 bars
  - confusion_matrix.png      best-epoch validation confusion matrix (counts)
  - confusion_matrix_norm.png row-normalised confusion matrix (recall view)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LABELS = ("angry", "disgust", "fear", "happy", "neutral", "sad")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Run directory containing history.json.")
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def _load(run_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    history = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))["epochs"]
    if not history:
        raise ValueError(f"No epochs recorded in {run_dir / 'history.json'}.")
    metrics_path = run_dir / "metrics.json"
    best = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    return history, best


def _save(fig: plt.Figure, path: Path, dpi: int) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def plot_loss_curves(history: list[dict[str, Any]], out: Path, dpi: int) -> None:
    epochs = [record["epoch"] for record in history]
    train = [record["train_loss"] for record in history]
    val = [record["val_loss"] for record in history]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, train, marker="o", ms=3, label="train")
    ax.plot(epochs, val, marker="s", ms=3, label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("total loss")
    ax.set_title("Training vs. validation loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save(fig, out, dpi)


def plot_metric_curves(history: list[dict[str, Any]], out: Path, dpi: int) -> None:
    epochs = [record["epoch"] for record in history]
    keys = ("macro_f1", "uar", "accuracy", "weighted_f1")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for key in keys:
        values = [record["val_metrics"][key] for record in history]
        ax.plot(epochs, values, marker="o", ms=3, label=key)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation score")
    ax.set_title("Validation metrics per epoch")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save(fig, out, dpi)


def plot_loss_components(history: list[dict[str, Any]], out: Path, dpi: int) -> None:
    epochs = [record["epoch"] for record in history]
    component_keys = sorted(
        key for key in history[0]["train_loss_components"] if key != "total"
    )
    fig, (ax_train, ax_val) = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for key in component_keys:
        ax_train.plot(
            epochs,
            [record["train_loss_components"].get(key, float("nan")) for record in history],
            marker="o", ms=3, label=key,
        )
        ax_val.plot(
            epochs,
            [record["val_loss_components"].get(key, float("nan")) for record in history],
            marker="s", ms=3, label=key,
        )
    ax_train.set_title("Train loss components")
    ax_val.set_title("Val loss components")
    for ax in (ax_train, ax_val):
        ax.set_xlabel("epoch")
        ax.grid(True, alpha=0.3)
        ax.legend()
    ax_train.set_ylabel("cross-entropy")
    _save(fig, out, dpi)


def _best_record(history: list[dict[str, Any]], metric: str = "macro_f1") -> dict[str, Any]:
    return max(history, key=lambda record: record["val_metrics"][metric])


def plot_per_class_f1(record: dict[str, Any], out: Path, dpi: int) -> None:
    per_class = record["val_metrics"]["per_class"]
    labels = list(per_class.keys())
    precision = [per_class[name]["precision"] for name in labels]
    recall = [per_class[name]["recall"] for name in labels]
    f1 = [per_class[name]["f1"] for name in labels]
    x = np.arange(len(labels))
    width = 0.27
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - width, precision, width, label="precision")
    ax.bar(x, recall, width, label="recall")
    ax.bar(x + width, f1, width, label="f1")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20)
    ax.set_ylabel("score")
    ax.set_ylim(0, 1)
    ax.set_title(f"Per-class metrics (best epoch {record['epoch']})")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    _save(fig, out, dpi)


def _plot_confusion(matrix: np.ndarray, labels: list[str], title: str, out: Path,
                    dpi: int, normalise: bool) -> None:
    display = matrix.astype(float)
    if normalise:
        row_sums = display.sum(axis=1, keepdims=True)
        display = np.divide(display, row_sums, out=np.zeros_like(display), where=row_sums != 0)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    image = ax.imshow(display, cmap="Blues", vmin=0, vmax=display.max() if display.max() else 1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    threshold = display.max() / 2.0 if display.max() else 0.5
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = display[i, j]
            text = format(value, ".2f") if normalise else format(int(round(value)), "d")
            ax.text(j, i, text, ha="center", va="center",
                    color="white" if value > threshold else "black", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    _save(fig, out, dpi)


def plot_confusion_matrices(record: dict[str, Any], out_dir: Path, dpi: int) -> None:
    matrix = np.asarray(record["val_metrics"]["confusion_matrix"], dtype=int)
    labels = list(record["val_metrics"]["per_class"].keys()) or list(LABELS)
    _plot_confusion(matrix, labels, f"Confusion matrix (best epoch {record['epoch']})",
                    out_dir / "confusion_matrix.png", dpi, normalise=False)
    _plot_confusion(matrix, labels, f"Row-normalised confusion (best epoch {record['epoch']})",
                    out_dir / "confusion_matrix_norm.png", dpi, normalise=True)


def main() -> int:
    args = _parse_args()
    run_dir = args.run_dir.resolve()
    history, _ = _load(run_dir)
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    plot_loss_curves(history, figures_dir / "loss_curves.png", args.dpi)
    plot_metric_curves(history, figures_dir / "metric_curves.png", args.dpi)
    plot_loss_components(history, figures_dir / "loss_components.png", args.dpi)
    best = _best_record(history)
    plot_per_class_f1(best, figures_dir / "per_class_f1.png", args.dpi)
    plot_confusion_matrices(best, figures_dir, args.dpi)
    print(f"figures written to {figures_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
