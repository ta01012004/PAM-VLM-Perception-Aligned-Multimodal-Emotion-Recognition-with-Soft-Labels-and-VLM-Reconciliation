#!/usr/bin/env python3
"""Render the PAM-VLM architecture diagram (fig:arch) for the report.

Pure matplotlib boxes/arrows (no external draw tools). Three tiers: frozen
encoders (gray), trainable fusion + heads, and the optional VLM branch (dashed).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "report/figures/architecture.png"

GRAY = "#d9d9d9"
BLUE = "#cfe2f3"
GREEN = "#d9ead3"
ORANGE = "#fce5cd"


def box(ax, x, y, w, h, text, color, *, dashed=False, fontsize=8):
    style = "round,pad=0.02,rounding_size=0.03"
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle=style, linewidth=1.2,
        edgecolor="#555555", facecolor=color,
        linestyle="--" if dashed else "-",
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, wrap=True)


def arrow(ax, x1, y1, x2, y2, *, dashed=False):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11,
        linewidth=1.1, color="#555555", linestyle="--" if dashed else "-",
    ))


def main() -> int:
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Tier 1 — inputs + frozen encoders (gray).
    box(ax, 0.2, 8.3, 2.1, 0.9, "audio.wav\n16 kHz mono", "white")
    box(ax, 0.2, 6.7, 2.1, 1.1, "WavLM-base-plus\n(frozen, 94M)", GRAY)
    box(ax, 7.5, 8.3, 2.3, 0.9, "video.flv\n16 frames, face crop", "white")
    box(ax, 7.5, 6.5, 2.3, 1.3, "EfficientNet-B0 (frozen)\n+ temporal Transformer", GRAY)
    arrow(ax, 1.25, 8.3, 1.25, 7.8)
    arrow(ax, 8.65, 8.3, 8.65, 7.8)

    # Projectors -> tokens.
    box(ax, 0.4, 5.4, 1.7, 0.8, "proj → 256", BLUE)
    box(ax, 7.9, 5.4, 1.7, 0.8, "proj → 256", BLUE)
    arrow(ax, 1.25, 6.7, 1.25, 6.2)
    arrow(ax, 8.65, 6.5, 8.65, 6.2)

    # Tier 2 — fusion (trainable, green).
    box(ax, 2.7, 4.0, 4.6, 1.1,
        "Bidirectional cross-attention\n+ gated pooling", GREEN, fontsize=8)
    arrow(ax, 1.25, 5.4, 3.3, 5.1)
    arrow(ax, 8.65, 5.4, 6.7, 5.1)

    # Heads.
    box(ax, 0.5, 2.4, 2.0, 0.9, "audio head\n→ p_audio(6)", GREEN)
    box(ax, 4.0, 2.4, 2.0, 0.9, "fusion head\n→ p_av(6)", GREEN)
    box(ax, 7.5, 2.4, 2.0, 0.9, "video head\n→ p_video(6)", GREEN)
    arrow(ax, 3.0, 4.0, 1.5, 3.3)
    arrow(ax, 5.0, 4.0, 5.0, 3.3)
    arrow(ax, 7.0, 4.0, 8.5, 3.3)

    # Supervision annotations.
    ax.text(1.5, 2.1, "KL / CE vs votes(voice)", ha="center", va="top", fontsize=6.5, color="#666")
    ax.text(5.0, 2.1, "PRIMARY metric\nKL / CE vs votes(AV)", ha="center", va="top",
            fontsize=6.5, color="#666")
    ax.text(8.5, 2.1, "KL / CE vs votes(face)", ha="center", va="top", fontsize=6.5, color="#666")

    # Tier 3 — VLM branch (optional, dashed, orange).
    box(ax, 3.0, 0.3, 4.0, 1.0,
        "projector → Qwen2.5-0.5B + LoRA\n→ <emotion>..</emotion> + reason",
        ORANGE, dashed=True, fontsize=7.5)
    arrow(ax, 5.0, 2.4, 5.0, 1.3, dashed=True)
    ax.text(7.1, 0.8, "optional\n(reconciler /\nzero-shot)", ha="left", va="center",
            fontsize=6.5, color="#999")

    # Tier labels on the left.
    for ty, tlabel in [(7.0, "Tier 1\nencoders\n(frozen)"),
                       (4.2, "Tier 2\nfusion\n(trainable)"),
                       (0.6, "Tier 3\nVLM (opt.)")]:
        ax.text(-0.15, ty, tlabel, ha="right", va="center", fontsize=6.5,
                color="#333", fontweight="bold")

    ax.set_xlim(-1.3, 10)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
