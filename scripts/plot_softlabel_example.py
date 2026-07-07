#!/usr/bin/env python3
"""Render the per-modality vote-distribution figure (fig:softlabel) for the report.

Picks one clip with high inter-condition disagreement (voice vs face vs AV) and
plots the three soft-label distributions as grouped bars, illustrating what the
soft-label objective uses and hard labels discard.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad"]
REPO = Path(__file__).resolve().parents[1]
SPLITS = REPO / "data/splits"
OUT = REPO / "report/figures/softlabel_example.png"


def main() -> int:
    voice = pd.read_parquet(SPLITS / "soft_labels_voice.parquet").set_index("clip_id")
    face = pd.read_parquet(SPLITS / "soft_labels_face.parquet").set_index("clip_id")
    av = pd.read_parquet(SPLITS / "soft_labels_av.parquet").set_index("clip_id")

    common = voice.index.intersection(face.index).intersection(av.index)
    # Rank clips by how much voice and face distributions disagree (L1 distance),
    # preferring genuine crowd disagreement (avoid one-hot fallback rows).
    best_clip, best_score = None, -1.0
    for clip in common:
        v = voice.loc[clip, LABELS].to_numpy(float)
        f = face.loc[clip, LABELS].to_numpy(float)
        if v.max() > 0.95 or f.max() > 0.95:  # skip near one-hot (fallback) clips
            continue
        score = np.abs(v - f).sum()
        if score > best_score:
            best_score, best_clip = score, clip
    if best_clip is None:
        best_clip = common[0]

    v = voice.loc[best_clip, LABELS].to_numpy(float)
    f = face.loc[best_clip, LABELS].to_numpy(float)
    a = av.loc[best_clip, LABELS].to_numpy(float)

    x = np.arange(len(LABELS))
    width = 0.27
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width, v, width, label="voice-only (cond. 1)")
    ax.bar(x, f, width, label="face-only (cond. 2)")
    ax.bar(x + width, a, width, label="audio+visual (cond. 3)")
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, rotation=20)
    ax.set_ylabel("vote probability")
    ax.set_ylim(0, 1)
    ax.set_title(f"Per-modality vote distributions — clip {best_clip}")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT} (clip={best_clip}, voice-face L1={best_score:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
