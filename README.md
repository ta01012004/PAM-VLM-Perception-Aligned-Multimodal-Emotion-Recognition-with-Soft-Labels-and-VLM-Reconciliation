# PAM: Perception-Aligned Audio–Visual Emotion Recognition

PAM combines WavLM speech features, EfficientNet-B0 facial features,
bidirectional cross-attention, learned gated fusion, and modality-matched
human emotion-rating distributions from CREMA-D. Cross-entropy supervises
intended labels; perceptual KL supervises corresponding audio, video, and
fused branches during training. Inference uses the fused six-class argmax.
**The current actor-disjoint study does not use the language-model reconciler.**

## Current evidence: unseen actors

The [2026 actor-disjoint package](experiments/actor_disjoint_2026/README.md)
contains predictions, recorded configurations, exact actor assignments,
lightweight training histories, code snapshots, and CPU validation for
**59 training runs**:

| Block | Runs | What varies |
|---|---:|---|
| Primary PAM | 16 | Five outer test folds, ten actor-disjoint folds, one fixed actor holdout |
| Ten-fold CE-only control | 10 | KL removed; label smoothing remains 0.05 |
| Fresh component/target controls | 33 | Eleven variants × training seeds 42, 43, 44 on one fixed actor assignment |

The ten primary CE+KL folds are reused as the comparison arm of the ten-fold
control; they are **not ten additional training runs**. Fold SD and seed SD
are different summaries. The three primary split settings are not a single
leaderboard, and the three-seed holdout does not replace cross-validation.

### Recognition and probability scores

| Ten-fold setting | Macro-F1, mean ± fold SD | ECE15, mean ± fold SD |
|---|---:|---:|
| CE-only, smoothing 0.05 | **0.7569 ± 0.0426** | 0.0965 ± 0.0408 |
| PAM CE+KL | 0.7549 ± 0.0428 | **0.0576 ± 0.0126** |

KL lowers mean ECE by 40.3% relative in this setting, while mean Macro-F1 is
slightly lower. Mean NLL and Brier are also lower. This is a descriptive
recognition–calibration tradeoff, **not a recognition gain or a significance claim**.

All eleven fresh variants, including literal one-hot CE, are included.
One-hot CE has the highest mean recognition on the fixed holdout but worse
probability scores. Routing effects are small/mixed; encoder substitutions
test operational portability, not a controlled ranking of backbones.
See the [complete results and limitations](experiments/actor_disjoint_2026/RESULTS.md).

## Verify the exported results on CPU

```bash
pip install -r experiments/actor_disjoint_2026/requirements-analysis.txt
python experiments/actor_disjoint_2026/analysis/validate_release.py
python -m pytest experiments/actor_disjoint_2026/analysis -q
```

The validator recomputes Macro-F1, UAR, accuracy, ECE15, NLL and Brier from all
59 test prediction files, checks actor partitions and paired clip identities,
and checks primary fold/OOF and ten-fold control summaries. No GPU, raw media,
trained checkpoint, or HPC account is needed for this exported-results audit.

Training material and its prerequisites are documented in
[REPRODUCE.md](experiments/actor_disjoint_2026/REPRODUCE.md).
The source snapshots are isolated so current evidence does not silently use
legacy code. Weights and raw CREMA-D data must be obtained separately.

## Historical PAM-VLM work

The repository name and root `src/`, `configs/`, `scripts/`, `tests/`,
`results/` (outside the package above), and `report/` predate this update.
They contain the **actor-overlapping clip-split PAM-VLM / Qwen experiments**,
including their negative held-out reconciliation results. They are retained
for traceability, not presented as current unseen-actor results.
The [archived README](docs/legacy_pam_vlm_README.md) records that older study;
its references to “primary paper results” apply only to that historical version.

The current journal manuscript, submission letters, author portraits, raw
media/human-rating dumps, credentials, checkpoints, and private HPC access
configuration are not part of this release. The historical `report/` is not
the current submission. The GitHub repository was not renamed.
