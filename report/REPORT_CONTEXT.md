# PAM-VLM report context

This file records the evidence boundaries and confirmed values used by
`main.tex`.

## Task and labels

PAM-VLM classifies six emotions from paired speech and face video:

`angry`, `disgust`, `fear`, `happy`, `neutral`, and `sad`.

CREMA-D contains 7,442 clips from 91 actors and provides separate voice-only,
face-only, and audio-visual vote distributions. RAVDESS is used only for
cross-corpus zero-shot evaluation after mapping to the same six labels.

## Primary protocol

- Protocol ID: `cremad-6class-clip-801010-s42-v2`.
- Deterministic emotion-stratified seed-42 split.
- Train/validation/test: 5,954/744/744 clips.
- Clips are disjoint; all 91 actors intentionally overlap.
- The score measures within-corpus clip recognition, not unseen actors.
- Checkpoint selection uses validation Macro-F1.

## Architecture

- WavLM-base-plus audio encoder.
- EfficientNet-B0 frame encoder and two-layer temporal Transformer.
- 256-dimensional bidirectional cross-attention and gated fusion.
- Separate audio, video, and fusion classification heads.
- Label-smoothed cross-entropy plus modality-matched CREMA-D vote-distribution
  KL terms.
- Selective fine-tuning of the final two WavLM layers and final two
  EfficientNet stages.
- Qwen2.5-0.5B-Instruct base weights frozen with rank-16 LoRA.
- Projector input: 256-D fusion embedding, two six-class probability vectors,
  and one disagreement scalar.
- Projector output: four 896-dimensional soft-prefix tokens.
- Inference scores only the six valid emotion tags by conditional NLL.

## Confirmed results

### Primary PAM classifier

| Model | Test Macro-F1 | UAR | Accuracy | ECE |
|---|---:|---:|---:|---:|
| Audio-only | 0.720 | 0.720 | 0.719 | 0.057 |
| Video-only | 0.702 | 0.697 | 0.696 | 0.062 |
| Frozen hard-label fusion | 0.849 | 0.845 | 0.847 | 0.038 |
| PAM high-accuracy | **0.888** | **0.886** | **0.887** | 0.090 |

### Reconciler

| Evaluation | Samples | PAM | Qwen | PAM+VLM routing |
|---|---:|---:|---:|---:|
| Mini-CREMA validation | 120 | 0.8750 | 0.8919 | **0.8924** |
| CREMA-D held-out test | 744 | **0.8870** | 0.8750 | 0.8817 |
| RAVDESS zero-shot | 1,056 | **0.5618** | 0.5004 | 0.5071 |

Table values are Macro-F1. Mini-CREMA is selection data. The complete
CREMA-D and RAVDESS evaluations show that the validation gain does not
generalize; PAM remains the strongest tested decision system.

## Reporting rules

- Keep clip-level and actor-independent results separate.
- Do not compare the six-class RAVDESS zero-shot result directly with
  supervised eight-class RAVDESS scores.
- Do not claim the Qwen branch improves held-out accuracy.
- Preserve the negative VLM result; it motivates calibration and routing work.
- Do not include local paths, CSV links, checkpoint paths, or hardware details
  in the paper body.

## Current report assets

- `pineline.png`: architecture figure.
- `figures/pam_highacc_loss_curves.png`: training and validation loss.
- `figures/pam_highacc_metric_curves.png`: validation metrics.
- `figures/pam_highacc_test_confusion_matrix_norm.png`: primary test confusion
  matrix.
- `main.pdf`: compiled IEEE report.
