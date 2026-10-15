> **Historical PAM-VLM material.** For the current actor-disjoint PAM study, see
> [the 2026 evidence package](experiments/actor_disjoint_2026/README.md).
> Actor-overlapping scores below are not current unseen-actor results.

# PAM-VLM high-accuracy experiment protocol

## Primary score-optimization track

Every new controlled comparison uses
`cremad-6class-clip-801010-s42-v2`:

- all 7,442 CREMA-D clips and the fixed six-class label mapping;
- deterministic, emotion-stratified clip split with seed 42;
- train/validation/test = 5,954/744/744 clips (80/10/10);
- actors may occur in all three partitions; clips may not overlap;
- augmentation applies only to training clips;
- checkpoint selection uses validation Macro-F1 only;
- final metrics are Macro-F1, UAR, accuracy, weighted-F1 and ECE;
- manifest checksum:
  `a53d3ba52bba1aedff4c66c8b507d92d0b2daa9c78d7e66a65c892438eab098d`.

This track intentionally optimizes the score under a clip-level protocol. It is
not actor-independent and must not be described as unseen-speaker
generalization. The existing 70/10/11 actor-independent results remain a
separate robustness track.

## Fair paper comparison

Use two tables:

1. **Published results:** copy each paper's own split, labels and metric exactly;
   this is context, not an apples-to-apples ranking.
2. **Controlled reproduction:** run every available baseline and PAM-VLM on the
   exact clip manifests above with the same preprocessing, training budget,
   checkpoint rule and evaluator. Only this table may bold the best method.

The controlled table should contain audio-only, video-only, late fusion,
cross-attention hard-label fusion, soft-label fusion, the high-accuracy model,
and the Qwen reconciler. A paper architecture enters this table only after its
code has been adapted to consume the shared manifests. AVT-CA has public code,
but its repository documents RAVDESS preprocessing, so a CREMA-D adapter and a
protocol audit are required before claiming a reproduction.

After architecture selection, rerun optimization seeds 42, 43 and 44 while
keeping the clip manifest fixed. Report mean and standard deviation. Never tune
ensemble weights, thresholds, prompts or hyperparameters on test predictions.

## High-accuracy model changes

`configs/high_accuracy.yaml` is the main candidate. Relative to the frozen
baseline it:

- fine-tunes the last two WavLM transformer layers and last two EfficientNet
  feature stages with a 20x smaller backbone learning rate;
- initializes fusion toward the stronger audio stream;
- uses modality dropout so the fusion layer remains useful when video is noisy;
- reduces video auxiliary-loss weight, adds label smoothing, and retains
  per-modality soft-label distillation;
- uses early stopping and a longer maximum schedule.

Next improvements are selected in this order: validation-fitted checkpoint
ensemble, stronger face alignment/video encoder, explicit audio-video temporal
alignment, then high-disagreement Qwen reconciliation. Each must be evaluated
as a one-factor ablation on the same manifest.

## Pooled-data track

CREMA-D and RAVDESS can be merged later after mapping to the same six labels.
That is a separate pooled supervised track and must split by clip 80/10/10 with
dataset-stratified reporting. It cannot also be called RAVDESS zero-shot: the
zero-shot track must train on CREMA-D only and evaluate the frozen model once on
RAVDESS.

## Completed VLM evaluation

The selected Mini-CREMA reconciler is fixed before complete evaluation. It
uses:

- Qwen2.5-0.5B-Instruct with frozen base weights and rank-16 LoRA;
- four learned 896-dimensional prefix tokens;
- constrained negative-log-likelihood scoring over six emotion tags;
- disagreement threshold `0.3242354617`;
- maximum PAM confidence threshold `0.6649658203`.

Final complete evaluations:

| Evaluation | Samples | PAM Macro-F1 | Qwen Macro-F1 | Routed Macro-F1 |
|---|---:|---:|---:|---:|
| CREMA-D test | 744 | **0.8870** | 0.8750 | 0.8817 |
| RAVDESS zero-shot | 1,056 | **0.5618** | 0.5004 | 0.5071 |

The Mini-CREMA validation gain does not transfer. Future routing or
calibration methods must be selected on development data and evaluated with
these test partitions unchanged.
