> **Historical PAM-VLM material.** For the current actor-disjoint PAM study, see
> [the 2026 evidence package](../experiments/actor_disjoint_2026/README.md).
> Actor-overlapping scores below are not current unseen-actor results.

# PAM-VLM Results

This directory stores lightweight evidence that can be reviewed without model
checkpoints or raw media: metric JSON files, per-clip prediction CSV files,
resolved configurations, training histories, and figures.

## Primary CREMA-D clip-level benchmark

Protocol: `cremad-6class-clip-801010-s42-v2`

- Seed 42, emotion-stratified 80/10/10 split.
- Train/validation/test: 5,954/744/744 clips.
- All 91 actors may occur in every partition; clips are disjoint.
- Checkpoint selection uses validation Macro-F1.
- The test partition is not used for training or checkpoint selection.

| Model | Val. Macro-F1 | Test Macro-F1 | Test UAR | Test accuracy | ECE |
|---|---:|---:|---:|---:|---:|
| Audio-only | 0.738 | 0.720 | 0.720 | 0.719 | 0.057 |
| Video-only | 0.736 | 0.702 | 0.697 | 0.696 | 0.062 |
| Frozen hard-label fusion | 0.832 | 0.849 | 0.845 | 0.847 | **0.038** |
| **PAM high-accuracy** | **0.906** | **0.888** | **0.886** | **0.887** | 0.090 |

The high-accuracy configuration selectively fine-tunes the final two WavLM
layers and final two EfficientNet stages. It combines label-smoothed branch
cross-entropy with CREMA-D voice, face, and audio-visual vote distributions.

## Language-model reconciler

Qwen2.5-0.5B base weights are frozen. Rank-16 LoRA and a projector producing
four soft-prefix tokens are trained on balanced Mini-CREMA. Inference scores
the six valid emotion tags by conditional negative log-likelihood.

Routing uses thresholds selected only on Mini-CREMA:

- audio-video disagreement >= `0.3242354617`;
- PAM confidence <= `0.6649658203`.

### Mini-CREMA validation

| System | Samples | Macro-F1 | UAR | Accuracy |
|---|---:|---:|---:|---:|
| PAM | 120 | 0.8750 | 0.8750 | 0.8750 |
| Qwen | 120 | 0.8919 | 0.8917 | 0.8917 |
| **PAM+VLM routing** | 120 | **0.8924** | **0.8917** | **0.8917** |

This set is used for checkpoint and routing selection. It is not an independent
test result.

### CREMA-D held-out test

| System | Samples | Routed | Macro-F1 | UAR | Accuracy |
|---|---:|---:|---:|---:|---:|
| **PAM cache baseline** | 744 | 0% | **0.8870** | **0.8846** | **0.8858** |
| Qwen | 744 | 100% | 0.8750 | 0.8727 | 0.8737 |
| PAM+VLM routing | 744 | 60.62% | 0.8817 | 0.8791 | 0.8804 |

The VLM does not improve the held-out CREMA-D classifier. Safety routing
recovers part of the direct-Qwen loss but remains 0.53 Macro-F1 points below
the cached PAM baseline.

Files:

- `vlm_mini_on_cremad_test/metrics.json`
- `vlm_mini_on_cremad_test/predictions.csv`

### RAVDESS zero-shot

The frozen CREMA-D model is evaluated on all 1,056 prepared RAVDESS clips from
24 actors after mapping to the six shared emotion labels. RAVDESS labels are
not used for adaptation or routing selection.

| System | Samples | Routed | Macro-F1 | UAR | Accuracy |
|---|---:|---:|---:|---:|---:|
| **PAM** | 1,056 | 0% | **0.5618** | **0.5573** | **0.5634** |
| Qwen | 1,056 | 100% | 0.5004 | 0.5226 | 0.5312 |
| PAM+VLM routing | 1,056 | 35.42% | 0.5071 | 0.5278 | 0.5369 |

The full transfer result confirms substantial cross-corpus shift. PAM is
strongly biased toward anger on RAVDESS (186/192 angry clips are recalled, but
many happy, neutral, and sad clips are predicted as angry). Qwen and the
Mini-CREMA routing thresholds do not correct this shift.

Files:

- `zeroshot_ravdess/metrics.json`
- `zeroshot_ravdess/predictions.csv`

## Legacy actor-independent evidence

The existing `audio/`, `video/`, `fusion_hardlabel/`, and
`fusion_softlabel/` directories contain the earlier 70/10/11 actor-disjoint
experiment. Those values answer unseen-actor generalization and must remain
separate from the primary clip-level benchmark.

| Model | Test Macro-F1 | UAR | Accuracy | ECE |
|---|---:|---:|---:|---:|
| Audio-only | 0.684 | 0.683 | 0.683 | 0.059 |
| Video-only | 0.488 | 0.498 | 0.502 | 0.384 |
| Hard-label fusion | 0.680 | 0.693 | 0.689 | 0.091 |
| Soft-label fusion | **0.700** | **0.712** | **0.711** | **0.053** |

## Artifact policy

Committed:

- JSON metrics and resolved configurations;
- CSV predictions;
- learning curves and confusion matrices;
- report source and compiled PDF.

Not committed:

- raw CREMA-D or RAVDESS media;
- pretrained Hugging Face weights;
- PAM and LoRA checkpoints;
- cached feature tensors;
- SLURM logs and temporary LaTeX files.
