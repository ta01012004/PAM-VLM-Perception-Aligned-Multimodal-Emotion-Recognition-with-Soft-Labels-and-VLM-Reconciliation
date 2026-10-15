> **Historical PAM-VLM material.** For the current actor-disjoint PAM study, see
> [the 2026 evidence package](experiments/actor_disjoint_2026/README.md).
> Actor-overlapping scores below are not current unseen-actor results.

# Current project state

## Completed

- Fixed CREMA-D clip-level 80/10/10 protocol:
  `cremad-6class-clip-801010-s42-v2`.
- High-accuracy PAM training and held-out 744-clip CREMA-D test evaluation.
- Qwen2.5-0.5B offline adapter with rank-16 LoRA.
- Fusion-to-LLM projector producing four 896-dimensional prefix tokens.
- Constrained six-label candidate scoring and confidence/disagreement routing.
- Balanced Mini-CREMA adaptation and validation.
- Full 744-clip CREMA-D reconciler evaluation.
- Full 1,056-clip, 24-actor RAVDESS zero-shot evaluation.
- IEEE report with training curves, confusion matrix, protocol-aware literature
  comparisons, and full two-dataset results.

## Confirmed results

| Evaluation | PAM Macro-F1 | Qwen Macro-F1 | Routed Macro-F1 |
|---|---:|---:|---:|
| Mini-CREMA validation, 120 clips | 0.8750 | 0.8919 | 0.8924 |
| CREMA-D held-out test, 744 clips | **0.8870** | 0.8750 | 0.8817 |
| RAVDESS zero-shot, 1,056 clips | **0.5618** | 0.5004 | 0.5071 |

The reconciler improves its selection set but not either complete evaluation.
PAM remains the strongest tested decision system.

## Next research priorities

1. Diagnose the RAVDESS anger bias and compare per-corpus probability
   calibration.
2. Replace fixed threshold routing with a learned calibration/router model
   trained on a larger development set.
3. Run component ablations and seeds 42, 43, and 44.
4. Improve face alignment and visual temporal modeling.
5. Evaluate supervised pooled CREMA-D+RAVDESS training as a separate protocol;
   do not call that track zero-shot.

## Evidence boundaries

- Primary CREMA-D uses actor-overlapping, clip-disjoint partitions.
- Mini-CREMA is selection data, not an independent test.
- RAVDESS is six-class zero-shot transfer with no target adaptation.
- Published RAVDESS scores use different labels and protocols and are
  contextual only.
