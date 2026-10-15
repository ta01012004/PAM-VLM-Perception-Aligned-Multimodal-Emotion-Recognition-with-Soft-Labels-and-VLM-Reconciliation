# Complete results and qualified interpretation

All scores below are fractions/raw scores. SD is the **sample SD**. No
significance tests or confidence intervals are claimed. Bold marks the best
observed mean only within a matched comparison, not a significant difference.

## Primary runs

These data splits measure different test populations and must not be ranked
against each other. The holdout row is the original one-run result, **not**
the fresh three-seed mean.

| Data split | Runs | Macro-F1 | UAR | Accuracy | ECE15 | NLL | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| 5-fold | 5 | 0.7516 ± 0.0171 | 0.7524 ± 0.0193 | 0.7520 ± 0.0173 | 0.0582 ± 0.0175 | 0.7320 ± 0.0405 | 0.3582 ± 0.0211 |
| 10-fold | 10 | 0.7549 ± 0.0428 | 0.7568 ± 0.0442 | 0.7559 ± 0.0425 | 0.0576 ± 0.0126 | 0.7122 ± 0.1022 | 0.3519 ± 0.0574 |
| 60/20/20 actor holdout | 1 | 0.7720 | 0.7628 | 0.7642 | 0.0552 | 0.6977 | 0.3454 |

The five-fold setting uses public outer test actor identities but reserves
nine outer-training actors for validation, so training membership differs
from the full published outer-training set. Four excluded clips are recorded
in the manifests/audits. Ten-fold uses newly assigned actors with split seed
42. The holdout is one fixed 55/18/18 actor assignment (split seed zero).
The development corpus had prior clip-level tuning; this is not a claim of
an entirely untouched development dataset or exact replication of other papers.

### Pooled out-of-fold summaries

Pooling changes the aggregation; it is not the arithmetic mean of fold scores.

| Data split | Unique clips | Macro-F1 | UAR | Accuracy | ECE15 | NLL | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| 5-fold | 7438 | 0.7529 | 0.7526 | 0.7522 | 0.0529 | 0.7317 | 0.3580 |
| 10-fold | 7442 | 0.7548 | 0.7557 | 0.7549 | 0.0262 | 0.7147 | 0.3532 |

## Ten-fold CE-only versus CE+KL

Both settings keep label smoothing at 0.05. The primary ten CE+KL models are
reused here; only the CE-only arm adds ten newly trained models. Mean ± SD
is across actor folds, not across optimization seeds. Training sets overlap.

| Setting | Macro-F1 | UAR | Accuracy | ECE15 | NLL | Brier |
|---|---:|---:|---:|---:|---:|---:|
| CE-only, smoothing 0.05 | 0.7569 ± 0.0426 | 0.7586 ± 0.0444 | 0.7581 ± 0.0424 | 0.0965 ± 0.0408 | 0.7447 ± 0.1272 | 0.3661 ± 0.0626 |
| PAM CE+KL | 0.7549 ± 0.0428 | 0.7568 ± 0.0442 | 0.7559 ± 0.0425 | 0.0576 ± 0.0126 | 0.7122 ± 0.1022 | 0.3519 ± 0.0574 |

CE+KL has 0.20 percentage points lower mean Macro-F1; it improves F1 on five
of ten folds. Mean ECE falls by 40.3% relative, and ECE improves on eight
folds. NLL is lower on five folds and Brier on six. Lower mean probability
scores do not imply an improvement on every fold or recognition gains.
The comparison changes both the target mixture and effective branch scales.

## Fresh eleven-variant controls

Mean ± SD across seeds 42/43/44 on **one unchanged retrospective actor split**:
55 training, 18 validation and 18 test actors; 4,496/1,470/1,476 clips. Three
seeds are not three new actor populations. Every variant is retained.

| ID | Variant | Macro-F1 | UAR | Accuracy | ECE15 | NLL | Brier |
|---|---|---:|---:|---:|---:|---:|---:|
| A0 | Full PAM CE+KL | 0.7655 ± 0.0077 | 0.7597 ± 0.0047 | 0.7588 ± 0.0065 | 0.0614 ± 0.0132 | 0.7030 ± 0.0133 | 0.3477 ± 0.0060 |
| A1 | CE-only, smoothing 0.05 | 0.7689 ± 0.0137 | 0.7636 ± 0.0124 | 0.7629 ± 0.0139 | 0.1099 ± 0.0078 | 0.7569 ± 0.0234 | 0.3670 ± 0.0143 |
| A2 | All branches use AV target | 0.7623 ± 0.0080 | 0.7595 ± 0.0059 | 0.7577 ± 0.0068 | 0.0582 ± 0.0119 | 0.7110 ± 0.0090 | 0.3513 ± 0.0006 |
| A3 | Voice/face targets swapped | 0.7626 ± 0.0144 | 0.7579 ± 0.0129 | 0.7568 ± 0.0141 | 0.0610 ± 0.0041 | 0.6970 ± 0.0117 | 0.3446 ± 0.0110 |
| A4 | Bypass cross-attention | 0.7485 ± 0.0194 | 0.7429 ± 0.0237 | 0.7425 ± 0.0214 | **0.0472** ± 0.0115 | 0.6919 ± 0.0232 | 0.3554 ± 0.0189 |
| A5 | Equal 0.5 gate | 0.7473 ± 0.0042 | 0.7399 ± 0.0025 | 0.7405 ± 0.0024 | 0.0511 ± 0.0089 | 0.7252 ± 0.0178 | 0.3657 ± 0.0062 |
| A6 | Fused losses only | 0.7537 ± 0.0013 | 0.7491 ± 0.0050 | 0.7484 ± 0.0027 | 0.0645 ± 0.0262 | 0.7418 ± 0.0038 | 0.3673 ± 0.0017 |
| A7 | wav2vec2 audio replacement | 0.6908 ± 0.0181 | 0.6829 ± 0.0183 | 0.6841 ± 0.0187 | 0.0815 ± 0.0257 | 0.9055 ± 0.0153 | 0.4503 ± 0.0157 |
| A8 | ResNet-18 video replacement | 0.7687 ± 0.0177 | 0.7667 ± 0.0178 | 0.7651 ± 0.0177 | 0.0596 ± 0.0164 | **0.6826** ± 0.0434 | **0.3344** ± 0.0201 |
| A9 | Both encoder replacements | 0.7190 ± 0.0099 | 0.7191 ± 0.0093 | 0.7188 ± 0.0088 | 0.0586 ± 0.0302 | 0.8119 ± 0.0301 | 0.3960 ± 0.0188 |
| H0 | Literal one-hot CE, smoothing 0 | **0.7757** ± 0.0064 | **0.7694** ± 0.0062 | **0.7692** ± 0.0064 | 0.1722 ± 0.0113 | 1.0747 ± 0.1116 | 0.3899 ± 0.0165 |

### What these controls do and do not establish

- A0 versus A1 lowers mean ECE by 44.1% relative and lowers mean NLL/Brier,
  while mean Macro-F1 is 0.34 percentage points lower. Both ECE and NLL are
  lower in all three paired seeds. No mean recognition gain from adding KL.
- H0 has the highest observed recognition mean, but substantially worse
  ECE/NLL. H0 vs A1 removes only smoothing; H0 vs A0 changes **both** smoothing
  and KL. The ten-fold `hard` filename is not the literal one-hot H0 arm.
- A0 has small mean F1 advantages over A2/A3 (about 0.32/0.29 points).
  A2 has lower mean ECE and A3 lower mean NLL/Brier than A0. Matching is not
  uniformly superior across seeds/metrics; generic soft-target effects and
  scale-matched alternatives remain untested.
- Bypassing attention or fixing the gate lowers mean F1 by about 1.70/1.83
  points. Attention bypass also lowers ECE/NLL and improves one seed's F1.
  This supports configuration-specific choices, not universal necessity.
- Removing auxiliary CE and KL lowers mean F1 by about 1.18 points but also
  changes objective scale. The result does not isolate auxiliary information
  from optimization weighting.
- Audio/both-encoder replacements reduce mean F1. ResNet-18 has a slightly
  higher mean F1 but mixed paired effects, and the lowest mean NLL/Brier.
  Encoders differ in capacity, pretraining and checkpoint-required
  preprocessing; these are portability checks, not controlled backbone ranking.

Still missing: scale-matched CE, uniform/shuffled-vote controls, KL-only,
replicated actor assignments for routing/components, matched external system
retraining, cross-corpus tests for this current recipe, and controlled
latency/FLOPs evidence. The historical VLM/RAVDESS experiment does not fill
those gaps for the current actor-disjoint PAM recipe.

## Source of every number

`runs.csv` links all 59 original probability CSVs and corresponding recorded
configurations. Primary and paired ten-fold original reports retain full
precision. Reviewer reports include per-seed deltas and training resource
records. `analysis/validate_release.py` recomputes all six per-run metrics,
primary fold/OOF statistics, and ten-fold comparison from exported predictions.
All results are descriptive; no SOTA claim, invented measurement, hidden weak
seed, or replaced historical result is introduced here.
