# Results — PAM-VLM on CREMA-D (actor-independent split)

All numbers below are from a strict **actor-independent** split (seed 42):
train 70 actors / 5,721 clips, val 10 / 820, test 11 / 901. Labels:
`[angry, disgust, fear, happy, neutral, sad]`. The test set was evaluated
**once**, after configuration freeze (no test-set model selection).

All runs below use **train-time augmentation** (audio: noise/gain/time-shift/
polarity; video: flip/brightness/contrast) and `head_dropout=0.3`. Encoders
(WavLM-base-plus, EfficientNet-B0) are frozen; only fusion + heads train.

## Test set (frozen, evaluated once)

| Run | Macro-F1 | UAR | Accuracy | ECE |
|---|---:|---:|---:|---:|
| Audio-only            | 0.684 | 0.683 | 0.683 | 0.059 |
| Video-only            | 0.488 | 0.498 | 0.502 | 0.384 |
| Fusion (hard label)   | 0.680 | 0.693 | 0.689 | 0.091 |
| **Fusion (soft label / distillation)** | **0.700** | **0.712** | **0.711** | **0.053** |

## Validation (model selection)

| Run | Macro-F1 | UAR | Accuracy |
|---|---:|---:|---:|
| Audio-only          | 0.767 | 0.769 | 0.767 |
| Video-only          | 0.548 | 0.552 | 0.552 |
| Fusion (hard label) | 0.770 | 0.777 | 0.774 |
| Fusion (soft label) | 0.756 | 0.767 | 0.766 |

**Observations.**
- The **soft-label distillation** model (hard CE anchor + per-modality KL on the
  three CREMA-D vote conditions) is the best on the test set across every metric
  and has the **lowest ECE (0.053)** — soft labels improve both accuracy and
  calibration, the core claim of the method.
- Pure-KL training (no CE anchor) collapsed (val Macro-F1 ~0.45); the CE anchor
  is required. See `configs/softlabel.yaml`.
- Augmentation substantially reduced train/val overfitting versus the earlier
  un-augmented runs (train F1 no longer saturates at ~1.0 as fast).
- Audio remains the strongest single modality; video is weakest and least
  calibrated.

## Layout

```
results/<run>/
├── val_metrics.json        # best-epoch validation metrics (+ per-class, confusion matrix)
├── test_metrics.json       # frozen test-set metrics (+ ECE, per-class, confusion matrix)
├── history.json            # full per-epoch train/val history
├── resolved_config.json    # exact resolved config for the run
├── test_predictions.csv    # per-clip predictions + class probabilities on test
├── figures_val/            # train/val curves, confusion matrices, per-class F1
└── figures_test/           # test-set confusion matrices + per-class F1
```

## Reproduce

```bash
pip install -r requirements.txt
# prepare CREMA-D under data/raw/crema_d (AudioWAV/, VideoFlash/, finishedResponses.csv)
python scripts/build_manifest.py
python scripts/build_soft_labels.py
sbatch scripts/train_all_augmented.sbatch   # trains fusion / audio / video / softlabel
sbatch scripts/eval_test.sbatch             # frozen test-set evaluation of the latest runs
```

Checkpoints (~400 MB each) are not committed. Retrain from the configs above,
or request the weights.
