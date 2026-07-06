# Results — PAM-VLM on CREMA-D (actor-independent split)

All numbers below are from a strict **actor-independent** split (seed 42):
train 70 actors / 5,721 clips, val 10 / 820, test 11 / 901. Labels:
`[angry, disgust, fear, happy, neutral, sad]`. The test set was evaluated
**once**, after configuration freeze (no test-set model selection).

## Hard-label baselines

Best checkpoint selected by **validation Macro-F1**. Training: 30 epochs,
frozen WavLM-base-plus + EfficientNet-B0 encoders, bidirectional
cross-attention fusion, AdamW (lr 3e-4, cosine), effective batch 32, AMP.

### Validation (model selection)

| Baseline | Accuracy | Macro-F1 | UAR | Weighted-F1 |
|---|---:|---:|---:|---:|
| Audio-only        | 0.759 | 0.760 | 0.762 | 0.757 |
| Fusion (A+V)      | 0.731 | 0.731 | 0.732 | 0.731 |
| Video-only        | 0.504 | 0.497 | 0.503 | 0.497 |

### Test (frozen, evaluated once)

| Baseline | Accuracy | Macro-F1 | UAR | ECE |
|---|---:|---:|---:|---:|
| Audio-only        | 0.693 | 0.692 | 0.695 | 0.077 |
| Fusion (A+V)      | 0.684 | 0.685 | 0.686 | 0.094 |
| Video-only        | 0.471 | 0.462 | 0.469 | 0.457 |

**Observations.** Audio is the strongest single modality on CREMA-D, consistent
with the literature. The fusion head overfits early (val peak at epoch 2) and
does not yet exceed audio-only — the audio-visual gain is a target for the
soft-label objective and stronger regularization. Video is weakest and poorly
calibrated (high ECE).

## Layout

```
results/<baseline>/
├── val_metrics.json        # best-epoch validation metrics (+ per-class, confusion matrix)
├── test_metrics.json       # frozen test-set metrics (+ ECE, per-class, confusion matrix)
├── history.json            # full per-epoch train/val history
├── resolved_config.json    # exact resolved config for the run
├── test_predictions.csv    # per-clip predictions + class probabilities on test
├── figures_val/            # train/val curves, confusion matrices, per-class F1
└── figures_test/           # test-set confusion matrices + per-class F1
```

## Soft-label (step-05) — pending

The per-modality soft-label KL objective (label-distribution learning on the
three CREMA-D vote conditions) is implemented and training; results and
calibration comparison will be added here on completion.

## Reproduce

```bash
pip install -r requirements.txt
# prepare CREMA-D under data/raw/crema_d (AudioWAV/, VideoFlash/, finishedResponses.csv)
python scripts/build_manifest.py
python scripts/build_soft_labels.py
sbatch scripts/train_fusion_gpu.sbatch   # or train_audio_gpu / train_video_gpu
sbatch scripts/eval_test.sbatch          # frozen test-set evaluation
```

Checkpoints (~400 MB each) are not committed. Retrain from the configs above,
or request the weights.
