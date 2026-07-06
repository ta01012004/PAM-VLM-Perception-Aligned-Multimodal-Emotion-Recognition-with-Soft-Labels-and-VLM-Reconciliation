# PAM-VLM: Perception-Aligned Multimodal Emotion Recognition

> Audio-visual emotion recognition on CREMA-D with per-modality soft labels and an optional VLM reconciler.

---

## Overview

Most published CREMA-D results (89–95%) are inflated by **random clip splits** that leak actor identity into the test set.
This project re-benchmarks the problem under a **strict actor-independent split** and introduces three contributions:

| Contribution | What it does |
|---|---|
| **Benchmark critique** | Shows that high-accuracy results correlate with random splits; establishes honest range of 70–85% |
| **Per-modality soft labels** | Exploits CREMA-D's 3 vote conditions (voice-only / face-only / AV) to build label distributions per clip per modality |
| **VLM reconciler** | Qwen2.5-0.5B + LoRA resolves audio-video conflicts and enables prompt-only zero-shot transfer |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        PAM-VLM (3 tiers)                        │
│                                                                  │
│  TIER 1 — ENCODERS (frozen)                                     │
│  ┌──────────────────────┐   ┌──────────────────────────────┐   │
│  │  WavLM-base-plus     │   │  EfficientNet-B0 (per frame) │   │
│  │  16kHz wav → 768-dim │   │  + 2-layer Transformer       │   │
│  │  → Linear proj 256   │   │  → 256-dim video tokens      │   │
│  └──────────┬───────────┘   └──────────────┬───────────────┘   │
│             │  audio tokens [B,Ta,256]      │ video tokens [B,F,256]
│             └──────────────┬───────────────┘                   │
│                            ▼                                    │
│  TIER 2 — FUSION (trainable, ~1.1M params)                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Bidirectional Cross-Attention                          │   │
│  │  audio queries video  <->  video queries audio          │   │
│  │  -> Gated pooling -> fusion embedding [B,256]           │   │
│  │  -> fusion head  -> logits [B,6]  <- PRIMARY METRIC     │   │
│  │  -> audio head   -> logits [B,6]  <- supervised voice   │   │
│  │  -> video head   -> logits [B,6]  <- supervised face    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  TIER 3 — VLM RECONCILER (optional, disabled at baseline)      │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  KL(p_audio || p_video) > threshold?                    │   │
│  │    -> Qwen2.5-0.5B-Instruct + LoRA (rank 16, ~3M)       │   │
│  │    -> "<emotion>angry</emotion> + rationale"             │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

**Training loss (soft-label mode):**

```
L = 0.6 x KL(fusion_logits || soft_av)
  + 0.2 x KL(audio_logits  || soft_voice)
  + 0.2 x KL(video_logits  || soft_face)
```

---

## Why soft labels?

CREMA-D is unique: each clip was rated by crowd workers under **3 separate conditions** — hearing only the voice, seeing only the face, or seeing+hearing together. Standard approaches collapse all votes into one hard label and throw away the rest.

PAM-VLM keeps all three distributions and supervises each encoder branch with its matching condition. This is the key novelty.

```
Clip: actor says "HAPPY" but looks sad
  voice-only votes:  [happy:0.7, neutral:0.2, sad:0.1]
  face-only votes:   [sad:0.5, neutral:0.3, happy:0.2]
  AV votes:          [happy:0.5, sad:0.3, neutral:0.2]
  hard label (usual): happy (one-hot)  ← discards all disagreement
  PAM-VLM:           uses all three distributions per branch
```

---

## Dataset Split

| Split | Clips | Actors |
|-------|-------|--------|
| Train | 5,721 | 70 |
| Val   | 820   | 10 |
| Test  | 901   | 11 |
| **Total** | **7,442** | **91** |

- 6 emotions: `angry · disgust · fear · happy · neutral · sad`
- **Actor-independent** (no actor appears in both train and test)
- Soft labels: 100% coverage, Laplace smoothing α=1

---

## Benchmark

| Method | Modality | Split | Score |
|---|---|---|---|
| Hybrid DL+FE (2025) | Audio | Random† | 95.1% |
| Transformer-CNN (2025) | Audio | Random†‡ | 90.0% |
| Spectral+Attn (2025) | Audio | Random† | 89.3% |
| Human ceiling (2014) | A+V | Human | ~63% |
| Raw waveform CNN (2023) | Audio | Indep.★ | 69.7% |
| DistilHuBERT (2025) | Audio | Indep.★ | 70.6% |
| Transfer+Emb. (2024) | Audio | Indep.★ | UAR 80.3% |
| Nature hybrid fusion (2026) | A+V | Indep.★ | ~80%+ |
| **PAM-VLM audio-only (ours)** | **Audio** | **Indep.★** | **test Macro-F1 0.692** |
| **PAM-VLM fusion (ours)** | **A+V** | **Indep.★** | **test Macro-F1 0.685** |

† random/augment-before-split suspected · ‡ <6 emotion classes · ★ actor/speaker-independent confirmed

Full validation + test tables, per-class metrics, confusion matrices, and
training curves are in [`results/`](results/README.md). Soft-label (step-05)
results pending.

Methods marked † exceed human agreement ceiling (~63% for AV condition) — physically impossible without data leakage.

---

## Quickstart

### 1. Install
```bash
pip install -r requirements.txt
```

### 2. Prepare CREMA-D
```
data/raw/crema_d/
├── AudioWAV/            # 7442 .wav files
├── VideoFlash/          # 7442 .flv files
└── finishedResponses.csv
```

### 3. Build splits + soft labels
```bash
python scripts/build_manifest.py
python scripts/build_soft_labels.py
```

### 4. Train on SLURM cluster
```bash
sbatch scripts/train_fusion_gpu.sbatch
```

### 5. Train on Kaggle (audio-only)
Use `kaggle_train.py` with dataset `ejlok1/cremad` — no upload needed.

---

## Project Structure

```
.
├── src/av_mer/
│   ├── data/           # Dataset, splits, audio/video I/O, soft labels
│   ├── models/         # AudioEncoder, VideoEncoder, BidirectionalCrossAttn, PAMFusion
│   ├── training/       # Train loop, losses (CE + KL), checkpointing
│   └── evaluation/     # Macro-F1, UAR, ECE, confusion matrix
├── configs/
│   ├── baseline_audio.yaml
│   ├── baseline_video.yaml
│   └── baseline_fusion.yaml
├── scripts/
│   ├── train.py
│   └── train_fusion_gpu.sbatch
├── report/             # LaTeX report (IEEEtran 2-column)
├── plan/               # Step-by-step implementation plan
├── kaggle_train.py     # Self-contained Kaggle notebook (audio-only baseline)
└── requirements.txt
```

---

## Model Parameters

| Component | Params | Trainable |
|---|---|---|
| WavLM-base-plus | ~94M | Frozen |
| EfficientNet-B0 | ~5.3M | Frozen |
| Fusion + heads | ~1.1M | Yes |
| Qwen2.5-0.5B + LoRA (optional) | ~500M + ~3M | LoRA only |

---

## Expected Results

| Run | Macro-F1 target |
|---|---|
| Audio-only | 65–72% |
| Video-only | 55–65% |
| Fusion (hard label) | 75–82% |
| Fusion (soft label) | 77–84% |
| + VLM reconciler | ± on conflict cases |

---

## Training Config

```yaml
epochs: 30
batch_size: 8              # effective 32 (grad_accum=4)
optimizer: AdamW lr=3e-4
scheduler: CosineAnnealingLR
amp: true
early_stopping_patience: 6
primary_metric: macro_f1
audio_encoder: microsoft/wavlm-base-plus  (FROZEN)
video_encoder: efficientnet_b0            (FROZEN)
hardware: NVIDIA A100 (SLURM, partition=gpu)
```
