# PAM-VLM — Idea & Design (read once)

## One-line
Strong audio+video encoders form the **scoring backbone**; each branch is trained
to match **human per-modality perception** (from CREMA-D vote conditions); a small
LLM (Qwen2.5-0.5B + LoRA) acts as a **reasoning reconciler** on disagreement cases
and enables **prompt-based zero-shot transfer** to other tasks.

## Why this design (context, do not re-litigate)
- Pure VLA/VLM drops audio, which is the strongest emotion signal on CREMA-D → rejected.
- Published 89–95% numbers are mostly random-split / augment-before-split → not our target.
- Our honest target: be competitive on **actor-independent** split (~80–85% region),
  with better calibration + explainability + cross-task generalization.

## Unique asset we exploit
`finishedResponses.csv` stores annotator votes under three conditions:

| queryType | condition   | use |
|---|---|---|
| 1 | voice-only  | supervise audio branch (soft label) |
| 2 | face-only   | supervise video branch (soft label) |
| 3 | audio+video | supervise fusion head (soft label)  |

Most papers ignore this. We turn it into **per-modality soft labels** (label
distribution learning), which is our main novelty and our best bet for a real
Macro-F1 / calibration gain on noisy labels.

## Architecture (3 tiers)
```text
audio.wav ─► WavLM ─────► audio head  ─► p_audio(6)   ◄─ KL vs votes(voice)
video.flv ─► face crop ─► visual enc ─► video head ─► p_video(6) ◄─ KL vs votes(face)
                          ▼
              cross-attention fusion ─► fusion head ─► p_av(6)  ◄─ KL vs votes(AV)
                          │                              └─ PRIMARY METRIC (deterministic)
                          ▼ fusion tokens + p_audio + p_video + disagreement(KL)
              projector ─► Qwen2.5-0.5B + LoRA ─► "<emotion>..</emotion>" + reason
                          └─ reconciler on high-disagreement; prompt-based zero-shot
```

## Loss (each term logged, each ablatable)
```text
total = 0.6*KL(fusion‖votes_AV) + 0.2*KL(audio‖votes_voice)
      + 0.2*KL(video‖votes_face) + λ*LM_loss   (LM only when VLM branch on)
```

## Roles — say it plainly
- **Fusion head** = source of benchmark metrics. Deterministic. Always trainable path.
- **VLM branch** = explanation + disagreement reconciliation + zero-shot transfer.
  NOT expected to beat the classifier on in-domain accuracy. Optional, toggleable.

## Known risks (Codex must watch)
- Mapping votes ↔ clip: `clipName` like `1010_MTI_SAD_XX` (no extension). Verify no clip dropped.
- `.flv` decode + face crop is the #1 technical risk. Smoke-test on a few clips early.
- Zero-shot cross-corpus emotion is usually 30–50%; frame it as *generalization
  measurement*, not a number to maximize.
- Frozen vs fine-tuned encoders is a real tradeoff: deep fine-tune → higher in-domain,
  worse zero-shot. Keep encoders mostly frozen to preserve transfer.

## Hardware reality
Cluster has A100 via `dgx-*` SLURM partitions. Backbone (WavLM-base ~94M frozen +
EfficientNet-B0 ~5M + LoRA on 0.5B) fits comfortably on A100. Login node has no GPU —
all training via `sbatch`.
