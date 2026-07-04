# PAM-VLM Roadmap (CREMA-D Audio-Visual Emotion Recognition)

> Đọc file này trước. Nó là bản đồ tổng. Mỗi bước có 1 file riêng trong `plan/`.
> Codex chỉ làm **một step mỗi lần**. Không nhảy bước. Không làm gộp.

## How to use this plan (for Codex)

1. Read `plan/00-IDEA.md` once to understand the whole system.
2. Read `AGENTS.md`, `ARCHITECTURE.md`, `EXPERIMENT_PROTOCOL.md` (kit rules — binding).
3. Open the **single** step file you are assigned (e.g. `plan/step-01-*.md`).
4. Copy that step's content into `CURRENT_TASK.md` before coding.
5. Implement ONLY that step's scope. Respect its "Out of scope".
6. Run the step's verification commands. Report changed files + results.
7. Do NOT start the next step until the current one passes and is reviewed.

## Hard constraints (never violate)

- **NEVER run training/eval on the login node.** Submit via `sbatch` to a
  `dgx-*` partition (A100 available). CPU smoke tests are the only thing
  allowed to run interactively.
- Actor-independent split only. See `EXPERIMENT_PROTOCOL.md`.
- Labels fixed order: `["angry","disgust","fear","happy","neutral","sad"]`.
- No dataset / checkpoint / cache committed to Git.
- No model download inside unit tests.

## Dependency graph (build order)

```text
step-01 manifest + soft-labels ─┐
                                ├─► step-03 model skeleton ─► step-04 training + baseline
step-02 dataloader + collator ──┘                                     │
                                                                       ▼
                                              step-05 soft-label (KL) training + calibration
                                                                       │
                                                                       ▼
                                              step-06 VLM reconciler (Qwen + LoRA)
                                                                       │
                                                                       ▼
                                              step-07 evaluation suite + zero-shot cross-task
```

## Step index

| Step | File | Output | Blocks |
|---|---|---|---|
| 1 | `step-01-manifest-and-soft-labels.md` | manifests + soft-label tables | 3,4 |
| 2 | `step-02-dataloader.md` | Dataset + collator + fixtures | 3,4 |
| 3 | `step-03-model-skeleton.md` | encoders + fusion + heads (shape-only) | 4 |
| 4 | `step-04-training-and-baseline.md` | train loop + hard-label baselines | 5 |
| 5 | `step-05-softlabel-training.md` | KL training + calibration (ECE) | 6 |
| 6 | `step-06-vlm-reconciler.md` | Qwen+LoRA reconciler on disagreement | 7 |
| 7 | `step-07-evaluation-and-zeroshot.md` | full metrics + zero-shot transfer | — |

## Definition of done for the whole project

- Actor-independent baselines (audio / video / fusion) reported with Macro-F1 + UAR.
- Soft-label training shows calibration and/or F1 effect vs hard-label baseline.
- VLM reconciler measured specifically on high-disagreement cases.
- One zero-shot cross-task evaluation run through the VLM branch (prompt-only).
- Every run has a full reproducibility record (see protocol §4).
