# Step 07 — Full evaluation suite + zero-shot cross-task

## Objective
Run the frozen-config final evaluation on the actor-independent test set, and
demonstrate **prompt-based zero-shot transfer** through the VLM branch to a
different task/dataset — measuring generalization, not chasing a high number.

## In scope
- `src/av_mer/evaluation/report.py`: assemble the full result table (protocol §9):
  Macro-F1, UAR, accuracy, per-class P/R/F1, confusion matrix, ECE, params, latency.
- Per-sample prediction export (clip_id, actor_id, pred, probs) for classifier AND
  VLM branch; generative invalid-output stats.
- `src/av_mer/inference/predictor.py`: single high-level `predict(video_path)`
  (ARCHITECTURE §8) returning label, confidence, optional generated text, diagnostics.
- Zero-shot: `configs/zeroshot_<target>.yaml` + `scripts/zeroshot_eval.py` —
  run the VLM branch on a target task by **changing only the prompt/label set**,
  no weight update. Report accuracy + invalid-format rate + honest analysis of
  domain shift.
- `scripts/eval.sbatch`, `scripts/zeroshot.sbatch`.

## Out of scope
- Any retraining. Any change to the frozen classifier config.

## Input contract
- Frozen best checkpoint (from step-05 backbone, optionally step-06 VLM).
- Immutable test manifest. Target-task data for zero-shot (defined with user).

## Output contract
- Final in-domain result table (test set, run once, config frozen — protocol §8).
- Two-column comparison for transfer: fusion-head (label-locked, cannot transfer)
  vs VLM-branch (prompt-transfers) on the target task.
- Written interpretation: what transferred, what broke, why (label mismatch,
  domain shift). No cherry-picking.

## Acceptance criteria
- [ ] Test evaluation run exactly once after config freeze; checkpoint + config recorded.
- [ ] Per-sample predictions exported for both branches.
- [ ] Confusion matrix + per-class metrics saved.
- [ ] Zero-shot run through prompt-only path; invalid-output rate reported.
- [ ] `predict(video_path)` works end-to-end on one real clip (compute node).
- [ ] Honest generalization narrative written (not a number to beat).

## Constraints
- Eval/generation via `sbatch` on `dgx-*` only.
- Do NOT tune prompts on the test set or on the zero-shot target's test split.
- Confirm the zero-shot target task with the user before running (RAVDESS / IEMOCAP
  / sentiment / Vietnamese — each has different expected difficulty).

## Verification commands
```bash
python -m pytest tests/evaluation/test_report.py tests/inference/test_predictor.py -q
sbatch scripts/eval.sbatch
sbatch scripts/zeroshot.sbatch
```

## Notes and decisions
- Zero-shot cross-corpus emotion is typically 30–50%. The contribution is showing
  the VLM branch *can* transfer at all where the classifier structurally cannot.
- Frame the whole story: accurate + calibrated in-domain (steps 04–05), explainable
  + reconciling (step-06), generalizable (step-07). Three real contributions, no
  reliance on the inflated 89–95% literature numbers.
