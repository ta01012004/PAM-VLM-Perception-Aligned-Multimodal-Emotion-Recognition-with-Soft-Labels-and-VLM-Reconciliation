# Step 04 — Training loop + hard-label baselines

## Objective
Build the training/checkpoint/eval scaffolding and produce the three
**hard-label baselines** (audio-only, video-only, fusion) on the actor-independent
split. These are the honest reference numbers everything else is compared to.

## In scope
- `src/av_mer/training/loop.py`: train/val loop, mixed precision (cfg), gradient
  accumulation (report effective batch), early stopping on primary metric.
- `src/av_mer/training/checkpoint.py`: save/resume model+optim+sched+scaler+epoch+
  step+RNG; save label mapping + resolved config + split checksum.
- `src/av_mer/training/losses.py`: cross-entropy for hard-label baseline; scaffold
  the weighted multi-term loss (soft-label terms behind flags, default off).
- `src/av_mer/evaluation/metrics.py`: Macro-F1, UAR(=macro recall), accuracy,
  per-class P/R/F1, confusion matrix.
- `scripts/train.py` + `configs/baseline_{audio,video,fusion}.yaml`.
- `scripts/*.sbatch` templates targeting a `dgx-*` partition.
- Reproducibility record writer (protocol §4).

## Out of scope
- Soft-label KL training (step-05). LLM (step-06). Zero-shot (step-07).

## Input contract
- Dataloader (step-02), model (step-03), configs.

## Output contract
- Three trained baselines with saved checkpoints + `metrics.json` on val (test frozen).
- Each run writes a full reproducibility record + result-table row (protocol §9).

## Acceptance criteria
- [ ] MANDATORY smoke test (protocol §7) passes on tiny fixture: 1 train + 1 val
      step, finite loss, logits [B,6], checkpoint save+reload, 1 eval batch.
- [ ] Seeds set (py/numpy/torch); resume restores full state (tested).
- [ ] Fail-fast on NaN/Inf loss + invalid labels (tested).
- [ ] Metrics unit-tested against known confusion matrix.
- [ ] Baselines trained via sbatch on dgx partition; val metrics reported.
- [ ] Test set NOT touched during selection (protocol §8).

## Constraints
- **Training runs via `sbatch` on `dgx-*` ONLY. Never on login node.**
- GPU: A100 available. Use AMP; report effective batch size.
- Do not change split/label/metric during tuning.

## Verification commands
```bash
python -m pytest tests/training/test_smoke.py tests/training/test_resume.py \
                 tests/evaluation/test_metrics.py -q
sbatch scripts/train_fusion.sbatch     # real training on compute node
```

## Notes and decisions
- Keep the loss module extensible: step-05 flips soft-label terms on via config,
  no rewrite. Hard-label CE is just the special case.
- Baseline numbers define "did the fancy stuff actually help?" — treat as sacred.
