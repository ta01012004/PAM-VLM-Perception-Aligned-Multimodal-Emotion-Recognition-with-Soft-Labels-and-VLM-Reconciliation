# Step 05 — Soft-label (KL) training + calibration

## Objective
Switch supervision from hard labels to **per-modality vote distributions** (label
distribution learning) and measure the effect on Macro-F1, UAR, and calibration
(ECE) versus the step-04 hard-label baselines. This is the main novelty bet.

## In scope
- Enable the weighted KL loss in `training/losses.py`:
  `0.6*KL(fusion‖votes_av) + 0.2*KL(audio‖votes_voice) + 0.2*KL(video‖votes_face)`.
  Each term individually logged; each behind an ablation flag.
- `src/av_mer/evaluation/calibration.py`: Expected Calibration Error (ECE) +
  reliability diagram data.
- `configs/softlabel.yaml`; extend result table with ECE column.
- Ablations: (a) hard vs soft, (b) drop each auxiliary term, (c) weight sensitivity.
- Unit tests: KL loss correctness on known distributions; ECE on synthetic probs.

## Out of scope
- LLM branch (step-06). Any change to split/metric/data.

## Input contract
- Soft-label tables (step-01), model (step-03), training scaffold (step-04).

## Output contract
- Soft-label run(s) with val Macro-F1/UAR/ECE, compared head-to-head with baseline
  fusion under identical split/preprocessing/budget (protocol §6).
- Ablation table isolating each loss term's contribution.

## Acceptance criteria
- [ ] KL loss unit-tested (0 when pred==target dist; positive otherwise).
- [ ] ECE unit-tested on synthetic calibrated/miscalibrated cases.
- [ ] Each loss term appears separately in logs.
- [ ] Soft vs hard comparison holds every other factor fixed (stated in table).
- [ ] Multi-seed mean±std reported (not a single lucky seed).
- [ ] Test set still frozen.

## Constraints
- Training via `sbatch` on `dgx-*` only.
- If soft labels do NOT improve F1, KEEP the result and report honestly — a
  calibration-only gain is still a valid contribution. Do not tweak the eval to
  make it look better.

## Verification commands
```bash
python -m pytest tests/training/test_kl_loss.py tests/evaluation/test_calibration.py -q
sbatch scripts/train_softlabel.sbatch
```

## Notes and decisions
- Soft labels are our best shot at a *real* metric gain; VLM (step-06) is not.
- Report both: does it help accuracy? does it help calibration? Either is publishable.
- Do not silently renormalize away the fallback one-hot clips; account for them.
