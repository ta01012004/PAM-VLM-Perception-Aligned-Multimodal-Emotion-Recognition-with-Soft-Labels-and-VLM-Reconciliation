# Reproduction levels and prerequisites

## 1. Audit the released test evidence (CPU, self-contained)

From the repository root:

```bash
python -m venv .venv
# Activate the environment using the command appropriate to your OS.
pip install -r experiments/actor_disjoint_2026/requirements-analysis.txt
python experiments/actor_disjoint_2026/analysis/validate_release.py
python -m pytest experiments/actor_disjoint_2026/analysis -q
```

Expected: PASS; 16 primary, ten CE-only, 33 reviewer runs; 7,438 five-fold and
7,442 ten-fold pooled clips, and 1,476 unique test clips per reviewer run.
All six metrics are recalculated from probabilities. Macro-F1/UAR average
six classes; ECE has 15 equal-width bins and includes confidence=1 in the last
bin; NLL uses natural logarithms and a 1e-12 floor; Brier sums across classes.
Fold means/sample SD, pooled metrics, and seed means/sample SD are different
summaries. The exported-results check requires neither torch nor SLURM.

## 2. Inspect exact training and model configurations

Use `runs.csv` to locate each recorded `resolved_config.json`. The original
model and loader configurations are in `evidence/primary/inputs/`; all 34
reviewer model definitions are in `evidence/reviewer/models/`.
`reproducibility.json` files record actual package versions, CUDA, source commit,
and parameter counts where collected. They are a record of the run, **not a
claim that root requirements.txt reproduces that environment**. The primary
record uses torch 2.5.0+cu121, torchvision 0.20.0+cu121, transformers 4.57.6,
numpy 1.26.4 and pandas 2.1.4. Consult each block's records for differences.

The primary/control source is isolated at
`tools/pam_hard_ablation/code_bundle/`. Reviewer switches and the deployed
source are in `tools/pam_reviewer_ablation/` and its `code_bundle/`.
Legacy `PamVlmModel` filenames remain in the frozen source; the current
`baseline=fusion` training does not train/use Qwen reconciliation.

## 3. Rerun training (external data/weights, GPU allocation required)

This is not a one-command dataset-free training release. Supply separately:

1. CREMA-D speech/video and its perception annotations from the original
   dataset, respecting its license. Do not substitute model-generated votes.
2. The three modality-specific normalized vote tables required by the frozen
   loader (`soft_labels_voice/face/av.parquet`). They are not redistributed.
   Check the original input hashes if recreating an immutable suite. No vote
   table generator is newly asserted or fabricated by this release.
3. Offline pretrained WavLM, EfficientNet, and (for A7/A8/A9) wav2vec2/ResNet
   caches. The reviewer model JSONs retain snapshot commits and weight filenames.
4. A matching training environment and a scheduler allocation appropriate to
   your cluster. Do not copy the old partition/time/GPU defaults without checking
   local permissions and quota. Never train on an HPC login node.

Copy a run's resolved JSON to a **new local working directory**. JSON is valid
YAML for the frozen train.py loader. Bind the symbolic path roots to your own
absolute directories, point its manifests at the released primary manifests,
set the model/loader/label-map paths and three vote-table paths, and assign a
new output directory. Also localize the model JSON's checkpoint/cache paths.
For reviewer runs, localize `canonical_model_config` to the matching canonical
A0 definition; preserve variant switches, seed and loss coefficients.
Do not change the archived evidence files or reuse trained A0 weights to
initialize ablated branches. Use paired **untrained** initialization per seed.

Inside a permitted GPU allocation, the frozen entry points are:

```bash
# Choose the original bundle for primary/CE controls or reviewer bundle for ablations.
BUNDLE=/absolute/path/to/the/chosen/code_bundle
python "$BUNDLE/scripts/train.py" --config /absolute/path/to/new-local-config.json
python "$BUNDLE/scripts/eval_test.py" --run-dir /absolute/path/to/new-output/run
```

The supplied `prepare.py`, `run.py`, `submit.py` and sbatch files document the
original fail-fast suite preparation, seed/variant pairing and sequential
scheduler behavior. Their freeze checks expect an unmodified immutable
baseline and original-byte hash map. **Path-redacted evidence is not itself
that runnable baseline.** Restore/regenerate your baseline inputs and a new
freeze map after local path binding before using those orchestration tools;
do not disable hash checks to force the archived private-path suite to run.
The direct entry points above are for a new run after local preparation, not
bitwise reproduction certified by the CPU audit.

## 4. Additional tests (requires training dependencies)

```bash
python -m unittest discover -s experiments/actor_disjoint_2026/tools/pam_reviewer_ablation -p 'test_*.py' -v
python -m unittest discover -s experiments/actor_disjoint_2026/tools/pam_hard_ablation -p 'test_*.py' -v
```

These tests use small fixtures/mock schedulers; they do not launch real jobs.
Literal one-hot CE is H0 (smoothing zero). A1 and ten-fold `hard` outputs retain
smoothing 0.05. H0 vs A1 isolates smoothing; H0 vs A0 changes both smoothing
and KL. No missing external input or saved-weight reload should be inferred
from a passing metric audit.
