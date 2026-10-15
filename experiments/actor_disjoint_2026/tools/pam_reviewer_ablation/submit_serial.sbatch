#!/usr/bin/env bash
# One allocated GPU; no arrays, no nested sbatch, no resume, fail on any error.
set -euo pipefail
test -n "${SLURM_JOB_ID:-}" || { echo 'Submit through sbatch'; exit 2; }
SUITE="$(realpath "$1")"
source "${PAM_CONDA_INIT:?Set PAM_CONDA_INIT}"
conda activate "${PAM_CONDA_ENV:-ta}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MPLBACKEND=Agg PYTHONUNBUFFERED=1
export PYTHONPATH="${SUITE}/code/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$SUITE/code"
python "$SUITE/run.py" --suite "$SUITE" --smoke
TOTAL="$(python -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["runs"]))' "$SUITE/suite.json")"
for ((INDEX=0; INDEX<TOTAL; INDEX++)); do
    echo "Starting reviewer run $((INDEX+1))/$TOTAL in job $SLURM_JOB_ID"
    python "$SUITE/run.py" --suite "$SUITE" --index "$INDEX"
done
# CPU analysis uses this allocation; no extra GPU or CPU job is submitted.
python "$SUITE/analyze.py" --suite "$SUITE"
