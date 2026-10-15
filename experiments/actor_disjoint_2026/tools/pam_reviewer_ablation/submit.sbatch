#!/usr/bin/env bash
# Resources/partition/time set by submit.py, not guessed from historical jobs.
set -euo pipefail
test -n "${SLURM_JOB_ID:-}" || { echo 'Submit through sbatch'; exit 2; }
SUITE="$(realpath "$1")"
MODE="$2"
source "${PAM_CONDA_INIT:?Set PAM_CONDA_INIT}"
conda activate "${PAM_CONDA_ENV:-ta}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MPLBACKEND=Agg PYTHONUNBUFFERED=1
export PYTHONPATH="${SUITE}/code/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$SUITE/code"
case "$MODE" in
  smoke) python "$SUITE/run.py" --suite "$SUITE" --smoke ;;
  run) python "$SUITE/run.py" --suite "$SUITE" --index "${SLURM_ARRAY_TASK_ID:?}" ;;
  analyze) python "$SUITE/analyze.py" --suite "$SUITE" ;;
  *) echo "Invalid mode: $MODE"; exit 2 ;;
esac
