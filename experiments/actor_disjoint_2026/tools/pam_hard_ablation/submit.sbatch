#!/usr/bin/env bash
#SBATCH --job-name=pam-hard-m10
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=6-23:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
test -n "${1:-}" || { echo 'Usage: sbatch submit.sbatch NEW_SUITE'; exit 2; }
SUITE="$(realpath "$1")"
: "${PAM_CONDA_INIT:?Set PAM_CONDA_INIT to your conda.sh path}"
source "$PAM_CONDA_INIT"
conda activate "${PAM_CONDA_ENV:-ta}"
REPO="${PAM_REPO:-${SLURM_SUBMIT_DIR}}"
export HF_HOME="${REPO}/artifacts/pretrained/huggingface"
export TORCH_HOME="${REPO}/artifacts/pretrained/torch"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MPLBACKEND=Agg PYTHONUNBUFFERED=1
export PYTHONPATH="${SUITE}/code/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$SUITE/code"
python "$SUITE/run.py" --suite "$SUITE"
