#!/usr/bin/env bash
#SBATCH --job-name=pam-vlm-mini-eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

source /home/22011107/miniconda3/etc/profile.d/conda.sh
conda activate ta

export HF_HOME="${PWD}/artifacts/pretrained/huggingface"
export TORCH_HOME="${PWD}/artifacts/pretrained/torch"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="${PWD}/src:${PWD}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

: "${VLM_CHECKPOINT:?Set VLM_CHECKPOINT to a trained Mini-CREMA best.pt}"
srun python scripts/eval_vlm_reconciler.py \
  --config configs/vlm_mini.yaml \
  --checkpoint "${VLM_CHECKPOINT}" \
  --split val \
  --output-dir results/vlm_mini_validation
