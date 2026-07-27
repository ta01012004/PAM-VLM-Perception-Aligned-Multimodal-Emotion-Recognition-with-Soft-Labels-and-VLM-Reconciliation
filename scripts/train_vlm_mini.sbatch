#!/usr/bin/env bash
#SBATCH --job-name=pam-vlm-mini
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

source /home/22011107/miniconda3/etc/profile.d/conda.sh
conda activate ta

export HF_HOME="${PWD}/artifacts/pretrained/huggingface"
export TORCH_HOME="${PWD}/artifacts/pretrained/torch"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

nvidia-smi
srun python scripts/cache_vlm_features.py \
  --config configs/vlm_mini.yaml \
  --splits train val \
  --overwrite
srun python scripts/train_vlm_reconciler.py --config configs/vlm_mini.yaml
