#!/bin/bash
#SBATCH --job-name=pam-vlm-zeroshot
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

source /home/22011107/miniconda3/etc/profile.d/conda.sh
conda activate ta
cd "/home/22011107/TA/Tiengnoi/Xử lí tiếng nói"

export HF_HOME="${PWD}/artifacts/pretrained/huggingface"
export TORCH_HOME="${PWD}/artifacts/pretrained/torch"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="${PWD}/src"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

: "${RECONCILER_CHECKPOINT:?Set RECONCILER_CHECKPOINT to the Mini-CREMA VLM best.pt}"
python scripts/zeroshot_eval.py \
  --config configs/zeroshot_ravdess.yaml \
  --reconciler-checkpoint "${RECONCILER_CHECKPOINT}"
