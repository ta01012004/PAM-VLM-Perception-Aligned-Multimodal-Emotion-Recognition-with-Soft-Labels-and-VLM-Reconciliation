#!/usr/bin/env bash
#SBATCH --job-name=pam-audio
#SBATCH --partition=dgx-small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export HF_HOME="${PWD}/artifacts/pretrained/huggingface"
srun python scripts/train.py --config configs/baseline_audio.yaml

