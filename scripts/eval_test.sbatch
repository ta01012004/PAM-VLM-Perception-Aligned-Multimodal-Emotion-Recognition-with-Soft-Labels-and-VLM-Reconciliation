#!/usr/bin/env bash
#SBATCH --job-name=pam-eval-test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

# --- Conda environment ---
source /home/22011107/miniconda3/etc/profile.d/conda.sh
conda activate ta

# --- Offline model caches (populated at the gateway; no internet on compute) ---
export HF_HOME="${PWD}/artifacts/pretrained/huggingface"
export TORCH_HOME="${PWD}/artifacts/pretrained/torch"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="${PWD}/src:${PWD}/scripts:${PYTHONPATH:-}"

nvidia-smi
python -c "import torch; print('cuda available:', torch.cuda.is_available(), '| device:', torch.cuda.get_device_name(0))"

# Evaluate every completed baseline on the frozen TEST set, one run at a time.
RUNS=(
  "artifacts/checkpoints/20260705-090942_wavlm-effb0_audio_s42_actor-v1"
  "artifacts/checkpoints/20260705-053736_wavlm-effb0_xattn_s42_actor-v1"
  "artifacts/checkpoints/20260705-123801_wavlm-effb0_video_s42_actor-v1"
)
for run in "${RUNS[@]}"; do
  echo "=== TEST EVAL: ${run} ==="
  srun python scripts/eval_test.py --run-dir "${run}"
done
echo "=== all test evaluations done ==="
