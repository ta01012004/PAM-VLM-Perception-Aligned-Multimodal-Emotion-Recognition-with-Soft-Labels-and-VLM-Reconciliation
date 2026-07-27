#!/usr/bin/env bash
#SBATCH --job-name=pam-aug-all
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out

set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"

# --- Conda environment ---
source /home/22011107/miniconda3/etc/profile.d/conda.sh
conda activate ta

# --- Offline model caches (populated at the gateway; no internet on compute) ---
export HF_HOME="${PWD}/artifacts/pretrained/huggingface"
export TORCH_HOME="${PWD}/artifacts/pretrained/torch"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"

nvidia-smi
python -c "import torch; print('cuda available:', torch.cuda.is_available(), '| device:', torch.cuda.get_device_name(0))"

# Retrain all baselines + soft-label with augmentation ON and head_dropout=0.3.
# Sequential in one allocation to fit the gpu-partition QOS (1 running + 1 queued).
# Each run writes its own timestamped run dir; a failure in one does not abort the rest.
CONFIGS=(
  "configs/baseline_fusion.yaml"
  "configs/baseline_audio.yaml"
  "configs/baseline_video.yaml"
  "configs/softlabel.yaml"
  "configs/high_accuracy.yaml"
)
for cfg in "${CONFIGS[@]}"; do
  echo "=== TRAIN: ${cfg} ==="
  srun python scripts/train.py --config "${cfg}" || echo "!!! run failed: ${cfg} (continuing)"
done
echo "=== all augmented training runs done ==="
