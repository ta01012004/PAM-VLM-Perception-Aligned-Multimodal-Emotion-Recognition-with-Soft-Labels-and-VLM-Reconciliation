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

# Evaluate the latest completed run of each type on the frozen TEST set, so the
# augmented retrains are picked automatically (newest run dir per pattern).
RUNS=()
add_latest() {  # $1 = name substring; xattn must not match the softlabel runs
  for d in $(ls -td artifacts/checkpoints/*"$1"*/ 2>/dev/null); do
    if [ "$1" = "xattn" ]; then case "$d" in *softlabel*) continue;; esac; fi
    if [ -f "${d%/}/checkpoints/best.pt" ]; then RUNS+=("${d%/}"); return; fi
  done
}
add_latest audio
add_latest xattn
add_latest video
add_latest softlabel
add_latest highacc

for run in "${RUNS[@]}"; do
  echo "=== TEST EVAL: ${run} ==="
  srun python scripts/eval_test.py --run-dir "${run}"
done
echo "=== all test evaluations done ==="
