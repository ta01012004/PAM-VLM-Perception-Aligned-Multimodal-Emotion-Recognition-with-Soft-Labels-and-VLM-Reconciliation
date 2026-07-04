#!/usr/bin/env bash
#SBATCH --job-name=cremad-decode-smoke
#SBATCH --partition=dgx-tiny
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
python scripts/smoke_decode.py --raw data/raw/crema_d --config configs/data/loader.json
