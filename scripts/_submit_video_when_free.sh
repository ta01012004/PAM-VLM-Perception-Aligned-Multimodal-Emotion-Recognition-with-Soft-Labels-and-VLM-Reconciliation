#!/usr/bin/env bash
# Submit the video baseline once a QOS slot frees up (gpu QOS dgx-large: MaxSubmit=2).
# Chains after the audio job (76419) via afterany. Exits 0 once submitted.
set -uo pipefail
cd "$(dirname "$0")/.."
MARKER="logs/.video_submitted"
[ -f "$MARKER" ] && { echo "video already submitted: $(cat "$MARKER")"; exit 0; }
OUT=$(sbatch --dependency=afterany:76419 --parsable scripts/train_video_gpu.sbatch 2>&1)
if [[ "$OUT" =~ ^[0-9]+$ ]]; then
  echo "$OUT" > "$MARKER"
  echo "video submitted: job $OUT"
else
  echo "not submitted yet: $OUT"
  exit 1
fi
