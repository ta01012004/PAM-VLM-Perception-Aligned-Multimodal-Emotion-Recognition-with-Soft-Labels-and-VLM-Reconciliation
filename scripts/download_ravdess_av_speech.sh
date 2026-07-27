#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-data/raw/ravdess}"
FIRST_ACTOR="${2:-1}"
LAST_ACTOR="${3:-24}"
if command -v ffmpeg >/dev/null 2>&1; then
  FFMPEG_BIN=$(command -v ffmpeg)
else
  FFMPEG_BIN=$(
    /home/22011107/miniconda3/envs/ta/bin/python -c \
      'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())'
  )
fi
mkdir -p "${ROOT}/video" "${ROOT}/audio" "${ROOT}/.downloads"

for actor_number in $(seq "${FIRST_ACTOR}" "${LAST_ACTOR}"); do
  actor=$(printf '%02d' "${actor_number}")
  video_count=$(find "${ROOT}/video" -maxdepth 1 -type f -name "01-01-*-*-*-*-${actor}.mp4" | wc -l)
  audio_count=$(find "${ROOT}/audio" -maxdepth 1 -type f -name "01-01-*-*-*-*-${actor}.wav" | wc -l)
  if [[ "${video_count}" -eq 44 && "${audio_count}" -eq 44 ]]; then
    echo "RAVDESS actor ${actor} already prepared"
    continue
  fi
  archive="${ROOT}/.downloads/Video_Speech_Actor_${actor}.zip"
  url="https://zenodo.org/api/records/1188976/files/Video_Speech_Actor_${actor}.zip/content"
  if [[ -s "${archive}" ]] && unzip -tq "${archive}" >/dev/null 2>&1; then
    echo "Using complete RAVDESS archive for actor ${actor}"
  else
    echo "Downloading RAVDESS actor ${actor}"
    curl -L --fail --retry 5 --continue-at - "${url}" -o "${archive}"
  fi

  while IFS= read -r entry; do
    stem=$(basename "${entry}" .mp4)
    emotion=$(printf '%s' "${stem}" | cut -d- -f3)
    case "${emotion}" in
      01|03|04|05|06|07) ;;
      *) continue ;;
    esac
    video="${ROOT}/video/${stem}.mp4"
    audio="${ROOT}/audio/${stem}.wav"
    if [[ ! -s "${video}" ]]; then
      unzip -p "${archive}" "${entry}" > "${video}"
    fi
    if [[ ! -s "${audio}" ]]; then
      "${FFMPEG_BIN}" -nostdin -loglevel error -y -i "${video}" \
        -vn -ac 1 -ar 16000 "${audio}"
    fi
  done < <(unzip -Z1 "${archive}" | grep -E '(^|/)01-01-[0-9]{2}-[0-9]{2}-[0-9]{2}-[0-9]{2}-[0-9]{2}\.mp4$')
  unlink "${archive}"
done

echo "RAVDESS full-AV speech prepared under ${ROOT}"
