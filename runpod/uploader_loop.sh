#!/usr/bin/env bash
# Self-contained uploader: every INTERVAL seconds, push the latest metadata parquet
# (full snapshot, overwrites) and any NEW segment MP3s (incremental, manifest-based)
# to the HF dataset. Runs independently of the driving laptop; launch it detached:
#
#   cd /emilia && setsid bash -c 'bash runpod/uploader_loop.sh > /emilia/uploader.log 2>&1' </dev/null &
#
# Args: [repo] [interval_seconds]
set -uo pipefail

cd /emilia
[ -f /emilia/.env ] && { set -a; . /emilia/.env; set +a; }
export HF_HOME=/emilia/hf
export HF_HUB_ENABLE_HF_TRANSFER=1

REPO="${1:-Scicom-intl/YouTube-Cantonese-Emilia}"
INTERVAL="${2:-14400}"   # 4 hours
TAG="${3:-a}"            # per-pod tag so multiple pods don't clobber each other
PY=/emilia/emilia/bin/python3

echo "[uploader] repo=$REPO interval=${INTERVAL}s tag=$TAG starting $(date -u)"
while true; do
  echo "[uploader $(date -u)] === cycle start (tag=$TAG) ==="
  $PY runpod/push_verify.py --repo "$REPO" --path_in_repo "data/part-${TAG}.parquet" \
      || echo "[uploader] push_verify failed (will retry next cycle)"
  $PY runpod/push_audio.py  --repo "$REPO" --prefix "output-audio-${TAG}" \
      --manifest "/emilia/uploaded_${TAG}.txt" \
      || echo "[uploader] push_audio failed (will retry next cycle)"
  echo "[uploader $(date -u)] === cycle done, sleeping ${INTERVAL}s ==="
  sleep "$INTERVAL"
done
