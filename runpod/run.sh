#!/usr/bin/env bash
# Launch the Emilia orchestrator (one process, internal multiprocess.Pool over
# all GPUs x --replication). Extra args are passed straight through to main.py,
# e.g.  bash runpod/run.sh --num_shards 2 --max_rows 5   (smoke test)
#        bash runpod/run.sh --replication 2              (full run, 2 workers/GPU)
set -euo pipefail

cd /emilia
[ -f /emilia/.env ] && { set -a; . /emilia/.env; set +a; }

export HF_HOME=/emilia/hf
export HF_HUB_ENABLE_HF_TRANSFER=1
export OMP_NUM_THREADS=1

# Point the loader at the venv's bundled cuDNN, resolved dynamically so we don't
# hardcode a python version in the path.
CUDNN_LIB="$(./emilia/bin/python3 -c 'import os,nvidia.cudnn as c;print(os.path.join(os.path.dirname(c.__file__),"lib"))' 2>/dev/null || true)"
if [ -n "${CUDNN_LIB:-}" ]; then
  export LD_LIBRARY_PATH="${CUDNN_LIB}:${LD_LIBRARY_PATH:-}"
fi

exec ./emilia/bin/python3 -u main.py \
  --config_path config.json \
  --repo_id alvanlii/cantonese-youtube \
  --output_dir /emilia/output \
  --batch_size 4 \
  --compute_type bfloat16 \
  --whisper_arch large-v3 \
  --replication 1 \
  "$@"
