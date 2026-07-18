#!/usr/bin/env bash
# One-time environment setup on the RunPod box. Everything lives under / (the
# container disk), NOT /workspace. Re-runnable: existing downloads/venv are kept.
#
# The install ORDER below is load-bearing — Emilia pins fussy versions and the
# onnxruntime-gpu swap must happen last, after transformers==4.47.1. Do not
# reorder. (Extra packages needed by the orchestrator are appended at the very
# end so the critical chain is untouched.)
set -euo pipefail

cd /emilia

# Make HF caches live on / so they survive container restarts and are shared by
# every worker (avoids re-downloading whisper-large-v3 / pyannote per process).
export HF_HOME=/emilia/hf
mkdir -p "$HF_HOME" output
[ -f /emilia/.env ] && { set -a; . /emilia/.env; set +a; }

# ---- critical chain (exact order from README/virtualenv.sh) -----------------
apt update
apt install screen ffmpeg libavdevice-dev pkg-config -y

[ -f UVR-MDX-NET-Inst_HQ_3.onnx ] || \
  wget -nc https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/UVR-MDX-NET-Inst_HQ_3.onnx
[ -f sig_bak_ovr.onnx ] || \
  wget -nc https://github.com/microsoft/DNS-Challenge/raw/refs/heads/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx

[ -d emilia ] || python3 -m venv emilia
# Repair two bits of stock-image / upstream drift (neither changes a pinned
# runtime version — only build tooling):
#  1. pip 22.0.2 builds `av` (PyAV) from source and fails; a modern pip picks the
#     manylinux wheel.
#  2. setuptools >= 81 dropped `pkg_resources`, which whisperX's setup.py imports
#     at build time -> pin setuptools < 81. PIP_CONSTRAINT makes the pin apply
#     inside pip's build-isolation env too (where whisperX is actually built).
printf 'setuptools<81\n' > /emilia/pip-constraints.txt
export PIP_CONSTRAINT=/emilia/pip-constraints.txt
./emilia/bin/pip3 install -U pip wheel "setuptools<81"
./emilia/bin/pip3 install -r requirements.txt
./emilia/bin/pip3 install transformers==4.47.1
./emilia/bin/pip3 uninstall onnxruntime onnxruntime-gpu -y
./emilia/bin/pip3 install onnxruntime-gpu==1.20.0
# ---- end critical chain -----------------------------------------------------

# Orchestrator-only extras (safe to install after the pinned chain).
./emilia/bin/pip3 install librosa
./emilia/bin/pip3 install "multiprocess" "click" "hf_transfer"

echo "bootstrap done. python: $(./emilia/bin/python3 --version)"
./emilia/bin/python3 -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),'ngpu',torch.cuda.device_count())" || true
