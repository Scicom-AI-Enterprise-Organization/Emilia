# Emilia

Fork open-mmlab/Amphion Emilia.

## Requirements

1. Install dependencies,

```bash
apt update
apt install screen ffmpeg libavdevice-dev -y
wget https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/UVR-MDX-NET-Inst_HQ_3.onnx
wget https://github.com/microsoft/DNS-Challenge/raw/refs/heads/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx
python3 -m venv emilia
./emilia/bin/pip3 install -r requirements.txt
./emilia/bin/pip3 install transformers==4.47.1
./emilia/bin/pip3 uninstall onnxruntime onnxruntime-gpu -y
./emilia/bin/pip3 install onnxruntime-gpu==1.20.0
```

**We recommend to use virtual environment because Emilia required specific pinning version**.

2. Add your own config [config.json](config.json), follow [example_config.json](example_config.json).

## Run script

Simply,

```bash
LD_LIBRARY_PATH="$(pwd)/emilia/lib/python3.10/site-packages/nvidia/cudnn/lib" \
./emilia/bin/python3 main.py \
  --batch_size 4 \
  --compute_type bfloat16 \
  --whisper_arch large-v3
```

### Multi-GPUs

Multi-GPUs with auto split, I run command below,

```bash
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)

for ((i=0; i<NUM_GPUS; i++)); do
    screen -S "run_$i" -X quit 2>/dev/null

    screen -dmS "run_$i" bash -c "
        cd \"$(pwd)\" && \
        LD_LIBRARY_PATH=\"$(pwd)/emilia/lib/python3.10/site-packages/nvidia/cudnn/lib\" \
        CUDA_VISIBLE_DEVICES=$i \
        ./emilia/bin/python3 main.py \
        --batch_size 4 \
        --compute_type bfloat16 \
        --whisper_arch large-v3 \
        --global-size $NUM_GPUS \
        --local-index $i
    "
done
```