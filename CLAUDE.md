# Emilia — repo guide for Claude

Fork of open-mmlab/Amphion's **Emilia** speech-data pipeline, adapted to clean and
segment large HF audio datasets on multi-GPU RunPod boxes. Current target dataset:
[`alvanlii/cantonese-youtube`](https://huggingface.co/datasets/alvanlii/cantonese-youtube)
(1090 parquet shards, ~1.49M MP3 clips, ~527 GB, Cantonese).

## What the pipeline does

Per input clip (`pipeline.main_process`), in order:
0. **standardization** — 24 kHz / mono / 16-bit / loudness-normalized.
1. **source separation** — UVR-MDX-NET vocals extraction (`UVR-MDX-NET-Inst_HQ_3.onnx`).
2. **speaker diarization** — `pyannote/speaker-diarization-3.1` (gated, needs HF token).
3. **VAD + segmentation** — silero VAD, merged/trimmed to 3–30 s by speaker.
4. **ASR** — WhisperX / faster-whisper `large-v3`, per-segment language detect.
5. **MOS filter** — DNSMOS (`sig_bak_ovr.onnx`) + char-rate/duration filtering.
6. **export** — one MP3 per kept segment + a `<id>.json` manifest.

## Architecture — why it's split in two files

- **`main.py`** = the orchestrator. It is **CUDA-free at import time** and must stay
  that way. It lists parquet shards from the Hub, splits them across
  `replication × num_gpus` workers, and drives a `multiprocess.Pool`.
- **`pipeline.py`** = all the torch / onnxruntime / pyannote work. It is imported
  **inside each worker**, *after* the worker pins `CUDA_VISIBLE_DEVICES`. Importing
  it (or torch, or onnxruntime) in the parent would poison the forked children's
  CUDA context and every worker would fight over GPU 0.

  ⚠️ **Never add a top-level `import torch` (or `import pipeline`, or
  `from utils.tool import ...`, which pulls torch) to `main.py`.** Keep heavy
  imports inside `loop()`.

This replaces the old `screen`-per-GPU launcher — now it's one `python main.py`
process with an internal pool (pattern borrowed from
`dataset/multilingual-tts/convert_neucodec.py`: `devices = replication * devices`,
balanced `chunks()`, model loaded once per worker).

## Data flow — streaming, never materialized

The 527 GB dataset is **never downloaded whole**. `loop()` opens each parquet shard
over `HfFileSystem` with HTTP **range reads**, iterates one row-group at a time, and
writes only the ~240 KB MP3 of the clip being processed to a tmpfs scratch file
(`/emilia/output/.tmp/<pid>/`), deleted immediately after. Peak disk = output only.

## Checkpoint / resume (this is load-bearing — the full run takes weeks)

- Output layout: `/emilia/output/<shard-stem>/<id>/<id>.json` + segment MP3s.
- A clip is **done** iff its `<id>.json` exists and parses. `loop()` skips done clips
  before decoding; `main_process` double-checks. So any re-run — after a crash, a pod
  stop/start, or widening the shard range — resumes exactly where it left off and
  **never re-processes** a finished clip.
- `id` is globally unique & monotonic across shards (0 → ~1,490,590), but output is
  still namespaced by shard so directories stay ~1357 clips each.
- Sharding uses balanced contiguous `chunks()` — **no dropped tail** (the old
  `len//global_size` slicing silently dropped the remainder shard).

## Running it

```bash
# smoke test: a couple shards, a few clips per worker
bash runpod/run.sh --num_shards 2 --max_rows 5
# full run, all 1090 shards, 1 worker/GPU
bash runpod/run.sh
# push each GPU harder (more VRAM): 2 workers/GPU
bash runpod/run.sh --replication 2
```

`run.sh` sources `/emilia/.env`, sets `HF_HOME=/emilia/hf`, and resolves the venv's
bundled cuDNN dynamically (`nvidia.cudnn.__file__`) into `LD_LIBRARY_PATH` — **no
hardcoded `python3.10` path** (works on any venv python). Key flags on `main.py`:
`--replication`, `--batch_size`, `--compute_type`, `--whisper_arch`, `--shard_start`,
`--num_shards`, `--max_rows` (per-worker cap, for tests), `--output_dir`.

## Environment gotchas (do not "fix" these)

The venv pins fussy versions; the **install order in `runpod/bootstrap.sh` is
load-bearing** and must not be reordered:
1. `apt install screen ffmpeg libavdevice-dev`
2. `wget` both ONNX models (UVR + DNSMOS)
3. `python3 -m venv emilia`
4. `pip install -r requirements.txt`  (torch 2.5.1 + whisperX pinned commit)
5. `pip install transformers==4.47.1`
6. `pip uninstall onnxruntime onnxruntime-gpu -y`
7. `pip install onnxruntime-gpu==1.20.0`  ← must be last, after transformers

`multiprocess`, `click`, `hf_transfer` are installed **after** that chain so they
can't perturb it. `requirements.txt` is not to be repinned.

Two stock-image drifts the bootstrap repairs (build tooling only — no pinned
runtime version changes): **(1)** `pip install -U pip wheel` so PyAV installs from a
wheel instead of a failing source build (needs `pkg-config`, also apt-installed);
**(2)** `setuptools<81` + `PIP_CONSTRAINT` so whisperX's `setup.py` (which imports
the removed `pkg_resources`) still builds inside pip's isolation env.

- **MP3 duration bug (fixed):** upstream read duration with `soundfile.read`, which
  silently returns 0 on MP3 when libsndfile < 1.1.0 → the clip is dropped. `get_length`
  now falls back soundfile→librosa→pydub, so MP3 input works regardless of the
  libsndfile build. If you ever see *every* clip "skipped, too long", suspect this.
- **HF token** comes from `HF_TOKEN` in the env (`main.load_cfg` overrides
  `config.json`), so the token is never committed. pyannote 3.1 is gated — the token
  must have accepted its terms.
- **Cantonese language:** `config.json` sets `force: "yue"` — for a known
  single-language dataset this skips whisper's per-segment language detection (which
  mislabels Cantonese as `zh`) and transcribes every segment as `yue`, dropping
  nothing. `pipeline.asr()` has the fast path. `supported`/`multilingual` still apply
  when `force` is unset.
  **Transcript-quality caveat:** base whisper `large-v3` normalizes Cantonese audio
  to written/simplified Chinese even when forced to `yue` (the label is right, the
  characters lean Mandarin). For true Cantonese output swap `--whisper_arch` to a
  Cantonese-finetuned CTranslate2 model. The current run uses `large-v3` (user OK'd
  it); the dataset also ships native `transcript_whisper` if you need Cantonese text.

## RunPod ops (via claude-ping)

Everything lives under **`/emilia`** on the container disk (`/`), **not
`/workspace`** — there is no network volume (`volumeInGb: 0`). Current pod is **4×
RTX 3090** (24 GB, community, ~$0.88/hr), driver 580 / CUDA 13, running cu124 torch.

> **Why not 4090?** Every RunPod community 4090 landed on one datacenter (CA) whose
> hosts pass `nvidia-smi` but fail `cuInit` (CUDA err 999) — unfixable from the
> container (not UVM/nvidia-caps). A100 was overkill; 6× single-host 3090 had no
> stock. 4× 3090 on a *different* host verified `cuInit==0` and was used. Always run
> the CUDA check (see `scratchpad/provision.py`) before bootstrapping a new pod.

**Throughput:** on 24 GB, use `--replication 1` (1 worker/GPU, ~15 GB used).
`--replication 2` nearly fills VRAM (~24 GB) and risks OOM on a long unattended run —
avoid unless watched. Batch size 4.

Drive it from the laptop with the `claude-ping` binary (persistent SSH). Point it at
this repo's config:

```bash
export CLAUDE_PING_CONFIG=/Users/husein.z/Documents/Emilia/claude-ping.json
CP=/Users/husein.z/Documents/claude-ping/claude-ping
$CP up                    # open persistent master
$CP sync                  # rsync repo -> /emilia (remote needs rsync: apt install rsync)
$CP env-sync              # push HF_TOKEN -> /emilia/.env (600)
$CP bootstrap             # or: exec "cd /emilia && nohup bash runpod/bootstrap.sh > bootstrap.log 2>&1 &"
$CP exec "cd /emilia && nohup bash runpod/run.sh > train.log 2>&1 & echo pid \$!"   # launch
$CP logs 200              # tail train.log (one-shot)
$CP gpu                   # nvidia-smi summary
$CP exec "find /emilia/output -name '*.json' | wc -l"   # progress = clips done
$CP down                  # close master
```

- **Long-running remote commands** (bootstrap/run): always `nohup ... &` them — a
  blocking `claude-ping exec` will hit the 2-min tool timeout even though the SSH job
  keeps running. Poll with one-shot verbs, never hold a stream.
- **rsync must be installed on the pod** (`apt install rsync -y`) — the base image
  lacks it and `claude-ping sync` fails with "rsync: command not found" otherwise.

### Durability caveat

`/` is the container disk: it survives process crashes and pod **stop/start**, so
checkpoint/resume works across those. It does **not** survive pod **termination**.
For a run you can't afford to lose across termination, periodically push
`/emilia/output` to an HF dataset (the pushed data is the real progress; local JSONs
are just skip-stubs) or attach a network volume.

## Provisioning a fresh pod

Use the REST API (the legacy GraphQL `podFindAndDeployOnDemand` hit `SUPPLY_CONSTRAINT`
for 4×4090; REST found stock):

```bash
curl -X POST https://rest.runpod.io/v1/pods -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" -d '{
    "name":"emilia-cantonese","cloudType":"COMMUNITY",
    "gpuTypeIds":["NVIDIA GeForce RTX 4090"],"gpuCount":4,
    "containerDiskInGb":300,"volumeInGb":0,
    "imageName":"runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04",
    "ports":["22/tcp"],"env":{"PUBLIC_KEY":"'"$(cat ~/.ssh/id_rsa.pub)"'"}}'
```

Then poll the GraphQL `pod(...).runtime.ports` for the public port mapped to private
port 22, and write host/port into `claude-ping.json`. Secrets are in `.env`
(`RUNPOD_API_KEY`, `HF_TOKEN`, `WANDB_API_KEY`) — gitignored.

## Files

| file | role |
|---|---|
| `main.py` | CUDA-free orchestrator: shard listing, `chunks()`, `multiprocess.Pool`, resume |
| `pipeline.py` | all GPU work: model loading (`init_models`) + `main_process` stages |
| `config.json` | Cantonese config (gitignored; HF token injected from env) |
| `models/` | `separate_fast`, `dnsmos`, `whisper_asr`, `silero_vad` |
| `utils/tool.py` | audio I/O, `calculate_audio_stats` MOS/char-rate filter |
| `runpod/bootstrap.sh` | one-time env setup (ordered install chain + drift fixes) |
| `runpod/run.sh` | launch orchestrator (LD_LIBRARY_PATH cuDNN + flags) |
| `runpod/push_verify.py` | flatten output JSONs → parquet, push to HF (metadata only) |
| `runpod/push_audio.py` | incremental zip+push of new segment MP3s (~5 GB parts) |
| `runpod/uploader_loop.sh` | detached loop: push_verify + push_audio every N seconds |
| `claude-ping.json` | persistent-SSH config for the pod (host/port filled per-pod) |

## Publishing results — HF dataset `Scicom-intl/YouTube-Cantonese-Emilia`

A detached loop uploads on a schedule (default every 4 h) — it is launched once and
runs independently of the laptop:

```bash
cd /emilia && setsid bash -c 'bash runpod/uploader_loop.sh Scicom-intl/YouTube-Cantonese-Emilia 14400 > /emilia/uploader.log 2>&1' </dev/null &
```

Each cycle:
- `push_verify.py` → rewrites `data/train-00000-of-00001.parquet` (full snapshot, one
  row per kept segment: id, shard, segment_index, audio_filename, text, start, end,
  speaker, language, dnsmos).
- `push_audio.py` → **incremental**: a manifest (`/emilia/uploaded_audio.txt`) tracks
  already-uploaded MP3s, so each run only zips/uploads NEW segments into uniquely-named
  `output-audio-<stamp>-<i>.zip` (~5 GB parts). `audio_filename` in the parquet matches
  the zip arcname.

Both read `HF_TOKEN` from `/emilia/.env`. This HF push is also the durability story:
`/` is wiped on pod stop/start, so the pushed dataset is the real progress record
(and on a fresh pod, `push_audio.py`'s manifest is gone → it re-derives "new" from the
repo? no — it re-zips everything; delete stale local manifest only, HF dedups by
content-addressing on re-upload of identical parts is NOT automatic, so prefer not to
recycle the pod mid-run).
