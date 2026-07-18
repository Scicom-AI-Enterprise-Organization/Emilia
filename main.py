# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Orchestrator for running the Emilia pipeline over the HF parquet dataset
# `alvanlii/cantonese-youtube` (1090 shards, ~1.49M MP3 clips) on N GPUs.
#
# Design notes
# ------------
# * This file is deliberately CUDA-free at import time. All torch/onnxruntime
#   work lives in pipeline.py, imported *inside* each worker AFTER pinning
#   CUDA_VISIBLE_DEVICES — importing it here would poison the forked children's
#   CUDA context. Keep it that way.
# * We stream parquet shards straight from the Hub with range reads
#   (HfFileSystem), one row-group at a time. The 527 GB dataset is never
#   materialized on disk; only the ~240 KB MP3 for the clip currently being
#   processed touches a tmpfs scratch file.
# * Work is sharded by parquet file across `replication * num_gpus` workers.
#   Checkpoint/resume is per clip: a valid `<out>/<shard>/<id>/<id>.json` means
#   done, so a re-run (after a crash, a pod restart, or bumping shard range)
#   simply skips finished clips. Nothing is ever re-processed.

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json
import subprocess
import tempfile

import click
from multiprocess import Pool

REPO_DEFAULT = "alvanlii/cantonese-youtube"


def load_cfg(cfg_path):
    """Minimal, CUDA-free config loader (keeps torch out of the parent)."""
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(
            f"{cfg_path} not found. Copy example_config.json to {cfg_path}."
        )
    with open(cfg_path) as f:
        cfg = json.load(f)
    # HF token: env wins over config so we never commit it.
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        cfg["huggingface_token"] = token
    return cfg


def resolve_devices(replication):
    """List of GPU ids to place workers on, WITHOUT importing torch (fork-safe).

    `CUDA_VISIBLE_DEVICES` wins; otherwise fall back to `nvidia-smi -L`.
    Each id is repeated `replication` times => that many worker replicas per GPU.
    """
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env:
        devices = [d.strip() for d in env.split(",") if d.strip() != ""]
    else:
        try:
            out = subprocess.check_output("nvidia-smi -L | wc -l", shell=True)
            n = int(out.decode().strip())
        except Exception:
            n = 1
        devices = [str(i) for i in range(max(n, 1))]
    return replication * devices


def chunks(l, devices):
    """Balanced contiguous split of `l` into len(devices) parts (no dropped tail)."""
    chunk_size = len(l) // len(devices)
    remainder = len(l) % len(devices)
    start = 0
    for i in range(len(devices)):
        extra = 1 if i < remainder else 0
        end = start + chunk_size + extra
        yield (l[start:end], devices[i])
        start = end


def list_shards(repo_id, split, token):
    """Sorted list of parquet shard paths in the dataset repo."""
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem(token=token)
    base = f"datasets/{repo_id}/data"
    paths = fs.glob(f"{base}/{split}-*.parquet")
    return sorted(paths)


# Config shared with every worker (set in main(), read in loop()).
_COMMON = {}


def loop(shards_device_pair):
    """Worker body: pin one GPU, load models once, stream assigned shards."""
    shards, device = shards_device_pair
    if not shards:
        return (0, 0)  # nothing assigned (e.g. more workers than shards) — don't load models
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)

    # Heavy, CUDA-touching imports happen HERE, after the fork + device pin.
    from huggingface_hub import HfFileSystem
    import pyarrow.parquet as pq
    import pipeline

    cfg = _COMMON["cfg"]
    out_dir = _COMMON["out_dir"]
    token = _COMMON["token"]
    max_rows = _COMMON["max_rows"]

    pipeline.init_models(
        cfg,
        whisper_arch=_COMMON["whisper_arch"],
        compute_type=_COMMON["compute_type"],
        threads=_COMMON["threads"],
        bs=_COMMON["batch_size"],
    )

    log = pipeline.logger
    fs = HfFileSystem(token=token)
    tmp_dir = os.path.join(out_dir, ".tmp", str(os.getpid()))
    os.makedirs(tmp_dir, exist_ok=True)

    done = 0
    processed = 0
    for shard_path in shards:
        stem = os.path.splitext(os.path.basename(shard_path))[0]  # train-XXXXX-of-01090
        shard_out = os.path.join(out_dir, stem)
        try:
            with fs.open(shard_path, "rb") as f:
                pf = pq.ParquetFile(f)
                for rg in range(pf.metadata.num_row_groups):
                    table = pf.read_row_group(rg, columns=["id", "audio"])
                    ids = table.column("id").to_pylist()
                    audios = table.column("audio").to_pylist()
                    for rid, audio in zip(ids, audios):
                        save_path = os.path.join(shard_out, rid)
                        final_path = os.path.join(save_path, rid + ".json")
                        if _is_done(final_path):
                            done += 1
                            continue
                        ext = os.path.splitext(audio.get("path") or "")[1] or ".mp3"
                        tmp_path = os.path.join(tmp_dir, rid + ext)
                        try:
                            with open(tmp_path, "wb") as w:
                                w.write(audio["bytes"])
                            pipeline.main_process(
                                tmp_path, save_path=save_path, audio_name=rid
                            )
                            processed += 1
                        except Exception as e:
                            log.warning(f"[gpu{device}] {stem}/{rid} failed: {e}")
                        finally:
                            if os.path.exists(tmp_path):
                                os.remove(tmp_path)
                        if processed and processed % 25 == 0:
                            _cleanup(pipeline)
                        if max_rows and processed >= max_rows:
                            log.info(f"[gpu{device}] hit max_rows={max_rows}, stopping")
                            _cleanup(pipeline)
                            return (processed, done)
            log.info(f"[gpu{device}] finished shard {stem} (processed={processed}, skipped={done})")
        except Exception as e:
            log.warning(f"[gpu{device}] shard {stem} errored, will retry next run: {e}")
        _cleanup(pipeline)
    return (processed, done)


def _is_done(final_path):
    if not os.path.exists(final_path):
        return False
    try:
        with open(final_path) as f:
            json.load(f)
        return True
    except Exception:
        return False


def _cleanup(pipeline):
    import gc

    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass


@click.command()
@click.option("--config_path", default="config.json", help="config path")
@click.option("--repo_id", default=REPO_DEFAULT, help="HF dataset repo id")
@click.option("--split", default="train", help="parquet split prefix")
@click.option("--output_dir", default="/emilia/output", help="where results land")
@click.option("--replication", default=1, type=int, help="worker replicas per GPU")
@click.option("--batch_size", default=16, type=int)
@click.option("--compute_type", default="bfloat16")
@click.option("--whisper_arch", default="large-v3")
@click.option("--threads", default=4, type=int)
@click.option("--shard_start", default=0, type=int, help="first shard index (inclusive)")
@click.option("--num_shards", default=0, type=int, help="how many shards (0 = all)")
@click.option("--max_rows", default=0, type=int, help="cap clips PER WORKER (0 = no cap); for smoke tests")
def main(
    config_path,
    repo_id,
    split,
    output_dir,
    replication,
    batch_size,
    compute_type,
    whisper_arch,
    threads,
    shard_start,
    num_shards,
    max_rows,
):
    cfg = load_cfg(config_path)
    token = cfg.get("huggingface_token", "") or None
    os.makedirs(output_dir, exist_ok=True)

    shards = list_shards(repo_id, split, token)
    if num_shards > 0:
        shards = shards[shard_start : shard_start + num_shards]
    else:
        shards = shards[shard_start:]
    print(f"{len(shards)} shard(s) queued from {repo_id} (start={shard_start})")

    devices = resolve_devices(replication)
    print(f"devices (with replication={replication}): {devices}")

    _COMMON.update(
        cfg=cfg,
        out_dir=output_dir,
        token=token,
        whisper_arch=whisper_arch,
        compute_type=compute_type,
        threads=threads,
        batch_size=batch_size,
        max_rows=max_rows,
    )

    splits = list(chunks(shards, devices))
    for (s, d) in splits:
        print(f"  gpu{d}: {len(s)} shard(s)")

    with Pool(len(devices)) as pool:
        results = pool.map(loop, splits)

    total_proc = sum(r[0] for r in results if r)
    total_skip = sum(r[1] for r in results if r)
    print(f"DONE. processed={total_proc} skipped(already done)={total_skip}")


if __name__ == "__main__":
    main()
