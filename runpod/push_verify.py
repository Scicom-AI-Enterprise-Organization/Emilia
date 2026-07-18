#!/usr/bin/env python3
"""Flatten Emilia output JSON manifests into a single parquet and push it to an
HF dataset for verification (metadata only — NO audio). One row per kept segment.

Audio (the segment MP3s) is uploaded separately later, zipped in ~5 GB parts
(see the gist convention), keyed by `audio_filename` below.

  python3 runpod/push_verify.py --out /emilia/output --repo Scicom-intl/YouTube-Cantonese-Emilia
"""
import argparse
import glob
import json
import os

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/emilia/output")
    ap.add_argument("--repo", default="Scicom-intl/YouTube-Cantonese-Emilia")
    ap.add_argument("--private", action="store_true", default=True)
    ap.add_argument("--path_in_repo", default="data/train-00000-of-00001.parquet")
    ap.add_argument("--local_parquet", default="/emilia/verify.parquet")
    args = ap.parse_args()

    rows = []
    clips = empty = 0
    for jf in glob.glob(os.path.join(args.out, "*", "*", "*.json")):
        parts = jf.split(os.sep)
        shard, cid = parts[-3], parts[-2]
        try:
            segs = json.load(open(jf))
        except Exception:
            continue  # mid-write / corrupt — skip in this snapshot
        clips += 1
        if not segs:
            empty += 1
            continue
        for i, s in enumerate(segs):
            rows.append({
                "id": cid,
                "shard": shard,
                "segment_index": i,
                "audio_filename": f"{shard}/{cid}/{cid}_{i}.mp3",
                "text": s.get("text"),
                "start": s.get("start"),
                "end": s.get("end"),
                "speaker": s.get("speaker"),
                "language": s.get("language"),
                "dnsmos": s.get("dnsmos"),
            })

    print(f"clips_with_json={clips} empty={empty} segment_rows={len(rows)}")
    if not rows:
        print("no rows yet — nothing to push")
        return

    table = pa.Table.from_pylist(rows)
    pq.write_table(table, args.local_parquet)
    print(f"wrote {args.local_parquet} ({os.path.getsize(args.local_parquet)/1e6:.1f} MB)")

    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    api.create_repo(args.repo, repo_type="dataset", private=args.private, exist_ok=True)
    api.upload_file(
        path_or_fileobj=args.local_parquet,
        path_in_repo=args.path_in_repo,
        repo_id=args.repo,
        repo_type="dataset",
    )
    print(f"pushed -> https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
