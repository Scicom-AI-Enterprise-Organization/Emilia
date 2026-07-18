#!/usr/bin/env python3
"""Upload segment MP3s to the HF dataset, zipped in ~5 GB parts (matches
huseinzol05's gist convention).

INCREMENTAL by default so it is safe to run on a schedule (e.g. every 4h): a local
manifest records already-uploaded MP3s, so each run only zips/uploads *new* audio.
Part zips are uniquely named per run (`output-audio-<stamp>-<i>.zip`) so growing the
dataset never shifts an earlier part's contents.

Zip arcnames are relative to --out so each entry matches the parquet's
`audio_filename` (<shard>/<id>/<id>_<i>.mp3).

  python3 runpod/push_audio.py --repo Scicom-intl/YouTube-Cantonese-Emilia
"""
import argparse
import glob
import os
import time
import zipfile

from huggingface_hub import HfApi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/emilia/output")
    ap.add_argument("--repo", default="Scicom-intl/YouTube-Cantonese-Emilia")
    ap.add_argument("--partition_size", type=float, default=5e9)
    ap.add_argument("--manifest", default="/emilia/uploaded_audio.txt")
    ap.add_argument("--stamp", default=None, help="override run stamp (default: UTC time)")
    args = ap.parse_args()

    stamp = args.stamp or time.strftime("%Y%m%d-%H%M%S", time.gmtime())

    uploaded = set()
    if os.path.exists(args.manifest):
        with open(args.manifest) as f:
            uploaded = {ln.strip() for ln in f if ln.strip()}

    all_mp3 = sorted(glob.glob(os.path.join(args.out, "*", "*", "*.mp3")))
    rels = [os.path.relpath(f, args.out) for f in all_mp3]
    new = [(f, r) for f, r in zip(all_mp3, rels) if r not in uploaded]
    print(f"total_mp3={len(all_mp3)} already_uploaded={len(uploaded)} new={len(new)}")
    if not new:
        print("nothing new to upload")
        return

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(args.repo, repo_type="dataset", private=True, exist_ok=True)

    def flush(batch, idx):
        part = f"output-audio-{stamp}-{idx}.zip"
        with zipfile.ZipFile(part, "w", zipfile.ZIP_DEFLATED) as z:
            for f, r in batch:
                z.write(f, arcname=r)
        while True:
            try:
                api.upload_file(path_or_fileobj=part, path_in_repo=part,
                                repo_id=args.repo, repo_type="dataset")
                break
            except Exception as e:
                print("  upload retry:", e)
                time.sleep(60)
        os.remove(part)
        # commit these to the manifest only after a successful upload
        with open(args.manifest, "a") as m:
            for _, r in batch:
                m.write(r + "\n")
        print(f"  pushed {part} ({len(batch)} files)")

    batch, total, idx = [], 0, 0
    for f, r in new:
        s = os.path.getsize(f)
        if batch and total + s >= args.partition_size:
            flush(batch, idx)
            idx += 1
            batch, total = [], 0
        batch.append((f, r))
        total += s
    if batch:
        flush(batch, idx)
    print(f"done -> https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
