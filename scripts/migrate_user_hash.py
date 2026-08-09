#!/usr/bin/env python3
"""
migrate_user_hash.py — move R2 data from one per-user hash prefix to another.

Used when the user_hash derivation changes. The 2026-05-27 switch from
sha256(apple_sub) → sha256(email) means existing Andrew data at
`highlights/u_666f1a02/` needs to move to `highlights/u_ae629639/`.

Usage:
  python scripts/migrate_user_hash.py --from u_666f1a02 --to u_ae629639 --dry-run
  python scripts/migrate_user_hash.py --from u_666f1a02 --to u_ae629639 --commit

What moves:
  highlights/<from>/<...>  →  highlights/<to>/<...>
  processed/<from>/<...>   →  processed/<to>/<...>
  uploads/<vid>.json       in-place edit: user_hash and uploaded_by fields
                           updated from <from> to <to>
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def _load_env():
    for path in [Path(__file__).resolve().parent.parent / ".env",
                 Path.home() / "tennis_analysis" / ".env"]:
        if path.exists():
            for line in path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line: continue
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
            return path
    return None
_load_env()

import boto3  # noqa: E402
from botocore.config import Config  # noqa: E402

BUCKET = "tennis-videos"


def s3_client():
    return boto3.client(
        "s3",
        endpoint_url=f"https://{os.environ['CF_ACCOUNT_ID']}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["CF_R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["CF_R2_SECRET_ACCESS_KEY"],
        config=Config(retries={"max_attempts": 5, "mode": "standard"},
                      read_timeout=900, connect_timeout=30),
    )


def list_keys(s3, prefix):
    p = s3.get_paginator("list_objects_v2")
    for page in p.paginate(Bucket=BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            yield obj["Key"]


def plan_moves(s3, src, dst):
    """Yield (src_key, dst_key) for every key under highlights/<src>/ and
    processed/<src>/."""
    for top in ("highlights", "processed"):
        prefix = f"{top}/{src}/"
        for key in list_keys(s3, prefix):
            rest = key[len(prefix):]
            yield (key, f"{top}/{dst}/{rest}")


def copy_then_delete(s3, src, dst):
    s3.copy_object(Bucket=BUCKET,
                   CopySource={"Bucket": BUCKET, "Key": src}, Key=dst)
    s3.delete_object(Bucket=BUCKET, Key=src)


def stamp_markers(s3, old_hash, new_hash, commit):
    """Rewrite uploads/<vid>.json: user_hash/uploaded_by old → new."""
    touched = 0
    for key in list_keys(s3, "uploads/"):
        if not key.endswith(".json") or "_inflight_" in key: continue
        if key == "uploads/_allowlist.json": continue
        body = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
        try:
            m = json.loads(body)
        except json.JSONDecodeError:
            continue
        changed = False
        if m.get("user_hash") == old_hash:
            m["user_hash"] = new_hash; changed = True
        if m.get("uploaded_by") == old_hash:
            m["uploaded_by"] = new_hash; changed = True
        if changed:
            touched += 1
            if commit:
                s3.put_object(
                    Bucket=BUCKET, Key=key,
                    Body=json.dumps(m).encode("utf-8"),
                    ContentType="application/json",
                )
    return touched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src", required=True,
                    help="Source user_hash (e.g. u_666f1a02)")
    ap.add_argument("--to", dest="dst", required=True,
                    help="Destination user_hash (e.g. u_ae629639)")
    ap.add_argument("--commit", action="store_true",
                    help="Actually perform copy+delete. Default is dry-run.")
    ap.add_argument("--concurrency", type=int, default=16)
    args = ap.parse_args()

    for v in (args.src, args.dst):
        if not v.startswith("u_") or len(v) != 10:
            print(f"ERROR: hashes must look like u_xxxxxxxx, got {v}",
                  file=sys.stderr); sys.exit(1)
    if args.src == args.dst:
        print("ERROR: --from and --to are the same", file=sys.stderr); sys.exit(1)

    s3 = s3_client()
    print(f"Planning {args.src} → {args.dst} …")
    moves = list(plan_moves(s3, args.src, args.dst))
    print(f"  {len(moves)} keys to move")
    if moves:
        print("  Sample (first 5):")
        for src, dst in moves[:5]: print(f"    {src}  →  {dst}")

    if not args.commit:
        print("\nDry-run only. Re-run with --commit to perform the move.")
        n = stamp_markers(s3, args.src, args.dst, commit=False)
        print(f"  {n} uploads/<vid>.json markers would be re-stamped.")
        return

    if moves:
        print(f"\nMoving {len(moves)} keys (concurrency={args.concurrency}) …")
        t0 = time.time(); done = 0; failed = []
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = {ex.submit(copy_then_delete, s3, s, d): (s, d) for s, d in moves}
            for fut in as_completed(futs):
                src, dst = futs[fut]
                try: fut.result()
                except Exception as e: failed.append((src, dst, str(e)))
                done += 1
                if done % 200 == 0 or done == len(moves):
                    print(f"  {done}/{len(moves)} ({time.time()-t0:.1f}s)")
        if failed:
            print(f"\n{len(failed)} failures (first 5):")
            for s, d, e in failed[:5]:
                print(f"  {s} → {d}: {e}")

    print("\nStamping markers …")
    n = stamp_markers(s3, args.src, args.dst, commit=True)
    print(f"  Re-stamped {n} markers ({args.src} → {args.dst})")
    print("\nDone.")


if __name__ == "__main__":
    main()
