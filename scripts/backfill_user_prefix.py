#!/usr/bin/env python3
"""
backfill_user_prefix.py — move existing R2 objects under a per-user prefix.

The per-user gallery refactor namespaces gallery outputs under
`highlights/<user_hash>/...` and pipeline intermediates under
`processed/<user_hash>/...`. Before the refactor, all data lived flat at
`highlights/<vid>/...` and `processed/<vid>/...`. This script moves the
existing flat data to a single owner's prefix (Andrew's, by default).

What it does NOT touch:
  • source/<vid>.{mov,mp4}   — raw uploads; the Hetzner poller still
                                scans this prefix flat. Attribute via
                                `uploaded_by` in the marker instead.
  • uploads/<vid>.json       — work-queue markers; the poller reads these
                                flat. We DO write `user_hash` into them.
  • highlights/index.html    — replaced separately with a signed-in
                                landing page (out of scope here).

Usage:
  python scripts/backfill_user_prefix.py --dry-run
  python scripts/backfill_user_prefix.py --user-hash u_666f1a02
  python scripts/backfill_user_prefix.py --user-hash u_666f1a02 --commit

Default is dry-run. --commit actually performs the copy + delete.
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Load .env from a few candidate locations. The Mac dev keeps the auth
# secrets in the main tree's .env (`~/tennis_analysis/.env`), but this
# script can also be run from a worktree without its own .env.
def _load_env():
    candidates = [
        Path(__file__).resolve().parent.parent / ".env",
        Path.home() / "tennis_analysis" / ".env",
    ]
    for path in candidates:
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
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
        config=Config(retries={"max_attempts": 5, "mode": "standard"}),
    )


def list_keys(s3, prefix):
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            yield obj["Key"], obj["Size"]


def plan_moves(s3, user_hash):
    """Yield (src, dst) tuples for every key that should be moved."""
    user_prefix_h = f"highlights/{user_hash}/"
    user_prefix_p = f"processed/{user_hash}/"

    # highlights/<vid>/<...>   →   highlights/<user_hash>/<vid>/<...>
    # highlights/thumbs/<vid>.jpg → highlights/<user_hash>/thumbs/<vid>.jpg
    for key, _size in list_keys(s3, "highlights/"):
        if key.startswith(user_prefix_h):
            continue  # already moved
        # Don't touch the root landing page or other static helpers.
        if key in ("highlights/index.html",):
            continue
        rest = key[len("highlights/"):]
        # Skip any other top-level non-vid files (e.g. shared.html, robots.txt).
        if "/" not in rest:
            continue
        dst = f"highlights/{user_hash}/{rest}"
        yield (key, dst)

    # processed/<vid>/<...>    →   processed/<user_hash>/<vid>/<...>
    for key, _size in list_keys(s3, "processed/"):
        if key.startswith(user_prefix_p):
            continue
        rest = key[len("processed/"):]
        if "/" not in rest:
            continue
        dst = f"processed/{user_hash}/{rest}"
        yield (key, dst)


def copy_then_delete(s3, src, dst):
    s3.copy_object(
        Bucket=BUCKET,
        CopySource={"Bucket": BUCKET, "Key": src},
        Key=dst,
    )
    s3.delete_object(Bucket=BUCKET, Key=src)


def stamp_markers(s3, user_hash, commit):
    """Add `user_hash` to every uploads/<vid>.json marker that lacks one."""
    paginator = s3.get_paginator("list_objects_v2")
    stamped = 0
    for page in paginator.paginate(Bucket=BUCKET, Prefix="uploads/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".json"):
                continue
            if "/_inflight_" in key or key == "uploads/_allowlist.json":
                continue
            body = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
            try:
                meta = json.loads(body)
            except json.JSONDecodeError:
                continue
            if meta.get("user_hash"):
                continue
            meta["user_hash"] = user_hash
            if commit:
                s3.put_object(
                    Bucket=BUCKET, Key=key,
                    Body=json.dumps(meta).encode("utf-8"),
                    ContentType="application/json",
                )
            stamped += 1
    return stamped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user-hash", default="u_666f1a02",
                    help="Target user_hash prefix (default: Andrew)")
    ap.add_argument("--commit", action="store_true",
                    help="Actually perform copy+delete. Default is dry-run.")
    ap.add_argument("--concurrency", type=int, default=16)
    args = ap.parse_args()

    if not args.user_hash.startswith("u_") or len(args.user_hash) != 10:
        print(f"ERROR: --user-hash must look like u_xxxxxxxx, got {args.user_hash}",
              file=sys.stderr)
        sys.exit(1)

    s3 = s3_client()
    print(f"Planning moves under user_hash={args.user_hash} …")
    moves = list(plan_moves(s3, args.user_hash))
    print(f"  {len(moves)} keys to move")

    if not moves:
        print("  (nothing to do)")
    else:
        print("  Sample (first 5):")
        for src, dst in moves[:5]:
            print(f"    {src}  →  {dst}")

    if not args.commit:
        print("\nDry-run only. Re-run with --commit to perform the move.")
        # Still preview marker stamping count.
        print("\nMarkers without user_hash:")
        n = stamp_markers(s3, args.user_hash, commit=False)
        print(f"  {n} markers would be stamped with user_hash={args.user_hash}")
        return

    print(f"\nMoving {len(moves)} keys with concurrency={args.concurrency} …")
    t0 = time.time()
    done = 0
    failed = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {ex.submit(copy_then_delete, s3, src, dst): (src, dst) for src, dst in moves}
        for fut in as_completed(futs):
            src, dst = futs[fut]
            try:
                fut.result()
            except Exception as e:
                failed.append((src, dst, str(e)))
            done += 1
            if done % 200 == 0 or done == len(moves):
                print(f"  {done}/{len(moves)} ({time.time()-t0:.1f}s)")
    if failed:
        print(f"\n{len(failed)} failures (first 5):")
        for src, dst, err in failed[:5]:
            print(f"  {src} → {dst}: {err}")

    print("\nStamping markers …")
    n = stamp_markers(s3, args.user_hash, commit=True)
    print(f"  Stamped {n} markers with user_hash={args.user_hash}")
    print(f"\nDone. Elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
