#!/usr/bin/env python3
"""Generate per-shot user-vs-pro comparison FILMSTRIPS and upload to R2.

The static analog of the per-shot comparison videos: for each shot that has a
pro match (the comparisons_index.json), render the stacked, contact-aligned
user+pro filmstrip (via compare_filmstrip) and upload to
highlights/<user>/<vid>/sequences/compare_shot_NNN.jpg. /inspect shows it
inline so you can read your swing against the pro frame-by-frame.

Usage:
    python scripts/gen_compare_filmstrips.py iphone_9ca0a615 --user-hash u_ae629639
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
# Load .env BEFORE importing config.settings — settings snapshots the R2
# credentials from env at import time, so a late load_dotenv leaves them empty
# (the AKID-malformed error). Standalone runs don't have env preset like the
# pipeline process does.
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")
from config.settings import PROJECT_ROOT
from storage.r2_client import R2Client


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="video id (e.g. iphone_9ca0a615)")
    ap.add_argument("--user-hash", default=None)
    ap.add_argument("--shots", default=None,
                    help="comma list of global shot indices (default: from comparisons_index.json)")
    args = ap.parse_args()
    vid = args.video
    uh = args.user_hash

    c = R2Client(); b = c.bucket_name
    prefix = f"highlights/{uh}/{vid}" if uh else f"highlights/{vid}"

    # Which shots to render — default to the ones that have comparison clips.
    if args.shots:
        shots = [int(x) for x in args.shots.split(",") if x.strip()]
    else:
        try:
            idx = json.loads(c.client.get_object(
                Bucket=b, Key=f"{prefix}/{vid}_comparisons_index.json")["Body"].read())
            shots = idx.get("shots", [])
        except Exception as e:
            print(f"[ERROR] no comparisons_index.json ({e}); pass --shots")
            return 1

    print(f"{vid}: generating {len(shots)} comparison filmstrips -> {prefix}/sequences/")
    done = 0
    for gi in shots:
        out = os.path.join(tempfile.gettempdir(), f"cmp_{vid}_{gi:03d}.png")
        r = subprocess.run(
            [sys.executable, str(Path(PROJECT_ROOT) / "scripts" / "compare_filmstrip.py"),
             "--user", vid, "--global-shot", str(gi), "--output", out,
             "--pro-only", "--no-skeleton"],   # pro-only film strip; user's is in the card
            cwd=PROJECT_ROOT, capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(out):
            print(f"  shot {gi}: FAILED — {r.stderr.strip()[-160:]}")
            continue
        key = f"{prefix}/sequences/compare_shot_{gi:03d}.jpg"
        # compare_filmstrip writes PNG; re-encode to jpg key is fine (R2 serves bytes).
        c.upload(out, key, content_type="image/png")
        done += 1
        print(f"  shot {gi}: uploaded compare_shot_{gi:03d}.jpg")
    print(f"done: {done}/{len(shots)}")
    return 0 if done else 1


if __name__ == "__main__":
    sys.exit(main())
