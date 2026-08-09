#!/usr/bin/env python3
"""Populate the protected holdout pose cache (eval/holdout/poses/).

eval_holdout reads poses from this dir instead of the churning POSES_DIR, so
model evals are fast (no re-extraction) and reproducible. The pipeline never
writes here, so these poses can't get cleared/overwritten — which is what
caused the bogus partial-holdout baselines.

For each video in the holdout manifest: if its poses aren't already cached,
copy them from POSES_DIR if present, else extract fresh. Run on a GPU machine.

Usage:
    python scripts/cache_holdout_poses.py            # fill any missing
    python scripts/cache_holdout_poses.py --force    # re-extract all
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import PROJECT_ROOT, POSES_DIR, PREPROCESSED_DIR

MANIFEST = Path(PROJECT_ROOT) / "eval" / "holdout" / "manifest.json"
CACHE_DIR = Path(PROJECT_ROOT) / "eval" / "holdout" / "poses"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="re-extract poses even if cached")
    args = ap.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text())
    vids = [v["video_id"] for v in manifest["videos"]]
    print(f"holdout videos: {len(vids)} -> cache {CACHE_DIR}")

    for vid in vids:
        dst = CACHE_DIR / f"{vid}.json"
        if dst.exists() and not args.force:
            print(f"  {vid}: cached ({dst.stat().st_size // 1024}KB)")
            continue
        src = Path(POSES_DIR) / f"{vid}.json"
        if src.exists() and not args.force:
            shutil.copy2(src, dst)
            print(f"  {vid}: copied from POSES_DIR")
            continue
        # Extract fresh from the preprocessed video.
        vp = Path(PREPROCESSED_DIR) / f"{vid}.mp4"
        if not vp.exists():
            print(f"  {vid}: [SKIP] no preprocessed video at {vp}")
            continue
        print(f"  {vid}: extracting (this is slow)...")
        # Extract to POSES_DIR (the script's default), then copy into cache.
        if src.exists():
            src.unlink()
        r = subprocess.run(
            [sys.executable, str(Path(PROJECT_ROOT) / "scripts" / "extract_poses.py"), str(vp)],
            cwd=PROJECT_ROOT)
        if r.returncode == 0 and src.exists():
            shutil.copy2(src, dst)
            print(f"  {vid}: extracted + cached")
        else:
            print(f"  {vid}: [FAIL] extraction returned {r.returncode}")

    cached = sorted(p.name for p in CACHE_DIR.glob("*.json"))
    print(f"\ncache now holds {len(cached)}/{len(vids)}: {cached}")
    if len(cached) < len(vids):
        print("WARNING: cache incomplete — eval_holdout will report INVALID.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
