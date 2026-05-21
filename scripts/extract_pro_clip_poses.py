#!/usr/bin/env python3
"""Extract MediaPipe pose for each curated pro clip.

Runs on the GPU machine (Golden Rule #1: pose extraction is GPU-only).
Loops over pros/<slug>/*.mp4 (skipping --raw subdir), shells out to
extract_poses.py for each, writes <stem>.pose.json next to the clip.
Idempotent — skips clips whose pose JSON already exists.

Usage (on tmassena):
    venv/Scripts/python.exe scripts/extract_pro_clip_poses.py --players sinner
    venv/Scripts/python.exe scripts/extract_pro_clip_poses.py --all
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PROS_DIR = REPO_ROOT / "pros"
PYTHON = sys.executable


def discover_clips(players: list[str] | None) -> list[Path]:
    """Return list of pro clip mp4s, optionally filtered to specific slugs."""
    clips = []
    for slug_dir in sorted(PROS_DIR.iterdir()):
        if not slug_dir.is_dir() or slug_dir.name == "_raw":
            continue
        if players and slug_dir.name not in players:
            continue
        for mp4 in sorted(slug_dir.glob("*.mp4")):
            clips.append(mp4)
    return clips


def extract_pose(clip: Path) -> bool:
    pose_path = clip.with_suffix(".pose.json")
    if pose_path.exists():
        return True
    cmd = [
        PYTHON,
        str(REPO_ROOT / "scripts" / "extract_poses.py"),
        str(clip),
        "-o", str(pose_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [FAIL] {clip.name}: exit {result.returncode}")
        print(f"    stderr tail: {result.stderr.strip()[-200:]}", file=sys.stderr)
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--players", help="Comma-separated slugs (default: all)")
    ap.add_argument("--all", action="store_true", help="Process every pro")
    args = ap.parse_args()

    players = None
    if args.players:
        players = [s.strip() for s in args.players.split(",") if s.strip()]
    elif not args.all:
        ap.error("specify --players or --all")

    clips = discover_clips(players)
    print(f"Found {len(clips)} pro clips to process")

    t0 = time.time()
    ok = 0
    fail = 0
    skip = 0
    for i, clip in enumerate(clips, 1):
        pose_path = clip.with_suffix(".pose.json")
        if pose_path.exists():
            skip += 1
            continue
        print(f"[{i}/{len(clips)}] {clip.parent.name}/{clip.name}")
        if extract_pose(clip):
            ok += 1
        else:
            fail += 1

    elapsed = time.time() - t0
    print(f"\n=== {ok} ok, {skip} skipped, {fail} failed in {elapsed/60:.1f} min ===")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
