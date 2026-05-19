#!/usr/bin/env python3
"""Phase 2 batch driver — runs MediaPipe pose extraction + shot detection
on every YouTube reel in pros/_raw/<slug>/<id>.mp4.

Designed to run on a GPU machine (tmassena/andrew-pc), not on Mac. The
pipeline is the same one used for user videos:
  extract_poses.py -> poses_full_videos/<id>.json
  detect_shots_sequence.py -> pros/_raw/<slug>/<id>_shots.json

Each reel takes ~5-15 minutes for pose extraction depending on duration
and fps. Detection itself is ~5 sec.

Idempotent: skips any reel whose `<id>_shots.json` already exists.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PROS_RAW = REPO_ROOT / "pros" / "_raw"
PREPROCESSED_DIR = REPO_ROOT / "preprocessed"
POSES_DIR = REPO_ROOT / "poses_full_videos"
PYTHON = sys.executable  # current venv's python


def discover_reels() -> list[tuple[str, Path]]:
    """Return list of (slug, mp4_path) tuples sorted for deterministic order."""
    reels = []
    for slug_dir in sorted(PROS_RAW.iterdir()):
        if not slug_dir.is_dir():
            continue
        for mp4 in sorted(slug_dir.glob("*.mp4")):
            reels.append((slug_dir.name, mp4))
    return reels


def shots_path_for(mp4: Path) -> Path:
    return mp4.with_name(f"{mp4.stem}_shots.json")


def run_extract(preprocessed_mp4: Path) -> bool:
    """Run extract_poses.py with --skip-dead. Returns True on success."""
    cmd = [
        PYTHON,
        str(REPO_ROOT / "scripts" / "extract_poses.py"),
        str(preprocessed_mp4),
        "--skip-dead",
    ]
    print(f"  [pose] {' '.join(cmd[1:])}")
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    return result.returncode == 0


def run_detect(preprocessed_mp4: Path, output: Path) -> bool:
    """Run detect_shots_sequence.py, writing JSON to `output`."""
    cmd = [
        PYTHON,
        str(REPO_ROOT / "scripts" / "detect_shots_sequence.py"),
        str(preprocessed_mp4),
        "--output", str(output),
    ]
    print(f"  [detect] {' '.join(cmd[1:])}")
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    return result.returncode == 0


def process_one(slug: str, mp4: Path, cleanup: bool) -> dict:
    """Returns a result dict with status + timing."""
    vid_id = mp4.stem
    out_path = shots_path_for(mp4)
    if out_path.exists():
        return {"slug": slug, "id": vid_id, "status": "skip-exists"}

    PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    POSES_DIR.mkdir(parents=True, exist_ok=True)

    preprocessed = PREPROCESSED_DIR / f"{vid_id}.mp4"
    pose_json = POSES_DIR / f"{vid_id}.json"

    t0 = time.time()

    # Stage to PREPROCESSED_DIR if not already there (we accept the raw mp4
    # as-is — these are CFR YouTube uploads, no NVENC pass needed).
    if not preprocessed.exists():
        print(f"  [stage] copying {mp4.name} -> preprocessed/")
        shutil.copy2(mp4, preprocessed)

    # Pose extraction (the slow step)
    if not pose_json.exists():
        if not run_extract(preprocessed):
            return {"slug": slug, "id": vid_id, "status": "fail-pose",
                    "elapsed_s": time.time() - t0}
    else:
        print(f"  [pose] cached: {pose_json.name}")

    # Shot detection
    if not run_detect(preprocessed, out_path):
        return {"slug": slug, "id": vid_id, "status": "fail-detect",
                "elapsed_s": time.time() - t0}

    elapsed = time.time() - t0

    # Optional cleanup of intermediates (preprocessed mp4 + pose JSON).
    # Saves disk if processing all 32 reels. Skip if you want to re-run
    # detection with different thresholds later.
    if cleanup:
        try:
            preprocessed.unlink()
            pose_json.unlink()
        except FileNotFoundError:
            pass

    # Read summary back for logging
    try:
        with out_path.open() as f:
            data = json.load(f)
        summary = data.get("summary", {})
    except Exception:
        summary = {}

    return {
        "slug": slug,
        "id": vid_id,
        "status": "ok",
        "elapsed_s": round(elapsed, 1),
        "by_type": summary.get("by_type", {}),
        "total": summary.get("total_detections", 0),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--players", help="Comma-separated slugs to process (default: all)")
    ap.add_argument("--cleanup", action="store_true",
                    help="Remove preprocessed/<id>.mp4 + poses_full_videos/<id>.json after each reel "
                         "(saves disk, but blocks re-running detection without re-extracting pose)")
    ap.add_argument("--dry-run", action="store_true", help="List reels that would be processed and exit")
    args = ap.parse_args()

    targets = discover_reels()
    if args.players:
        wanted = {s.strip() for s in args.players.split(",") if s.strip()}
        targets = [(slug, mp4) for (slug, mp4) in targets if slug in wanted]

    if not targets:
        print("No reels found.")
        return 0

    print(f"Found {len(targets)} reels across {len({s for s, _ in targets})} pros.")
    if args.dry_run:
        for slug, mp4 in targets:
            done = "x" if shots_path_for(mp4).exists() else " "
            print(f"  [{done}] {slug}/{mp4.name}")
        return 0

    overall_t0 = time.time()
    results = []
    for i, (slug, mp4) in enumerate(targets, 1):
        print(f"\n[{i}/{len(targets)}] {slug}/{mp4.name}")
        result = process_one(slug, mp4, cleanup=args.cleanup)
        results.append(result)
        status = result["status"]
        if status == "ok":
            print(f"  -> {result['total']} shots ({result['by_type']}) in {result['elapsed_s']}s")
        elif status == "skip-exists":
            print(f"  -> skipped (shots JSON already exists)")
        else:
            print(f"  -> FAIL: {status}")

    total_elapsed = time.time() - overall_t0
    ok = sum(1 for r in results if r["status"] == "ok")
    skip = sum(1 for r in results if r["status"] == "skip-exists")
    fail = sum(1 for r in results if r["status"].startswith("fail"))
    print(f"\n=== Batch complete: {ok} ok, {skip} skipped, {fail} failed, "
          f"{total_elapsed/60:.1f} min total ===")

    # Print per-reel summary table
    print("\nslug                id              status         shots   elapsed")
    print("-" * 75)
    for r in results:
        elapsed = f"{r.get('elapsed_s', 0):.0f}s" if "elapsed_s" in r else "-"
        total = r.get("total", "-")
        print(f"{r['slug']:18s} {r['id']:15s} {r['status']:14s} {str(total):6s} {elapsed}")

    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
