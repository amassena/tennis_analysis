#!/usr/bin/env python3
"""Validate 3D joint-angle computation against 2D baseline.

For each video with local detection + pose files, computes knee / trunk /
arm angles in both 2D (image-plane) and 3D (world_landmarks) modes at
each shot's contact frame. Reports:

  - Sample coverage (how many shots produce a valid measurement per mode)
  - Distribution summaries per shot type (mean, stdev, median, min, max)
  - Anatomical-plausibility failures per mode
    (e.g. knee >180° or <60° = impossible)
  - Grade-changing disagreements between 2D and 3D
  - Cross-angle consistency: same-player-same-shot-type variance
    across videos with different camera angles

Usage:
    python scripts/validate_3d_angles.py                # all local videos
    python scripts/validate_3d_angles.py IMG_0864 IMG_6713  # specific
"""
import argparse
import json
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.render_graded import (
    compute_shot_metrics, grade_shot, IDEALS,
    FLAG_YELLOW, FLAG_RED,
)

PROJECT_ROOT = Path(__file__).parent.parent
DETECTIONS_DIR = PROJECT_ROOT / "detections"
POSES_DIR = PROJECT_ROOT / "poses_full_videos"

# Anatomically plausible ranges (degrees). Anything outside = broken measurement.
PLAUSIBLE = {
    "knee":  (60, 180),
    "trunk": (0, 120),
    "arm":   (30, 180),
}


def load_detections(vid):
    for name in [f"{vid}_fused.json", f"{vid}_fused_detections.json"]:
        p = DETECTIONS_DIR / name
        if p.exists():
            with open(p) as f:
                return json.load(f)
    return None


def fmt_stats(values):
    if not values:
        return "n=0"
    return (f"n={len(values):3d}  "
            f"mean={statistics.mean(values):6.1f}  "
            f"median={statistics.median(values):6.1f}  "
            f"stdev={statistics.stdev(values) if len(values)>1 else 0:5.1f}  "
            f"range=[{min(values):.0f},{max(values):.0f}]")


def analyze_video(vid):
    det = load_detections(vid)
    if not det:
        return None
    pose_path = POSES_DIR / f"{vid}.json"
    if not pose_path.exists():
        return None

    with open(pose_path) as f:
        poses = json.load(f)
    pose_frames = poses.get("frames", [])

    per_shot = []  # list of (type, m2d, m3d, grade2d, grade3d)
    for d in det.get("detections", []):
        st = d.get("shot_type", "unknown")
        cf = int(d.get("frame", 0))
        m2 = compute_shot_metrics(pose_frames, cf, st, use_3d=False)
        m3 = compute_shot_metrics(pose_frames, cf, st, use_3d=True)
        g2 = grade_shot(m2, st)[0] if m2 else "?"
        g3 = grade_shot(m3, st)[0] if m3 else "?"
        per_shot.append((st, m2, m3, g2, g3))
    return {"vid": vid, "shots": per_shot}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="*")
    ap.add_argument("--json", help="Write full report to this JSON path")
    args = ap.parse_args()

    if args.videos:
        targets = args.videos
    else:
        targets = sorted({
            p.stem.replace("_fused_detections", "").replace("_fused", "")
            for p in DETECTIONS_DIR.glob("*_fused*.json")
        })

    results = []
    for vid in targets:
        r = analyze_video(vid)
        if r:
            results.append(r)

    if not results:
        print("No videos analysable locally.")
        return

    # Aggregate
    by_type_2d = defaultdict(lambda: {"knee": [], "trunk": [], "arm": []})
    by_type_3d = defaultdict(lambda: {"knee": [], "trunk": [], "arm": []})
    coverage_2d = coverage_3d = total_shots = 0
    implausible_2d = defaultdict(int)
    implausible_3d = defaultdict(int)
    grade_disagreement = 0
    grade_shift = defaultdict(int)
    total_graded = 0

    for r in results:
        for st, m2, m3, g2, g3 in r["shots"]:
            total_shots += 1
            if m2:
                coverage_2d += 1
                for k in ("knee", "trunk", "arm"):
                    by_type_2d[st][k].append(m2[k])
                    lo, hi = PLAUSIBLE[k]
                    if not (lo <= m2[k] <= hi):
                        implausible_2d[k] += 1
            if m3:
                coverage_3d += 1
                for k in ("knee", "trunk", "arm"):
                    by_type_3d[st][k].append(m3[k])
                    lo, hi = PLAUSIBLE[k]
                    if not (lo <= m3[k] <= hi):
                        implausible_3d[k] += 1
            if m2 and m3:
                total_graded += 1
                if g2 != g3:
                    grade_disagreement += 1
                    grade_shift[f"{g2}→{g3}"] += 1

    # Print report
    print(f"\n{'='*80}")
    print(f"3D vs 2D ANGLE VALIDATION — {len(results)} videos, {total_shots} shots")
    print(f"{'='*80}\n")

    print(f"Coverage (shots with valid measurement):")
    print(f"  2D:  {coverage_2d} / {total_shots}  ({coverage_2d/total_shots:.0%})")
    print(f"  3D:  {coverage_3d} / {total_shots}  ({coverage_3d/total_shots:.0%})")
    print()

    print(f"Implausible readings (outside anatomical ranges):")
    for k in ("knee", "trunk", "arm"):
        lo, hi = PLAUSIBLE[k]
        print(f"  {k:5s} (plausible {lo}-{hi}°):  2D={implausible_2d[k]:3d}  3D={implausible_3d[k]:3d}")
    print()

    print(f"Per-shot-type distributions:")
    for st in sorted(set(list(by_type_2d) + list(by_type_3d))):
        ideals = IDEALS.get(st, {})
        print(f"\n  === {st} === (ideal knee={ideals.get('knee','-')}°  "
              f"trunk={ideals.get('trunk','-')}°  arm={ideals.get('arm','-')}°)")
        for k in ("knee", "trunk", "arm"):
            v2 = by_type_2d[st][k]; v3 = by_type_3d[st][k]
            print(f"    {k:5s}  2D: {fmt_stats(v2)}")
            print(f"           3D: {fmt_stats(v3)}")

    print(f"\nGrade-changing disagreements (2D vs 3D):")
    print(f"  {grade_disagreement} / {total_graded} shots differ  "
          f"({grade_disagreement/max(1,total_graded):.0%})")
    for shift, n in sorted(grade_shift.items(), key=lambda x: -x[1])[:15]:
        print(f"    {shift}: {n}")

    # Cross-angle consistency: same shot type across multiple videos
    print(f"\nCross-video per-type variance (lower variance = more camera-invariant):")
    by_vid_type_mean_knee_2d = defaultdict(lambda: defaultdict(list))
    by_vid_type_mean_knee_3d = defaultdict(lambda: defaultdict(list))
    for r in results:
        for st, m2, m3, _, _ in r["shots"]:
            if m2: by_vid_type_mean_knee_2d[st][r["vid"]].append(m2["knee"])
            if m3: by_vid_type_mean_knee_3d[st][r["vid"]].append(m3["knee"])
    for st in sorted(by_vid_type_mean_knee_2d):
        means_2d = [statistics.mean(v) for v in by_vid_type_mean_knee_2d[st].values() if v]
        means_3d = [statistics.mean(v) for v in by_vid_type_mean_knee_3d[st].values() if v]
        if len(means_2d) < 2 or len(means_3d) < 2:
            continue
        sd2 = statistics.stdev(means_2d); sd3 = statistics.stdev(means_3d)
        print(f"  {st:20s}  knee cross-video stdev: 2D={sd2:5.1f}°  3D={sd3:5.1f}°  "
              f"{'(3D more consistent ✓)' if sd3 < sd2 else '(2D more consistent)'}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"results": results, "implausible_2d": dict(implausible_2d),
                       "implausible_3d": dict(implausible_3d),
                       "grade_shift": dict(grade_shift)}, f, indent=2, default=str)
        print(f"\nFull report → {args.json}")


if __name__ == "__main__":
    main()
