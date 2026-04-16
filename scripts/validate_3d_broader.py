#!/usr/bin/env python3
"""Broader validation of 3D vs 2D angle computation.

Tests the REAL value propositions of 3D, not just internal consistency:

1. Camera-invariance: same player, same shot type, different camera angles.
   2D should swing wildly; 3D should stay similar. Uses known labeled
   camera angles (IMG_6713 = left-side, IMG_0991/0994/0996/1003 = front-facing).

2. Temporal stability: does 3D jitter frame-to-frame during a single shot?
   For each shot, compute angles at contact_frame ± 0,1,2,3 frames.
   Report intra-shot stdev per metric per mode.

3. Pose-quality sensitivity: bin shots by MediaPipe visibility score at
   contact. Report angle stdev per bin. Healthy signal: accuracy roughly
   constant across confidence bins; red flag: accuracy degrades badly
   at low confidence.

4. Known-bad cases: IMG_1001 (into-the-sun, terrible pose detection) —
   how noisy are BOTH 2D and 3D there?
"""
import argparse
import json
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.render_graded import compute_shot_metrics

PROJECT_ROOT = Path(__file__).parent.parent
DETECTIONS_DIR = PROJECT_ROOT / "detections"
POSES_DIR = PROJECT_ROOT / "poses_full_videos"

# Camera angle labels from MEMORY
CAMERA_ANGLE = {
    "IMG_6713": "left-side",
    "IMG_0991": "front-facing",
    "IMG_0994": "front-facing",
    "IMG_0995": "front-facing",
    "IMG_0996": "front-facing",
    "IMG_0997": "front-facing",
    "IMG_0999": "front-facing",
    "IMG_1001": "front-facing-into-sun",
    "IMG_1003": "front-facing",
    "IMG_1004": "front-right",
    "IMG_1005": "side_right",
    "IMG_1007": "front_right",
    "IMG_1008": "front-right",
    "IMG_6665": "back-court",
    "IMG_0864": "back-court",
    "IMG_0865": "back-court",
    "IMG_0866": "back-court",
    "IMG_0867": "back-court",
    "IMG_0868": "back-court",
    "IMG_0869": "back-court",
    "IMG_0870": "back-court",
}


def load_det(vid):
    for n in [f"{vid}_fused.json", f"{vid}_fused_detections.json"]:
        p = DETECTIONS_DIR / n
        if p.exists():
            with open(p) as f: return json.load(f)
    return None


def load_poses(vid):
    p = POSES_DIR / f"{vid}.json"
    if p.exists():
        with open(p) as f: return json.load(f)
    return None


def frame_visibility(pf):
    """Average visibility of key tennis landmarks at a frame."""
    if not pf or not pf.get("detected"):
        return 0.0
    lms = pf.get("landmarks")
    if not lms: return 0.0
    # key landmarks: shoulders(11,12), elbows(13,14), wrists(15,16),
    # hips(23,24), knees(25,26), ankles(27,28)
    KEY = [11,12,13,14,15,16,23,24,25,26,27,28]
    vals = []
    for i in KEY:
        if i >= len(lms): continue
        e = lms[i]
        if isinstance(e, dict):
            vals.append(e.get("visibility", 0))
        elif len(e) >= 4:
            vals.append(e[3])
    return statistics.mean(vals) if vals else 0.0


def stats_block(values, label=""):
    if not values: return f"{label}  n=0"
    return (f"{label}  n={len(values):4d}  "
            f"mean={statistics.mean(values):6.1f}  "
            f"stdev={statistics.stdev(values) if len(values)>1 else 0:5.1f}  "
            f"median={statistics.median(values):6.1f}  "
            f"p10={sorted(values)[int(len(values)*0.1)]:.0f}  "
            f"p90={sorted(values)[int(len(values)*0.9)]:.0f}")


def test_camera_invariance():
    """Same player's forehands/backhands across different camera angles."""
    print("\n" + "="*80)
    print("TEST 1: CAMERA INVARIANCE")
    print("="*80)
    print("Question: for the same shot type on the same player, do 3D angles")
    print("stay similar across camera angles? (Lower cross-camera stdev = better)\n")

    # Group: (shot_type, camera_angle_family) → [all angle values from any video
    # in that group]
    for shot_type in ["forehand", "backhand", "serve"]:
        print(f"\n--- {shot_type} ---")
        by_cam_2d = defaultdict(lambda: defaultdict(list))
        by_cam_3d = defaultdict(lambda: defaultdict(list))
        for vid, cam in CAMERA_ANGLE.items():
            det = load_det(vid); poses = load_poses(vid)
            if not det or not poses: continue
            for d in det.get("detections", []):
                if d.get("shot_type") != shot_type: continue
                cf = int(d.get("frame", 0))
                m2 = compute_shot_metrics(poses["frames"], cf, shot_type, use_3d=False)
                m3 = compute_shot_metrics(poses["frames"], cf, shot_type, use_3d=True)
                for metric in ("knee", "trunk", "arm"):
                    if m2: by_cam_2d[metric][cam].append(m2[metric])
                    if m3: by_cam_3d[metric][cam].append(m3[metric])

        for metric in ("knee", "trunk", "arm"):
            if len(by_cam_2d[metric]) < 2: continue
            # Cross-camera stdev = stdev of the mean angle per camera angle
            means_2d = [statistics.mean(v) for v in by_cam_2d[metric].values() if v]
            means_3d = [statistics.mean(v) for v in by_cam_3d[metric].values() if v]
            if len(means_2d) < 2 or len(means_3d) < 2: continue
            sd2 = statistics.stdev(means_2d); sd3 = statistics.stdev(means_3d)
            winner = "3D ✓" if sd3 < sd2 else "2D" if sd2 < sd3 else "tie"
            print(f"  {metric:5s}  cross-camera stdev: 2D={sd2:5.1f}°  3D={sd3:5.1f}°  → {winner}")
            for cam in sorted(by_cam_2d[metric]):
                v2 = by_cam_2d[metric][cam]; v3 = by_cam_3d[metric][cam]
                if not v2: continue
                print(f"    {cam:25s}  2D: mean={statistics.mean(v2):5.1f} n={len(v2):3d}  "
                      f"3D: mean={statistics.mean(v3):5.1f} n={len(v3):3d}")


def test_temporal_stability():
    """For each shot, compute angles at ±0,1,2,3 frames. Lower intra-shot stdev = more stable."""
    print("\n" + "="*80)
    print("TEST 2: TEMPORAL STABILITY (intra-shot jitter)")
    print("="*80)
    print("Question: for a single shot at its contact frame, do 3D angles jitter more")
    print("or less than 2D across adjacent frames? Lower stdev = more stable measurement.\n")

    # Sample a subset to keep runtime reasonable
    jitter_2d = defaultdict(list)
    jitter_3d = defaultdict(list)

    for vid in list(CAMERA_ANGLE.keys())[:10]:
        det = load_det(vid); poses = load_poses(vid)
        if not det or not poses: continue
        for d in det.get("detections", [])[:30]:
            cf = int(d.get("frame", 0)); st = d.get("shot_type", "unknown")
            vals_2d = {"knee": [], "trunk": [], "arm": []}
            vals_3d = {"knee": [], "trunk": [], "arm": []}
            for delta in (-3, -2, -1, 0, 1, 2, 3):
                m2 = compute_shot_metrics(poses["frames"], cf + delta, st, use_3d=False)
                m3 = compute_shot_metrics(poses["frames"], cf + delta, st, use_3d=True)
                if m2:
                    for k in ("knee","trunk","arm"): vals_2d[k].append(m2[k])
                if m3:
                    for k in ("knee","trunk","arm"): vals_3d[k].append(m3[k])
            for k in ("knee","trunk","arm"):
                if len(vals_2d[k]) >= 3:
                    jitter_2d[k].append(statistics.stdev(vals_2d[k]))
                if len(vals_3d[k]) >= 3:
                    jitter_3d[k].append(statistics.stdev(vals_3d[k]))

    print(f"  Per-shot stdev of angle across ±3 frames:")
    for k in ("knee", "trunk", "arm"):
        if not jitter_2d[k] or not jitter_3d[k]: continue
        m2 = statistics.mean(jitter_2d[k]); m3 = statistics.mean(jitter_3d[k])
        winner = "3D smoother ✓" if m3 < m2 else "2D smoother"
        print(f"    {k:5s}  2D mean-stdev={m2:4.1f}°  3D mean-stdev={m3:4.1f}°  → {winner}")


def test_pose_quality():
    """Bin shots by MediaPipe visibility at contact. See if measurement quality degrades."""
    print("\n" + "="*80)
    print("TEST 3: POSE-QUALITY SENSITIVITY")
    print("="*80)
    print("Question: as MediaPipe detection confidence drops, does 3D degrade more")
    print("than 2D? Group shots by average visibility at contact frame.\n")

    # Bins: high (>0.85), medium (0.6-0.85), low (<0.6)
    bins_2d = defaultdict(lambda: defaultdict(list))  # bin → metric → vals
    bins_3d = defaultdict(lambda: defaultdict(list))
    for vid in CAMERA_ANGLE:
        det = load_det(vid); poses = load_poses(vid)
        if not det or not poses: continue
        for d in det.get("detections", []):
            cf = int(d.get("frame", 0)); st = d.get("shot_type", "unknown")
            if cf >= len(poses["frames"]): continue
            v = frame_visibility(poses["frames"][cf])
            if v >= 0.85: b = "high (>0.85)"
            elif v >= 0.60: b = "med (0.60-0.85)"
            else: b = "low (<0.60)"
            m2 = compute_shot_metrics(poses["frames"], cf, st, use_3d=False)
            m3 = compute_shot_metrics(poses["frames"], cf, st, use_3d=True)
            for k in ("knee","trunk","arm"):
                if m2: bins_2d[b][k].append(m2[k])
                if m3: bins_3d[b][k].append(m3[k])

    for bin_name in ("high (>0.85)", "med (0.60-0.85)", "low (<0.60)"):
        if not bins_2d[bin_name]["knee"]: continue
        print(f"\n  --- {bin_name} ---  n={len(bins_2d[bin_name]['knee'])}")
        for k in ("knee", "trunk", "arm"):
            v2 = bins_2d[bin_name][k]; v3 = bins_3d[bin_name][k]
            if not v2 or not v3: continue
            sd2 = statistics.stdev(v2) if len(v2)>1 else 0
            sd3 = statistics.stdev(v3) if len(v3)>1 else 0
            print(f"    {k:5s}  2D stdev={sd2:5.1f}°  3D stdev={sd3:5.1f}°")


def test_known_bad():
    """IMG_1001 is documented as 'terrible pose detection' due to into-sun lighting."""
    print("\n" + "="*80)
    print("TEST 4: KNOWN-BAD CASE (IMG_1001, into-the-sun lighting)")
    print("="*80)
    vid = "IMG_1001"
    det = load_det(vid); poses = load_poses(vid)
    if not det or not poses:
        print(f"  IMG_1001 not available locally"); return

    # Stats for all detected shots
    shots = det.get("detections", [])
    print(f"  {len(shots)} shots in IMG_1001")
    vis_scores = []
    for d in shots:
        cf = int(d.get("frame", 0))
        if cf < len(poses["frames"]):
            vis_scores.append(frame_visibility(poses["frames"][cf]))
    if vis_scores:
        print(f"  MediaPipe visibility at contact — "
              f"mean={statistics.mean(vis_scores):.2f} "
              f"median={statistics.median(vis_scores):.2f} "
              f"min={min(vis_scores):.2f}")

    for use_3d in (False, True):
        label = "3D" if use_3d else "2D"
        stats = defaultdict(list)
        valid = 0
        for d in shots:
            cf = int(d.get("frame", 0)); st = d.get("shot_type", "unknown")
            m = compute_shot_metrics(poses["frames"], cf, st, use_3d=use_3d)
            if m:
                valid += 1
                for k in ("knee","trunk","arm"):
                    stats[k].append(m[k])
        print(f"  {label}: {valid}/{len(shots)} valid measurements")
        for k in ("knee","trunk","arm"):
            if stats[k]:
                print(f"    {stats_block(stats[k], label=f'{k:5s}')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip", nargs="+", default=[],
                    choices=["camera","temporal","quality","known-bad"])
    args = ap.parse_args()

    if "camera" not in args.skip:
        test_camera_invariance()
    if "temporal" not in args.skip:
        test_temporal_stability()
    if "quality" not in args.skip:
        test_pose_quality()
    if "known-bad" not in args.skip:
        test_known_bad()


if __name__ == "__main__":
    main()
