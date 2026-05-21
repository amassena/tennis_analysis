#!/usr/bin/env python3
"""Phase 1 audit: flag shots where biomech outputs look implausible.

Runs the production biomech analysis (which prefers MediaPipe's
`world_landmarks` over 2D `landmarks`) on a corpus of GT shots, then flags
any shot hitting a heuristic-implausibility criterion. Output is a JSON
structure that can be diffed/rendered downstream.

Locked criteria (do not edit post-hoc — see eval/3d-lifting/README.md):
  - knee_angle_at_contact outside [120°, 180°]
  - trunk_rotation_at_contact magnitude > 90°
  - arm_extension_at_contact outside [60°, 180°]
  - kinetic_chain_timing_ms with reversed order (any later-segment
    arrival before an earlier one in hip→shoulder→elbow→wrist)
  - Any of the above scalar fields reported as exactly 0.0 or 180.0
    (degenerate signal — math collapsed to a default)

Usage:
    python scripts/audit_world_landmarks.py \\
        --videos IMG_0996 IMG_6874 IMG_6851 \\
        --output eval/3d-lifting/audit_phase1.json
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.biomechanical_analysis import analyze_shot

PROJECT_ROOT = "/Users/andrewhome/tennis_analysis"
POSES_DIR = os.path.join(PROJECT_ROOT, "poses_full_videos")
DETECTIONS_DIR = os.path.join(PROJECT_ROOT, "detections")

ANALYZABLE = {"forehand", "backhand", "serve"}

# Locked plausibility ranges
KNEE_RANGE = (120.0, 180.0)
TRUNK_ROT_MAX = 90.0
ARM_EXT_RANGE = (60.0, 180.0)


def is_degenerate(v):
    """Return True if v is exactly 0.0 or exactly 180.0 (degenerate marker)."""
    if not isinstance(v, (int, float)):
        return False
    if isinstance(v, bool):
        return False
    return v == 0.0 or v == 180.0


def kinetic_chain_reversed(timing):
    """Return True if hip→shoulder→elbow→wrist order isn't monotonic.

    timing is a dict {hip, shoulder, elbow, wrist} of ms offsets.
    """
    if not isinstance(timing, dict):
        return False
    order = ["hip", "shoulder", "elbow", "wrist"]
    vals = [timing.get(k) for k in order]
    if any(v is None for v in vals):
        return False
    # Allow ±10ms tolerance — equal-time stages are fine, slight noise OK
    for i in range(len(vals) - 1):
        if vals[i + 1] < vals[i] - 10:
            return True
    return False


def flag_shot(biomech):
    """Return list of flag reasons (empty = clean shot)."""
    flags = []

    knee = biomech.get("knee_angle_at_contact")
    trunk = biomech.get("trunk_rotation_at_contact")
    arm = biomech.get("arm_extension_at_contact")
    timing = biomech.get("kinetic_chain_timing_ms")

    if isinstance(knee, (int, float)) and not isinstance(knee, bool):
        if is_degenerate(knee):
            flags.append("knee_degenerate")
        elif knee < KNEE_RANGE[0] or knee > KNEE_RANGE[1]:
            flags.append(f"knee_out_of_range:{knee:.1f}")

    if isinstance(trunk, (int, float)) and not isinstance(trunk, bool):
        if is_degenerate(trunk):
            flags.append("trunk_degenerate")
        elif abs(trunk) > TRUNK_ROT_MAX:
            flags.append(f"trunk_extreme:{trunk:.1f}")

    if isinstance(arm, (int, float)) and not isinstance(arm, bool):
        if is_degenerate(arm):
            flags.append("arm_degenerate")
        elif arm < ARM_EXT_RANGE[0] or arm > ARM_EXT_RANGE[1]:
            flags.append(f"arm_out_of_range:{arm:.1f}")

    if kinetic_chain_reversed(timing):
        flags.append(f"chain_reversed:{timing}")

    return flags


def dense_frames(frames, total):
    by_idx = {f["frame_idx"]: f for f in frames}
    return [
        by_idx.get(i, {"frame_idx": i, "detected": False,
                       "landmarks": None, "world_landmarks": None})
        for i in range(total)
    ]


def audit_video(vid):
    pose_path = os.path.join(POSES_DIR, f"{vid}.json")
    gt_path = os.path.join(DETECTIONS_DIR, f"{vid}_fused.json")
    if not (os.path.exists(pose_path) and os.path.exists(gt_path)):
        return {"video": vid, "error": "missing pose or GT"}

    with open(pose_path) as f:
        pose = json.load(f)
    with open(gt_path) as f:
        gt = json.load(f)

    fps = gt.get("fps", 60.0)
    dominant_hand = gt.get("dominant_hand", "right")
    total = pose["video_info"]["total_frames"]

    frames = dense_frames(pose["frames"], total)
    has_world = any(
        f.get("detected") and f.get("world_landmarks") is not None
        for f in pose["frames"][:200]
    )

    shots_audited = []
    for shot in gt.get("detections", []):
        st = shot.get("shot_type")
        if st not in ANALYZABLE:
            continue
        cf = int(shot.get("frame", 0))
        try:
            biomech = analyze_shot(frames, cf, fps, dominant_hand=dominant_hand)
        except Exception as e:
            shots_audited.append({
                "frame": cf, "shot_type": st,
                "error": f"analyze_shot failed: {e}",
                "flags": ["analyze_shot_exception"],
            })
            continue
        flags = flag_shot(biomech)
        shots_audited.append({
            "frame": cf,
            "timestamp": shot.get("timestamp"),
            "shot_type": st,
            "flags": flags,
            "knee_angle_at_contact": biomech.get("knee_angle_at_contact"),
            "trunk_rotation_at_contact": biomech.get("trunk_rotation_at_contact"),
            "arm_extension_at_contact": biomech.get("arm_extension_at_contact"),
            "kinetic_chain_timing_ms": biomech.get("kinetic_chain_timing_ms"),
            "knee_bend_depth": biomech.get("knee_bend_depth"),
            "peak_swing_speed": biomech.get("peak_swing_speed"),
            "contact_swing_speed": biomech.get("contact_swing_speed"),
        })

    return {
        "video": vid,
        "fps": fps,
        "dominant_hand": dominant_hand,
        "has_world_landmarks": has_world,
        "shots_audited": shots_audited,
    }


def aggregate(reports):
    by_flag = Counter()
    by_shot_type = defaultdict(lambda: {"total": 0, "flagged": 0})
    total_shots = 0
    total_flagged = 0
    for r in reports:
        if "error" in r:
            continue
        for s in r["shots_audited"]:
            total_shots += 1
            by_shot_type[s["shot_type"]]["total"] += 1
            if s["flags"]:
                total_flagged += 1
                by_shot_type[s["shot_type"]]["flagged"] += 1
                # Count distinct flag categories (strip the numeric suffix
                # so "knee_out_of_range:127.3" -> "knee_out_of_range")
                for f in s["flags"]:
                    by_flag[f.split(":")[0]] += 1
    return {
        "total_shots": total_shots,
        "total_flagged": total_flagged,
        "flag_rate": (total_flagged / total_shots) if total_shots else 0,
        "by_flag_category": dict(by_flag),
        "by_shot_type": {k: dict(v) for k, v in by_shot_type.items()},
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--videos", nargs="+", required=True,
                   help="Video IDs (e.g. IMG_0996)")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    print(f"[audit] analyzing {len(args.videos)} videos", flush=True)
    reports = []
    for vid in args.videos:
        print(f"[audit] {vid} ...", flush=True)
        r = audit_video(vid)
        reports.append(r)
        if "error" not in r:
            n_total = len(r["shots_audited"])
            n_flagged = sum(1 for s in r["shots_audited"] if s["flags"])
            print(f"        {n_flagged}/{n_total} shots flagged", flush=True)

    summary = {
        "videos": reports,
        "aggregate": aggregate(reports),
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[audit] wrote {args.output}", flush=True)

    agg = summary["aggregate"]
    print()
    print("=== Aggregate ===")
    print(f"Total shots: {agg['total_shots']}")
    print(f"Flagged:     {agg['total_flagged']} ({agg['flag_rate']*100:.1f}%)")
    print(f"By flag category:")
    for cat, n in sorted(agg["by_flag_category"].items(), key=lambda x: -x[1]):
        print(f"  {cat:30s} {n}")
    print(f"By shot type:")
    for st, stats in agg["by_shot_type"].items():
        rate = stats["flagged"] / stats["total"] if stats["total"] else 0
        print(f"  {st:10s} {stats['flagged']}/{stats['total']} ({rate*100:.0f}%)")


if __name__ == "__main__":
    main()
