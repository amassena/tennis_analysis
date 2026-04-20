#!/usr/bin/env python3
"""Compute calibrated ball speed and line calls using court homography.

Combines:
  - Ball tracking data (TrackNet per-frame ball positions)
  - Court calibration (homography from court_detect.py)
  - Shot detections (timestamps of each shot)

Outputs per-shot:
  - ball_speed_mph: calibrated speed at contact
  - ball_landing: (x, y) in court coordinates (meters)
  - line_call: "IN" or "OUT" with margin
  - shot_depth: distance from net to landing point

Usage:
    python scripts/shot_speed.py --video IMG_1141
    python scripts/shot_speed.py --all
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT_ROOT = Path(__file__).parent.parent
BALL_TRACKING_DIR = PROJECT_ROOT / "ball_tracking"
COURT_DIR = PROJECT_ROOT / "court_calibrations"
DETECTIONS_DIR = PROJECT_ROOT / "detections"

from scripts.court_detect import pixel_to_court, is_ball_in, COURT


def load_data(vid):
    """Load ball tracking, court calibration, and detection data."""
    ball_path = BALL_TRACKING_DIR / f"{vid}.json"
    court_path = COURT_DIR / f"{vid}.json"
    det_path = DETECTIONS_DIR / f"{vid}_fused_detections.json"
    if not det_path.exists():
        det_path = DETECTIONS_DIR / f"{vid}_fused.json"

    data = {}
    for name, path in [("ball", ball_path), ("court", court_path), ("det", det_path)]:
        if path.exists():
            with open(path) as f:
                data[name] = json.load(f)
        else:
            print(f"  Missing {name}: {path}")
            data[name] = None
    return data


def compute_shot_speeds(vid, output_path=None):
    """Compute ball speed and line calls for each detected shot."""
    data = load_data(vid)
    if not data["ball"] or not data["court"] or not data["det"]:
        print(f"  Skipping {vid}: missing data")
        return None

    H = np.array(data["court"]["homography"], dtype=np.float64)
    fps = data["det"].get("fps", 60.0)
    ball_frames = {f["frame"]: f for f in data["ball"].get("frames", [])
                   if f.get("x") is not None}

    results = []
    for d in data["det"].get("detections", []):
        frame = int(d.get("frame", 0))
        shot_type = d.get("shot_type", "unknown")

        # Find ball positions around contact for speed calculation
        # Look at frames [contact-5, contact+5] for the fastest segment
        speeds = []
        for offset in range(-5, 5):
            f1 = ball_frames.get(frame + offset)
            f2 = ball_frames.get(frame + offset + 1)
            if f1 and f2 and f1.get("x") and f2.get("x"):
                try:
                    x1, y1 = pixel_to_court(H, f1["x"], f1["y"])
                    x2, y2 = pixel_to_court(H, f2["x"], f2["y"])
                    dist = math.sqrt((x2-x1)**2 + (y2-y1)**2)
                    speed_ms = dist * fps  # meters per second
                    speed_mph = speed_ms * 2.237
                    # Per-shot-type plausible range (mph)
                    # Serves can be 70-140 mph; groundstrokes 30-80 mph
                    max_speed = 140 if shot_type == "serve" else 85
                    if 5 < speed_mph < max_speed:
                        speeds.append(speed_mph)
                except Exception:
                    pass

        ball_speed_mph = round(max(speeds), 1) if speeds else None

        # Find ball landing point (first frame after contact where ball
        # trajectory changes from descending to ascending = bounce)
        landing = None
        line_call = None
        margin = None
        for bounce_offset in range(3, 30):
            bf = ball_frames.get(frame + bounce_offset)
            bf_prev = ball_frames.get(frame + bounce_offset - 1)
            bf_next = ball_frames.get(frame + bounce_offset + 1)
            if bf and bf_prev and bf_next:
                # Ball descending then ascending = bounce
                if bf_prev["y"] < bf["y"] and bf_next["y"] < bf["y"]:
                    try:
                        cx, cy = pixel_to_court(H, bf["x"], bf["y"])
                        is_in, m = is_ball_in(cx, cy)
                        landing = {"x": round(cx, 2), "y": round(cy, 2)}
                        line_call = "IN" if is_in else "OUT"
                        margin = round(m, 3)
                    except Exception:
                        pass
                    break

        result = {
            "shot_idx": len(results),
            "frame": frame,
            "timestamp": round(frame / fps, 2),
            "shot_type": shot_type,
            "ball_speed_mph": ball_speed_mph,
            "landing": landing,
            "line_call": line_call,
            "margin_meters": margin,
        }
        if landing:
            result["shot_depth"] = round(abs(landing["y"]), 2)

        results.append(result)

    # Summary
    speeds = [r["ball_speed_mph"] for r in results if r["ball_speed_mph"]]
    landings = [r for r in results if r["landing"]]
    in_count = sum(1 for r in results if r["line_call"] == "IN")
    out_count = sum(1 for r in results if r["line_call"] == "OUT")

    output = {
        "video": vid,
        "total_shots": len(results),
        "shots_with_speed": len(speeds),
        "avg_speed_mph": round(sum(speeds)/len(speeds), 1) if speeds else None,
        "max_speed_mph": round(max(speeds), 1) if speeds else None,
        "shots_with_landing": len(landings),
        "in_count": in_count,
        "out_count": out_count,
        "shots": results,
    }

    if output_path:
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"  {vid}: {len(speeds)} speeds (avg {output['avg_speed_mph']} mph, "
              f"max {output['max_speed_mph']} mph), "
              f"{in_count} IN / {out_count} OUT")

    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", help="Video ID")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / "shot_analysis"
    out_dir.mkdir(exist_ok=True)

    if args.all:
        for ball_file in sorted(BALL_TRACKING_DIR.glob("*.json")):
            vid = ball_file.stem
            court_file = COURT_DIR / f"{vid}.json"
            if not court_file.exists():
                continue
            compute_shot_speeds(vid, str(out_dir / f"{vid}.json"))
    elif args.video:
        compute_shot_speeds(args.video, str(out_dir / f"{args.video}.json"))
    else:
        parser.error("Provide --video or --all")


if __name__ == "__main__":
    main()
