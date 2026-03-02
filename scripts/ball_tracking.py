#!/usr/bin/env python3
"""Ball tracking using TrackNet for tennis shot analysis.

Runs TrackNet on video frames to detect ball position per-frame,
then enriches detection JSONs with ball-derived features.

IMPORTANT: Run this on Windows GPU machine, not Mac.

Setup:
    1. Clone TrackNet: git clone <tracknet-repo> models/tracknet/
    2. Download pretrained weights to models/tracknet/
    3. pip install torch torchvision opencv-python

Usage:
    # Track ball in a video
    python scripts/ball_tracking.py preprocessed/IMG_6703.mp4

    # Enrich detection JSON with ball features
    python scripts/ball_tracking.py preprocessed/IMG_6703.mp4 \
        --detections detections/IMG_6703_fused_v5.json

    # Process all videos
    python scripts/ball_tracking.py --all
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import PROJECT_ROOT, BALL_TRACKING_DIR, PREPROCESSED_DIR

DETECTIONS_DIR = os.path.join(PROJECT_ROOT, "detections")


def track_ball(video_path, output_path=None, model_path=None):
    """Run TrackNet ball tracking on a video.

    Args:
        video_path: Path to preprocessed video
        output_path: Output JSON path (default: ball_tracking/{VIDEO}.json)
        model_path: Path to TrackNet weights

    Returns:
        List of per-frame dicts: {frame, x, y, confidence, visible}
    """
    try:
        import cv2
        import torch
    except ImportError:
        print("Required: pip install torch torchvision opencv-python")
        sys.exit(1)

    video_name = Path(video_path).stem
    if output_path is None:
        os.makedirs(BALL_TRACKING_DIR, exist_ok=True)
        output_path = os.path.join(BALL_TRACKING_DIR, f"{video_name}.json")

    # TODO: Load TrackNet model and run inference
    # For now, output format specification:
    print(f"  Ball tracking for {video_name}...")
    print(f"  TrackNet model path: {model_path or 'models/tracknet/'}")
    print(f"  This script requires TrackNet setup. See docstring for instructions.")
    print(f"  Output will be saved to: {output_path}")

    # Placeholder structure
    result = {
        "video": video_name,
        "video_path": str(video_path),
        "frames": [],  # Per-frame: {frame, x, y, confidence, visible}
        "status": "pending_setup",
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    return result


def enrich_detections(det_path, ball_path):
    """Add ball-derived features to detection JSON.

    New features per detection:
        - ball_speed_at_contact: Ball speed near contact frame
        - ball_direction_angle: Ball trajectory angle at contact
        - ball_near_wrist: Distance between ball and wrist at contact
        - ball_height_at_contact: Ball Y position at contact
        - ball_trajectory_change: Angle change through contact (indicates hit)
        - ball_visible_at_contact: Whether ball is detected at contact
    """
    with open(det_path) as f:
        det_data = json.load(f)
    with open(ball_path) as f:
        ball_data = json.load(f)

    if ball_data.get("status") == "pending_setup":
        print(f"  Ball tracking not yet run. Skipping enrichment.")
        return det_data

    ball_frames = {f["frame"]: f for f in ball_data.get("frames", [])}
    fps = det_data.get("fps", 60.0)

    for det in det_data.get("detections", []):
        frame = det.get("frame", 0)

        # Find ball position at and around contact
        ball_at = ball_frames.get(frame, {})
        ball_before = ball_frames.get(frame - 3, {})
        ball_after = ball_frames.get(frame + 3, {})

        visible = ball_at.get("visible", False)
        det["ball_visible_at_contact"] = 1.0 if visible else 0.0
        det["ball_height_at_contact"] = ball_at.get("y", 0.0) if visible else 0.0

        # Ball speed: distance between before and after positions
        if ball_before.get("visible") and ball_after.get("visible"):
            dx = ball_after["x"] - ball_before["x"]
            dy = ball_after["y"] - ball_before["y"]
            dist = math.sqrt(dx * dx + dy * dy)
            det["ball_speed_at_contact"] = round(dist * fps / 6, 2)
            det["ball_direction_angle"] = round(math.degrees(math.atan2(dy, dx)), 1)
        else:
            det["ball_speed_at_contact"] = 0.0
            det["ball_direction_angle"] = 0.0

        # Ball near wrist: would need pose data too
        det["ball_near_wrist"] = 0.0

        # Trajectory change through contact
        if (ball_before.get("visible") and ball_at.get("visible")
                and ball_after.get("visible")):
            pre_angle = math.atan2(
                ball_at["y"] - ball_before["y"],
                ball_at["x"] - ball_before["x"]
            )
            post_angle = math.atan2(
                ball_after["y"] - ball_at["y"],
                ball_after["x"] - ball_at["x"]
            )
            det["ball_trajectory_change"] = round(
                abs(math.degrees(post_angle - pre_angle)), 1
            )
        else:
            det["ball_trajectory_change"] = 0.0

    # Save enriched detections
    with open(det_path, "w") as f:
        json.dump(det_data, f, indent=2)
    print(f"  Enriched {len(det_data.get('detections', []))} detections with ball features")

    return det_data


def main():
    parser = argparse.ArgumentParser(description="Ball tracking with TrackNet")
    parser.add_argument("video", nargs="?", help="Video file to process")
    parser.add_argument("--detections", help="Detection JSON to enrich")
    parser.add_argument("--model", help="TrackNet model path")
    parser.add_argument("--all", action="store_true", help="Process all videos")
    parser.add_argument("--output", help="Output path override")
    args = parser.parse_args()

    if args.all:
        for vf in sorted(os.listdir(PREPROCESSED_DIR)):
            if not vf.endswith(".mp4"):
                continue
            video_path = os.path.join(PREPROCESSED_DIR, vf)
            track_ball(video_path, model_path=args.model)
        return

    if not args.video:
        parser.error("Provide a video path or use --all")

    result = track_ball(args.video, output_path=args.output, model_path=args.model)

    if args.detections:
        video_name = Path(args.video).stem
        ball_path = os.path.join(BALL_TRACKING_DIR, f"{video_name}.json")
        if os.path.exists(ball_path):
            enrich_detections(args.detections, ball_path)


if __name__ == "__main__":
    main()
