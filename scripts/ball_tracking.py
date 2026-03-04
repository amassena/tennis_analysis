#!/usr/bin/env python3
"""Ball tracking using TrackNet for tennis shot analysis.

Runs TrackNet on video frames to detect ball position per-frame,
then enriches detection JSONs with ball-derived features.

IMPORTANT: Run this on Windows GPU machine, not Mac.

Requirements: pip install torch torchvision opencv-python scipy

Usage:
    # Track ball in a video
    python scripts/ball_tracking.py preprocessed/IMG_6703.mp4

    # Enrich detection JSON with ball features
    python scripts/ball_tracking.py preprocessed/IMG_6703.mp4 \
        --detections detections/IMG_6703_fused.json

    # Process all videos
    python scripts/ball_tracking.py --all
"""

import argparse
import json
import math
import os
import sys
from itertools import groupby
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import PROJECT_ROOT, BALL_TRACKING_DIR, PREPROCESSED_DIR

DETECTIONS_DIR = os.path.join(PROJECT_ROOT, "detections")
TRACKNET_DIR = os.path.join(PROJECT_ROOT, "models", "tracknet")
DEFAULT_WEIGHTS = os.path.join(TRACKNET_DIR, "tracknet_weights.pt")

# TrackNet input resolution
TN_WIDTH = 640
TN_HEIGHT = 360


def _postprocess(feature_map, scale=2):
    """Convert TrackNet heatmap output to ball coordinates.

    Returns (x, y) in original video coordinates, or (None, None) if not detected.
    """
    import cv2

    feature_map = (feature_map * 255).reshape((TN_HEIGHT, TN_WIDTH)).astype(np.uint8)
    _, heatmap = cv2.threshold(feature_map, 127, 255, cv2.THRESH_BINARY)
    circles = cv2.HoughCircles(
        heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=1,
        param1=50, param2=2, minRadius=2, maxRadius=7,
    )
    if circles is not None and len(circles) == 1:
        return circles[0][0][0] * scale, circles[0][0][1] * scale
    return None, None


def _remove_outliers(ball_track, dists, max_dist=100):
    """Remove outlier detections based on distance between consecutive points."""
    outliers = list(np.where(np.array(dists) > max_dist)[0])
    for i in outliers:
        if i + 1 < len(dists) and ((dists[i + 1] > max_dist) or (dists[i + 1] == -1)):
            ball_track[i] = (None, None)
        elif i > 0 and dists[i - 1] == -1:
            ball_track[i - 1] = (None, None)
    return ball_track


def _split_track(ball_track, max_gap=4, max_dist_gap=80, min_track=5):
    """Split ball track into subtracks for interpolation."""
    from scipy.spatial import distance

    list_det = [0 if x[0] is not None else 1 for x in ball_track]
    groups = [(k, sum(1 for _ in g)) for k, g in groupby(list_det)]

    cursor = 0
    min_value = 0
    result = []
    for i, (k, length) in enumerate(groups):
        if k == 1 and i > 0 and i < len(groups) - 1:
            if ball_track[cursor - 1][0] is not None and ball_track[cursor + length][0] is not None:
                dist = distance.euclidean(ball_track[cursor - 1], ball_track[cursor + length])
                if length >= max_gap or dist / length > max_dist_gap:
                    if cursor - min_value > min_track:
                        result.append([min_value, cursor])
                    min_value = cursor + length - 1
        cursor += length
    if len(list_det) - min_value > min_track:
        result.append([min_value, len(list_det)])
    return result


def _interpolate(coords):
    """Interpolate missing ball positions within a subtrack."""
    x = np.array([c[0] if c[0] is not None else np.nan for c in coords])
    y = np.array([c[1] if c[1] is not None else np.nan for c in coords])

    nans_x = np.isnan(x)
    nans_y = np.isnan(y)
    if nans_x.all() or (~nans_x).sum() < 2:
        return coords

    idx = np.arange(len(x))
    x[nans_x] = np.interp(idx[nans_x], idx[~nans_x], x[~nans_x])
    y[nans_y] = np.interp(idx[nans_y], idx[~nans_y], y[~nans_y])

    return list(zip(x.tolist(), y.tolist()))


def track_ball(video_path, output_path=None, model_path=None, batch_size=4):
    """Run TrackNet ball tracking on a video.

    Args:
        video_path: Path to preprocessed video
        output_path: Output JSON path (default: ball_tracking/{VIDEO}.json)
        model_path: Path to TrackNet weights
        batch_size: Inference batch size

    Returns:
        dict with video name, frames list, and detection stats
    """
    import cv2
    import torch

    # Add tracknet model dir to path for import
    sys.path.insert(0, TRACKNET_DIR)
    from model import BallTrackerNet

    video_name = Path(video_path).stem
    if output_path is None:
        os.makedirs(BALL_TRACKING_DIR, exist_ok=True)
        output_path = os.path.join(BALL_TRACKING_DIR, f"{video_name}.json")

    if model_path is None:
        model_path = DEFAULT_WEIGHTS
    if not os.path.exists(model_path):
        print(f"  ERROR: TrackNet weights not found at {model_path}")
        print(f"  Download from: https://drive.google.com/file/d/1XEYZ4myUN7QT-NeBYJI0xteLsvs-ZAOl")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Ball tracking {video_name} on {device}...")

    # Load model
    model = BallTrackerNet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Read video
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Scale factor: TrackNet outputs at 2x its input (640x360 -> 1280x720)
    # We need to scale from 1280x720 to actual video resolution
    scale_x = orig_w / (TN_WIDTH * 2)
    scale_y = orig_h / (TN_HEIGHT * 2)

    print(f"  Video: {orig_w}x{orig_h} @ {fps:.1f}fps, {total_frames} frames")

    # Process frames
    from scipy.spatial import distance

    ball_track = [(None, None)] * 2  # First 2 frames have no triplet
    dists = [-1.0] * 2
    prev_frame = None
    preprev_frame = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        resized = cv2.resize(frame, (TN_WIDTH, TN_HEIGHT))

        if frame_idx >= 2:
            # Concatenate 3 consecutive frames (current, prev, preprev)
            imgs = np.concatenate((resized, prev_frame, preprev_frame), axis=2)
            imgs = imgs.astype(np.float32) / 255.0
            imgs = np.rollaxis(imgs, 2, 0)
            inp = np.expand_dims(imgs, axis=0)

            with torch.no_grad():
                out = model(torch.from_numpy(inp).float().to(device))
            output = out.argmax(dim=1).detach().cpu().numpy()
            x_pred, y_pred = _postprocess(output[0])

            # Scale coordinates to original video resolution
            if x_pred is not None:
                x_pred = x_pred * scale_x
                y_pred = y_pred * scale_y

            ball_track.append((x_pred, y_pred))

            if ball_track[-1][0] is not None and ball_track[-2][0] is not None:
                dist = distance.euclidean(ball_track[-1], ball_track[-2])
            else:
                dist = -1.0
            dists.append(dist)
        else:
            ball_track.append((None, None)) if frame_idx < 2 else None
            dists.append(-1.0) if frame_idx < 2 else None

        preprev_frame = prev_frame
        prev_frame = resized
        frame_idx += 1

        if frame_idx % 1000 == 0:
            detected = sum(1 for b in ball_track if b[0] is not None)
            print(f"    Frame {frame_idx}/{total_frames} ({detected} detections)")

    cap.release()

    # Post-process: remove outliers and interpolate
    ball_track = _remove_outliers(ball_track, dists)
    subtracks = _split_track(ball_track)
    for r in subtracks:
        ball_subtrack = ball_track[r[0]:r[1]]
        ball_subtrack = _interpolate(ball_subtrack)
        ball_track[r[0]:r[1]] = ball_subtrack

    # Build output
    detected_count = sum(1 for b in ball_track if b[0] is not None)
    detection_rate = detected_count / len(ball_track) if ball_track else 0

    frames_out = []
    for i, (x, y) in enumerate(ball_track):
        entry = {
            "frame": i,
            "x": round(float(x), 1) if x is not None else None,
            "y": round(float(y), 1) if y is not None else None,
            "visible": x is not None,
        }
        frames_out.append(entry)

    result = {
        "video": video_name,
        "video_path": str(video_path),
        "resolution": [int(orig_w), int(orig_h)],
        "fps": float(fps),
        "total_frames": len(ball_track),
        "detected_frames": detected_count,
        "detection_rate": round(detection_rate, 3),
        "status": "complete",
        "frames": frames_out,
    }

    with open(output_path, "w") as f:
        json.dump(result, f)
    print(f"  Saved {output_path}: {detected_count}/{len(ball_track)} frames detected ({detection_rate:.1%})")

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

    if ball_data.get("status") != "complete":
        print(f"  Ball tracking not complete (status={ball_data.get('status')}). Skipping.")
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
                ball_at["x"] - ball_before["x"],
            )
            post_angle = math.atan2(
                ball_after["y"] - ball_at["y"],
                ball_after["x"] - ball_at["x"],
            )
            det["ball_trajectory_change"] = round(
                abs(math.degrees(post_angle - pre_angle)), 1,
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
            video_name = Path(vf).stem
            out_path = os.path.join(BALL_TRACKING_DIR, f"{video_name}.json")
            if os.path.exists(out_path):
                print(f"  Skipping {video_name} (already exists)")
                continue
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
