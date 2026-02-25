#!/usr/bin/env python3
"""Heuristic-based shot detection as a baseline.

No ML model - just geometric rules based on pose keypoints.
Expected accuracy: ~90% serve, ~70% forehand/backhand.

This serves as:
1. A fallback when the ML model fails
2. A baseline to compare ML model accuracy against
3. A fast alternative that doesn't require GPU

Heuristics:
- SERVE: Wrist raised above head level
- FOREHAND: Dominant wrist crosses body midline (right-to-left for right-handed)
- BACKHAND: Non-dominant wrist crosses body midline (left-to-right for right-handed)
- NEUTRAL: Low arm movement, stance is relatively still

Usage:
    python heuristic_detect.py poses/IMG_1234.json -o shots_heuristic.json
    python heuristic_detect.py poses/IMG_1234.json --visualize
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class Detection:
    """A detected shot segment."""
    shot_type: str
    start_frame: int
    end_frame: int
    confidence: float
    trigger: str  # What triggered the detection


# ─────────────────────────────────────────────────────────────
# Keypoint indices (MediaPipe 33-keypoint model)
# ─────────────────────────────────────────────────────────────

# MediaPipe pose landmarks
NOSE = 0
LEFT_EYE_INNER = 1
LEFT_EYE = 2
LEFT_EYE_OUTER = 3
RIGHT_EYE_INNER = 4
RIGHT_EYE = 5
RIGHT_EYE_OUTER = 6
LEFT_EAR = 7
RIGHT_EAR = 8
MOUTH_LEFT = 9
MOUTH_RIGHT = 10
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_PINKY = 17
RIGHT_PINKY = 18
LEFT_INDEX = 19
RIGHT_INDEX = 20
LEFT_THUMB = 21
RIGHT_THUMB = 22
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
LEFT_HEEL = 29
RIGHT_HEEL = 30
LEFT_FOOT_INDEX = 31
RIGHT_FOOT_INDEX = 32


# ─────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────

def get_keypoint(frame_data, idx):
    """Get keypoint coordinates from frame data.

    Returns (x, y, z) or None if not available.
    """
    # Try world_landmarks first (real-world coordinates)
    world_lm = frame_data.get("world_landmarks")
    if world_lm and idx < len(world_lm):
        return world_lm[idx]

    # Fall back to normalized keypoints
    keypoints = frame_data.get("keypoints")
    if keypoints and idx < len(keypoints):
        kp = keypoints[idx]
        return [kp.get("x", 0), kp.get("y", 0), kp.get("z", 0)]

    return None


def body_midline_x(frame_data):
    """Calculate x-coordinate of body midline (average of shoulders)."""
    left_shoulder = get_keypoint(frame_data, LEFT_SHOULDER)
    right_shoulder = get_keypoint(frame_data, RIGHT_SHOULDER)

    if left_shoulder and right_shoulder:
        return (left_shoulder[0] + right_shoulder[0]) / 2
    return None


def head_level_y(frame_data):
    """Calculate y-coordinate of head level (nose or eye average)."""
    nose = get_keypoint(frame_data, NOSE)
    if nose:
        return nose[1]

    left_eye = get_keypoint(frame_data, LEFT_EYE)
    right_eye = get_keypoint(frame_data, RIGHT_EYE)
    if left_eye and right_eye:
        return (left_eye[1] + right_eye[1]) / 2

    return None


def shoulder_level_y(frame_data):
    """Calculate y-coordinate of shoulder level."""
    left_shoulder = get_keypoint(frame_data, LEFT_SHOULDER)
    right_shoulder = get_keypoint(frame_data, RIGHT_SHOULDER)

    if left_shoulder and right_shoulder:
        return (left_shoulder[1] + right_shoulder[1]) / 2
    return None


# ─────────────────────────────────────────────────────────────
# Detection rules
# ─────────────────────────────────────────────────────────────

def detect_serve(frame_data) -> tuple:
    """Detect serve: wrist raised above head level.

    In MediaPipe world coordinates, Y increases downward in image space,
    but in world coordinates Y is often positive upward. We need to check
    if wrist.y > head.y (world coords) or wrist.y < head.y (image coords).

    Returns: (is_serve, confidence, trigger_description)
    """
    right_wrist = get_keypoint(frame_data, RIGHT_WRIST)
    left_wrist = get_keypoint(frame_data, LEFT_WRIST)
    head_y = head_level_y(frame_data)
    shoulder_y = shoulder_level_y(frame_data)

    if not head_y or not shoulder_y:
        return False, 0.0, ""

    # Check both wrists
    for wrist, name in [(right_wrist, "right"), (left_wrist, "left")]:
        if not wrist:
            continue

        # In world coordinates, more negative Y = higher
        # Serve = wrist at least 0.3m above head
        height_above_head = head_y - wrist[1]  # positive = wrist higher

        if height_above_head > 0.15:  # 15cm above head
            confidence = min(1.0, height_above_head / 0.4)  # Max at 40cm
            return True, confidence, f"{name} wrist {height_above_head:.2f}m above head"

    return False, 0.0, ""


def detect_forehand(frame_data, dominant_hand="right") -> tuple:
    """Detect forehand: dominant wrist crosses body to non-dominant side.

    For right-handed player: right wrist moves to left of body midline.
    Returns: (is_forehand, confidence, trigger_description)
    """
    midline = body_midline_x(frame_data)
    if midline is None:
        return False, 0.0, ""

    if dominant_hand == "right":
        wrist = get_keypoint(frame_data, RIGHT_WRIST)
        # Forehand = right wrist crosses to left of midline
        if wrist and wrist[0] < midline - 0.1:  # 10cm past midline
            offset = midline - wrist[0]
            confidence = min(1.0, offset / 0.3)
            return True, confidence, f"right wrist {offset:.2f}m left of midline"
    else:
        wrist = get_keypoint(frame_data, LEFT_WRIST)
        # Forehand = left wrist crosses to right of midline
        if wrist and wrist[0] > midline + 0.1:
            offset = wrist[0] - midline
            confidence = min(1.0, offset / 0.3)
            return True, confidence, f"left wrist {offset:.2f}m right of midline"

    return False, 0.0, ""


def detect_backhand(frame_data, dominant_hand="right") -> tuple:
    """Detect backhand: non-dominant side leads, wrist on dominant side.

    For right-handed player: right wrist stays on right side but body rotates.
    More reliably: left shoulder forward, right wrist behind midline.

    Returns: (is_backhand, confidence, trigger_description)
    """
    midline = body_midline_x(frame_data)
    if midline is None:
        return False, 0.0, ""

    left_shoulder = get_keypoint(frame_data, LEFT_SHOULDER)
    right_shoulder = get_keypoint(frame_data, RIGHT_SHOULDER)

    if dominant_hand == "right":
        wrist = get_keypoint(frame_data, RIGHT_WRIST)
        # Backhand indicators:
        # 1. Right wrist on right side of body (not crossed over)
        # 2. Left shoulder rotated forward (smaller x than right)
        if wrist and left_shoulder and right_shoulder:
            wrist_right_of_midline = wrist[0] > midline
            shoulder_rotation = left_shoulder[0] - right_shoulder[0]

            # Left shoulder forward = more positive rotation (closer to camera)
            if wrist_right_of_midline and shoulder_rotation > 0.05:
                confidence = min(1.0, shoulder_rotation / 0.2)
                return True, confidence, f"backhand stance, rotation={shoulder_rotation:.2f}m"
    else:
        wrist = get_keypoint(frame_data, LEFT_WRIST)
        if wrist and left_shoulder and right_shoulder:
            wrist_left_of_midline = wrist[0] < midline
            shoulder_rotation = right_shoulder[0] - left_shoulder[0]

            if wrist_left_of_midline and shoulder_rotation > 0.05:
                confidence = min(1.0, shoulder_rotation / 0.2)
                return True, confidence, f"backhand stance, rotation={shoulder_rotation:.2f}m"

    return False, 0.0, ""


def classify_frame(frame_data, dominant_hand="right") -> tuple:
    """Classify a single frame using heuristics.

    Returns: (shot_type, confidence, trigger)
    Priority: serve > forehand > backhand > neutral
    """
    # Check serve first (highest priority)
    is_serve, conf, trigger = detect_serve(frame_data)
    if is_serve and conf > 0.3:
        return "serve", conf, trigger

    # Check forehand
    is_forehand, conf, trigger = detect_forehand(frame_data, dominant_hand)
    if is_forehand and conf > 0.3:
        return "forehand", conf, trigger

    # Check backhand
    is_backhand, conf, trigger = detect_backhand(frame_data, dominant_hand)
    if is_backhand and conf > 0.3:
        return "backhand", conf, trigger

    return "neutral", 0.5, "no shot detected"


# ─────────────────────────────────────────────────────────────
# Segment merging
# ─────────────────────────────────────────────────────────────

def merge_detections(frame_results: list, fps: float = 60.0,
                     min_duration_frames: int = 15,
                     max_gap_frames: int = 10) -> List[Detection]:
    """Merge consecutive same-label frames into segments.

    Args:
        frame_results: List of (shot_type, confidence, trigger) per frame
        fps: Frames per second
        min_duration_frames: Minimum segment length to keep
        max_gap_frames: Maximum gap to bridge within same shot type

    Returns:
        List of Detection objects
    """
    if not frame_results:
        return []

    segments = []
    current_type = frame_results[0][0]
    current_start = 0
    current_confidences = [frame_results[0][1]]
    current_trigger = frame_results[0][2]

    for i, (shot_type, conf, trigger) in enumerate(frame_results[1:], 1):
        if shot_type == current_type:
            current_confidences.append(conf)
        else:
            # End current segment
            if current_type != "neutral" and len(current_confidences) >= min_duration_frames:
                segments.append(Detection(
                    shot_type=current_type,
                    start_frame=current_start,
                    end_frame=i - 1,
                    confidence=sum(current_confidences) / len(current_confidences),
                    trigger=current_trigger,
                ))

            # Start new segment
            current_type = shot_type
            current_start = i
            current_confidences = [conf]
            current_trigger = trigger

    # Handle final segment
    if current_type != "neutral" and len(current_confidences) >= min_duration_frames:
        segments.append(Detection(
            shot_type=current_type,
            start_frame=current_start,
            end_frame=len(frame_results) - 1,
            confidence=sum(current_confidences) / len(current_confidences),
            trigger=current_trigger,
        ))

    # Merge nearby segments of same type
    merged = []
    for seg in segments:
        if merged and merged[-1].shot_type == seg.shot_type:
            gap = seg.start_frame - merged[-1].end_frame
            if gap <= max_gap_frames:
                # Merge
                merged[-1].end_frame = seg.end_frame
                merged[-1].confidence = (merged[-1].confidence + seg.confidence) / 2
                continue
        merged.append(seg)

    return merged


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def process_poses_file(poses_path: str, dominant_hand: str = "right") -> dict:
    """Process a poses JSON file and detect shots.

    Returns dict compatible with detect_shots.py output format.
    """
    with open(poses_path) as f:
        poses = json.load(f)

    fps = poses.get("fps", 60)
    frames = poses.get("frames", [])

    if not frames:
        return {
            "source_video": Path(poses_path).stem,
            "fps": fps,
            "total_frames": 0,
            "shot_counts": {},
            "segments": [],
            "detector": "heuristic",
        }

    # Classify each frame
    frame_results = []
    for frame in frames:
        if frame.get("keypoints") or frame.get("world_landmarks"):
            shot_type, conf, trigger = classify_frame(frame, dominant_hand)
            frame_results.append((shot_type, conf, trigger))
        else:
            frame_results.append(("neutral", 0.0, "no pose"))

    # Merge into segments
    detections = merge_detections(frame_results, fps)

    # Build output
    segments = []
    shot_counts = {"forehand": 0, "backhand": 0, "serve": 0}

    for det in detections:
        shot_counts[det.shot_type] = shot_counts.get(det.shot_type, 0) + 1
        segments.append({
            "shot_type": det.shot_type,
            "start_frame": det.start_frame,
            "end_frame": det.end_frame,
            "start_time": round(det.start_frame / fps, 2),
            "end_time": round(det.end_frame / fps, 2),
            "confidence": round(det.confidence, 3),
            "trigger": det.trigger,
        })

    return {
        "source_video": Path(poses_path).stem.replace("_poses", ""),
        "fps": fps,
        "total_frames": len(frames),
        "shot_counts": shot_counts,
        "segments": segments,
        "detector": "heuristic",
        "dominant_hand": dominant_hand,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Heuristic-based shot detection (no ML model)")
    parser.add_argument("poses_file", help="Path to poses JSON file")
    parser.add_argument("-o", "--output", help="Output JSON file")
    parser.add_argument("--dominant-hand", choices=["left", "right"],
                        default="right", help="Player's dominant hand")
    parser.add_argument("--visualize", action="store_true",
                        help="Print detection summary")
    args = parser.parse_args()

    if not os.path.exists(args.poses_file):
        print(f"Error: {args.poses_file} not found")
        sys.exit(1)

    # Process
    result = process_poses_file(args.poses_file, args.dominant_hand)

    # Output
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Wrote {args.output}")

    if args.visualize or not args.output:
        print(f"\n=== Heuristic Detection Results ===")
        print(f"Source: {result['source_video']}")
        print(f"FPS: {result['fps']}")
        print(f"Total frames: {result['total_frames']}")
        print(f"Dominant hand: {result['dominant_hand']}")
        print(f"\nShot counts: {result['shot_counts']}")
        print(f"\nDetected {len(result['segments'])} segments:")
        for i, seg in enumerate(result["segments"][:20], 1):
            print(f"  {i:2d}. {seg['shot_type']:10s} "
                  f"{seg['start_time']:6.1f}s - {seg['end_time']:6.1f}s "
                  f"(conf: {seg['confidence']:.2f}) - {seg['trigger']}")
        if len(result["segments"]) > 20:
            print(f"  ... and {len(result['segments']) - 20} more")


if __name__ == "__main__":
    main()
