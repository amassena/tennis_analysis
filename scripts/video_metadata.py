"""Utilities for managing per-video metadata files."""

import json
import os
import sys
from typing import Optional, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import PROJECT_ROOT, POSES_DIR, DEFAULT_VIEW_ANGLE, VIEW_ANGLES


def _metadata_path(video_name: str) -> str:
    """Return the path to a video's metadata JSON file."""
    # Strip extension if present
    base = os.path.splitext(video_name)[0]
    return os.path.join(PROJECT_ROOT, f"{base}_metadata.json")


def load_video_metadata(video_name: str) -> dict:
    """Load metadata for a video, returning empty dict if not found."""
    path = _metadata_path(video_name)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def save_video_metadata(video_name: str, metadata: dict) -> None:
    """Save metadata for a video to its JSON file."""
    path = _metadata_path(video_name)
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)


def get_view_angle(video_name: str) -> str:
    """Get the view angle for a video, defaulting to back-court."""
    metadata = load_video_metadata(video_name)
    return metadata.get("view_angle", DEFAULT_VIEW_ANGLE)


def set_view_angle(video_name: str, view_angle: str) -> None:
    """Set the view angle for a video."""
    metadata = load_video_metadata(video_name)
    metadata["view_angle"] = view_angle
    save_video_metadata(video_name, metadata)


def detect_view_angle_from_pose(pose_data: dict, sample_size: int = 500) -> str:
    """Auto-detect view angle from pose landmarks.

    Analyzes shoulder positions (landmarks 11, 12) and wrist positions
    to determine camera position relative to player.

    MediaPipe world_landmarks z-axis: positive = away from camera

    Returns one of: back-court, left-side, right-side, front, overhead
    """
    frames = pose_data.get("frames", [])

    # Collect samples from throughout the video (not just beginning)
    total_frames = len(frames)
    step = max(1, total_frames // sample_size)

    shoulder_diffs = []  # right_z - left_z (positive = left side view)
    wrist_vs_shoulder = []  # avg_wrist_z - avg_shoulder_z (positive = arms forward = back-court)

    for i in range(0, total_frames, step):
        frame = frames[i]
        if not frame.get("detected") or not frame.get("world_landmarks"):
            continue

        wl = frame["world_landmarks"]
        if len(wl) < 17:  # Need at least up to wrists
            continue

        # Landmarks: 11=left_shoulder, 12=right_shoulder, 15=left_wrist, 16=right_wrist
        left_shoulder = wl[11]
        right_shoulder = wl[12]
        left_wrist = wl[15]
        right_wrist = wl[16]

        # z difference between shoulders (positive = viewing from left side)
        shoulder_diff = right_shoulder[2] - left_shoulder[2]
        shoulder_diffs.append(shoulder_diff)

        # Wrist vs shoulder midpoint z (positive = arms extended forward = facing away)
        avg_shoulder_z = (left_shoulder[2] + right_shoulder[2]) / 2
        avg_wrist_z = (left_wrist[2] + right_wrist[2]) / 2
        wrist_diff = avg_wrist_z - avg_shoulder_z
        wrist_vs_shoulder.append(wrist_diff)

        if len(shoulder_diffs) >= sample_size:
            break

    if not shoulder_diffs:
        return DEFAULT_VIEW_ANGLE

    # Compute averages
    avg_shoulder_diff = sum(shoulder_diffs) / len(shoulder_diffs)
    avg_wrist_diff = sum(wrist_vs_shoulder) / len(wrist_vs_shoulder)

    # Thresholds (in meters, world coordinates)
    side_threshold = 0.10  # If shoulders differ by >10cm in z, it's a side view

    # Determine view angle
    if abs(avg_shoulder_diff) > side_threshold:
        # Clear side view
        if avg_shoulder_diff > 0:
            return "left-side"  # Right shoulder further = viewing from left
        else:
            return "right-side"  # Left shoulder further = viewing from right
    else:
        # Not a clear side view - default to back-court
        # This is the most common tennis filming angle
        # Front view is rare (filming from the net toward baseline)
        return "back-court"


def detect_view_angle_from_file(video_name: str) -> Optional[str]:
    """Load pose file and auto-detect view angle.

    Returns detected view angle, or None if pose file not found.
    """
    base = os.path.splitext(video_name)[0]
    pose_path = os.path.join(POSES_DIR, f"{base}.json")

    if not os.path.exists(pose_path):
        return None

    try:
        with open(pose_path, "r") as f:
            pose_data = json.load(f)
        return detect_view_angle_from_pose(pose_data)
    except (json.JSONDecodeError, OSError):
        return None


def get_view_angle_auto(video_name: str) -> str:
    """Get view angle, auto-detecting from pose if no metadata exists.

    Priority:
    1. Existing metadata file
    2. Auto-detect from pose landmarks
    3. Default (back-court)
    """
    # Check for existing metadata
    metadata = load_video_metadata(video_name)
    if "view_angle" in metadata:
        return metadata["view_angle"]

    # Try auto-detection
    detected = detect_view_angle_from_file(video_name)
    if detected:
        # Cache the detected value
        metadata["view_angle"] = detected
        metadata["view_angle_auto_detected"] = True
        save_video_metadata(video_name, metadata)
        return detected

    return DEFAULT_VIEW_ANGLE
