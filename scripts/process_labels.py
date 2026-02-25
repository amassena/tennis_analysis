#!/usr/bin/env python3
"""Convert manual label files into training clips and pose data.

Reads a label file (timestamps + shot types), extracts clip windows around
each timestamp, extracts pose data for each clip, and saves everything to
the training directory structure.

Label file format (see labels/README.md):
    # Video: IMG_1234.mp4
    0:05 serve
    0:12 forehand
    0:15 backhand
    1:02 backhand
    1:15 forehand

Usage:
    python process_labels.py labels/IMG_1234_labels.txt
    python process_labels.py labels/IMG_1234_labels.txt --dry-run
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import (
    PROJECT_ROOT, PREPROCESSED_DIR, TRAINING_DIR, CLIPS_DIR,
    TRAINING_DATA_DIR, LABELS_DIR, MODEL, SHOT_TYPES,
)


# ── Clip extraction parameters ───────────────────────────────

# Seconds before/after contact point per shot type
CLIP_BUFFERS = {
    "serve": (1.5, 1.5),           # 3s window around contact
    "forehand": (1.5, 1.5),        # 3s window
    "backhand": (1.5, 1.5),        # 3s window
    "forehand_volley": (1.0, 1.0), # 2s window (shorter motion)
    "backhand_volley": (1.0, 1.0), # 2s window
    "overhead": (1.5, 1.0),        # 2.5s window
    "neutral": (1.5, 1.5),         # 3s window
}
DEFAULT_BUFFER = (1.5, 1.5)

# Training targets per shot type
TRAINING_TARGETS = {
    "serve": 100,
    "forehand": 100,
    "backhand": 100,
    "forehand_volley": 50,
    "backhand_volley": 50,
    "overhead": 50,
    "neutral": 80,
}


def parse_timestamp(ts_str):
    """Parse MM:SS or H:MM:SS timestamp to seconds.

    Returns float seconds, or None if parse fails.
    """
    parts = ts_str.strip().split(":")
    try:
        if len(parts) == 2:
            minutes, seconds = int(parts[0]), float(parts[1])
            return minutes * 60 + seconds
        elif len(parts) == 3:
            hours, minutes, seconds = int(parts[0]), int(parts[1]), float(parts[2])
            return hours * 3600 + minutes * 60 + seconds
    except (ValueError, IndexError):
        pass
    return None


def parse_label_file(label_path):
    """Parse a label file into a list of (timestamp_sec, shot_type) tuples.

    Also extracts video name from header comments.
    Returns (video_name, labels_list).
    """
    video_name = None
    labels = []

    with open(label_path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Parse header comments
            if line.startswith("#"):
                # Look for "# Video: IMG_1234.mp4"
                match = re.match(r"#\s*Video:\s*(.+)", line, re.IGNORECASE)
                if match:
                    video_name = os.path.splitext(match.group(1).strip())[0]
                continue

            # Parse label line: "MM:SS shot_type" or "MM:SS,shot_type"
            parts = re.split(r"[\s,]+", line, maxsplit=1)
            if len(parts) != 2:
                print(f"  [WARN] Line {line_num}: could not parse '{line}'")
                continue

            ts_str, shot_type = parts
            timestamp = parse_timestamp(ts_str)
            if timestamp is None:
                print(f"  [WARN] Line {line_num}: invalid timestamp '{ts_str}'")
                continue

            shot_type = shot_type.strip().lower()
            labels.append((timestamp, shot_type))

    # If no video name in header, try to derive from filename
    if video_name is None:
        base = os.path.basename(label_path)
        # Strip _labels.txt suffix
        video_name = re.sub(r"_labels\.txt$", "", base, flags=re.IGNORECASE)

    return video_name, labels


def find_video(video_name):
    """Find the preprocessed video file."""
    for ext in [".mp4", ".mov"]:
        path = os.path.join(PREPROCESSED_DIR, video_name + ext)
        if os.path.exists(path):
            return path
    return None


def get_video_fps(video_path):
    """Get video FPS using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "csv=p=0", video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    fps_str = result.stdout.strip()
    if "/" in fps_str:
        num, den = fps_str.split("/")
        return float(num) / float(den)
    return float(fps_str) if fps_str else 60.0


def extract_training_clip(video_path, start_time, end_time, output_path):
    """Extract a training clip using FFmpeg."""
    duration = end_time - start_time

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_time:.3f}",
        "-i", video_path,
        "-t", f"{duration:.3f}",
        "-c:v", "libx264", "-crf", "20", "-preset", "fast",
        "-an",
        "-pix_fmt", "yuv420p",
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def extract_clip_poses(video_path, start_time, end_time, fps):
    """Extract pose data for a clip window.

    Returns a dict with clip pose data suitable for training,
    or None on failure.
    """
    try:
        from scripts.extract_poses import init_pose_model, extract_poses
        from config.settings import MEDIAPIPE, POSES_DIR as poses_dir
    except ImportError as e:
        print(f"    [WARN] Cannot extract poses: {e}")
        return None

    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    pose_model = init_pose_model(MEDIAPIPE)

    import mediapipe as mp
    landmark_names = [lm.name for lm in mp.solutions.pose.PoseLandmark]

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    frame_idx = start_frame

    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = pose_model.process(rgb)

        if result.pose_landmarks:
            world_landmarks = [
                [round(lm.x, 6), round(lm.y, 6), round(lm.z, 6)]
                for lm in result.pose_world_landmarks.landmark
            ]
            frames.append({
                "frame_idx": frame_idx - start_frame,
                "timestamp": round((frame_idx - start_frame) / fps, 6),
                "detected": True,
                "world_landmarks_xyz": world_landmarks,
            })
        else:
            frames.append({
                "frame_idx": frame_idx - start_frame,
                "timestamp": round((frame_idx - start_frame) / fps, 6),
                "detected": False,
                "world_landmarks_xyz": None,
            })

        frame_idx += 1

    cap.release()
    pose_model.close()

    return {
        "frames": frames,
        "landmark_names": landmark_names,
        "fps": fps,
        "total_frames": len(frames),
    }


def load_metadata():
    """Load training metadata.json."""
    meta_path = os.path.join(TRAINING_DIR, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            return json.load(f)
    return {
        "version": 1,
        "last_updated": "",
        "camera_angles": {},
        "angle_options": ["back", "side_left", "side_right", "front", "unknown"],
        "clips": [],
    }


def save_metadata(metadata):
    """Save training metadata.json."""
    meta_path = os.path.join(TRAINING_DIR, "metadata.json")
    metadata["last_updated"] = datetime.now().strftime("%Y-%m-%d")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)


def count_existing_clips():
    """Count existing training clips per shot type."""
    counts = {}
    for shot_type in os.listdir(CLIPS_DIR) if os.path.isdir(CLIPS_DIR) else []:
        type_dir = os.path.join(CLIPS_DIR, shot_type)
        if os.path.isdir(type_dir):
            clip_count = len([f for f in os.listdir(type_dir) if f.endswith(".mp4")])
            if clip_count > 0:
                counts[shot_type] = clip_count
    return counts


def count_existing_poses():
    """Count existing training pose files per shot type."""
    counts = {}
    for shot_type in os.listdir(TRAINING_DATA_DIR) if os.path.isdir(TRAINING_DATA_DIR) else []:
        type_dir = os.path.join(TRAINING_DATA_DIR, shot_type)
        if os.path.isdir(type_dir):
            pose_count = len([f for f in os.listdir(type_dir) if f.endswith(".json")])
            if pose_count > 0:
                counts[shot_type] = pose_count
    return counts


def print_progress(clip_counts, pose_counts):
    """Print progress toward training targets."""
    print(f"\nTraining Data Progress:")
    print(f"  {'Shot Type':<20} {'Clips':>6} {'Poses':>6} {'Target':>7} {'Status':>10}")
    print(f"  {'─'*55}")

    for shot_type in ["serve", "forehand", "backhand", "forehand_volley",
                       "backhand_volley", "overhead", "neutral"]:
        clips = clip_counts.get(shot_type, 0)
        poses = pose_counts.get(shot_type, 0)
        target = TRAINING_TARGETS.get(shot_type, 50)
        pct = poses / target * 100 if target > 0 else 0
        status = "DONE" if poses >= target else f"{pct:.0f}%"
        print(f"  {shot_type:<20} {clips:>6} {poses:>6} {target:>7} {status:>10}")

    total_poses = sum(pose_counts.values())
    total_target = sum(TRAINING_TARGETS.values())
    print(f"  {'─'*55}")
    print(f"  {'TOTAL':<20} {sum(clip_counts.values()):>6} {total_poses:>6} "
          f"{total_target:>7} {total_poses/total_target*100:.0f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Convert label files into training clips and pose data"
    )
    parser.add_argument("label_file", help="Path to label file (.txt)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Parse and validate only, don't extract anything")
    parser.add_argument("--skip-poses", action="store_true",
                        help="Skip pose extraction (clips only)")
    parser.add_argument("--camera-angle", default="unknown",
                        help="Camera angle for this video (back, side_left, side_right, front)")
    args = parser.parse_args()

    label_path = args.label_file
    if not os.path.isabs(label_path):
        label_path = os.path.join(os.getcwd(), label_path)

    if not os.path.exists(label_path):
        print(f"[ERROR] Label file not found: {label_path}")
        sys.exit(1)

    # ── Parse labels ─────────────────────────────────────────
    print(f"Parsing: {os.path.basename(label_path)}")
    video_name, labels = parse_label_file(label_path)
    print(f"  Video: {video_name}")
    print(f"  Labels: {len(labels)}")

    if not labels:
        print("[ERROR] No valid labels found")
        sys.exit(1)

    # Count by type
    type_counts = {}
    for ts, st in labels:
        type_counts[st] = type_counts.get(st, 0) + 1
    for st, count in sorted(type_counts.items()):
        print(f"    {st}: {count}")
    print()

    # ── Find video ───────────────────────────────────────────
    video_path = find_video(video_name)
    if not video_path:
        print(f"[ERROR] Video not found: {video_name} in {PREPROCESSED_DIR}/")
        print(f"  Run preprocess_videos.py first")
        sys.exit(1)

    fps = get_video_fps(video_path)
    print(f"  Video: {os.path.basename(video_path)} ({fps:.0f} fps)")

    if args.dry_run:
        print("\n[DRY RUN] Would extract:")
        for ts, st in labels:
            pre, post = CLIP_BUFFERS.get(st, DEFAULT_BUFFER)
            start = max(0, ts - pre)
            end = ts + post
            print(f"  {st:<20} {ts:.1f}s (clip: {start:.1f}-{end:.1f}s)")
        print_progress(count_existing_clips(), count_existing_poses())
        return

    # ── Create directories ───────────────────────────────────
    for shot_type in type_counts:
        os.makedirs(os.path.join(CLIPS_DIR, shot_type), exist_ok=True)
        os.makedirs(os.path.join(TRAINING_DATA_DIR, shot_type), exist_ok=True)

    # ── Process each label ───────────────────────────────────
    metadata = load_metadata()

    # Update camera angle for this video
    metadata["camera_angles"][video_name] = args.camera_angle

    clips_extracted = 0
    poses_extracted = 0
    counters = {}

    seq_len = MODEL["sequence_length"]  # 90 frames = 1.5s at 60fps

    for i, (timestamp, shot_type) in enumerate(labels):
        counters[shot_type] = counters.get(shot_type, 0) + 1

        # Calculate clip window
        pre, post = CLIP_BUFFERS.get(shot_type, DEFAULT_BUFFER)
        start_time = max(0, timestamp - pre)
        end_time = timestamp + post

        clip_name = f"{video_name}_{shot_type}_{counters[shot_type]:03d}.mp4"
        pose_name = f"{video_name}_{shot_type}_{counters[shot_type]:03d}.json"

        clip_path = os.path.join(CLIPS_DIR, shot_type, clip_name)
        pose_path = os.path.join(TRAINING_DATA_DIR, shot_type, pose_name)

        print(f"  [{i+1}/{len(labels)}] {shot_type} @ {timestamp:.1f}s", end="")

        # Extract clip
        if os.path.exists(clip_path):
            print(f" [clip exists]", end="")
        else:
            if extract_training_clip(video_path, start_time, end_time, clip_path):
                clips_extracted += 1
                print(f" [clip OK]", end="")
            else:
                print(f" [clip FAILED]", end="")

        # Extract poses
        if args.skip_poses:
            print(f" [poses skipped]")
        elif os.path.exists(pose_path):
            print(f" [poses exist]")
        else:
            pose_data = extract_clip_poses(video_path, start_time, end_time, fps)
            if pose_data:
                # Save in training format
                training_clip_data = {
                    "version": 2,
                    "shot_type": shot_type,
                    "source_video": video_name,
                    "contact_time": timestamp,
                    "start_time": start_time,
                    "end_time": end_time,
                    "start_frame": int(start_time * fps),
                    "end_frame": int(end_time * fps),
                    "camera_angle": args.camera_angle,
                    "fps": fps,
                    "sequence_length": seq_len,
                    "frames": pose_data["frames"],
                }
                with open(pose_path, "w") as f:
                    json.dump(training_clip_data, f, separators=(",", ":"))
                poses_extracted += 1
                print(f" [poses OK]")
            else:
                print(f" [poses FAILED]")

        # Add to metadata
        clip_entry = {
            "filename": clip_name,
            "shot_type": shot_type,
            "source_video": video_name,
            "contact_time": timestamp,
            "verified": True,  # Manual labels are pre-verified
            "has_pose": os.path.exists(pose_path),
            "camera_angle": args.camera_angle,
        }

        # Check if already in metadata (by filename)
        existing_idx = next(
            (i for i, c in enumerate(metadata["clips"])
             if c["filename"] == clip_name),
            None
        )
        if existing_idx is not None:
            metadata["clips"][existing_idx] = clip_entry
        else:
            metadata["clips"].append(clip_entry)

    # ── Save metadata ────────────────────────────────────────
    save_metadata(metadata)

    # ── Summary ──────────────────────────────────────────────
    print()
    print(f"{'='*50}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*50}")
    print(f"  Labels processed: {len(labels)}")
    print(f"  Clips extracted: {clips_extracted}")
    print(f"  Poses extracted: {poses_extracted}")
    print(f"  Metadata updated: {os.path.join(TRAINING_DIR, 'metadata.json')}")

    # Print overall progress
    print_progress(count_existing_clips(), count_existing_poses())

    print()
    print(f"Next steps:")
    print(f"  1. Verify clips: python scripts/verify_clips.py --source {video_name}")
    print(f"  2. Retrain model: python scripts/train_model.py")


if __name__ == "__main__":
    main()
