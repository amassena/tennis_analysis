#!/usr/bin/env python3
"""Interactive terminal tool to label shot segments from pose data."""

import argparse
import csv
import json
import os
import sys
import time

# Add project root to path so config is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import (
    POSES_DIR,
    TRAINING_DATA_DIR,
    MODEL,
    SHOT_TYPES,
)
from scripts.video_metadata import get_view_angle


# ── Helpers ──────────────────────────────────────────────────


def format_size(num_bytes):
    """Return human-readable file size string."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(num_bytes) < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"


def format_duration(seconds):
    """Format seconds as MM:SS or HH:MM:SS."""
    seconds = int(seconds)
    if seconds >= 3600:
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        return f"{h}:{m:02d}:{s:02d}"
    m = seconds // 60
    s = seconds % 60
    return f"{m}:{s:02d}"


# ── Data Loading ─────────────────────────────────────────────


def load_pose_data(path):
    """Load and validate a pose JSON file.

    Returns the parsed dict, or None on failure.
    """
    if not os.path.isfile(path):
        print(f"  [ERROR] File not found: {path}")
        return None

    try:
        with open(path, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  [ERROR] Cannot read {os.path.basename(path)}: {e}")
        return None

    # Validate required keys
    required = ("version", "source_video", "video_info", "frames")
    missing = [k for k in required if k not in data]
    if missing:
        print(f"  [ERROR] Missing keys in {os.path.basename(path)}: {missing}")
        return None

    if not data["frames"]:
        print(f"  [ERROR] No frames in {os.path.basename(path)}")
        return None

    return data


def find_pose_files(poses_dir):
    """List .json pose files (excluding _skeleton files), sorted by name."""
    if not os.path.isdir(poses_dir):
        return []
    files = [
        f for f in os.listdir(poses_dir)
        if f.lower().endswith(".json") and "_skeleton" not in f.lower()
    ]
    files.sort()
    return files


# ── Clip Discovery ───────────────────────────────────────────


def get_existing_clips(training_dir, source_video):
    """Find already-labeled clips matching a source video across all shot types."""
    name = os.path.splitext(source_video)[0]
    existing = []
    for shot_type in SHOT_TYPES:
        type_dir = os.path.join(training_dir, shot_type)
        if not os.path.isdir(type_dir):
            continue
        for f in os.listdir(type_dir):
            if f.startswith(name) and f.endswith(".json"):
                existing.append(os.path.join(type_dir, f))
    return existing


def count_clips_per_type(training_dir):
    """Count clips per shot type directory.

    Returns dict like {"forehand": 12, "backhand": 8, ...}
    """
    counts = {}
    for shot_type in SHOT_TYPES:
        type_dir = os.path.join(training_dir, shot_type)
        if not os.path.isdir(type_dir):
            counts[shot_type] = 0
            continue
        counts[shot_type] = len([
            f for f in os.listdir(type_dir)
            if f.endswith(".json")
        ])
    return counts


# ── Frame Processing ─────────────────────────────────────────


def interpolate_missing_frames(frames, start, end):
    """Linear interpolation for detected=false frames in a range.

    Returns list of frame dicts with world_landmarks_xyz (visibility dropped),
    plus counts of interpolated frames.
    """
    result = []
    interpolated_indices = []

    # Collect detected frame indices and their landmarks within range
    detected = {}
    for frame in frames:
        idx = frame["frame_idx"]
        if start <= idx <= end and frame["detected"] and frame["world_landmarks"]:
            # Drop visibility (4th element), keep xyz only
            detected[idx] = [lm[:3] for lm in frame["world_landmarks"]]

    for frame_idx in range(start, end + 1):
        if frame_idx in detected:
            result.append({
                "frame_idx": frame_idx,
                "world_landmarks_xyz": detected[frame_idx],
            })
        else:
            # Find nearest detected neighbors
            prev_idx = None
            next_idx = None
            for i in range(frame_idx - 1, start - 1, -1):
                if i in detected:
                    prev_idx = i
                    break
            for i in range(frame_idx + 1, end + 1):
                if i in detected:
                    next_idx = i
                    break

            if prev_idx is not None and next_idx is not None:
                # Linear interpolation
                t = (frame_idx - prev_idx) / (next_idx - prev_idx)
                interp_landmarks = []
                for kp in range(len(detected[prev_idx])):
                    px, py, pz = detected[prev_idx][kp]
                    nx, ny, nz = detected[next_idx][kp]
                    interp_landmarks.append([
                        round(px + t * (nx - px), 6),
                        round(py + t * (ny - py), 6),
                        round(pz + t * (nz - pz), 6),
                    ])
                result.append({
                    "frame_idx": frame_idx,
                    "world_landmarks_xyz": interp_landmarks,
                })
                interpolated_indices.append(frame_idx)
            elif prev_idx is not None:
                # Copy from previous
                result.append({
                    "frame_idx": frame_idx,
                    "world_landmarks_xyz": [lm[:] for lm in detected[prev_idx]],
                })
                interpolated_indices.append(frame_idx)
            elif next_idx is not None:
                # Copy from next
                result.append({
                    "frame_idx": frame_idx,
                    "world_landmarks_xyz": [lm[:] for lm in detected[next_idx]],
                })
                interpolated_indices.append(frame_idx)
            else:
                # No neighbors at all — zero fill
                result.append({
                    "frame_idx": frame_idx,
                    "world_landmarks_xyz": [[0.0, 0.0, 0.0]] * 33,
                })
                interpolated_indices.append(frame_idx)

    return result, interpolated_indices


# ── Sliding Window ───────────────────────────────────────────


def generate_clip_windows(start, end, seq_len, stride):
    """Generate sliding window (start, end) tuples from a labeled range.

    Each window is seq_len frames. Windows that would extend past end are skipped.
    """
    windows = []
    pos = start
    while pos + seq_len - 1 <= end:
        windows.append((pos, pos + seq_len - 1))
        pos += stride
    return windows


def validate_window(frames, start, end):
    """Reject windows with >30% missing (detected=false) frames.

    Returns (valid, missing_count, total_count).
    """
    total = end - start + 1
    missing = 0
    for frame in frames:
        idx = frame["frame_idx"]
        if start <= idx <= end and not frame["detected"]:
            missing += 1
    threshold = 0.3 * total
    return missing <= threshold, missing, total


# ── Clip Saving ──────────────────────────────────────────────


def save_clip(clip_data, shot_type, training_dir):
    """Atomic write of a clip JSON to training_data/{type}/.

    Filename: {source}_{start_frame:06d}.json
    """
    type_dir = os.path.join(training_dir, shot_type)
    os.makedirs(type_dir, exist_ok=True)

    source_name = os.path.splitext(clip_data["source_video"])[0]
    filename = f"{source_name}_{clip_data['start_frame']:06d}.json"
    final_path = os.path.join(type_dir, filename)
    part_path = final_path + ".part"

    with open(part_path, "w") as f:
        json.dump(clip_data, f, separators=(",", ":"))
    os.replace(part_path, final_path)

    return final_path


# ── CSV Batch Mode ───────────────────────────────────────────


def parse_labels_csv(csv_path, shot_types, max_frame):
    """Parse a CSV file with columns: shot_type, start, end.

    Returns list of (shot_type, start_frame, end_frame) tuples.
    Validates shot types and frame ranges.
    """
    if not os.path.isfile(csv_path):
        print(f"  [ERROR] CSV not found: {csv_path}")
        return None

    labels = []
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        for line_num, row in enumerate(reader, 1):
            # Skip empty lines and comments
            if not row or row[0].strip().startswith("#"):
                continue
            if len(row) < 3:
                print(f"  [WARN] Line {line_num}: expected 3 columns, got {len(row)} — skipping")
                continue

            shot_type = row[0].strip().lower()
            if shot_type not in shot_types:
                print(f"  [WARN] Line {line_num}: unknown type '{shot_type}' — skipping")
                continue

            try:
                start = int(row[1].strip())
                end = int(row[2].strip())
            except ValueError:
                print(f"  [WARN] Line {line_num}: invalid frame numbers — skipping")
                continue

            if start < 0 or end < 0 or start >= end:
                print(f"  [WARN] Line {line_num}: invalid range {start}-{end} — skipping")
                continue

            if end > max_frame:
                print(f"  [WARN] Line {line_num}: end frame {end} exceeds max {max_frame} — skipping")
                continue

            labels.append((shot_type, start, end))

    return labels


# ── Display ──────────────────────────────────────────────────


def display_progress(clip_counts, target=50):
    """Print progress bars per shot type."""
    bar_width = 20
    print("\nCurrent progress:")
    for shot_type in SHOT_TYPES:
        count = clip_counts.get(shot_type, 0)
        filled = min(int(bar_width * count / target), bar_width) if target > 0 else 0
        bar = "#" * filled + "." * (bar_width - filled)
        label = f"  {shot_type}:"
        print(f"{label:<14} [{bar}] {count}/{target}")
    print()


# ── Interactive Mode ─────────────────────────────────────────


def _parse_frame_input(text, fps):
    """Parse a frame number or seconds input (e.g. '1200' or '20.0s').

    Returns integer frame number, or None on error.
    """
    text = text.strip()
    if not text:
        return None

    if text.lower().endswith("s"):
        try:
            seconds = float(text[:-1])
            return int(seconds * fps)
        except ValueError:
            return None
    else:
        try:
            return int(text)
        except ValueError:
            return None


def interactive_mode(pose_data, source_video, training_dir, stride, seq_len):
    """Primary interactive labeling loop."""
    frames = pose_data["frames"]
    video_info = pose_data["video_info"]
    fps = video_info["fps"]
    total_frames = video_info["total_frames"]
    duration = video_info.get("duration_seconds", total_frames / fps)
    pose_file = pose_data.get("source_pose_file", os.path.splitext(source_video)[0] + ".json")

    # Get view angle for this video
    view_angle = get_view_angle(source_video)

    # Shortcut map
    shortcut_map = {
        "f": "forehand",
        "b": "backhand",
        "s": "serve",
        "n": "neutral",
    }

    print("=" * 50)
    print("=== Tennis Shot Labeling Tool ===")
    print("=" * 50)
    print(f"Video: {source_video} ({total_frames} frames, {duration:.1f}s @ {fps:.0f}fps)")

    clip_counts = count_clips_per_type(training_dir)
    display_progress(clip_counts)

    print("Commands: [f]orehand  [b]ackhand  [s]erve  [n]eutral  [status]  [q]uit")
    print(f"Window: {seq_len} frames, stride: {stride}")
    print(f"Input frames or seconds (e.g. 1200 or 20.0s)")
    print()

    while True:
        try:
            cmd = input("> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nQuitting.")
            break

        if not cmd:
            continue

        if cmd == "q" or cmd == "quit":
            print("Done.")
            break

        if cmd == "status":
            clip_counts = count_clips_per_type(training_dir)
            display_progress(clip_counts)
            continue

        # Resolve shot type
        shot_type = shortcut_map.get(cmd, cmd if cmd in SHOT_TYPES else None)
        if shot_type is None:
            print(f"  Unknown command: '{cmd}'")
            print("  Use [f]orehand [b]ackhand [s]erve [n]eutral [status] [q]uit")
            continue

        # Get start frame
        try:
            start_input = input(f"  Start (frame or seconds, e.g. 1200 or 20.0s): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nQuitting.")
            break

        start_frame = _parse_frame_input(start_input, fps)
        if start_frame is None:
            print("  [ERROR] Invalid start input")
            continue

        # Get end frame
        try:
            end_input = input(f"  End (frame or seconds): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nQuitting.")
            break

        end_frame = _parse_frame_input(end_input, fps)
        if end_frame is None:
            print("  [ERROR] Invalid end input")
            continue

        # Validate range
        if start_frame >= end_frame:
            print(f"  [ERROR] Start ({start_frame}) must be before end ({end_frame})")
            continue

        if start_frame < 0:
            print(f"  [ERROR] Start frame cannot be negative")
            continue

        if end_frame >= total_frames:
            print(f"  [ERROR] End frame {end_frame} exceeds video length ({total_frames} frames)")
            continue

        range_frames = end_frame - start_frame
        range_seconds = range_frames / fps

        # Generate windows
        windows = generate_clip_windows(start_frame, end_frame, seq_len, stride)
        if not windows:
            print(f"  [ERROR] Range too short for {seq_len}-frame windows")
            continue

        print(f"  Range: frame {start_frame} - {end_frame} ({range_frames} frames, {range_seconds:.2f}s)")
        print(f"  Windows (stride={stride}): {len(windows)} clips")

        # Confirm
        try:
            confirm = input("  Confirm? [Y/n]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nQuitting.")
            break

        if confirm and confirm != "y":
            print("  Skipped.")
            continue

        # Process windows
        saved = 0
        skipped = 0
        for w_start, w_end in windows:
            valid, missing, total = validate_window(frames, w_start, w_end)
            if not valid:
                skipped += 1
                continue

            interp_frames, interp_indices = interpolate_missing_frames(frames, w_start, w_end)

            clip_data = {
                "version": 2,
                "source_video": source_video,
                "source_pose_file": pose_file,
                "shot_type": shot_type,
                "view_angle": view_angle,
                "start_frame": w_start,
                "end_frame": w_end,
                "sequence_length": len(interp_frames),
                "features_per_frame": 99,
                "num_interpolated": len(interp_indices),
                "interpolated_frame_indices": interp_indices,
                "frames": interp_frames,
            }

            save_clip(clip_data, shot_type, training_dir)
            saved += 1

        if skipped > 0:
            print(f"  [OK] Saved {saved} {shot_type} clips ({skipped} skipped — >30% missing)")
        else:
            print(f"  [OK] Saved {saved} {shot_type} clips")

        # Refresh counts
        clip_counts = count_clips_per_type(training_dir)
        print(f"  Total {shot_type}: {clip_counts[shot_type]}")
        print()


# ── Main ─────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Label shot segments from pose data for GRU training."
    )
    parser.add_argument(
        "--pose",
        help="Specific pose JSON file to label (default: auto-detect)",
    )
    parser.add_argument(
        "--csv",
        help="CSV file with labels (columns: shot_type, start_frame, end_frame)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=15,
        help="Sliding window stride in frames (default: 15)",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show clip counts only, then exit",
    )
    args = parser.parse_args()

    seq_len = MODEL["sequence_length"]

    # ── Status-only mode ──────────────────────────────────────
    if args.status:
        clip_counts = count_clips_per_type(TRAINING_DATA_DIR)
        display_progress(clip_counts)
        total = sum(clip_counts.values())
        print(f"Total clips: {total}")
        return

    # ── Find pose file ────────────────────────────────────────
    if args.pose:
        pose_path = args.pose
        if not os.path.isabs(pose_path):
            pose_path = os.path.join(POSES_DIR, pose_path)
    else:
        pose_files = find_pose_files(POSES_DIR)
        if not pose_files:
            print(f"No pose files found in {POSES_DIR}/")
            print("Run extract_poses.py first.")
            sys.exit(1)

        if len(pose_files) == 1:
            pose_path = os.path.join(POSES_DIR, pose_files[0])
            print(f"Auto-selected: {pose_files[0]}")
        else:
            print("Available pose files:")
            for i, f in enumerate(pose_files, 1):
                size = os.path.getsize(os.path.join(POSES_DIR, f))
                print(f"  {i}. {f} ({format_size(size)})")
            try:
                choice = input(f"Select [1-{len(pose_files)}]: ").strip()
                idx = int(choice) - 1
                if idx < 0 or idx >= len(pose_files):
                    print("Invalid selection.")
                    sys.exit(1)
            except (ValueError, EOFError, KeyboardInterrupt):
                print("\nAborted.")
                sys.exit(1)
            pose_path = os.path.join(POSES_DIR, pose_files[idx])

    # ── Load pose data ────────────────────────────────────────
    print(f"Loading {os.path.basename(pose_path)}...")
    t0 = time.time()
    pose_data = load_pose_data(pose_path)
    if pose_data is None:
        sys.exit(1)

    elapsed = time.time() - t0
    source_video = pose_data["source_video"]
    total_frames = pose_data["video_info"]["total_frames"]
    pose_data["source_pose_file"] = os.path.basename(pose_path)
    print(f"  Loaded {total_frames} frames in {elapsed:.1f}s")

    # Ensure training directories exist
    for shot_type in SHOT_TYPES:
        os.makedirs(os.path.join(TRAINING_DATA_DIR, shot_type), exist_ok=True)

    # ── CSV batch mode ────────────────────────────────────────
    if args.csv:
        labels = parse_labels_csv(args.csv, SHOT_TYPES, total_frames - 1)
        if labels is None:
            sys.exit(1)
        if not labels:
            print("  No valid labels found in CSV.")
            sys.exit(1)

        # Get view angle for this video
        view_angle = get_view_angle(source_video)

        print(f"\nProcessing {len(labels)} label(s) from CSV...")
        print(f"  View angle: {view_angle}")
        frames = pose_data["frames"]
        total_saved = 0

        for shot_type, start, end in labels:
            windows = generate_clip_windows(start, end, seq_len, args.stride)
            saved = 0
            for w_start, w_end in windows:
                valid, missing, total = validate_window(frames, w_start, w_end)
                if not valid:
                    continue

                interp_frames, interp_indices = interpolate_missing_frames(
                    frames, w_start, w_end
                )
                clip_data = {
                    "version": 2,
                    "source_video": source_video,
                    "source_pose_file": os.path.basename(pose_path),
                    "shot_type": shot_type,
                    "view_angle": view_angle,
                    "start_frame": w_start,
                    "end_frame": w_end,
                    "sequence_length": len(interp_frames),
                    "features_per_frame": 99,
                    "num_interpolated": len(interp_indices),
                    "interpolated_frame_indices": interp_indices,
                    "frames": interp_frames,
                }
                save_clip(clip_data, shot_type, TRAINING_DATA_DIR)
                saved += 1

            total_saved += saved
            print(f"  {shot_type}: frames {start}-{end} → {saved} clips")

        print(f"\n[OK] Saved {total_saved} clips total")
        clip_counts = count_clips_per_type(TRAINING_DATA_DIR)
        display_progress(clip_counts)
        return

    # ── Interactive mode (primary) ────────────────────────────
    interactive_mode(
        pose_data,
        source_video,
        TRAINING_DATA_DIR,
        args.stride,
        seq_len,
    )


if __name__ == "__main__":
    main()
