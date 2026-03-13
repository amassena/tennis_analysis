#!/usr/bin/env python3
"""Prepare training data for sequence model.

Extracts 90-frame pose windows from GT files + pose JSONs, saves as NPZ.

Positive samples: centered on labeled shots (forehand, backhand, serve).
Negative samples (3:1 ratio):
  - Random background (2x): windows >=3s from any shot
  - Hard negatives (0.5x): windows 1.5-3.0s from shots
  - Edge negatives (0.5x): windows offset 1.0-1.5s from shot center

Usage:
    .venv/bin/python scripts/prepare_sequence_data.py
    .venv/bin/python scripts/prepare_sequence_data.py --neg-ratio 4.0
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import PROJECT_ROOT, POSES_DIR, TRAINING_DIR
from scripts.extract_training_features import load_pose_frames
from scripts.sequence_model import (
    SEQUENCE_LENGTH, FEATURES_PER_FRAME, NUM_LANDMARKS,
    LEFT_HIP_IDX, RIGHT_HIP_IDX, CLASS_TO_IDX,
)

DETECTIONS_DIR = os.path.join(PROJECT_ROOT, "detections")

# Shot types to include as positives
TRAINABLE_TYPES = {"forehand", "backhand", "serve"}

# Serve-only videos: skip FH/BH (may be mislabeled)
SERVE_ONLY_VIDEOS = {
    "IMG_0864", "IMG_0865", "IMG_0866", "IMG_0867",
    "IMG_0868", "IMG_0869", "IMG_0870",
}

# Videos to exclude entirely
EXCLUDE_VIDEOS = {
    "IMG_0993",  # GT timestamps invalid (slo-mo mismatch)
    "IMG_0995",  # GT = detection file (identical)
    "IMG_6712",  # 0 shots
}

# Shoulder indices for normalization
LEFT_SHOULDER_IDX = 11
RIGHT_SHOULDER_IDX = 12


def extract_normalized_window(frames, center_frame):
    """Extract a 90-frame window, hip-centered and shoulder-normalized.

    Args:
        frames: List of pose frame dicts
        center_frame: Frame index at window center

    Returns:
        (90, 99) float32 array, or None if insufficient pose data
    """
    n = len(frames)
    half = SEQUENCE_LENGTH // 2
    start = center_frame - half

    sequence = np.zeros((SEQUENCE_LENGTH, FEATURES_PER_FRAME), dtype=np.float32)
    valid_count = 0

    for seq_idx in range(SEQUENCE_LENGTH):
        frame_idx = start + seq_idx
        if frame_idx < 0 or frame_idx >= n:
            continue

        frame = frames[frame_idx]
        if not frame:
            continue

        landmarks = frame.get("world_landmarks")
        if not landmarks or len(landmarks) < NUM_LANDMARKS:
            landmarks = frame.get("landmarks")
            if not landmarks or len(landmarks) < NUM_LANDMARKS:
                continue

        coords = []
        for lm in landmarks[:NUM_LANDMARKS]:
            if isinstance(lm, dict):
                coords.extend([lm.get("x", 0), lm.get("y", 0), lm.get("z", 0)])
            elif isinstance(lm, (list, tuple)):
                coords.extend(list(lm[:3]))
                while len(coords) % 3 != 0:
                    coords.append(0.0)
            else:
                coords.extend([0.0, 0.0, 0.0])

        if len(coords) >= FEATURES_PER_FRAME:
            sequence[seq_idx] = coords[:FEATURES_PER_FRAME]
            valid_count += 1

    # Need at least 30% valid frames
    if valid_count < SEQUENCE_LENGTH * 0.3:
        return None

    # Hip-center normalization
    for i in range(SEQUENCE_LENGTH):
        lh_start = LEFT_HIP_IDX * 3
        rh_start = RIGHT_HIP_IDX * 3
        lh = sequence[i, lh_start:lh_start + 3]
        rh = sequence[i, rh_start:rh_start + 3]

        if np.sum(np.abs(lh)) < 1e-8 and np.sum(np.abs(rh)) < 1e-8:
            continue

        hip_center = (lh + rh) / 2
        for j in range(NUM_LANDMARKS):
            s = j * 3
            sequence[i, s:s + 3] -= hip_center

    # Shoulder-width normalization
    for i in range(SEQUENCE_LENGTH):
        ls_start = LEFT_SHOULDER_IDX * 3
        rs_start = RIGHT_SHOULDER_IDX * 3
        ls = sequence[i, ls_start:ls_start + 3]
        rs = sequence[i, rs_start:rs_start + 3]

        shoulder_width = np.linalg.norm(ls - rs)
        if shoulder_width > 0.01:  # avoid div by zero
            sequence[i] /= shoulder_width

    return sequence


def load_video_data(video_name):
    """Load GT detections and pose frames for a video.

    Returns (detections_list, frames, fps) or (None, None, None) if missing.
    """
    # GT file
    gt_path = os.path.join(DETECTIONS_DIR, f"{video_name}_fused.json")
    if not os.path.exists(gt_path):
        return None, None, None

    # Pose file
    pose_path = os.path.join(POSES_DIR, f"{video_name}.json")
    if not os.path.exists(pose_path):
        return None, None, None

    with open(gt_path) as f:
        gt_data = json.load(f)

    frames, fps, total = load_pose_frames(pose_path)
    return gt_data, frames, fps


def main():
    parser = argparse.ArgumentParser(description="Prepare sequence training data")
    parser.add_argument("--neg-ratio", type=float, default=3.0,
                        help="Negative:positive ratio (default: 3.0)")
    parser.add_argument("--output", default=os.path.join(TRAINING_DIR, "sequence_data.npz"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Discover GT files
    gt_files = sorted(
        f for f in os.listdir(DETECTIONS_DIR)
        if f.endswith("_fused.json")
        and "_detections" not in f
        and "_v2" not in f and "_v5" not in f
        and "_ml" not in f and "_pre" not in f and "_baseline" not in f
    )

    all_X = []
    all_y = []
    all_videos = []
    all_timestamps = []

    total_positives = 0
    total_negatives = 0
    skipped_no_pose = []

    for gt_file in gt_files:
        video_name = gt_file.replace("_fused.json", "")

        if video_name in EXCLUDE_VIDEOS:
            print(f"  SKIP {video_name} (excluded)")
            continue

        gt_data, frames, fps = load_video_data(video_name)
        if gt_data is None:
            skipped_no_pose.append(video_name)
            continue

        is_serve_only = video_name in SERVE_ONLY_VIDEOS
        detections = gt_data.get("detections", [])

        # --- Positive samples ---
        shot_frames = []
        shot_timestamps = []
        positives_this_video = 0

        for det in detections:
            shot_type = det.get("shot_type", "")
            if shot_type not in TRAINABLE_TYPES:
                continue
            if is_serve_only and shot_type in ("forehand", "backhand"):
                continue

            frame_idx = det.get("frame", 0)
            timestamp = det.get("timestamp", frame_idx / fps)

            seq = extract_normalized_window(frames, frame_idx)
            if seq is None:
                continue

            label_idx = CLASS_TO_IDX.get(shot_type)
            if label_idx is None:
                continue

            all_X.append(seq)
            all_y.append(label_idx)
            all_videos.append(video_name)
            all_timestamps.append(timestamp)

            shot_frames.append(frame_idx)
            shot_timestamps.append(timestamp)
            positives_this_video += 1

        total_positives += positives_this_video

        if positives_this_video == 0:
            print(f"  {video_name}: 0 positives, skipping negatives")
            continue

        # --- Negative samples ---
        n_total_neg = int(positives_this_video * args.neg_ratio)
        n_random = int(n_total_neg * 0.67)    # 2/3 random background
        n_hard = int(n_total_neg * 0.17)      # 1/6 hard negatives
        n_edge = n_total_neg - n_random - n_hard  # 1/6 edge negatives

        n_frames = len(frames)
        not_shot_idx = CLASS_TO_IDX["not_shot"]

        # Random background: >=3s from any shot
        random_candidates = []
        step_frames = int(fps * 0.5)  # 0.5s grid
        for fi in range(SEQUENCE_LENGTH, n_frames - SEQUENCE_LENGTH, step_frames):
            t = fi / fps
            if all(abs(t - st) >= 3.0 for st in shot_timestamps):
                random_candidates.append(fi)

        if random_candidates:
            chosen = np.random.choice(
                random_candidates,
                size=min(n_random, len(random_candidates)),
                replace=False
            )
            for fi in chosen:
                seq = extract_normalized_window(frames, fi)
                if seq is not None:
                    all_X.append(seq)
                    all_y.append(not_shot_idx)
                    all_videos.append(video_name)
                    all_timestamps.append(fi / fps)
                    total_negatives += 1

        # Hard negatives: 1.5-3.0s from any shot
        hard_candidates = []
        for fi in range(SEQUENCE_LENGTH, n_frames - SEQUENCE_LENGTH, step_frames):
            t = fi / fps
            min_dist = min((abs(t - st) for st in shot_timestamps), default=999)
            if 1.5 <= min_dist <= 3.0:
                hard_candidates.append(fi)

        if hard_candidates:
            chosen = np.random.choice(
                hard_candidates,
                size=min(n_hard, len(hard_candidates)),
                replace=False
            )
            for fi in chosen:
                seq = extract_normalized_window(frames, fi)
                if seq is not None:
                    all_X.append(seq)
                    all_y.append(not_shot_idx)
                    all_videos.append(video_name)
                    all_timestamps.append(fi / fps)
                    total_negatives += 1

        # Edge negatives: offset 1.0-1.5s from shot center
        edge_count = 0
        for sf in shot_frames:
            if edge_count >= n_edge:
                break
            offset_sec = np.random.uniform(1.0, 1.5)
            sign = np.random.choice([-1, 1])
            fi = sf + int(sign * offset_sec * fps)
            if fi < SEQUENCE_LENGTH or fi >= n_frames - SEQUENCE_LENGTH:
                continue
            seq = extract_normalized_window(frames, fi)
            if seq is not None:
                all_X.append(seq)
                all_y.append(not_shot_idx)
                all_videos.append(video_name)
                all_timestamps.append(fi / fps)
                total_negatives += 1
                edge_count += 1

        neg_this = n_total_neg  # approximate
        print(f"  {video_name}: {positives_this_video} positives, ~{total_negatives} negatives so far")

    # Convert to arrays
    X = np.array(all_X, dtype=np.float32)
    y = np.array(all_y, dtype=np.int64)
    videos = np.array(all_videos)
    timestamps = np.array(all_timestamps, dtype=np.float32)

    # Save
    np.savez_compressed(
        args.output,
        X=X, y=y, videos=videos, timestamps=timestamps
    )

    # Summary
    from collections import Counter
    from scripts.sequence_model import CLASSES
    label_counts = Counter(y)

    print(f"\n{'='*50}")
    print(f"Saved {len(X)} samples to {args.output}")
    print(f"  Shape: X={X.shape}, y={y.shape}")
    print(f"  Positives: {total_positives}")
    print(f"  Negatives: {total_negatives}")
    print(f"  Ratio: {total_negatives/max(1,total_positives):.1f}:1")
    print(f"\nClass distribution:")
    for idx in sorted(label_counts.keys()):
        print(f"  {CLASSES[idx]}: {label_counts[idx]}")
    print(f"\nVideos: {len(set(all_videos))}")
    if skipped_no_pose:
        print(f"Skipped (no pose): {', '.join(skipped_no_pose)}")
    print(f"File size: {os.path.getsize(args.output) / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
