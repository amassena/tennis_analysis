#!/usr/bin/env python3
"""Mine hard negatives from false positive detections.

Runs validation, collects FP timestamps, extracts their pose windows,
and appends them to the training NPZ as not_shot samples.

Usage:
    .venv/bin/python scripts/mine_hard_negatives.py
    .venv/bin/python scripts/mine_hard_negatives.py --input training/sequence_data.npz --output training/sequence_data_v2.npz
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
from scripts.prepare_sequence_data import extract_normalized_window
from scripts.validate_pipeline import validate_video, GT_VIDEOS

DETECTIONS_DIR = os.path.join(PROJECT_ROOT, "detections")


def collect_false_positives():
    """Run validation on all GT videos, collect FP timestamps.

    Returns list of (video_name, timestamp, shot_type, confidence) tuples.
    """
    fps_list = []

    for video_name in sorted(GT_VIDEOS.keys()):
        r = validate_video(video_name, verbose=False)
        if not r:
            continue

        for fp in r.get("false_positives", []):
            fps_list.append((
                video_name,
                fp["timestamp"],
                fp.get("shot_type", "unknown"),
                fp.get("confidence", 0),
            ))

    return fps_list


def extract_fp_windows(fps_list):
    """Extract normalized pose windows for each FP.

    Returns (X_fp, videos_fp, timestamps_fp) arrays.
    """
    pose_cache = {}
    X_list = []
    videos_list = []
    timestamps_list = []

    for video_name, timestamp, shot_type, conf in fps_list:
        # Load poses
        if video_name not in pose_cache:
            pose_path = os.path.join(POSES_DIR, f"{video_name}.json")
            if not os.path.exists(pose_path):
                continue
            frames, fps, total = load_pose_frames(pose_path)
            pose_cache[video_name] = (frames, fps)

        frames, fps = pose_cache[video_name]
        frame_idx = int(round(timestamp * fps))

        seq = extract_normalized_window(frames, frame_idx)
        if seq is not None:
            X_list.append(seq)
            videos_list.append(video_name)
            timestamps_list.append(timestamp)

    if not X_list:
        return np.array([]), np.array([]), np.array([])

    return (
        np.array(X_list, dtype=np.float32),
        np.array(videos_list),
        np.array(timestamps_list, dtype=np.float32),
    )


def main():
    parser = argparse.ArgumentParser(description="Mine hard negatives from FPs")
    parser.add_argument("--input", default=os.path.join(TRAINING_DIR, "sequence_data.npz"),
                        help="Original NPZ file")
    parser.add_argument("--output", default=os.path.join(TRAINING_DIR, "sequence_data_v2.npz"),
                        help="Output NPZ with hard negatives appended")
    args = parser.parse_args()

    # Load original data
    print(f"Loading original data from {args.input}...")
    orig = np.load(args.input, allow_pickle=True)
    X_orig = orig["X"]
    y_orig = orig["y"]
    videos_orig = orig["videos"]
    timestamps_orig = orig["timestamps"]

    print(f"  Original: {len(X_orig)} samples")
    not_shot_idx = CLASS_TO_IDX["not_shot"]
    n_orig_neg = np.sum(y_orig == not_shot_idx)
    print(f"  Original not_shot: {n_orig_neg}")

    # Collect FPs
    print(f"\nCollecting false positives from validation...")
    fps_list = collect_false_positives()
    print(f"  Found {len(fps_list)} false positives")

    if not fps_list:
        print("No FPs found. Nothing to do.")
        return

    # Show breakdown
    from collections import Counter
    fp_by_video = Counter(v for v, _, _, _ in fps_list)
    print(f"\n  FPs by video:")
    for video, count in fp_by_video.most_common(10):
        print(f"    {video}: {count}")
    if len(fp_by_video) > 10:
        print(f"    ... and {len(fp_by_video) - 10} more videos")

    # Extract windows
    print(f"\nExtracting pose windows for FPs...")
    X_fp, videos_fp, timestamps_fp = extract_fp_windows(fps_list)
    print(f"  Extracted {len(X_fp)} windows")

    if len(X_fp) == 0:
        print("No valid windows extracted. Nothing to do.")
        return

    # Merge
    y_fp = np.full(len(X_fp), not_shot_idx, dtype=np.int64)

    X_merged = np.concatenate([X_orig, X_fp], axis=0)
    y_merged = np.concatenate([y_orig, y_fp], axis=0)
    videos_merged = np.concatenate([videos_orig, videos_fp])
    timestamps_merged = np.concatenate([timestamps_orig, timestamps_fp])

    # Save
    np.savez_compressed(
        args.output,
        X=X_merged, y=y_merged, videos=videos_merged, timestamps=timestamps_merged
    )

    # Summary
    from scripts.sequence_model import CLASSES
    label_counts = Counter(y_merged)
    print(f"\n{'='*50}")
    print(f"Saved {len(X_merged)} samples to {args.output}")
    print(f"  Added {len(X_fp)} hard negatives (not_shot)")
    print(f"  New not_shot count: {np.sum(y_merged == not_shot_idx)} (was {n_orig_neg})")
    print(f"\nClass distribution:")
    for idx in sorted(label_counts.keys()):
        print(f"  {CLASSES[idx]}: {label_counts[idx]}")
    print(f"\nFile size: {os.path.getsize(args.output) / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
