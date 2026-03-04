#!/usr/bin/env python3
"""Sliding window shot detector.

Slides a window across the entire video, extracts temporal features at
each position, scores with the window detector model, and applies NMS
to produce final detections.

Outputs the same JSON format as fused_detect.py for compatibility with
validate_pipeline.py.

Usage:
    .venv/bin/python scripts/window_detect.py preprocessed/IMG_6703.mp4
    .venv/bin/python scripts/window_detect.py preprocessed/IMG_6703.mp4 --step 0.1 --threshold 0.5
    .venv/bin/python scripts/window_detect.py --all  # run all GT videos
"""

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from scripts.extract_window_features import (
    load_pose_data, extract_window_features, extract_audio_rms,
    GT_VIDEOS, IGNORE_TYPES, KEY_JOINTS, JOINT_NAMES,
)

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DETECTIONS_DIR = os.path.join(PROJECT_ROOT, "detections")
POSES_DIR = os.path.join(PROJECT_ROOT, "poses_full_videos")


def load_model():
    """Load window detector model and metadata."""
    model_path = os.path.join(MODELS_DIR, "window_detector.pkl")
    meta_path = os.path.join(MODELS_DIR, "window_detector_meta.json")

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(meta_path) as f:
        meta = json.load(f)

    return model, meta


def nms(detections, min_gap=1.5):
    """Non-maximum suppression on detection list.

    Greedily selects highest-confidence detection, suppresses all
    detections within min_gap seconds, repeats.

    Args:
        detections: list of dicts with 'timestamp' and 'confidence'
        min_gap: minimum seconds between detections

    Returns: filtered list of detections
    """
    if not detections:
        return []

    # Sort by confidence descending
    sorted_dets = sorted(detections, key=lambda d: d["confidence"], reverse=True)

    selected = []
    suppressed = set()

    for i, det in enumerate(sorted_dets):
        if i in suppressed:
            continue
        selected.append(det)
        # Suppress all lower-confidence detections within min_gap
        for j in range(i + 1, len(sorted_dets)):
            if j in suppressed:
                continue
            if abs(sorted_dets[j]["timestamp"] - det["timestamp"]) < min_gap:
                suppressed.add(j)

    # Sort selected by timestamp
    selected.sort(key=lambda d: d["timestamp"])
    return selected


def detect_video(video_path, model, meta, step=0.2, threshold=0.5,
                 nms_gap=1.5, verbose=True):
    """Run sliding window detection on a video.

    Args:
        video_path: path to preprocessed video
        model: trained classifier
        meta: model metadata (feature_names, etc.)
        step: window step in seconds
        threshold: minimum shot probability to consider
        nms_gap: NMS suppression window in seconds
        verbose: print progress

    Returns: list of detection dicts
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    feature_names = meta["feature_names"]

    # Load pose data
    frames, fps, total_frames, duration = load_pose_data(video_name)
    if frames is None or len(frames) == 0:
        print(f"No pose data for {video_name}")
        return []

    half_window = 45  # ±45 frames = 1.5s window at 60fps

    # Load audio
    if verbose:
        print(f"  Extracting audio...")
    rms_envelope, time_per_sample = extract_audio_rms(video_path)
    if rms_envelope is not None and verbose:
        print(f"  Audio: {len(rms_envelope)} RMS samples")

    # Slide window across video
    step_frames = max(1, int(step * fps))
    start_frame = half_window
    end_frame = len(frames) - half_window

    if verbose:
        total_steps = (end_frame - start_frame) // step_frames
        print(f"  Sliding window: {start_frame} to {end_frame}, step={step_frames} frames, {total_steps} steps")

    shot_idx = list(model.classes_).index("shot")
    candidates = []

    t_start = time.time()
    steps_done = 0
    for center_frame in range(start_frame, end_frame, step_frames):
        # Extract features
        feats = extract_window_features(
            frames, fps, center_frame, half_window,
            rms_envelope, time_per_sample
        )

        # Build feature vector in correct order
        x = np.array([[feats.get(f, 0) for f in feature_names]])

        # Handle NaN/Inf
        nan_mask = ~np.isfinite(x)
        if nan_mask.any():
            x[nan_mask] = 0

        # Score
        probs = model.predict_proba(x)
        shot_prob = float(probs[0][shot_idx])

        if shot_prob >= threshold:
            timestamp = center_frame / fps
            candidates.append({
                "timestamp": round(timestamp, 3),
                "frame": center_frame,
                "confidence": round(shot_prob, 4),
                "shot_prob": round(shot_prob, 4),
            })

        steps_done += 1
        if verbose and steps_done % 100 == 0:
            elapsed = time.time() - t_start
            rate = steps_done / elapsed
            remaining = (total_steps - steps_done) / rate if rate > 0 else 0
            print(f"\r  Step {steps_done}/{total_steps} ({steps_done/total_steps:.0%}), "
                  f"{rate:.1f} steps/s, "
                  f"candidates={len(candidates)}, "
                  f"ETA={remaining:.0f}s", end="", flush=True)

    if verbose:
        elapsed = time.time() - t_start
        print(f"\r  Done: {steps_done} steps in {elapsed:.1f}s, "
              f"{len(candidates)} candidates above threshold")

    # Apply NMS
    if verbose:
        print(f"  Applying NMS (gap={nms_gap}s)...")
    detections = nms(candidates, min_gap=nms_gap)

    if verbose:
        print(f"  After NMS: {len(detections)} detections")

    # Format output
    for det in detections:
        det["shot_type"] = "forehand"  # placeholder — window detector doesn't classify type
        det["source"] = "window_detector"
        det["tier"] = "medium"

    return detections


def format_output(video_path, detections, fps=60.0, total_frames=0, duration=0):
    """Format detections in the same JSON structure as fused_detect.py."""
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    output = {
        "version": 1,
        "detector": "window_detector",
        "source_video": video_name,
        "video_path": os.path.abspath(video_path),
        "pose_path": os.path.join(POSES_DIR, f"{video_name}.json"),
        "fps": fps,
        "total_frames": total_frames,
        "duration": duration,
        "parameters": {
            "window_sec": 1.5,
            "step_sec": 0.2,
            "nms_gap_sec": 1.5,
        },
        "summary": {
            "total_detections": len(detections),
            "by_type": {},
            "by_source": {"window_detector": len(detections)},
        },
        "detections": detections,
    }

    # Count by type
    for det in detections:
        t = det.get("shot_type", "unknown")
        output["summary"]["by_type"][t] = output["summary"]["by_type"].get(t, 0) + 1

    return output


def main():
    parser = argparse.ArgumentParser(description="Sliding window shot detector")
    parser.add_argument("video", nargs="?", help="Path to preprocessed video")
    parser.add_argument("--all", action="store_true", help="Run on all GT videos")
    parser.add_argument("--step", type=float, default=0.2, help="Step size in seconds (default: 0.2)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection threshold (default: 0.5)")
    parser.add_argument("--nms-gap", type=float, default=1.5, help="NMS gap in seconds (default: 1.5)")
    parser.add_argument("--output-suffix", default="_window_detections.json",
                        help="Output filename suffix")
    args = parser.parse_args()

    if not args.video and not args.all:
        parser.error("Provide a video path or use --all")

    model, meta = load_model()
    print(f"Loaded model: {meta['n_features']} features, threshold={args.threshold}")

    if args.all:
        videos = []
        for video_name in sorted(GT_VIDEOS.keys()):
            video_path = os.path.join(PROJECT_ROOT, "preprocessed", f"{video_name}.mp4")
            if os.path.exists(video_path):
                videos.append((video_name, video_path))
            else:
                print(f"Skipping {video_name} (no preprocessed video)")
        print(f"Processing {len(videos)} videos...")
    else:
        video_name = os.path.splitext(os.path.basename(args.video))[0]
        videos = [(video_name, args.video)]

    for video_name, video_path in videos:
        print(f"\n{'='*60}")
        print(f"Processing {video_name}")
        print(f"{'='*60}")

        detections = detect_video(
            video_path, model, meta,
            step=args.step,
            threshold=args.threshold,
            nms_gap=args.nms_gap,
        )

        # Get video info for output
        frames, fps, total_frames, duration = load_pose_data(video_name)

        output = format_output(video_path, detections, fps, total_frames, duration)

        out_path = os.path.join(DETECTIONS_DIR, f"{video_name}{args.output_suffix}")
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"  Saved {len(detections)} detections to {out_path}")


if __name__ == "__main__":
    main()
