#!/usr/bin/env python3
"""Run the trained GRU model on a full video's pose data to detect shots."""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import POSES_DIR, MODELS_DIR, PROJECT_ROOT, VIEW_ANGLES, DEFAULT_VIEW_ANGLE
from scripts.video_metadata import get_view_angle


def load_model_and_meta(model_dir):
    """Load the trained .h5 model and its metadata JSON."""
    model_path = os.path.join(model_dir, "shot_classifier.h5")
    meta_path = os.path.join(model_dir, "shot_classifier_meta.json")

    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found: {model_path}")
        sys.exit(1)
    if not os.path.exists(meta_path):
        print(f"[ERROR] Metadata not found: {meta_path}")
        sys.exit(1)

    import numpy as np
    from tensorflow import keras

    model = keras.models.load_model(model_path)

    with open(meta_path) as f:
        meta = json.load(f)

    mean = np.array(meta["normalization"]["mean"], dtype=np.float32)
    std = np.array(meta["normalization"]["std"], dtype=np.float32)
    std[std < 1e-8] = 1.0

    return model, meta, mean, std


def load_pose_frames(pose_path):
    """Load pose JSON and extract world_landmarks_xyz per frame.

    Returns list of length total_frames. Each entry is either a list of
    33 [x,y,z] landmarks, or None if no pose was detected.
    """
    with open(pose_path) as f:
        data = json.load(f)

    video_info = data.get("video_info", {})
    fps = video_info.get("fps", 60.0)
    raw_frames = data["frames"]
    total = video_info.get("total_frames", len(raw_frames))

    # Build indexed array (some frames may be missing from the list)
    frames = [None] * total
    for fr in raw_frames:
        idx = fr["frame_idx"]
        if idx < total and fr.get("detected") and fr.get("world_landmarks"):
            frames[idx] = [lm[:3] for lm in fr["world_landmarks"]]

    return frames, fps, total


def run_inference(model, frames, mean, std, seq_len, stride, inverse_label_map, view_angle_one_hot=None):
    """Slide a window over all frames and predict shot type for each window.

    Args:
        model: Trained Keras model
        frames: List of pose landmarks per frame
        mean, std: Normalization parameters
        seq_len: Sequence length for model input
        stride: Window stride
        inverse_label_map: Dict mapping int label to string
        view_angle_one_hot: 5-element one-hot for view angle (None for old 99-feature models)

    Returns list of (center_frame, predicted_label, confidence) tuples.
    """
    import numpy as np

    total = len(frames)
    predictions = []

    # Determine feature count based on model
    n_features = 99
    if view_angle_one_hot is not None:
        n_features = 104

    for start in range(0, total - seq_len + 1, stride):
        window = []
        for i in range(start, start + seq_len):
            if frames[i] is not None:
                flat = []
                for kp in frames[i]:
                    flat.extend(kp[:3])
                while len(flat) < 99:
                    flat.append(0.0)
                pose_features = flat[:99]
            else:
                pose_features = [0.0] * 99

            # Append view_angle one-hot if using 104-feature model
            if view_angle_one_hot is not None:
                frame_features = pose_features + view_angle_one_hot
            else:
                frame_features = pose_features

            window.append(frame_features)

        X = np.array([window], dtype=np.float32)
        X = (X - mean) / std

        probs = model.predict(X, verbose=0)[0]
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        label = inverse_label_map[str(pred_idx)]

        center = start + seq_len // 2
        predictions.append((center, label, confidence))

    return predictions


def merge_segments(predictions, fps, min_gap_frames=15, min_segment_frames=15):
    """Merge consecutive same-label predictions into segments.

    Returns list of dicts with shot_type, start_frame, end_frame,
    start_time, end_time, avg_confidence.
    """
    if not predictions:
        return []

    segments = []
    current_label = predictions[0][1]
    current_start = predictions[0][0]
    current_end = predictions[0][0]
    confidences = [predictions[0][2]]

    for center, label, conf in predictions[1:]:
        if label == current_label and (center - current_end) <= min_gap_frames:
            current_end = center
            confidences.append(conf)
        else:
            segments.append({
                "shot_type": current_label,
                "start_frame": current_start,
                "end_frame": current_end,
                "start_time": round(current_start / fps, 2),
                "end_time": round(current_end / fps, 2),
                "avg_confidence": round(sum(confidences) / len(confidences), 3),
            })
            current_label = label
            current_start = center
            current_end = center
            confidences = [conf]

    # Final segment
    segments.append({
        "shot_type": current_label,
        "start_frame": current_start,
        "end_frame": current_end,
        "start_time": round(current_start / fps, 2),
        "end_time": round(current_end / fps, 2),
        "avg_confidence": round(sum(confidences) / len(confidences), 3),
    })

    # Filter out very short segments
    segments = [s for s in segments if (s["end_frame"] - s["start_frame"]) >= min_segment_frames]

    return segments


def print_results(segments, fps, total_frames):
    """Print detection results in a readable table."""
    print()
    print(f"{'Type':<10} {'Start':>8} {'End':>8} {'Duration':>8} {'Confidence':>10}")
    print("-" * 50)

    counts = {}
    for seg in segments:
        st = seg["shot_type"]
        counts[st] = counts.get(st, 0) + 1
        duration = seg["end_time"] - seg["start_time"]
        start_ts = f"{seg['start_time']:.1f}s"
        end_ts = f"{seg['end_time']:.1f}s"
        dur_ts = f"{duration:.1f}s"
        conf = f"{seg['avg_confidence']:.1%}"
        print(f"  {st:<10} {start_ts:>6} {end_ts:>8} {dur_ts:>8} {conf:>10}")

    print("-" * 50)
    total_duration = total_frames / fps
    print(f"  Total: {len(segments)} segments in {total_duration:.1f}s video")
    for st, c in sorted(counts.items()):
        print(f"    {st}: {c}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Detect shots in a full video using the trained model")
    parser.add_argument("pose_json", nargs="?", help="Path to pose JSON (auto-discovers if omitted)")
    parser.add_argument("-o", "--output", help="Output JSON path (default: shots_detected.json)")
    parser.add_argument("--stride", type=int, default=5, help="Window stride in frames (default: 5)")
    parser.add_argument("--min-confidence", type=float, default=0.0,
                        help="Minimum confidence to include a segment (default: 0.0)")
    parser.add_argument("--view-angle", choices=VIEW_ANGLES,
                        help=f"Override view angle (default: load from metadata or {DEFAULT_VIEW_ANGLE})")
    args = parser.parse_args()

    # Find pose JSON
    if args.pose_json:
        pose_path = args.pose_json
    else:
        jsons = sorted([
            os.path.join(POSES_DIR, f)
            for f in os.listdir(POSES_DIR)
            if f.endswith(".json")
        ])
        if not jsons:
            print("[ERROR] No pose JSON files found in", POSES_DIR)
            sys.exit(1)
        pose_path = jsons[0]
        print(f"Auto-selected: {os.path.basename(pose_path)}")

    video_name = os.path.splitext(os.path.basename(pose_path))[0]
    output_path = args.output or os.path.join(PROJECT_ROOT, "shots_detected.json")

    print(f"Loading model...")
    model, meta, mean, std = load_model_and_meta(MODELS_DIR)

    seq_len = meta["sequence_length"]
    n_features = meta.get("num_features", 99)
    inverse_label_map = meta["inverse_label_map"]
    label_map = meta["label_map"]
    view_angle_map = meta.get("view_angle_map", None)

    print(f"  Classes: {list(label_map.keys())}")
    print(f"  Sequence length: {seq_len} frames")
    print(f"  Features: {n_features}")
    print()

    # Determine view angle
    if args.view_angle:
        view_angle = args.view_angle
        print(f"View angle (CLI override): {view_angle}")
    else:
        view_angle = get_view_angle(video_name)
        print(f"View angle (from metadata): {view_angle}")

    # Build view_angle one-hot if model supports it (104 features)
    view_angle_one_hot = None
    if n_features == 104 and view_angle_map is not None:
        if view_angle not in view_angle_map:
            print(f"  [WARN] Unknown view angle '{view_angle}', using {DEFAULT_VIEW_ANGLE}")
            view_angle = DEFAULT_VIEW_ANGLE
        view_angle_idx = view_angle_map[view_angle]
        view_angle_one_hot = [0.0] * len(VIEW_ANGLES)
        view_angle_one_hot[view_angle_idx] = 1.0
    elif n_features == 99:
        print("  [INFO] Using legacy 99-feature model (view angle not supported)")
    print()

    print(f"Loading poses: {os.path.basename(pose_path)}...")
    frames, fps, total_frames = load_pose_frames(pose_path)
    detected = sum(1 for f in frames if f is not None)
    print(f"  {total_frames} frames, {fps} fps, {detected} with pose")
    print()

    print(f"Running inference (stride={args.stride})...")
    predictions = run_inference(model, frames, mean, std, seq_len, args.stride, inverse_label_map, view_angle_one_hot)
    print(f"  {len(predictions)} windows evaluated")

    # Merge into segments
    segments = merge_segments(predictions, fps)

    # Filter by confidence
    if args.min_confidence > 0:
        before = len(segments)
        segments = [s for s in segments if s["avg_confidence"] >= args.min_confidence]
        if len(segments) < before:
            print(f"  Filtered {before - len(segments)} low-confidence segments")

    print_results(segments, fps, total_frames)

    # Save output
    output = {
        "version": 2,
        "source_video": video_name,
        "pose_json": os.path.basename(pose_path),
        "model": meta["model_file"],
        "view_angle": view_angle,
        "fps": fps,
        "total_frames": total_frames,
        "stride": args.stride,
        "segments": segments,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved {len(segments)} segments to {output_path}")


if __name__ == "__main__":
    main()
