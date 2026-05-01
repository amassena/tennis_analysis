#!/usr/bin/env python3
"""Detect tennis shots using the 1D-CNN sequence model.

Slides a 90-frame window across pose data, runs batch inference, finds peaks
in P(shot), applies NMS. Three tunable parameters: threshold, NMS gap, step.

Replaces the 2,253-line fused_detect.py with ~250 lines.

Usage:
    .venv/bin/python scripts/detect_shots_sequence.py preprocessed/IMG_0866.mp4
    .venv/bin/python scripts/detect_shots_sequence.py --all
    .venv/bin/python scripts/detect_shots_sequence.py --all --threshold 0.5 --nms-gap 1.5
    .venv/bin/python scripts/detect_shots_sequence.py --all --sweep  # threshold sweep
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import PROJECT_ROOT, POSES_DIR, MODELS_DIR, PREPROCESSED_DIR
from scripts.extract_training_features import load_pose_frames
from scripts.sequence_model import (
    ShotClassifierCNN, CLASSES, CLASS_TO_IDX,
    SEQUENCE_LENGTH, FEATURES_PER_FRAME, NUM_LANDMARKS,
    LEFT_HIP_IDX, RIGHT_HIP_IDX, LEFT_SHOULDER_IDX, RIGHT_SHOULDER_IDX,
    load_model,
)

DETECTIONS_DIR = os.path.join(PROJECT_ROOT, "detections")

# Videos known to have invalid GT
EXCLUDE_VIDEOS = {"IMG_0993", "IMG_0995", "IMG_6712"}


def normalize_window(sequence):
    """Apply hip-centering and shoulder-width normalization to a (90, 99) array."""
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

    for i in range(SEQUENCE_LENGTH):
        ls_start = LEFT_SHOULDER_IDX * 3
        rs_start = RIGHT_SHOULDER_IDX * 3
        ls = sequence[i, ls_start:ls_start + 3]
        rs = sequence[i, rs_start:rs_start + 3]

        shoulder_width = np.linalg.norm(ls - rs)
        if shoulder_width > 0.01:
            sequence[i] /= shoulder_width

    return sequence


def extract_raw_window(frames, center_frame):
    """Extract raw 90-frame pose window (no normalization).

    Returns (90, 99) array or None if <30% valid frames.
    """
    n = len(frames)
    half = SEQUENCE_LENGTH // 2
    start = center_frame - half

    sequence = np.zeros((SEQUENCE_LENGTH, FEATURES_PER_FRAME), dtype=np.float32)
    valid = 0

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
            valid += 1

    if valid < SEQUENCE_LENGTH * 0.3:
        return None
    return sequence


def sliding_window_inference(model, device, frames, fps, step_sec=0.1, batch_size=256):
    """Run sliding window inference over all frames.

    Returns:
        timestamps: (N,) array of window center times
        probs: (N, 4) array of class probabilities
    """
    import torch

    n_frames = len(frames)
    step_frames = max(1, int(fps * step_sec))
    half = SEQUENCE_LENGTH // 2

    # Pre-extract all windows
    centers = list(range(half, n_frames - half, step_frames))
    windows = []
    valid_centers = []

    for center in centers:
        seq = extract_raw_window(frames, center)
        if seq is not None:
            seq = normalize_window(seq)
            windows.append(seq)
            valid_centers.append(center)

    if not windows:
        return np.array([]), np.array([])

    windows = np.array(windows, dtype=np.float32)
    timestamps = np.array(valid_centers, dtype=np.float32) / fps

    # Batch inference
    all_probs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(windows), batch_size):
            batch = torch.from_numpy(windows[i:i + batch_size]).to(device)
            logits = model(batch)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)

    probs = np.concatenate(all_probs, axis=0)
    return timestamps, probs


SHOT_CLASS_NAMES = ("backhand", "forehand", "serve")


def find_shots(timestamps, probs, threshold=0.5, nms_gap=1.5, prominence=0.1,
               per_class_thresholds=None, min_class_conf=0.0,
               demote_below_class_conf=None):
    """Find shot detections from probability curves.

    Default mode: peak-find on P(shot) = 1 - P(not_shot), then assign type
    via argmax over shot classes. Tuned by `threshold` (height of P(shot) peak).

    Per-class mode (when `per_class_thresholds` is set): peak-find separately
    on each class probability curve. Allows a lower bar for serves than
    forehand/backhand. NMS is global by class probability so overlapping
    peaks pick the most confident class.

    Args:
        threshold: P(shot) peak height (used when per_class_thresholds is None).
        nms_gap: minimum seconds between kept detections.
        prominence: scipy.signal.find_peaks prominence.
        per_class_thresholds: dict like {"serve": 0.55, "forehand": 0.70,
            "backhand": 0.70}. When provided, switches to per-class peak finding.
        min_class_conf: drop detections whose top-class probability is below this.
        demote_below_class_conf: if set, candidates whose top-class probability
            is below this floor have shot_type re-tagged as "unknown_shot"
            instead of being dropped. Useful to keep shot count accurate while
            letting downstream filters skip ambiguous types.

    Returns list of dicts with timestamp, shot_type, confidence, p_shot, probabilities.
    """
    from scipy.signal import find_peaks

    not_shot_idx = CLASS_TO_IDX["not_shot"]
    p_shot = 1.0 - probs[:, not_shot_idx]

    raw_detections = []

    if per_class_thresholds:
        # Peak-find each class probability curve independently
        for cname in SHOT_CLASS_NAMES:
            t_c = per_class_thresholds.get(cname)
            if t_c is None:
                continue
            cidx = CLASS_TO_IDX[cname]
            peaks, _ = find_peaks(probs[:, cidx], height=t_c, prominence=prominence)
            for peak_idx in peaks:
                shot_probs = probs[peak_idx]
                raw_detections.append({
                    "timestamp": round(float(timestamps[peak_idx]), 3),
                    "p_shot": float(p_shot[peak_idx]),
                    "shot_type": cname,
                    "confidence": float(shot_probs[cidx]),
                    "probabilities": {CLASSES[c]: round(float(shot_probs[c]), 4)
                                      for c in range(len(CLASSES))},
                })
    else:
        # Original mode: peak on P(shot), classify via argmax
        peaks, _ = find_peaks(p_shot, height=threshold, prominence=prominence)
        if len(peaks) == 0:
            return []
        shot_class_idxs = [CLASS_TO_IDX[c] for c in SHOT_CLASS_NAMES]
        for peak_idx in peaks:
            shot_probs = probs[peak_idx]
            best_class = max(shot_class_idxs, key=lambda c: shot_probs[c])
            raw_detections.append({
                "timestamp": round(float(timestamps[peak_idx]), 3),
                "p_shot": float(p_shot[peak_idx]),
                "shot_type": CLASSES[best_class],
                "confidence": float(shot_probs[best_class]),
                "probabilities": {CLASSES[c]: round(float(shot_probs[c]), 4)
                                  for c in range(len(CLASSES))},
            })

    if not raw_detections:
        return []

    if min_class_conf > 0.0:
        raw_detections = [d for d in raw_detections if d["confidence"] >= min_class_conf]
        if not raw_detections:
            return []

    # Greedy NMS: prefer higher confidence; per-class mode dedups overlapping classes
    sort_key = "confidence" if per_class_thresholds else "p_shot"
    raw_detections.sort(key=lambda d: d[sort_key], reverse=True)
    kept = []
    for det in raw_detections:
        if all(abs(det["timestamp"] - k["timestamp"]) >= nms_gap for k in kept):
            kept.append(det)

    if demote_below_class_conf is not None:
        for det in kept:
            if det["confidence"] < demote_below_class_conf:
                det["shot_type"] = "unknown_shot"

    # Round confidence for output
    for det in kept:
        det["confidence"] = round(det["confidence"], 3)

    kept.sort(key=lambda d: d["timestamp"])
    return kept


def detect_video(video_path, model, device, threshold=0.5, nms_gap=1.5,
                 step_sec=0.1, batch_size=256, per_class_thresholds=None,
                 min_class_conf=0.0, demote_below_class_conf=None):
    """Run full detection pipeline on a video.

    Returns result dict in fused_detect.py format.
    """
    video_name = Path(video_path).stem
    pose_path = os.path.join(POSES_DIR, f"{video_name}.json")

    if not os.path.exists(pose_path):
        print(f"  No pose file for {video_name}")
        return None

    print(f"  Loading poses: {video_name}")
    frames, fps, total_frames = load_pose_frames(pose_path)

    print(f"  Sliding window ({len(frames)} frames, step={step_sec}s)...")
    t0 = time.time()
    timestamps, probs = sliding_window_inference(
        model, device, frames, fps, step_sec=step_sec, batch_size=batch_size
    )
    inference_time = time.time() - t0

    if len(timestamps) == 0:
        print(f"  No valid windows")
        return None

    print(f"  Inference: {inference_time:.1f}s ({len(timestamps)} windows)")

    detections = find_shots(
        timestamps, probs,
        threshold=threshold, nms_gap=nms_gap,
        per_class_thresholds=per_class_thresholds,
        min_class_conf=min_class_conf,
        demote_below_class_conf=demote_below_class_conf,
    )

    # Build output detections in fused_detect format
    output_detections = []
    for det in detections:
        frame_idx = int(round(det["timestamp"] * fps))
        output_detections.append({
            "timestamp": det["timestamp"],
            "frame": frame_idx,
            "shot_type": det["shot_type"],
            "confidence": det["confidence"],
            "tier": "high" if det["p_shot"] > 0.8 else ("medium" if det["p_shot"] > 0.6 else "low"),
            "source": "sequence_cnn",
            "p_shot": round(det["p_shot"], 4),
            "probabilities": det["probabilities"],
        })

    # Type counts
    type_counts = {}
    tier_counts = {"high": 0, "medium": 0, "low": 0}
    for d in output_detections:
        st = d["shot_type"]
        type_counts[st] = type_counts.get(st, 0) + 1
        tier_counts[d["tier"]] += 1

    params = {
        "threshold": threshold,
        "nms_gap": nms_gap,
        "step_sec": step_sec,
        "model": "sequence_detector.pt",
    }
    if per_class_thresholds:
        params["per_class_thresholds"] = per_class_thresholds
    if min_class_conf > 0.0:
        params["min_class_conf"] = min_class_conf
    if demote_below_class_conf is not None:
        params["demote_below_class_conf"] = demote_below_class_conf

    result = {
        "version": 6,
        "detector": "sequence_cnn",
        "source_video": video_name,
        "video_path": str(video_path),
        "pose_path": pose_path,
        "fps": fps,
        "total_frames": total_frames,
        "duration": round(total_frames / fps, 2),
        "parameters": params,
        "summary": {
            "total_detections": len(output_detections),
            "by_tier": tier_counts,
            "by_type": type_counts,
            "inference_time": round(inference_time, 2),
            "windows_evaluated": len(timestamps),
        },
        "detections": output_detections,
    }

    # Print summary
    type_str = ", ".join(f"{k}={v}" for k, v in sorted(type_counts.items()))
    print(f"  Found {len(output_detections)} shots: {type_str}")

    return result


def run_threshold_sweep(model, device, videos, args):
    """Sweep thresholds efficiently: run inference once, vary peak finding."""
    from scripts.validate_pipeline import validate_video

    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.92, 0.95]

    # Step 1: Run inference ONCE per video, cache probability curves
    print("\nRunning inference on all videos (one-time)...")
    prob_cache = {}  # video_name -> (timestamps, probs, fps)
    for video_path in videos:
        video_name = Path(video_path).stem
        pose_path = os.path.join(POSES_DIR, f"{video_name}.json")
        if not os.path.exists(pose_path):
            continue

        frames, fps, total_frames = load_pose_frames(pose_path)
        timestamps, probs = sliding_window_inference(
            model, device, frames, fps, step_sec=args.step
        )
        if len(timestamps) > 0:
            prob_cache[video_name] = (timestamps, probs, fps, total_frames, str(video_path), pose_path)
            print(f"  {video_name}: {len(timestamps)} windows")

    # Step 2: Sweep thresholds using cached probs
    print(f"\n{'='*60}")
    print(f"THRESHOLD SWEEP ({len(prob_cache)} videos)")
    print(f"{'='*60}")
    print(f"{'Thresh':>8s} {'TP':>5s} {'FP':>5s} {'FN':>5s} {'Prec':>7s} {'Rec':>7s} {'F1':>7s} {'Err':>5s}")
    print(f"{'-'*55}")

    best_f1 = 0
    best_thresh = 0.5

    for thresh in thresholds:
        total_tp = total_fp = total_fn = 0

        for video_name, (timestamps, probs, fps, total_frames, vpath, ppath) in prob_cache.items():
            detections = find_shots(timestamps, probs, threshold=thresh, nms_gap=args.nms_gap)

            # Build output
            output_dets = []
            for det in detections:
                frame_idx = int(round(det["timestamp"] * fps))
                output_dets.append({
                    "timestamp": det["timestamp"],
                    "frame": frame_idx,
                    "shot_type": det["shot_type"],
                    "confidence": det["confidence"],
                    "tier": "high" if det["p_shot"] > 0.8 else ("medium" if det["p_shot"] > 0.6 else "low"),
                    "source": "sequence_cnn",
                    "p_shot": round(det["p_shot"], 4),
                    "probabilities": det["probabilities"],
                })

            type_counts = {}
            tier_counts = {"high": 0, "medium": 0, "low": 0}
            for d in output_dets:
                type_counts[d["shot_type"]] = type_counts.get(d["shot_type"], 0) + 1
                tier_counts[d["tier"]] += 1

            result = {
                "version": 6, "detector": "sequence_cnn",
                "source_video": video_name, "video_path": vpath,
                "pose_path": ppath, "fps": fps,
                "total_frames": total_frames,
                "duration": round(total_frames / fps, 2),
                "parameters": {"threshold": thresh, "nms_gap": args.nms_gap,
                              "step_sec": args.step, "model": "sequence_detector.pt"},
                "summary": {"total_detections": len(output_dets),
                           "by_tier": tier_counts, "by_type": type_counts},
                "detections": output_dets,
            }

            out_path = os.path.join(DETECTIONS_DIR, f"{video_name}_fused_detections.json")
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)

            r = validate_video(video_name, verbose=False)
            if r and r.get("overall"):
                total_tp += r["overall"]["tp"]
                total_fp += r["overall"]["fp"]
                total_fn += r["overall"]["fn"]

        prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        errors = total_fp + total_fn

        marker = " ***" if f1 > best_f1 else ""
        print(f"{thresh:>8.2f} {total_tp:>5d} {total_fp:>5d} {total_fn:>5d} "
              f"{prec:>7.3f} {rec:>7.3f} {f1:>7.3f} {errors:>5d}{marker}")

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    print(f"\nBest: threshold={best_thresh}, F1={best_f1:.3f}")
    return best_thresh


def get_all_gt_videos():
    """Get video paths for all videos that have GT files."""
    gt_videos = []
    for f in sorted(os.listdir(DETECTIONS_DIR)):
        if not f.endswith("_fused.json"):
            continue
        if "_detections" in f or "_v2" in f or "_v5" in f or "_ml" in f or "_pre" in f or "_baseline" in f:
            continue
        video_name = f.replace("_fused.json", "")
        if video_name in EXCLUDE_VIDEOS:
            continue
        # Check pose file exists
        pose_path = os.path.join(POSES_DIR, f"{video_name}.json")
        if not os.path.exists(pose_path):
            continue
        video_path = os.path.join(PREPROCESSED_DIR, f"{video_name}.mp4")
        gt_videos.append(video_path)
    return gt_videos


def main():
    parser = argparse.ArgumentParser(description="Detect shots with sequence CNN")
    parser.add_argument("video", nargs="?", help="Path to preprocessed video")
    parser.add_argument("--all", action="store_true", help="Run on all GT videos")
    parser.add_argument("--threshold", type=float, default=0.95,
                        help="P(shot) peak threshold (default: 0.95)")
    parser.add_argument("--nms-gap", type=float, default=1.5,
                        help="NMS gap in seconds (default: 1.5)")
    parser.add_argument("--step", type=float, default=0.1,
                        help="Sliding window step in seconds (default: 0.1)")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--model", default=os.path.join(MODELS_DIR, "sequence_detector.pt"))
    parser.add_argument("--sweep", action="store_true", help="Run threshold sweep")
    parser.add_argument("--output", help="Override output path")
    parser.add_argument("--per-class-thresh",
                        help='Per-class peak thresholds, e.g. "serve=0.55,forehand=0.70,backhand=0.70". '
                             "When set, peak-find independently on each class probability curve "
                             "(overrides --threshold).")
    parser.add_argument("--min-class-conf", type=float, default=0.0,
                        help="Drop detections whose top-class probability is below this floor.")
    parser.add_argument("--demote-below", type=float, default=None,
                        help='Detections with class prob below this floor are re-tagged as '
                             '"unknown_shot" rather than dropped.')
    args = parser.parse_args()

    if not args.video and not args.all:
        parser.error("Specify a video path or --all")

    # Parse per-class thresholds string
    per_class_thresholds = None
    if args.per_class_thresh:
        per_class_thresholds = {}
        for kv in args.per_class_thresh.split(","):
            if not kv.strip():
                continue
            k, _, v = kv.partition("=")
            per_class_thresholds[k.strip()] = float(v.strip())

    # Load model
    import torch
    model, device = load_model(args.model)
    if model is None:
        print(f"Model not found at {args.model}")
        sys.exit(1)
    print(f"Model loaded from {args.model}")
    print(f"Device: {device}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    if args.all:
        videos = get_all_gt_videos()
        print(f"\nProcessing {len(videos)} videos...")

        if args.sweep:
            run_threshold_sweep(model, device, videos, args)
            return

        total_detections = 0
        for video_path in videos:
            video_name = Path(video_path).stem
            print(f"\n--- {video_name} ---")

            result = detect_video(
                video_path, model, device,
                threshold=args.threshold, nms_gap=args.nms_gap,
                step_sec=args.step, batch_size=args.batch_size,
                per_class_thresholds=per_class_thresholds,
                min_class_conf=args.min_class_conf,
                demote_below_class_conf=args.demote_below,
            )
            if result is None:
                continue

            out_path = os.path.join(DETECTIONS_DIR, f"{video_name}_fused_detections.json")
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            total_detections += len(result["detections"])

        print(f"\n{'='*50}")
        print(f"Total: {total_detections} detections across {len(videos)} videos")
    else:
        video_path = args.video
        if not os.path.isabs(video_path):
            video_path = os.path.join(PROJECT_ROOT, video_path)

        video_name = Path(video_path).stem
        print(f"\n--- {video_name} ---")

        result = detect_video(
            video_path, model, device,
            threshold=args.threshold, nms_gap=args.nms_gap,
            step_sec=args.step, batch_size=args.batch_size,
            per_class_thresholds=per_class_thresholds,
            min_class_conf=args.min_class_conf,
            demote_below_class_conf=args.demote_below,
        )
        if result is None:
            sys.exit(1)

        out_path = args.output or os.path.join(DETECTIONS_DIR, f"{video_name}_fused_detections.json")

        # Backup user-labeled files
        if os.path.exists(out_path):
            try:
                with open(out_path) as f:
                    existing = json.load(f)
                has_edits = (
                    existing.get("last_saved")
                    or any(d.get("source") == "manual" for d in existing.get("detections", []))
                )
                if has_edits:
                    from datetime import datetime
                    import shutil
                    backup = out_path.replace(".json", f"_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
                    shutil.copy2(out_path, backup)
                    print(f"  [BACKUP] {os.path.basename(backup)}")
            except (json.JSONDecodeError, KeyError):
                pass

        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
