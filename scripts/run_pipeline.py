#!/usr/bin/env python3
"""End-to-end tennis shot detection pipeline.

Processes a raw video from input to extracted highlight clips in one command.

Steps:
  1. Preprocess  (240fps VFR → 60fps CFR)
  2. Extract poses (MediaPipe or YOLO)
  3. Detect shots (fused audio + heuristic)
  4. Classify shots (GRU model on detected windows)
  5. Filter by confidence
  6. Extract clips
  7. Print summary + gap report

Usage:
    python run_pipeline.py /path/to/video.mp4
    python run_pipeline.py /path/to/video.mp4 --review
    python run_pipeline.py /path/to/video.mp4 --skip-preprocess --skip-poses
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import (
    PROJECT_ROOT, RAW_DIR, PREPROCESSED_DIR, POSES_DIR, MODELS_DIR,
    CLIPS_DIR, HIGHLIGHTS_DIR, LABELS_DIR, VIDEO, SHOT_TYPES,
)


def format_duration(seconds):
    """Format seconds as MM:SS or HH:MM:SS."""
    seconds = int(seconds)
    if seconds >= 3600:
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        return f"{h}:{m:02d}:{s:02d}"
    return f"{seconds // 60}:{seconds % 60:02d}"


def step_preprocess(video_path):
    """Step 1: Convert raw video to 60fps CFR mp4.

    Returns path to preprocessed video, or None on failure.
    """
    from scripts.preprocess_videos import convert_video, probe_video, is_already_processed

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    dst_path = os.path.join(PREPROCESSED_DIR, video_name + ".mp4")

    if os.path.exists(dst_path) and os.path.getsize(dst_path) > 0:
        print(f"    [SKIP] Already preprocessed: {os.path.basename(dst_path)}")
        return dst_path

    os.makedirs(PREPROCESSED_DIR, exist_ok=True)

    info = probe_video(video_path)
    if not info:
        print(f"    [ERROR] Could not probe video")
        return None

    print(f"    {info['codec']} {info['width']}x{info['height']} "
          f"{info['fps']:.0f}fps -> 60fps CFR")

    success = convert_video(video_path, dst_path, VIDEO)
    if not success:
        print(f"    [ERROR] Preprocessing failed")
        return None

    return dst_path


def step_extract_poses(preprocessed_path):
    """Step 2: Extract pose keypoints from preprocessed video.

    Returns path to pose JSON, or None on failure.
    """
    from scripts.extract_poses import process_video

    video_name = os.path.splitext(os.path.basename(preprocessed_path))[0]
    pose_path = os.path.join(POSES_DIR, video_name + ".json")

    if os.path.exists(pose_path) and os.path.getsize(pose_path) > 0:
        print(f"    [SKIP] Poses already exist: {os.path.basename(pose_path)}")
        return pose_path

    os.makedirs(POSES_DIR, exist_ok=True)

    success = process_video(preprocessed_path, pose_path)
    if not success:
        print(f"    [ERROR] Pose extraction failed")
        return None

    return pose_path


def step_fused_detect(preprocessed_path, pose_path, dominant_hand="right"):
    """Step 3: Run fused audio + heuristic detection.

    Returns fused detection result dict, or None on failure.
    """
    from scripts.fused_detect import fused_detect

    result = fused_detect(
        preprocessed_path, pose_path,
        dominant_hand=dominant_hand,
    )

    if not result or not result.get("detections"):
        print(f"    [WARN] No detections from fused detector")
        return result

    summary = result["summary"]
    print(f"    {summary['total_detections']} detections: "
          f"{summary['by_tier']['high']} high, "
          f"{summary['by_tier']['medium']} medium, "
          f"{summary['by_tier']['low']} low")

    return result


def step_classify_shots(pose_path, fused_result):
    """Step 4: Run GRU model classification on detected windows.

    Enriches fused detections with model predictions where possible.
    Returns updated result dict.
    """
    model_path = os.path.join(MODELS_DIR, "shot_classifier.h5")
    meta_path = os.path.join(MODELS_DIR, "shot_classifier_meta.json")

    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        print(f"    [SKIP] No trained model found at {MODELS_DIR}/")
        print(f"    Using fused detection results only")
        return fused_result

    from scripts.detect_shots import load_model_and_meta, load_pose_frames, run_inference

    model, meta, mean, std = load_model_and_meta(MODELS_DIR)
    seq_len = meta["sequence_length"]
    inverse_label_map = meta["inverse_label_map"]
    n_features = meta.get("num_features", 99)

    frames, fps, total_frames = load_pose_frames(pose_path)

    # Build view angle one-hot if model supports it
    from config.settings import VIEW_ANGLES, DEFAULT_VIEW_ANGLE
    view_angle_map = meta.get("view_angle_map", None)
    view_angle_one_hot = None
    if n_features == 104 and view_angle_map is not None:
        view_angle = DEFAULT_VIEW_ANGLE
        view_angle_idx = view_angle_map.get(view_angle, 0)
        view_angle_one_hot = [0.0] * len(VIEW_ANGLES)
        view_angle_one_hot[view_angle_idx] = 1.0

    # Run inference with a small stride for precision
    predictions = run_inference(
        model, frames, mean, std, seq_len, stride=5,
        inverse_label_map=inverse_label_map,
        view_angle_one_hot=view_angle_one_hot,
    )

    # Build a frame → prediction map for quick lookup
    pred_map = {}
    for center_frame, label, confidence in predictions:
        pred_map[center_frame] = (label, confidence)

    # Enrich fused detections with GRU predictions
    enriched = 0
    for det in fused_result["detections"]:
        det_frame = det["frame"]
        # Find closest GRU prediction within ±seq_len frames
        best_pred = None
        best_dist = float("inf")
        for pred_frame, (label, conf) in pred_map.items():
            dist = abs(pred_frame - det_frame)
            if dist < best_dist and dist <= seq_len:
                best_dist = dist
                best_pred = (label, conf)

        if best_pred:
            gru_label, gru_conf = best_pred
            det["gru_prediction"] = gru_label
            det["gru_confidence"] = round(gru_conf, 3)

            # If fused detection was "unknown_shot" and GRU is confident, adopt GRU label
            if det["shot_type"] == "unknown_shot" and gru_conf > 0.7:
                det["shot_type"] = gru_label
                det["confidence"] = round(
                    max(det["confidence"], gru_conf * 0.8), 3
                )
            enriched += 1

    print(f"    GRU enriched {enriched}/{len(fused_result['detections'])} detections")

    # Recalculate type counts
    type_counts = {}
    for det in fused_result["detections"]:
        st = det["shot_type"]
        type_counts[st] = type_counts.get(st, 0) + 1
    fused_result["summary"]["by_type"] = type_counts

    return fused_result


def step_filter(result, min_confidence=0.3, review_mode=False):
    """Step 5: Filter detections by confidence.

    If review_mode, outputs low-confidence detections to review queue.
    Returns (filtered_detections, review_queue).
    """
    all_dets = result["detections"]
    accepted = []
    review_queue = []

    for det in all_dets:
        if det["confidence"] >= min_confidence:
            accepted.append(det)
        elif review_mode:
            review_queue.append(det)

    print(f"    Accepted: {len(accepted)} (conf >= {min_confidence})")
    if review_queue:
        print(f"    Review queue: {len(review_queue)} low-confidence detections")

    return accepted, review_queue


def step_extract_clips(preprocessed_path, detections, video_name):
    """Step 6: Extract video clips for each detection.

    Returns number of clips extracted.
    """
    from scripts.extract_clips import extract_clip

    if not detections:
        print(f"    No detections to extract clips for")
        return 0

    os.makedirs(CLIPS_DIR, exist_ok=True)

    # Create type directories
    type_counts = {}
    for det in detections:
        st = det["shot_type"]
        type_dir = os.path.join(CLIPS_DIR, st)
        os.makedirs(type_dir, exist_ok=True)
        type_counts[st] = type_counts.get(st, 0) + 1

    extracted = 0
    counters = {}

    for i, det in enumerate(detections):
        st = det["shot_type"]
        counters[st] = counters.get(st, 0) + 1

        # Calculate clip window around detection
        timestamp = det["timestamp"]
        # Use shot-specific buffers
        buffers = {
            "serve": (5.0, 3.0),
            "forehand": (4.5, 3.5),
            "backhand": (4.5, 3.5),
        }
        pre, post = buffers.get(st, (4.5, 3.5))
        start_time = max(0, timestamp - pre)
        end_time = timestamp + post

        filename = f"{video_name}_{st}_{counters[st]:03d}.mp4"
        output_path = os.path.join(CLIPS_DIR, st, filename)

        success = extract_clip(preprocessed_path, start_time, end_time, output_path, pad=0.3)
        if success:
            extracted += 1
            if (i + 1) % 5 == 0 or i == len(detections) - 1:
                print(f"\r    Extracted {extracted}/{len(detections)} clips", end="", flush=True)

    print()
    return extracted


def step_summary(result, accepted, review_queue, elapsed_total):
    """Step 7: Print summary and gap report."""
    print()
    print(f"{'='*60}")
    print(f"PIPELINE SUMMARY")
    print(f"{'='*60}")
    print(f"Source: {result['source_video']}")
    print(f"Duration: {result['duration']:.1f}s")
    print(f"Processing time: {format_duration(elapsed_total)}")
    print()

    # Shot counts
    type_counts = {}
    for det in accepted:
        st = det["shot_type"]
        type_counts[st] = type_counts.get(st, 0) + 1

    print(f"Detected shots ({len(accepted)} total):")
    for st in ["serve", "forehand", "backhand", "unknown_shot"]:
        if st in type_counts:
            print(f"  {st}: {type_counts[st]}")
    for st, count in sorted(type_counts.items()):
        if st not in ["serve", "forehand", "backhand", "unknown_shot"]:
            print(f"  {st}: {count}")
    print()

    # Confidence tier breakdown
    tier_counts = {"high": 0, "medium": 0, "low": 0}
    for det in accepted:
        tier_counts[det["tier"]] += 1
    print(f"Confidence tiers:")
    print(f"  HIGH   (audio+heuristic): {tier_counts['high']}")
    print(f"  MEDIUM (audio only):      {tier_counts['medium']}")
    print(f"  LOW    (heuristic only):   {tier_counts['low']}")
    print()

    # Gap report: find long periods with no detections
    if accepted:
        print(f"Gap report (periods > 30s with no detections):")
        sorted_dets = sorted(accepted, key=lambda d: d["timestamp"])
        gaps = []
        # Check gap from start
        if sorted_dets[0]["timestamp"] > 30:
            gaps.append((0, sorted_dets[0]["timestamp"]))
        # Check gaps between detections
        for i in range(len(sorted_dets) - 1):
            gap = sorted_dets[i + 1]["timestamp"] - sorted_dets[i]["timestamp"]
            if gap > 30:
                gaps.append((sorted_dets[i]["timestamp"], sorted_dets[i + 1]["timestamp"]))
        # Check gap to end
        video_duration = result.get("duration", 0)
        if video_duration - sorted_dets[-1]["timestamp"] > 30:
            gaps.append((sorted_dets[-1]["timestamp"], video_duration))

        if gaps:
            for start, end in gaps:
                print(f"  {start:.0f}s - {end:.0f}s ({end - start:.0f}s gap)")
        else:
            print(f"  No significant gaps")
        print()

    if review_queue:
        print(f"Review queue: {len(review_queue)} low-confidence detections")
        print(f"  Run with --review to save these for manual verification")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end tennis shot detection pipeline"
    )
    parser.add_argument("video", help="Path to video file (raw .MOV or preprocessed .mp4)")
    parser.add_argument("--dominant-hand", choices=["left", "right"],
                        default="right", help="Player's dominant hand (default: right)")
    parser.add_argument("--min-confidence", type=float, default=0.3,
                        help="Minimum confidence threshold (default: 0.3)")
    parser.add_argument("--review", action="store_true",
                        help="Save low-confidence detections to review queue")
    parser.add_argument("--skip-preprocess", action="store_true",
                        help="Skip preprocessing (input is already 60fps mp4)")
    parser.add_argument("--skip-poses", action="store_true",
                        help="Skip pose extraction (poses already exist)")
    parser.add_argument("--skip-clips", action="store_true",
                        help="Skip clip extraction (detection only)")
    parser.add_argument("--skip-classify", action="store_true",
                        help="Skip GRU classification step")
    parser.add_argument("-o", "--output", help="Output JSON path for detections")
    args = parser.parse_args()

    video_path = args.video
    if not os.path.isabs(video_path):
        video_path = os.path.join(os.getcwd(), video_path)

    if not os.path.exists(video_path):
        print(f"[ERROR] Video not found: {video_path}")
        sys.exit(1)

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    is_raw = video_path.lower().endswith(".mov")

    print(f"{'='*60}")
    print(f"TENNIS SHOT DETECTION PIPELINE")
    print(f"{'='*60}")
    print(f"Input: {os.path.basename(video_path)}")
    print(f"Dominant hand: {args.dominant_hand}")
    print()

    pipeline_start = time.time()
    step_times = {}

    # ── Step 1: Preprocess ───────────────────────────────────
    print(f"[Step 1/7] Preprocess")
    t0 = time.time()
    if args.skip_preprocess or not is_raw:
        preprocessed_path = video_path
        if is_raw and args.skip_preprocess:
            # Even if skipping, check if preprocessed version exists
            preprocessed_path = os.path.join(PREPROCESSED_DIR, video_name + ".mp4")
            if not os.path.exists(preprocessed_path):
                preprocessed_path = video_path
        print(f"    Using: {os.path.basename(preprocessed_path)}")
    else:
        preprocessed_path = step_preprocess(video_path)
        if not preprocessed_path:
            print("[ABORT] Preprocessing failed")
            sys.exit(1)
    step_times["preprocess"] = time.time() - t0
    print()

    # ── Step 2: Extract Poses ────────────────────────────────
    print(f"[Step 2/7] Extract Poses")
    t0 = time.time()
    if args.skip_poses:
        pose_path = os.path.join(POSES_DIR, video_name + ".json")
        if not os.path.exists(pose_path):
            print(f"    [ERROR] --skip-poses but no pose file at {pose_path}")
            sys.exit(1)
        print(f"    Using: {os.path.basename(pose_path)}")
    else:
        pose_path = step_extract_poses(preprocessed_path)
        if not pose_path:
            print("[ABORT] Pose extraction failed")
            sys.exit(1)
    step_times["extract_poses"] = time.time() - t0
    print()

    # ── Step 3: Fused Detection ──────────────────────────────
    print(f"[Step 3/7] Fused Audio + Heuristic Detection")
    t0 = time.time()
    fused_result = step_fused_detect(
        preprocessed_path, pose_path,
        dominant_hand=args.dominant_hand,
    )
    if not fused_result:
        print("[ABORT] Detection failed")
        sys.exit(1)
    step_times["fused_detect"] = time.time() - t0
    print()

    # ── Step 4: GRU Classification ───────────────────────────
    print(f"[Step 4/7] GRU Shot Classification")
    t0 = time.time()
    if args.skip_classify:
        print(f"    [SKIP] Skipping GRU classification")
    else:
        fused_result = step_classify_shots(pose_path, fused_result)
    step_times["classify"] = time.time() - t0
    print()

    # ── Step 5: Filter ───────────────────────────────────────
    print(f"[Step 5/7] Confidence Filtering")
    t0 = time.time()
    accepted, review_queue = step_filter(
        fused_result,
        min_confidence=args.min_confidence,
        review_mode=args.review,
    )
    step_times["filter"] = time.time() - t0
    print()

    # ── Step 6: Extract Clips ────────────────────────────────
    print(f"[Step 6/7] Extract Clips")
    t0 = time.time()
    if args.skip_clips:
        print(f"    [SKIP] Skipping clip extraction")
        clips_extracted = 0
    else:
        clips_extracted = step_extract_clips(preprocessed_path, accepted, video_name)
        print(f"    {clips_extracted} clips extracted to {CLIPS_DIR}/")
    step_times["extract_clips"] = time.time() - t0
    print()

    # ── Save review queue ────────────────────────────────────
    if args.review and review_queue:
        os.makedirs(LABELS_DIR, exist_ok=True)
        review_path = os.path.join(LABELS_DIR, "review_queue.json")

        # Append to existing queue if present
        existing = []
        if os.path.exists(review_path):
            with open(review_path) as f:
                existing = json.load(f).get("detections", [])

        review_data = {
            "version": 1,
            "description": "Low-confidence detections for manual review",
            "detections": existing + [
                {**det, "source_video": video_name}
                for det in review_queue
            ],
        }
        with open(review_path, "w") as f:
            json.dump(review_data, f, indent=2)
        print(f"Review queue saved to {review_path}")
        print()

    # ── Save detection results ───────────────────────────────
    output_path = args.output or os.path.join(
        PROJECT_ROOT, f"{video_name}_pipeline_detections.json"
    )
    fused_result["accepted_detections"] = accepted
    fused_result["pipeline_step_times"] = step_times
    with open(output_path, "w") as f:
        json.dump(fused_result, f, indent=2)
    print(f"Detections saved to {output_path}")

    # ── Step 7: Summary ──────────────────────────────────────
    elapsed_total = time.time() - pipeline_start
    step_summary(fused_result, accepted, review_queue, elapsed_total)

    # Print step timing
    print(f"Step timing:")
    for step_name, elapsed in step_times.items():
        print(f"  {step_name:<20} {format_duration(elapsed)}")
    print(f"  {'TOTAL':<20} {format_duration(elapsed_total)}")
    print()


if __name__ == "__main__":
    main()
