#!/usr/bin/env python3
"""Fused audio + heuristic shot detection.

Combines audio peak detection (detect_audio_hits) with biomechanical pose
analysis (heuristic_detect) to produce higher-confidence shot detections.

Scoring logic:
  - Audio + heuristic agree  -> HIGH confidence
  - Audio only (no pose match) -> MEDIUM confidence (real hit, unknown shot type)
  - Heuristic only (no audio)  -> LOW confidence (likely false positive)

Usage:
    python fused_detect.py /path/to/video.mp4 --poses /path/to/poses.json
    python fused_detect.py /path/to/video.mp4 --poses /path/to/poses.json -o detections.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import POSES_DIR, PREPROCESSED_DIR, PROJECT_ROOT

from scripts.detect_audio_hits import extract_audio, detect_peaks, get_video_fps
from scripts.heuristic_detect import (
    analyze_stroke_pattern,
    detect_serve_pattern,
    compute_wrist_velocities,
    find_velocity_spikes,
    RIGHT_WRIST,
    LEFT_WRIST,
    THRESHOLDS,
)


# ── Configuration ────────────────────────────────────────────

# Window (in seconds) around an audio hit to search for a matching pose pattern
AUDIO_POSE_WINDOW = 0.5  # +/- 0.5s = 1s window

# Confidence thresholds for the output tiers
CONFIDENCE_TIERS = {
    "high": 0.8,    # Both signals agree
    "medium": 0.5,  # Audio only
    "low": 0.3,     # Heuristic only
}


def load_pose_frames(pose_path):
    """Load pose JSON and return (frames_list, fps, total_frames).

    Returns the raw frame dicts (with world_landmarks, keypoints, etc.)
    as expected by heuristic_detect functions.
    """
    with open(pose_path) as f:
        data = json.load(f)

    video_info = data.get("video_info", {})
    fps = video_info.get("fps", 60.0)
    raw_frames = data.get("frames", [])
    total_frames = video_info.get("total_frames", len(raw_frames))

    # Build indexed array so frame_idx lines up with list index
    frames = [{}] * total_frames
    for fr in raw_frames:
        idx = fr.get("frame_idx", 0)
        if idx < total_frames:
            frames[idx] = fr

    return frames, fps, total_frames


def fused_detect(video_path, pose_path, dominant_hand="right",
                 audio_threshold=92, audio_min_gap=400,
                 window_sec=AUDIO_POSE_WINDOW):
    """Run fused audio + heuristic detection.

    Args:
        video_path: Path to preprocessed video (.mp4)
        pose_path: Path to pose JSON
        dominant_hand: "left" or "right"
        audio_threshold: Percentile threshold for audio peak detection
        audio_min_gap: Minimum gap between audio hits in ms
        window_sec: Seconds around audio hit to search for pose match

    Returns:
        dict with detections list, metadata, and summary
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # ── Step 1: Audio detection ──────────────────────────────
    print(f"  Extracting audio...")
    sample_rate = 16000
    audio = extract_audio(video_path, sample_rate)
    audio_duration = len(audio) / sample_rate
    print(f"    Duration: {audio_duration:.1f}s")

    print(f"  Detecting audio peaks (threshold={audio_threshold}%, min_gap={audio_min_gap}ms)...")
    audio_peaks = detect_peaks(audio, sample_rate, audio_threshold, audio_min_gap)
    print(f"    Found {len(audio_peaks)} audio peaks")

    video_fps = get_video_fps(video_path)

    # ── Step 2: Load poses ───────────────────────────────────
    print(f"  Loading poses...")
    frames, pose_fps, total_frames = load_pose_frames(pose_path)
    fps = pose_fps or video_fps
    print(f"    {total_frames} frames at {fps} fps")

    # ── Step 3: Heuristic-only detection (velocity spikes) ───
    wrist_idx = RIGHT_WRIST if dominant_hand == "right" else LEFT_WRIST
    velocities = compute_wrist_velocities(frames, wrist_idx, fps)
    heuristic_spikes = find_velocity_spikes(
        velocities,
        min_vel=THRESHOLDS["velocity_min"],
        spike_ratio=THRESHOLDS["velocity_spike_ratio"],
        min_gap=THRESHOLDS["min_shot_gap_frames"],
    )
    heuristic_times = {spike_frame / fps: spike_frame for spike_frame, _ in heuristic_spikes}
    print(f"    {len(heuristic_spikes)} heuristic velocity spikes")

    # ── Step 4: Fuse audio + heuristic ───────────────────────
    window_frames = int(window_sec * fps)
    detections = []
    matched_heuristic_frames = set()

    # Process each audio peak
    for peak_time in audio_peaks:
        peak_frame = int(peak_time * fps)

        # Look for a heuristic spike within the window
        best_heuristic_frame = None
        best_heuristic_dist = float("inf")
        for h_time, h_frame in heuristic_times.items():
            dist = abs(peak_time - h_time)
            if dist <= window_sec and dist < best_heuristic_dist:
                best_heuristic_dist = dist
                best_heuristic_frame = h_frame

        if best_heuristic_frame is not None:
            # AUDIO + HEURISTIC: High confidence
            matched_heuristic_frames.add(best_heuristic_frame)
            center_frame = best_heuristic_frame

            # Classify the shot using stroke pattern analysis
            is_serve, serve_conf, serve_trigger = detect_serve_pattern(
                frames, center_frame, fps
            )
            if is_serve and serve_conf > 0.4:
                shot_type = "serve"
                pattern_conf = serve_conf
                trigger = serve_trigger
            else:
                shot_type, pattern_conf, trigger = analyze_stroke_pattern(
                    frames, center_frame, dominant_hand, fps
                )
                if shot_type == "neutral":
                    shot_type = "unknown_shot"
                    pattern_conf = 0.5

            # Fused confidence: boost from both signals agreeing
            fused_conf = min(1.0, (pattern_conf * 0.6 + CONFIDENCE_TIERS["high"] * 0.4))

            detections.append({
                "timestamp": round(peak_time, 3),
                "frame": center_frame,
                "shot_type": shot_type,
                "confidence": round(fused_conf, 3),
                "tier": "high",
                "source": "audio+heuristic",
                "audio_peak_time": round(peak_time, 3),
                "heuristic_frame": center_frame,
                "pattern_confidence": round(pattern_conf, 3),
                "trigger": trigger,
            })
        else:
            # AUDIO ONLY: Medium confidence - real hit but no pose pattern match
            detections.append({
                "timestamp": round(peak_time, 3),
                "frame": peak_frame,
                "shot_type": "unknown_shot",
                "confidence": round(CONFIDENCE_TIERS["medium"], 3),
                "tier": "medium",
                "source": "audio_only",
                "audio_peak_time": round(peak_time, 3),
                "heuristic_frame": None,
                "pattern_confidence": 0.0,
                "trigger": "audio peak only, no matching pose pattern",
            })

    # Process unmatched heuristic spikes (no corresponding audio peak)
    for spike_frame, velocity in heuristic_spikes:
        if spike_frame in matched_heuristic_frames:
            continue

        spike_time = spike_frame / fps

        # Classify the shot
        is_serve, serve_conf, serve_trigger = detect_serve_pattern(
            frames, spike_frame, fps
        )
        if is_serve and serve_conf > 0.4:
            shot_type = "serve"
            pattern_conf = serve_conf
            trigger = serve_trigger
        else:
            shot_type, pattern_conf, trigger = analyze_stroke_pattern(
                frames, spike_frame, dominant_hand, fps
            )

        if shot_type == "neutral":
            continue  # Skip neutral heuristic-only detections

        # HEURISTIC ONLY: Low confidence
        fused_conf = min(1.0, pattern_conf * 0.5 + CONFIDENCE_TIERS["low"] * 0.5)

        detections.append({
            "timestamp": round(spike_time, 3),
            "frame": spike_frame,
            "shot_type": shot_type,
            "confidence": round(fused_conf, 3),
            "tier": "low",
            "source": "heuristic_only",
            "audio_peak_time": None,
            "heuristic_frame": spike_frame,
            "pattern_confidence": round(pattern_conf, 3),
            "trigger": f"{trigger}, vel={velocity:.1f}",
        })

    # Sort by timestamp
    detections.sort(key=lambda d: d["timestamp"])

    # ── Step 5: Build summary ────────────────────────────────
    tier_counts = {"high": 0, "medium": 0, "low": 0}
    type_counts = {}
    for det in detections:
        tier_counts[det["tier"]] += 1
        st = det["shot_type"]
        type_counts[st] = type_counts.get(st, 0) + 1

    result = {
        "version": 1,
        "detector": "fused_audio_heuristic",
        "source_video": video_name,
        "video_path": video_path,
        "pose_path": pose_path,
        "fps": fps,
        "total_frames": total_frames,
        "duration": round(audio_duration, 2),
        "dominant_hand": dominant_hand,
        "parameters": {
            "audio_threshold_percentile": audio_threshold,
            "audio_min_gap_ms": audio_min_gap,
            "pose_window_sec": window_sec,
        },
        "summary": {
            "total_detections": len(detections),
            "by_tier": tier_counts,
            "by_type": type_counts,
            "audio_peaks": len(audio_peaks),
            "heuristic_spikes": len(heuristic_spikes),
        },
        "detections": detections,
    }

    return result


def print_results(result):
    """Print fused detection results."""
    print()
    print(f"{'='*65}")
    print(f"FUSED DETECTION RESULTS")
    print(f"{'='*65}")
    print(f"Source: {result['source_video']}")
    print(f"Duration: {result['duration']:.1f}s, FPS: {result['fps']}")
    print(f"Audio peaks: {result['summary']['audio_peaks']}, "
          f"Heuristic spikes: {result['summary']['heuristic_spikes']}")
    print()

    summary = result["summary"]
    print(f"Total detections: {summary['total_detections']}")
    print(f"  HIGH   (audio+heuristic): {summary['by_tier']['high']}")
    print(f"  MEDIUM (audio only):      {summary['by_tier']['medium']}")
    print(f"  LOW    (heuristic only):   {summary['by_tier']['low']}")
    print()

    print(f"By shot type:")
    for st, count in sorted(summary["by_type"].items()):
        print(f"  {st}: {count}")
    print()

    print(f"{'Tier':<8} {'Type':<14} {'Time':>7} {'Frame':>7} {'Conf':>6}  Trigger")
    print(f"{'─'*65}")

    for det in result["detections"][:50]:
        tier_str = det["tier"].upper()
        print(f"  {tier_str:<6} {det['shot_type']:<14} {det['timestamp']:6.1f}s "
              f"{det['frame']:>6}  {det['confidence']:.2f}   {det['trigger'][:40]}")

    if len(result["detections"]) > 50:
        print(f"  ... and {len(result['detections']) - 50} more")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Fused audio + heuristic shot detection"
    )
    parser.add_argument("video", help="Path to preprocessed video (.mp4)")
    parser.add_argument("--poses", help="Path to pose JSON (auto-discovers if omitted)")
    parser.add_argument("-o", "--output", help="Output JSON path")
    parser.add_argument("--dominant-hand", choices=["left", "right"],
                        default="right", help="Player's dominant hand (default: right)")
    parser.add_argument("--audio-threshold", type=float, default=92,
                        help="Audio peak percentile threshold (default: 92)")
    parser.add_argument("--audio-min-gap", type=float, default=400,
                        help="Min gap between audio hits in ms (default: 400)")
    parser.add_argument("--window", type=float, default=AUDIO_POSE_WINDOW,
                        help=f"Audio-pose matching window in seconds (default: {AUDIO_POSE_WINDOW})")
    parser.add_argument("--min-confidence", type=float, default=0.0,
                        help="Minimum confidence to include (default: 0.0)")
    args = parser.parse_args()

    video_path = args.video
    if not os.path.isabs(video_path):
        video_path = os.path.join(os.getcwd(), video_path)

    if not os.path.exists(video_path):
        print(f"[ERROR] Video not found: {video_path}")
        sys.exit(1)

    # Auto-discover pose file
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    if args.poses:
        pose_path = args.poses
    else:
        pose_path = os.path.join(POSES_DIR, video_name + ".json")
        if not os.path.exists(pose_path):
            # Try without _poses suffix
            alt = os.path.join(POSES_DIR, video_name + "_poses.json")
            if os.path.exists(alt):
                pose_path = alt

    if not os.path.exists(pose_path):
        print(f"[ERROR] Pose file not found: {pose_path}")
        print(f"  Run extract_poses.py on the video first.")
        sys.exit(1)

    output_path = args.output or os.path.join(
        PROJECT_ROOT, f"{video_name}_fused_detections.json"
    )

    print(f"Fused Detection: {video_name}")
    print(f"  Video: {os.path.basename(video_path)}")
    print(f"  Poses: {os.path.basename(pose_path)}")
    print()

    result = fused_detect(
        video_path, pose_path,
        dominant_hand=args.dominant_hand,
        audio_threshold=args.audio_threshold,
        audio_min_gap=args.audio_min_gap,
        window_sec=args.window,
    )

    # Filter by confidence
    if args.min_confidence > 0:
        before = len(result["detections"])
        result["detections"] = [
            d for d in result["detections"]
            if d["confidence"] >= args.min_confidence
        ]
        filtered = before - len(result["detections"])
        if filtered:
            print(f"  Filtered {filtered} detections below {args.min_confidence} confidence")

    print_results(result)

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
