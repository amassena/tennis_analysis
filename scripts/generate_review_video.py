#!/usr/bin/env python3
"""Generate a continuous review video with shot labels overlaid at the bottom.

Plays through the source video with a label bar at the bottom showing
detection events as they occur. Color coding:
  GREEN  = correct detection, type matches ground truth
  YELLOW = detected, type unknown (ground truth type shown)
  ORANGE = wrong classification
  RED    = false positive (no matching ground truth)
  CYAN   = missed shot from ground truth

Each label fades in when the detection timestamp is reached and stays
visible for a few seconds. Uses OpenCV for frame-by-frame overlay.
"""

import argparse
import json
import os
import re
import subprocess
import sys

import cv2
import numpy as np


def parse_ground_truth(gt_path):
    """Parse ground truth label file -> list of (timestamp_sec, shot_type)."""
    shots = []
    with open(gt_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "#" in line:
                line = line[:line.index("#")]
            line = line.strip()
            parts = line.split()
            if len(parts) < 2:
                continue
            m = re.match(r"(\d+):(\d+)", parts[0])
            if m:
                secs = int(m.group(1)) * 60 + int(m.group(2))
                shots.append((float(secs), parts[1]))
    return shots


def match_detections_to_gt(detections, gt_shots, match_window=3.0):
    """Match detections to ground truth. Returns events list sorted by time."""
    gt_matched = set()
    events = []

    for det in detections:
        det_time = det["timestamp"]
        best_i, best_dist = None, match_window + 1
        for i, (gt_time, gt_type) in enumerate(gt_shots):
            if i in gt_matched:
                continue
            dist = abs(det_time - gt_time)
            if dist < best_dist:
                best_i, best_dist = i, dist

        if best_i is not None and best_dist <= match_window:
            gt_matched.add(best_i)
            gt_type = gt_shots[best_i][1]
            det_type = det["shot_type"]

            if det_type == gt_type:
                match = "correct"
            elif det_type == "unknown_shot":
                match = "unknown"
            else:
                match = "wrong_type"

            events.append({
                "time": det_time,
                "match": match,
                "det_type": det_type,
                "gt_type": gt_type,
                "tier": det["tier"],
                "conf": det["confidence"],
                "amp": det.get("audio_amplitude", 0),
            })
        else:
            events.append({
                "time": det_time,
                "match": "false_positive",
                "det_type": det["shot_type"],
                "gt_type": None,
                "tier": det["tier"],
                "conf": det["confidence"],
                "amp": det.get("audio_amplitude", 0),
            })

    # Add missed ground truth as events
    for i, (gt_time, gt_type) in enumerate(gt_shots):
        if i not in gt_matched:
            events.append({
                "time": gt_time,
                "match": "missed",
                "det_type": None,
                "gt_type": gt_type,
                "tier": None,
                "conf": 0,
                "amp": 0,
            })

    events.sort(key=lambda e: e["time"])
    return events


# Colors (BGR for OpenCV)
COLORS = {
    "correct":       (0, 200, 0),
    "unknown":       (0, 220, 220),
    "wrong_type":    (0, 140, 255),
    "false_positive": (0, 0, 200),
    "missed":        (220, 200, 0),
}


def fmt_time(secs):
    m = int(secs) // 60
    s = int(secs) % 60
    return f"{m}:{s:02d}"


def make_label_text(event):
    """Create label string for an event."""
    match = event["match"]
    if match == "correct":
        return f"{event['det_type'].upper()}"
    elif match == "unknown":
        return f"SHOT ({event['gt_type']})"
    elif match == "wrong_type":
        return f"{event['det_type']} (was {event['gt_type']})"
    elif match == "false_positive":
        return f"FP: {event['det_type']}"
    elif match == "missed":
        return f"MISSED: {event['gt_type']}"
    return "?"


def draw_label_bar(frame, active_events, current_time, bar_height=60):
    """Draw the label bar at the bottom of the frame."""
    h, w = frame.shape[:2]
    bar_y = h - bar_height

    # Semi-transparent dark background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, bar_y), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Thin separator line
    cv2.line(frame, (0, bar_y), (w, bar_y), (80, 80, 80), 1)

    # Timestamp on the left
    font = cv2.FONT_HERSHEY_SIMPLEX
    ts_text = fmt_time(current_time)
    cv2.putText(frame, ts_text, (10, bar_y + 40), font, 1.0,
                (180, 180, 180), 2, cv2.LINE_AA)

    # Active labels (right-aligned, most recent first)
    x_offset = 130
    for event, alpha in active_events:
        color = COLORS[event["match"]]
        # Fade color by alpha
        faded = tuple(int(c * alpha) for c in color)

        label = make_label_text(event)
        tier_str = f" [{event['tier'].upper()}]" if event["tier"] else ""
        full_text = f"{fmt_time(event['time'])} {label}{tier_str}"

        (tw, _), _ = cv2.getTextSize(full_text, font, 0.7, 2)

        # Background pill
        pill_overlay = frame.copy()
        cv2.rectangle(pill_overlay,
                       (x_offset - 5, bar_y + 8),
                       (x_offset + tw + 10, bar_y + bar_height - 8),
                       faded, -1)
        cv2.addWeighted(pill_overlay, 0.3 * alpha, frame, 1 - 0.3 * alpha, 0, frame)

        cv2.putText(frame, full_text, (x_offset, bar_y + 40), font, 0.7,
                    faded, 2, cv2.LINE_AA)

        x_offset += tw + 25
        if x_offset > w - 100:
            break

    return frame


def draw_flash_indicator(frame, event, alpha):
    """Draw a brief colored flash at the top when a new detection appears."""
    h, w = frame.shape[:2]
    color = COLORS[event["match"]]

    indicator_h = 6
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, indicator_h), color, -1)
    cv2.addWeighted(overlay, alpha * 0.8, frame, 1 - alpha * 0.8, 0, frame)
    return frame


def generate_review_video(video_path, detections_path, gt_path, output_path,
                          end_time=None, match_window=3.0, tier_filter=None,
                          label_duration=3.0):
    """Generate continuous review video with label overlays."""

    with open(detections_path) as f:
        data = json.load(f)
    detections = data["detections"]
    fps = data["fps"]

    gt_shots = parse_ground_truth(gt_path)

    # Determine end time from ground truth range
    if end_time is None and gt_shots:
        end_time = max(t for t, _ in gt_shots) + 15

    # Filter detections to time range
    if end_time:
        detections = [d for d in detections if d["timestamp"] <= end_time]

    if tier_filter:
        tiers = set(tier_filter.split(","))
        detections = [d for d in detections if d["tier"] in tiers]

    # Match and build events
    events = match_detections_to_gt(detections, gt_shots, match_window)

    # Print summary
    counts = {}
    for e in events:
        counts[e["match"]] = counts.get(e["match"], 0) + 1
    print(f"Events: {len(events)} total")
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}")
    print()

    # Open source video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {video_path}")
        return False

    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    end_frame = int(end_time * fps) if end_time else total_frames
    end_frame = min(end_frame, total_frames)

    print(f"Video: {vid_w}x{vid_h} @ {fps}fps")
    print(f"Processing frames 0-{end_frame} ({end_frame/fps:.0f}s)")

    # Open output - write raw to temp file, then re-encode with FFmpeg
    temp_path = output_path + ".raw.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_path, fourcc, fps, (vid_w, vid_h))

    if not out.isOpened():
        print(f"ERROR: Cannot open VideoWriter")
        return False

    # Build event index for quick lookup
    event_idx = 0

    # Active labels: list of (event, start_time)
    active_labels = []

    frame_num = 0
    last_pct = -1

    while frame_num < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = frame_num / fps

        # Check for new events
        while event_idx < len(events) and events[event_idx]["time"] <= current_time:
            active_labels.append((events[event_idx], current_time))
            event_idx += 1

        # Remove expired labels
        active_labels = [(e, st) for e, st in active_labels
                          if current_time - st < label_duration]

        if active_labels:
            # Compute alpha for each label (fade out)
            active_with_alpha = []
            for event, start in active_labels:
                age = current_time - start
                if age < 0.2:
                    alpha = age / 0.2  # fade in
                elif age > label_duration - 0.5:
                    alpha = (label_duration - age) / 0.5  # fade out
                else:
                    alpha = 1.0
                active_with_alpha.append((event, max(0.1, alpha)))

            # Draw label bar
            frame = draw_label_bar(frame, active_with_alpha, current_time)

            # Flash indicator for very recent events
            for event, start in active_labels:
                age = current_time - start
                if age < 0.3:
                    frame = draw_flash_indicator(frame, event, 1.0 - age / 0.3)

        out.write(frame)
        frame_num += 1

        # Progress
        pct = int(frame_num * 100 / end_frame)
        if pct != last_pct and pct % 5 == 0:
            last_pct = pct
            print(f"  {pct}% ({frame_num}/{end_frame} frames)")

    out.release()
    cap.release()

    print(f"\nRaw video written. Re-encoding with FFmpeg...")

    # Re-encode with FFmpeg for better compression + compatibility
    cmd = [
        "ffmpeg", "-y",
        "-i", temp_path,
        "-c:v", "libx264", "-preset", "fast", "-crf", "22",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    if result.returncode == 0 and os.path.exists(output_path):
        os.unlink(temp_path)
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"Review video: {output_path} ({size_mb:.1f} MB)")
        return True
    else:
        print(f"FFmpeg re-encode failed, keeping raw: {temp_path}")
        if result.stderr:
            print(result.stderr[-300:])
        return False


def main():
    parser = argparse.ArgumentParser(description="Generate continuous review video")
    parser.add_argument("--video", default="preprocessed/IMG_6665.mp4")
    parser.add_argument("--detections", default="IMG_6665_fused_test5.json")
    parser.add_argument("--gt", default="labels/IMG_6665_ground_truth.txt")
    parser.add_argument("--output", default="IMG_6665_review.mp4")
    parser.add_argument("--end-time", type=float, default=None,
                        help="End time in seconds (default: end of ground truth)")
    parser.add_argument("--window", type=float, default=3.0)
    parser.add_argument("--tier", default=None,
                        help="Filter: 'high' or 'high,medium'")
    parser.add_argument("--label-duration", type=float, default=3.0,
                        help="How long labels stay visible (seconds)")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)

    generate_review_video(
        args.video, args.detections, args.gt, args.output,
        end_time=args.end_time,
        match_window=args.window,
        tier_filter=args.tier,
        label_duration=args.label_duration,
    )


if __name__ == "__main__":
    main()
