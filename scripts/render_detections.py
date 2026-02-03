#!/usr/bin/env python3
"""Render an annotated video with shot detection overlay."""

import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import POSES_DIR, PREPROCESSED_DIR, PROJECT_ROOT

# Colors (BGR)
COLORS = {
    "serve":    (0, 100, 255),   # orange
    "neutral":  (180, 180, 180), # gray
    "forehand": (0, 200, 0),     # green
    "backhand": (255, 100, 0),   # blue
}
TIMELINE_H = 36
LABEL_H = 48


def build_frame_lookup(segments, total_frames):
    """Map every frame index to its segment (or None)."""
    lookup = [None] * total_frames
    for seg in segments:
        for f in range(seg["start_frame"], min(seg["end_frame"] + 1, total_frames)):
            lookup[f] = seg
    return lookup


def draw_label_bar(frame, seg, frame_idx, fps, w):
    """Draw shot type label and confidence at the top of the frame."""
    if seg is None:
        return
    color = COLORS.get(seg["shot_type"], (200, 200, 200))
    shot = seg["shot_type"].upper()
    conf = seg["avg_confidence"]
    ts = frame_idx / fps

    # Background bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, LABEL_H), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Shot type label
    cv2.putText(frame, shot, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)

    # Confidence
    conf_text = f"{conf:.0%}"
    cv2.putText(frame, conf_text, (200, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    # Timestamp
    ts_text = f"{ts:.1f}s"
    (tw, _), _ = cv2.getTextSize(ts_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.putText(frame, ts_text, (w - tw - 12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)


def draw_timeline(frame, segments, frame_idx, total_frames, w, h):
    """Draw a colored timeline bar at the bottom with playhead."""
    y0 = h - TIMELINE_H
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Draw segments
    for seg in segments:
        x1 = int(seg["start_frame"] / total_frames * w)
        x2 = int(seg["end_frame"] / total_frames * w)
        color = COLORS.get(seg["shot_type"], (200, 200, 200))
        cv2.rectangle(frame, (x1, y0 + 4), (x2, h - 4), color, -1)

    # Playhead
    px = int(frame_idx / total_frames * w)
    cv2.line(frame, (px, y0), (px, h), (255, 255, 255), 2)


def main():
    parser = argparse.ArgumentParser(description="Render video with shot detection overlay")
    parser.add_argument("--detections", default=os.path.join(PROJECT_ROOT, "shots_detected.json"),
                        help="Path to shots_detected.json")
    parser.add_argument("--video", help="Source video (default: preprocessed or skeleton)")
    parser.add_argument("-o", "--output", help="Output video path")
    parser.add_argument("--skeleton", action="store_true", help="Use skeleton video instead of preprocessed")
    args = parser.parse_args()

    # Load detections
    with open(args.detections) as f:
        det = json.load(f)

    segments = det["segments"]
    fps = det["fps"]
    total_frames = det["total_frames"]
    video_name = det["source_video"]

    # Find source video
    if args.video:
        video_path = args.video
    elif args.skeleton:
        video_path = os.path.join(POSES_DIR, f"{video_name}_skeleton.mp4")
    else:
        video_path = os.path.join(PREPROCESSED_DIR, f"{video_name}.mp4")
        if not os.path.exists(video_path):
            video_path = os.path.join(POSES_DIR, f"{video_name}_skeleton.mp4")

    if not os.path.exists(video_path):
        print(f"[ERROR] Video not found: {video_path}")
        sys.exit(1)

    output_path = args.output or os.path.join(PROJECT_ROOT, "clips",
                                               f"{video_name}_detected.mp4")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Source:     {os.path.basename(video_path)}")
    print(f"Detections: {len(segments)} segments")
    print(f"Output:     {output_path}")
    print()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open {video_path}")
        sys.exit(1)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"  {w}x{h} @ {src_fps} fps, {frame_count} frames")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, src_fps, (w, h))

    lookup = build_frame_lookup(segments, total_frames)

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        seg = lookup[idx] if idx < len(lookup) else None
        draw_label_bar(frame, seg, idx, fps, w)
        draw_timeline(frame, segments, idx, total_frames, w, h)

        out.write(frame)
        idx += 1

        if idx % 500 == 0:
            pct = idx / frame_count * 100
            print(f"  {idx}/{frame_count} frames ({pct:.0f}%)")

    cap.release()
    out.release()

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n  Wrote {idx} frames to {os.path.basename(output_path)} ({size_mb:.1f} MB)")

    # Re-encode with ffmpeg for better compression and compatibility
    final_path = output_path.replace(".mp4", "_final.mp4")
    print(f"  Re-encoding with ffmpeg...")
    ret = os.system(
        f'ffmpeg -y -i "{output_path}" -c:v libx264 -crf 20 -preset medium '
        f'-pix_fmt yuv420p -movflags +faststart "{final_path}" 2>/dev/null'
    )
    if ret == 0 and os.path.exists(final_path):
        os.replace(final_path, output_path)
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"  Final: {os.path.basename(output_path)} ({size_mb:.1f} MB)")
    else:
        print(f"  [WARN] ffmpeg re-encode failed, keeping raw mp4v output")


if __name__ == "__main__":
    main()
