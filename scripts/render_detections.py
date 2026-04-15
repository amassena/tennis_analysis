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
    "serve":           (0, 100, 255),   # orange
    "neutral":         (180, 180, 180), # gray
    "forehand":        (0, 200, 0),     # green
    "backhand":        (255, 100, 0),   # blue
    "forehand_volley": (0, 255, 200),   # teal
    "false_positive":  (0, 0, 200),     # red
    "missed":          (0, 200, 255),   # yellow
}
TIMELINE_H = 36
LEGEND_H = 28
LABEL_H = 48

# Form deviation thresholds (degrees from ideal)
FORM_GREEN = 15   # within 15° = green
FORM_YELLOW = 30  # 15-30° = yellow, >30° = red

# Ideal joint angles per shot type (recreational → advanced targets)
IDEAL_ANGLES = {
    "serve":    {"knee_bend": 120, "trunk_rot": 45, "arm_ext": 165},
    "forehand": {"knee_bend": 120, "trunk_rot": 50, "arm_ext": 165},
    "backhand": {"knee_bend": 120, "trunk_rot": 55, "arm_ext": 165},
}


def form_color(delta):
    """BGR color for a joint-angle delta from ideal. Green <15°, yellow <30°, red else."""
    d = abs(delta)
    if d <= FORM_GREEN:
        return (80, 200, 80)    # green
    if d <= FORM_YELLOW:
        return (0, 200, 220)    # yellow
    return (60, 60, 220)        # red


def calc_angle(a, b, c):
    """Angle ABC in degrees, given three (x, y) points."""
    import math
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    dot = ba[0] * bc[0] + ba[1] * bc[1]
    mag = (math.hypot(*ba) * math.hypot(*bc)) or 1e-9
    cos = max(-1.0, min(1.0, dot / mag))
    return math.degrees(math.acos(cos))


def draw_form_panel(frame, seg, pose_frame, w, h):
    """Draw a small form-deviation panel with color-coded joint angle dots.

    pose_frame: landmark dict from poses JSON for the current frame.
    Shows knee bend, trunk rotation, arm extension vs ideals for the shot type.
    """
    if seg is None or not pose_frame or not pose_frame.get("detected"):
        return
    ideals = IDEAL_ANGLES.get(seg["shot_type"])
    if not ideals:
        return
    lms = pose_frame.get("landmarks")
    if not lms or len(lms) < 29:
        return

    # MediaPipe indexes: 12=R_shoulder, 14=R_elbow, 16=R_wrist,
    # 24=R_hip, 26=R_knee, 28=R_ankle
    # 11=L_shoulder, 13=L_elbow, 15=L_wrist, 23=L_hip, 25=L_knee, 27=L_ankle
    def pt(i):
        lm = lms[i]
        if lm.get("visibility", 0) < 0.3:
            return None
        return (lm["x"] * w, lm["y"] * h)

    measurements = []  # (label, measured, ideal, delta)
    # Dominant side: pick side with better visibility at knee
    side = "right" if (lms[26].get("visibility", 0) >= lms[25].get("visibility", 0)) else "left"
    if side == "right":
        sh, el, wr, hp, kn, an = pt(12), pt(14), pt(16), pt(24), pt(26), pt(28)
    else:
        sh, el, wr, hp, kn, an = pt(11), pt(13), pt(15), pt(23), pt(25), pt(27)

    if hp and kn and an:
        k = calc_angle(hp, kn, an)
        measurements.append(("KNEE", int(k), ideals["knee_bend"], int(k - ideals["knee_bend"])))
    if sh and el and wr:
        a = calc_angle(sh, el, wr)
        measurements.append(("ARM", int(a), ideals["arm_ext"], int(a - ideals["arm_ext"])))
    # Trunk rotation: angle between shoulder line and hip line, projected
    if sh and hp and pt(11) and pt(23):
        l_sh, r_sh = pt(11), pt(12)
        l_hp, r_hp = pt(23), pt(24)
        if l_sh and r_sh and l_hp and r_hp:
            import math
            sh_vec = (r_sh[0] - l_sh[0], r_sh[1] - l_sh[1])
            hp_vec = (r_hp[0] - l_hp[0], r_hp[1] - l_hp[1])
            a_sh = math.degrees(math.atan2(sh_vec[1], sh_vec[0]))
            a_hp = math.degrees(math.atan2(hp_vec[1], hp_vec[0]))
            rot = abs(((a_sh - a_hp) + 180) % 360 - 180)
            measurements.append(("TRUNK", int(rot), ideals["trunk_rot"], int(rot - ideals["trunk_rot"])))

    if not measurements:
        return

    # Draw panel top-right
    panel_w, panel_h = 180, 28 + 22 * len(measurements)
    px, py = w - panel_w - 12, LABEL_H + 12
    overlay = frame.copy()
    cv2.rectangle(overlay, (px, py), (px + panel_w, py + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    cv2.putText(frame, "FORM", (px + 10, py + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    for i, (lbl, measured, ideal, delta) in enumerate(measurements):
        yy = py + 28 + 22 * i + 16
        color = form_color(delta)
        cv2.circle(frame, (px + 16, yy - 4), 5, color, -1)
        cv2.putText(frame, f"{lbl:5s} {measured:3d}° vs {ideal}°",
                    (px + 28, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (230, 230, 230), 1, cv2.LINE_AA)


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


def draw_legend(frame, segment_types, w, h):
    """Draw a small color legend above the timeline bar."""
    # Only draw types that actually appear in the segments
    items = [(st, COLORS[st]) for st in segment_types if st in COLORS]
    if not items:
        return

    y0 = h - TIMELINE_H - LEGEND_H
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y0), (w, y0 + LEGEND_H), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    x = 10
    font = cv2.FONT_HERSHEY_SIMPLEX
    for label, color in items:
        # Color swatch
        cv2.rectangle(frame, (x, y0 + 6), (x + 16, y0 + 22), color, -1)
        x += 22
        # Label text
        cv2.putText(frame, label, (x, y0 + 20), font, 0.45,
                    (220, 220, 220), 1, cv2.LINE_AA)
        (tw, _), _ = cv2.getTextSize(label, font, 0.45, 1)
        x += tw + 18


def main():
    parser = argparse.ArgumentParser(description="Render video with shot detection overlay")
    parser.add_argument("--detections", default=os.path.join(PROJECT_ROOT, "shots_detected.json"),
                        help="Path to shots_detected.json")
    parser.add_argument("--video", help="Source video (default: preprocessed or skeleton)")
    parser.add_argument("-o", "--output", help="Output video path")
    parser.add_argument("--skeleton", action="store_true", help="Use skeleton video instead of preprocessed")
    parser.add_argument("--no-label", action="store_true", help="Skip text label bar at top (timeline only)")
    parser.add_argument("--show-form", action="store_true",
                        help="Overlay color-coded form deviation panel (requires poses JSON)")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0=all)")
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
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

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

    # Optional pose data for form overlay
    pose_frames = None
    if args.show_form:
        poses_path = os.path.join(POSES_DIR, f"{video_name}.json")
        if os.path.exists(poses_path):
            with open(poses_path) as pf:
                pd = json.load(pf)
            pose_frames = pd.get("frames", [])
            print(f"  Form overlay: loaded {len(pose_frames)} pose frames")
        else:
            print(f"  --show-form requested but no poses at {poses_path}")

    # Collect unique segment types for legend
    segment_types = list(dict.fromkeys(s["shot_type"] for s in segments))

    stop_at = args.max_frames if args.max_frames > 0 else frame_count

    idx = 0
    while idx < stop_at:
        ret, frame = cap.read()
        if not ret:
            break

        seg = lookup[idx] if idx < len(lookup) else None
        if not args.no_label:
            draw_label_bar(frame, seg, idx, fps, w)
        draw_timeline(frame, segments, idx, total_frames, w, h)
        draw_legend(frame, segment_types, w, h)
        if pose_frames and idx < len(pose_frames):
            draw_form_panel(frame, seg, pose_frames[idx], w, h)

        out.write(frame)
        idx += 1

        if idx % 3000 == 0:
            pct = idx / stop_at * 100
            print(f"  {idx}/{stop_at} frames ({pct:.0f}%)")

    cap.release()
    out.release()

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n  Wrote {idx} frames to {os.path.basename(output_path)} ({size_mb:.1f} MB)")

    # Re-encode with ffmpeg for better compression and compatibility
    final_path = output_path.replace(".mp4", "_final.mp4")
    print(f"  Re-encoding with ffmpeg...")
    # Try NVENC first (Windows GPU), fall back to libx264
    import subprocess
    nvenc_check = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"],
        capture_output=True, text=True
    )
    if "h264_nvenc" in nvenc_check.stdout:
        enc = "h264_nvenc -preset p4 -cq 22"
    else:
        enc = "libx264 -crf 20 -preset medium"
    ret = os.system(
        f'ffmpeg -y -i "{output_path}" -c:v {enc} '
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
