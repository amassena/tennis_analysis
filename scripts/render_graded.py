#!/usr/bin/env python3
"""Render a tennis video with per-shot biomechanical grade overlay.

Overlays each shot's:
  - Shot type + grade (A-F) + confidence
  - Knee bend, trunk rotation, arm extension angles vs ideal
  - Kinetic chain sequence tick/cross

Grade is computed per-shot from pose landmarks at the contact frame:
  A = 0 yellow + 0 red flags
  B = 1 yellow, 0 red
  C = 2 yellow, 0 red
  D = 1 red
  F = 2+ red

A flag is "yellow" at >15° delta from ideal, "red" at >30°.

Usage:
    python scripts/render_graded.py --video IMG_1141
    python scripts/render_graded.py --video IMG_1141 --output out.mp4
"""

import argparse
import json
import math
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREPROCESSED_DIR = os.path.join(PROJECT_ROOT, "preprocessed")
POSES_DIR = os.path.join(PROJECT_ROOT, "poses_full_videos")
DETECTIONS_DIR = os.path.join(PROJECT_ROOT, "detections")

# BGR colors
SHOT_COLORS = {
    "serve":    (0, 100, 255),
    "forehand": (0, 200, 0),
    "backhand": (255, 100, 0),
    "forehand_volley": (0, 255, 200),
    "backhand_volley": (200, 180, 0),
    "overhead":        (150, 120, 255),
    "unknown":         (180, 180, 180),
}
GRADE_COLORS = {
    "A": (80, 200, 80),     # green
    "B": (120, 220, 120),   # light green
    "C": (0, 200, 220),     # yellow
    "D": (0, 140, 240),     # orange
    "F": (60, 60, 220),     # red
}

# Ideal joint angles at/near contact per shot type
IDEALS = {
    "serve":           {"knee": 120, "trunk": 45, "arm": 165},
    "forehand":        {"knee": 120, "trunk": 50, "arm": 165},
    "backhand":        {"knee": 120, "trunk": 55, "arm": 165},
    "forehand_volley": {"knee": 130, "trunk": 30, "arm": 160},
    "backhand_volley": {"knee": 130, "trunk": 35, "arm": 160},
    "overhead":        {"knee": 115, "trunk": 50, "arm": 165},
}
FLAG_YELLOW = 15  # degrees delta from ideal
FLAG_RED = 30

# Label bar geometry
LABEL_H = 56
TIMELINE_H = 32
SHOT_WINDOW_SEC = 1.5  # overlay visible ±1.5s around contact


def calc_angle(a, b, c):
    """Angle ABC in degrees from three (x, y) points."""
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    dot = ba[0] * bc[0] + ba[1] * bc[1]
    mag = (math.hypot(*ba) * math.hypot(*bc)) or 1e-9
    return math.degrees(math.acos(max(-1.0, min(1.0, dot / mag))))


def lm(frame_entry, idx):
    """Pull landmark idx as (x, y) if visible enough, else None.

    Handles both formats:
      - dict: {"x": ..., "y": ..., "visibility": ...}
      - list/tuple: [x, y, z, visibility]
    """
    if not frame_entry or not frame_entry.get("detected"):
        return None
    lms = frame_entry.get("landmarks")
    if not lms or idx >= len(lms):
        return None
    entry = lms[idx]
    if isinstance(entry, dict):
        if entry.get("visibility", 0) < 0.3:
            return None
        return (entry["x"], entry["y"])
    # list/tuple form: [x, y, z, visibility]
    if len(entry) >= 4 and entry[3] < 0.3:
        return None
    return (entry[0], entry[1])


def compute_shot_metrics(pose_frames, contact_frame, shot_type):
    """Compute knee / trunk / arm angles at the frame closest to contact
    where all required landmarks are visible. Returns dict or None."""
    # Try the exact contact frame first, then expand outward ±12 frames
    for delta in [0] + [d for i in range(1, 13) for d in (-i, i)]:
        idx = contact_frame + delta
        if idx < 0 or idx >= len(pose_frames):
            continue
        pf = pose_frames[idx]
        # MediaPipe landmark indices
        # 11=L_shoulder 12=R_shoulder 13=L_elbow 14=R_elbow 15=L_wrist 16=R_wrist
        # 23=L_hip 24=R_hip 25=L_knee 26=R_knee 27=L_ankle 28=R_ankle
        l_sh, r_sh = lm(pf, 11), lm(pf, 12)
        l_el, r_el = lm(pf, 13), lm(pf, 14)
        l_wr, r_wr = lm(pf, 15), lm(pf, 16)
        l_hp, r_hp = lm(pf, 23), lm(pf, 24)
        l_kn, r_kn = lm(pf, 25), lm(pf, 26)
        l_an, r_an = lm(pf, 27), lm(pf, 28)

        if not (l_sh and r_sh and l_hp and r_hp):
            continue

        # Pick dominant side by knee visibility
        if r_kn and r_hp and r_an:
            hp, kn, an = r_hp, r_kn, r_an
        elif l_kn and l_hp and l_an:
            hp, kn, an = l_hp, l_kn, l_an
        else:
            continue
        knee = calc_angle(hp, kn, an)

        # Arm: dominant = same side as knee unless visibility differs
        if r_sh and r_el and r_wr:
            sh, el, wr = r_sh, r_el, r_wr
        elif l_sh and l_el and l_wr:
            sh, el, wr = l_sh, l_el, l_wr
        else:
            continue
        arm = calc_angle(sh, el, wr)

        # Trunk rotation: angle between shoulder line and hip line
        sh_vec = (r_sh[0] - l_sh[0], r_sh[1] - l_sh[1])
        hp_vec = (r_hp[0] - l_hp[0], r_hp[1] - l_hp[1])
        a_sh = math.degrees(math.atan2(sh_vec[1], sh_vec[0]))
        a_hp = math.degrees(math.atan2(hp_vec[1], hp_vec[0]))
        trunk = abs(((a_sh - a_hp) + 180) % 360 - 180)

        return {
            "knee": int(knee),
            "trunk": int(trunk),
            "arm": int(arm),
            "frame_used": idx,
        }
    return None


def grade_shot(metrics, shot_type):
    """Return (grade_letter, [(label, value, ideal, flag_level)])."""
    ideals = IDEALS.get(shot_type, IDEALS["forehand"])
    rows = []
    yellow = red = 0
    for key, label in [("knee", "Knee"), ("trunk", "Trunk"), ("arm", "Arm")]:
        v = metrics[key]
        ideal = ideals[key]
        d = abs(v - ideal)
        if d >= FLAG_RED:
            level = 2; red += 1
        elif d >= FLAG_YELLOW:
            level = 1; yellow += 1
        else:
            level = 0
        rows.append((label, v, ideal, level))

    if red >= 2:
        grade = "F"
    elif red == 1:
        grade = "D"
    elif yellow >= 2:
        grade = "C"
    elif yellow == 1:
        grade = "B"
    else:
        grade = "A"
    return grade, rows


def flag_color(level):
    return [(80, 200, 80), (0, 200, 220), (60, 60, 220)][level]


def draw_overlay(frame, w, h, det, metrics, grade, rows, ts):
    """Draw the label bar with grade + metric badges."""
    # Background bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, LABEL_H), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    shot = det.get("shot_type", "unknown").upper()
    color = SHOT_COLORS.get(det.get("shot_type"), (200, 200, 200))
    grade_color = GRADE_COLORS.get(grade, (200, 200, 200))

    # Shot type
    cv2.putText(frame, shot, (14, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.05, color, 2, cv2.LINE_AA)

    # Grade chip (colored rounded box)
    gx = 230
    cv2.rectangle(frame, (gx, 12), (gx + 58, 44), grade_color, -1)
    (gw, gh), _ = cv2.getTextSize(grade, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
    cv2.putText(frame, grade, (gx + (58 - gw) // 2, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3, cv2.LINE_AA)

    # Metric badges: Knee / Trunk / Arm
    mx = 310
    if rows:
        for label, v, ideal, level in rows:
            fcol = flag_color(level)
            text = f"{label} {v}"  # no unit — Hershey font lacks the degree symbol
            # dot
            cv2.circle(frame, (mx + 9, 28), 7, fcol, -1)
            cv2.putText(frame, text, (mx + 22, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1, cv2.LINE_AA)
            (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            mx += 22 + tw + 24
    else:
        cv2.putText(frame, "(no pose at contact)", (mx, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (170, 170, 170), 1, cv2.LINE_AA)

    # Confidence + timestamp on right
    conf = det.get("confidence", 0)
    right = f"{conf:.0%}  |  {ts:.1f}s"
    (tw, _), _ = cv2.getTextSize(right, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.putText(frame, right, (w - tw - 14, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # Grade color accent bar under the label bar
    cv2.rectangle(frame, (0, LABEL_H), (w, LABEL_H + 4), grade_color, -1)


def draw_timeline(frame, detections, fps, total_frames, frame_idx, w, h):
    """Draw a timeline bar at bottom with colored shot markers + playhead."""
    y0 = h - TIMELINE_H
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    for d in detections:
        fi = d.get("frame", 0)
        x = int(fi / max(1, total_frames) * w)
        col = SHOT_COLORS.get(d.get("shot_type"), (200, 200, 200))
        cv2.rectangle(frame, (max(0, x - 2), y0 + 4), (min(w, x + 2), h - 4), col, -1)

    px = int(frame_idx / max(1, total_frames) * w)
    cv2.line(frame, (px, y0), (px, h), (255, 255, 255), 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Video ID, e.g. IMG_1141")
    parser.add_argument("--output", help="Output path (default: <video>_graded.mp4)")
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()

    vid = args.video
    video_path = os.path.join(PREPROCESSED_DIR, f"{vid}.mp4")
    detections_path = os.path.join(DETECTIONS_DIR, f"{vid}_fused_detections.json")
    if not os.path.exists(detections_path):
        detections_path = os.path.join(DETECTIONS_DIR, f"{vid}_fused.json")
    poses_path = os.path.join(POSES_DIR, f"{vid}.json")
    out_path = args.output or os.path.join(PROJECT_ROOT, "exports", f"{vid}_graded.mp4")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    for p in (video_path, detections_path, poses_path):
        if not os.path.exists(p):
            print(f"[ERROR] Missing: {p}")
            sys.exit(1)

    print(f"Loading detections: {detections_path}")
    with open(detections_path) as f:
        det_data = json.load(f)
    detections = det_data.get("detections", [])
    fps = det_data.get("fps", 60.0)
    total_frames = det_data.get("total_frames", 0)
    print(f"  {len(detections)} shots @ {fps} fps, {total_frames} frames")

    print(f"Loading poses: {poses_path}")
    with open(poses_path) as f:
        pose_data = json.load(f)
    pose_frames = pose_data.get("frames", [])
    if not total_frames:
        total_frames = len(pose_frames)
    print(f"  {len(pose_frames)} pose frames")

    # Pre-compute grade + metrics for each shot
    print("Computing per-shot grades...")
    shot_info = []  # (start_frame, end_frame, det, grade, rows)
    window = int(SHOT_WINDOW_SEC * fps)
    grade_counts = {}
    for d in detections:
        cf = int(d.get("frame", 0))
        metrics = compute_shot_metrics(pose_frames, cf, d.get("shot_type"))
        if metrics:
            grade, rows = grade_shot(metrics, d.get("shot_type"))
        else:
            grade, rows = "?", []
        grade_counts[grade] = grade_counts.get(grade, 0) + 1
        shot_info.append((max(0, cf - window), cf + window, d, grade, rows))
    print(f"  Grades: {dict(sorted(grade_counts.items()))}")

    # Build frame -> shot info lookup
    lookup = [None] * max(total_frames, len(pose_frames))
    for start, end, d, g, r in shot_info:
        for f_idx in range(start, min(end + 1, len(lookup))):
            lookup[f_idx] = (d, g, r)

    # Render
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open {video_path}")
        sys.exit(1)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  Video: {w}x{h} @ {src_fps} fps, {frame_count} frames")
    print(f"  Output: {out_path}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, src_fps, (w, h))

    stop = args.max_frames if args.max_frames > 0 else frame_count
    idx = 0
    while idx < stop:
        ok, frame = cap.read()
        if not ok:
            break
        entry = lookup[idx] if idx < len(lookup) else None
        if entry:
            d, grade, rows = entry
            ts = idx / src_fps
            metrics = {"knee": 0, "trunk": 0, "arm": 0}
            draw_overlay(frame, w, h, d, metrics, grade, rows, ts)
        draw_timeline(frame, detections, src_fps, total_frames, idx, w, h)
        out.write(frame)
        idx += 1
        if idx % 1800 == 0:
            pct = idx / stop * 100
            print(f"  {idx}/{stop} ({pct:.1f}%)")

    cap.release()
    out.release()
    print(f"Done: {out_path} ({os.path.getsize(out_path) / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
