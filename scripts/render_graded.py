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
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_font(size):
    """Load a TTF font that supports Unicode (degree symbol, etc.)."""
    candidates = [
        # macOS
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        # Windows
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\segoeui.ttf",
        # Linux
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()


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
    """Angle ABC in degrees from three N-D points (2D or 3D)."""
    # Works for any dimensionality by only using the components both share
    n = min(len(a), len(b), len(c))
    ba = [a[i] - b[i] for i in range(n)]
    bc = [c[i] - b[i] for i in range(n)]
    dot = sum(ba[i] * bc[i] for i in range(n))
    mag = (math.sqrt(sum(v * v for v in ba)) * math.sqrt(sum(v * v for v in bc))) or 1e-9
    return math.degrees(math.acos(max(-1.0, min(1.0, dot / mag))))


def _angle_between_vectors_on_plane(v1, v2, plane_normal):
    """Signed angle between v1 and v2 projected onto a plane defined by its normal.

    Returns unsigned magnitude in degrees [0, 180].
    """
    import numpy as np
    n = np.array(plane_normal, dtype=float)
    n = n / (np.linalg.norm(n) or 1e-9)
    def proj(v):
        v = np.array(v, dtype=float)
        return v - np.dot(v, n) * n
    p1 = proj(v1); p2 = proj(v2)
    m = (np.linalg.norm(p1) * np.linalg.norm(p2)) or 1e-9
    cos = max(-1.0, min(1.0, float(np.dot(p1, p2) / m)))
    return math.degrees(math.acos(cos))


def lm(frame_entry, idx):
    """Pull landmark idx as (x, y) in normalized image coords. Projection-dependent."""
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
    if len(entry) >= 4 and entry[3] < 0.3:
        return None
    return (entry[0], entry[1])


def lm3d(frame_entry, idx):
    """Pull landmark idx as (x, y, z) from world_landmarks — hip-centered metric 3D space.

    World landmarks are camera-invariant: same pose returns same angles
    regardless of shooting angle. Preferred over 2D `lm()` for angle calcs.
    """
    if not frame_entry or not frame_entry.get("detected"):
        return None
    lms = frame_entry.get("world_landmarks") or frame_entry.get("landmarks")
    if not lms or idx >= len(lms):
        return None
    entry = lms[idx]
    # Use visibility from 2D landmarks — world_landmarks doesn't always have it
    vis_source = frame_entry.get("landmarks")
    if vis_source and idx < len(vis_source):
        v2d = vis_source[idx]
        vis = v2d.get("visibility", 1.0) if isinstance(v2d, dict) else (
            v2d[3] if len(v2d) >= 4 else 1.0)
        if vis < 0.3:
            return None
    if isinstance(entry, dict):
        return (entry["x"], entry["y"], entry.get("z", 0.0))
    # [x, y, z, visibility] or [x, y, z]
    if len(entry) >= 3:
        return (entry[0], entry[1], entry[2])
    return None


def compute_shot_metrics(pose_frames, contact_frame, shot_type, use_3d=True):
    """Compute knee / trunk / arm angles at the frame closest to contact.

    When use_3d=True (default), uses MediaPipe's `world_landmarks` — a hip-
    centered, metric, camera-invariant 3D coordinate system. Joint angles
    are then TRUE angles in 3D space, not image-plane projections.

    Trunk rotation in 3D is measured as the angle between the shoulder line
    and the hip line projected onto the horizontal (X-Z) plane. Vertical
    component is ignored since we only care about rotation about the
    longitudinal (Y) axis.

    Falls back to 2D calc if world_landmarks aren't available.

    Returns dict or None.
    """
    picker = lm3d if use_3d else lm
    for delta in [0] + [d for i in range(1, 13) for d in (-i, i)]:
        idx = contact_frame + delta
        if idx < 0 or idx >= len(pose_frames):
            continue
        pf = pose_frames[idx]

        # If we asked for 3D but this frame has no world_landmarks, degrade to 2D
        if use_3d and not pf.get("world_landmarks"):
            picker = lm

        l_sh, r_sh = picker(pf, 11), picker(pf, 12)
        l_el, r_el = picker(pf, 13), picker(pf, 14)
        l_wr, r_wr = picker(pf, 15), picker(pf, 16)
        l_hp, r_hp = picker(pf, 23), picker(pf, 24)
        l_kn, r_kn = picker(pf, 25), picker(pf, 26)
        l_an, r_an = picker(pf, 27), picker(pf, 28)

        if not (l_sh and r_sh and l_hp and r_hp):
            continue

        # Always use right side (racket arm for right-handed player); fall back to left
        hp, kn, an = r_hp, r_kn, r_an
        if not (hp and kn and an):
            hp, kn, an = l_hp, l_kn, l_an
        if not (hp and kn and an):
            continue
        knee = calc_angle(hp, kn, an)

        sh, el, wr = r_sh, r_el, r_wr
        if not (sh and el and wr):
            sh, el, wr = l_sh, l_el, l_wr
        if not (sh and el and wr):
            continue
        arm = calc_angle(sh, el, wr)

        # Trunk rotation
        sh_vec = tuple(r_sh[i] - l_sh[i] for i in range(len(r_sh)))
        hp_vec = tuple(r_hp[i] - l_hp[i] for i in range(len(r_hp)))
        if use_3d and len(sh_vec) >= 3:
            # Project onto horizontal plane: in MediaPipe world coords, Y is vertical.
            # So rotation about the vertical axis = angle between shoulder_line and
            # hip_line after zeroing out the Y component.
            trunk = _angle_between_vectors_on_plane(sh_vec, hp_vec, (0.0, 1.0, 0.0))
        else:
            # 2D: atan2 image-plane angle difference
            a_sh = math.degrees(math.atan2(sh_vec[1], sh_vec[0]))
            a_hp = math.degrees(math.atan2(hp_vec[1], hp_vec[0]))
            trunk = abs(((a_sh - a_hp) + 180) % 360 - 180)

        return {
            "knee": int(knee),
            "trunk": int(trunk),
            "arm": int(arm),
            "frame_used": idx,
            "mode": "3d" if use_3d and picker is lm3d else "2d",
        }
    return None


def grade_shot(metrics, shot_type):
    """Return (grade_letter, [(label, value, ideal, flag_level)])."""
    ideals = IDEALS.get(shot_type, IDEALS["forehand"])
    rows = []
    yellow = red = 0
    for key, label in [("knee", "Knee"), ("trunk", "Trunk"), ("arm", "R.Arm")]:
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


def build_shot_tile(w, det, grade, rows, ts):
    """Build a numpy BGR tile (w × LABEL_H+4) with the label bar fully rendered.

    Uses PIL for all text so we get true Unicode (°) and better antialiasing.
    Returns BGR uint8 array to paste directly onto frames during playback.
    """
    # PIL wants RGB; we'll convert to BGR at the end
    img = Image.new("RGB", (w, LABEL_H + 4), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    font_lg = _load_font(32)
    font_md = _load_font(22)
    font_sm = _load_font(18)

    shot = det.get("shot_type", "unknown").upper()
    shot_color = SHOT_COLORS.get(det.get("shot_type"), (200, 200, 200))
    grade_color = GRADE_COLORS.get(grade, (200, 200, 200))

    # BGR → RGB for PIL
    def to_rgb(c): return (c[2], c[1], c[0])

    # Shot label
    draw.text((14, 10), shot, font=font_lg, fill=to_rgb(shot_color))

    # Grade chip
    gx = 230
    draw.rectangle([gx, 10, gx + 58, 46], fill=to_rgb(grade_color))
    gw = draw.textlength(grade, font=font_lg)
    draw.text((gx + (58 - gw) / 2, 8), grade, font=font_lg, fill=(0, 0, 0))

    # Metric badges
    mx = 310
    if rows:
        for label, v, ideal, level in rows:
            fcol = flag_color(level)
            # Dot
            draw.ellipse([mx + 2, 20, mx + 18, 36], fill=to_rgb(fcol))
            val_text = f"{label} {v}\u00b0"
            ideal_text = f"/{ideal}\u00b0"
            draw.text((mx + 24, 12), val_text, font=font_md, fill=(230, 230, 230))
            vw = draw.textlength(val_text, font=font_md)
            draw.text((mx + 24 + vw + 2, 16), ideal_text, font=font_sm, fill=(120, 120, 120))
            iw = draw.textlength(ideal_text, font=font_sm)
            mx += 24 + int(vw) + int(iw) + 22
    else:
        draw.text((mx, 14), "(no pose at contact)", font=font_md, fill=(170, 170, 170))

    # Confidence + timestamp on right
    conf = det.get("confidence", 0)
    right = f"{conf:.0%}  |  {ts:.1f}s"
    rw = draw.textlength(right, font=font_md)
    draw.text((w - rw - 14, 14), right, font=font_md, fill=(255, 255, 255))

    # Grade accent bar
    draw.rectangle([0, LABEL_H, w, LABEL_H + 4], fill=to_rgb(grade_color))

    rgb = np.array(img)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr


def apply_shot_tile(frame, tile):
    """Blend the pre-built shot tile onto the top of the frame with 65% opacity."""
    th, tw = tile.shape[:2]
    roi = frame[0:th, 0:tw]
    # Only the bar area (top LABEL_H rows) gets blended; the 4px accent is solid
    cv2.addWeighted(tile[:LABEL_H], 0.75, roi[:LABEL_H], 0.25, 0, roi[:LABEL_H])
    # Solid accent bar below
    roi[LABEL_H:LABEL_H + 4] = tile[LABEL_H:LABEL_H + 4]


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
    parser.add_argument("--variant", default="timeline",
                        help="Which output variant to grade: timeline | rally | "
                             "rally_slowmo | forehands | forehands_slowmo | "
                             "backhands | backhands_slowmo | serves | serves_slowmo | "
                             "volleys | volleys_slowmo. Non-timeline variants read the "
                             "variant .mp4 already in exports/ and use shots.json positions.")
    parser.add_argument("--input", help="Override input mp4 (useful for non-timeline variants)")
    parser.add_argument("--shots-json", help="Path to shots.json for variant positions")
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()

    vid = args.video
    variant = args.variant
    is_timeline = (variant == "timeline")

    # Input video
    if args.input:
        video_path = args.input
    elif is_timeline:
        video_path = os.path.join(PREPROCESSED_DIR, f"{vid}.mp4")
    else:
        # Look in exports/{vid}/ first, fall back to exports/ root
        c1 = os.path.join(PROJECT_ROOT, "exports", vid, f"{vid}_{variant}.mp4")
        c2 = os.path.join(PROJECT_ROOT, "exports", f"{vid}_{variant}.mp4")
        video_path = c1 if os.path.exists(c1) else c2
    detections_path = os.path.join(DETECTIONS_DIR, f"{vid}_fused_detections.json")
    if not os.path.exists(detections_path):
        detections_path = os.path.join(DETECTIONS_DIR, f"{vid}_fused.json")
    poses_path = os.path.join(POSES_DIR, f"{vid}.json")

    # Shots.json (only needed for non-timeline variants)
    shots_path = args.shots_json
    if not shots_path and not is_timeline:
        for cand in [os.path.join(PROJECT_ROOT, "exports", vid, "shots.json"),
                     os.path.join(PROJECT_ROOT, "exports", f"{vid}_shots.json"),
                     os.path.join(PROJECT_ROOT, "shots", f"{vid}.json")]:
            if os.path.exists(cand):
                shots_path = cand; break

    default_out = f"{vid}_{variant}_graded.mp4" if not is_timeline else f"{vid}_graded.mp4"
    out_path = args.output or os.path.join(PROJECT_ROOT, "exports", default_out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    required = [video_path, detections_path, poses_path]
    if not is_timeline:
        if not shots_path:
            print(f"[ERROR] --variant {variant} requires shots.json. Provide via "
                  f"--shots-json or place at exports/{vid}/shots.json")
            sys.exit(1)
        required.append(shots_path)
    for p in required:
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
    # We need actual video dims for tile width; open cap early to get w
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open {video_path}")
        sys.exit(1)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Load shots.json for non-timeline variants (maps source frame → output time)
    variant_positions = None
    if not is_timeline and shots_path:
        with open(shots_path) as sf:
            shots_data = json.load(sf)
        # idx → output time (seconds) in the target variant
        variant_positions = {}
        for s in shots_data.get("shots", []):
            pos = s.get("positions", {}).get(variant)
            if pos is not None:
                variant_positions[s.get("idx")] = float(pos)
        # For slow-mo variants the on-screen shot window should be stretched
        # to cover the slowed playback (e.g. 0.25x ⇒ 4× window)
        speed = 0.25 if variant.endswith("_slowmo") else 1.0
        shot_window_frames = int(SHOT_WINDOW_SEC * src_fps / speed)
    else:
        shot_window_frames = int(SHOT_WINDOW_SEC * src_fps)

    shot_info = []  # (start_frame, end_frame, tile)
    grade_counts = {}
    for idx, d in enumerate(detections):
        cf = int(d.get("frame", 0))
        metrics = compute_shot_metrics(pose_frames, cf, d.get("shot_type"))
        if metrics:
            grade, rows = grade_shot(metrics, d.get("shot_type"))
        else:
            grade, rows = "?", []
        grade_counts[grade] = grade_counts.get(grade, 0) + 1
        ts = cf / fps  # source-timeline ts (shown on the tile)
        tile = build_shot_tile(w, d, grade, rows, ts)

        if is_timeline:
            # 1:1 mapping: tile active around cf
            shot_info.append((max(0, cf - shot_window_frames),
                              cf + shot_window_frames, tile))
        else:
            # Shot may or may not appear in this variant
            if idx not in variant_positions:
                continue
            out_center = int(variant_positions[idx] * src_fps)
            shot_info.append((max(0, out_center - shot_window_frames),
                              out_center + shot_window_frames, tile))
    print(f"  Grades: {dict(sorted(grade_counts.items()))}  (variant={variant})")

    # Build frame -> tile lookup. For non-timeline variants the total output
    # frame count isn't total_frames — use the video's own frame count.
    lookup_size = max(total_frames, len(pose_frames))
    if not is_timeline:
        lookup_size = max(lookup_size, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    lookup = [None] * lookup_size
    for start, end, tile in shot_info:
        for f_idx in range(start, min(end + 1, len(lookup))):
            lookup[f_idx] = tile

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
        tile = lookup[idx] if idx < len(lookup) else None
        if tile is not None:
            apply_shot_tile(frame, tile)
        if is_timeline:
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
