#!/usr/bin/env python3
"""Generate swing-sequence composite images showing full shot motion.

For each detected shot, extracts 5-7 key frames around contact,
auto-crops to the player bounding box, and composites them side-by-side
into a single filmstrip image — like the Djokovic reference images.

Optionally draws:
  - Pose skeleton on each frame
  - Wrist motion trail (cyan line connecting wrist across frames)
  - Joint angle annotations at contact frame

Usage:
    python scripts/swing_composite.py --video IMG_1141
    python scripts/swing_composite.py --video IMG_1141 --shot 5    # specific shot
    python scripts/swing_composite.py --video IMG_1141 --upload    # upload to R2
    python scripts/swing_composite.py --video IMG_1141 --no-skeleton
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT_ROOT = Path(__file__).parent.parent
PREPROCESSED_DIR = PROJECT_ROOT / "preprocessed"
POSES_DIR = PROJECT_ROOT / "poses_full_videos"
DETECTIONS_DIR = PROJECT_ROOT / "detections"

# How many frames to sample per shot
NUM_FRAMES = 7
# Time window: frames from before_sec before contact to after_sec after
BEFORE_SEC = 0.5
AFTER_SEC = 0.3
# Output height per frame panel (width computed from aspect ratio)
PANEL_HEIGHT = 480
# Skeleton line colors (BGR)
SKELETON_COLOR = (255, 200, 0)  # cyan
WRIST_TRAIL_COLOR = (255, 255, 0)  # cyan-yellow
JOINT_DOT_COLOR = (0, 255, 200)
CONTACT_FRAME_BORDER = (0, 140, 255)  # orange border on the contact frame

# Skeleton connections (MediaPipe indices)
SKELETON_PAIRS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # shoulders + arms
    (11, 23), (12, 24), (23, 24),                        # torso
    (23, 25), (25, 27), (24, 26), (26, 28),              # legs
]


def load_data(vid):
    """Load detection + pose data for a video."""
    det_path = DETECTIONS_DIR / f"{vid}_fused_detections.json"
    if not det_path.exists():
        det_path = DETECTIONS_DIR / f"{vid}_fused.json"
    pose_path = POSES_DIR / f"{vid}.json"
    video_path = PREPROCESSED_DIR / f"{vid}.mp4"

    for p in (det_path, pose_path, video_path):
        if not p.exists():
            print(f"[ERROR] Missing: {p}")
            return None, None, None

    with open(det_path) as f:
        det = json.load(f)
    with open(pose_path) as f:
        poses = json.load(f)
    return det, poses, str(video_path)


def get_landmarks(pose_frames, frame_idx):
    """Get landmarks as list of (x, y) normalized coords, handling both formats."""
    if frame_idx >= len(pose_frames):
        return None
    pf = pose_frames[frame_idx]
    if not pf.get("detected") or not pf.get("landmarks"):
        return None
    lms = pf["landmarks"]
    result = []
    for lm in lms:
        if isinstance(lm, dict):
            vis = lm.get("visibility", 0)
            result.append((lm["x"], lm["y"], vis))
        else:
            vis = lm[3] if len(lm) >= 4 else 1.0
            result.append((lm[0], lm[1], vis))
    return result


def compute_player_bbox(landmarks, img_w, img_h, padding=0.15):
    """Compute tight bounding box around visible landmarks with padding."""
    visible = [(x * img_w, y * img_h) for x, y, v in landmarks if v > 0.3]
    if len(visible) < 4:
        return None
    xs = [p[0] for p in visible]
    ys = [p[1] for p in visible]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    w = x_max - x_min
    h = y_max - y_min
    # Add padding
    x_min -= w * padding
    y_min -= h * padding
    x_max += w * padding
    y_max += h * padding
    # Enforce 3:4 aspect ratio (portrait)
    target_ratio = 3.0 / 4.0
    bw = x_max - x_min
    bh = y_max - y_min
    if bw / bh > target_ratio:
        # too wide — increase height
        new_h = bw / target_ratio
        cy = (y_min + y_max) / 2
        y_min = cy - new_h / 2
        y_max = cy + new_h / 2
    else:
        # too tall — increase width
        new_w = bh * target_ratio
        cx = (x_min + x_max) / 2
        x_min = cx - new_w / 2
        x_max = cx + new_w / 2
    # Clamp
    x_min = max(0, int(x_min))
    y_min = max(0, int(y_min))
    x_max = min(img_w, int(x_max))
    y_max = min(img_h, int(y_max))
    return (x_min, y_min, x_max, y_max)


def draw_skeleton(img, landmarks, bbox):
    """Draw skeleton lines and joint dots on a cropped image."""
    bx, by, bx2, by2 = bbox
    bw, bh = bx2 - bx, by2 - by
    h, w = img.shape[:2]

    def to_px(lm_idx):
        if lm_idx >= len(landmarks):
            return None
        x, y, v = landmarks[lm_idx]
        if v < 0.3:
            return None
        px = int((x * w / bw * (bx2 - bx) - (bx - bx) * w / bw) if bw > 0 else 0)
        py = int((y * h / bh * (by2 - by) - (by - by) * h / bh) if bh > 0 else 0)
        # Simpler: map normalized coords relative to bbox
        px = int((x * (bx2 - bx + bx2 - bx) / 1.0 - bx) * w / (bx2 - bx)) if bw > 0 else 0
        py = int((y * (by2 - by + by2 - by) / 1.0 - by) * h / (by2 - by)) if bh > 0 else 0
        return (px, py)

    # Simpler coordinate mapping
    img_h_orig = int(bh / (landmarks[0][1] + 0.001))  # rough
    def lm_to_crop(lm_idx):
        if lm_idx >= len(landmarks):
            return None
        x_norm, y_norm, v = landmarks[lm_idx]
        if v < 0.3:
            return None
        # landmarks are in 0-1 normalized image coords
        # bbox is in pixel coords of the original image
        # crop is the bbox region scaled to (w, h)
        orig_x = x_norm  # we need to know original image size
        orig_y = y_norm
        # Map to crop coordinates
        crop_x = int((orig_x * (bx2 + bx2 - bx) - bx) * w / max(1, bx2 - bx))
        crop_y = int((orig_y * (by2 + by2 - by) - by) * h / max(1, by2 - by))
        return (crop_x, crop_y)

    # Draw connections
    for i, j in SKELETON_PAIRS:
        p1 = lm_to_crop(i)
        p2 = lm_to_crop(j)
        if p1 and p2:
            cv2.line(img, p1, p2, SKELETON_COLOR, 2, cv2.LINE_AA)

    # Draw joints
    for idx in range(min(len(landmarks), 33)):
        pt = lm_to_crop(idx)
        if pt:
            cv2.circle(img, pt, 3, JOINT_DOT_COLOR, -1, cv2.LINE_AA)


def remove_racket_from_crop(crop, racket_bbox, bx, by, bx2, by2, panel_w, panel_h, crop_w, crop_h):
    """Inpaint the racket region from a cropped frame panel.
    racket_bbox: [x1, y1, x2, y2] in original video pixel coords."""
    if racket_bbox is None:
        return crop

    rx1, ry1, rx2, ry2 = racket_bbox
    # Transform racket bbox from original video coords to crop coords then to panel coords
    px1 = int((rx1 - bx) / max(1, crop_w) * panel_w)
    py1 = int((ry1 - by) / max(1, crop_h) * panel_h)
    px2 = int((rx2 - bx) / max(1, crop_w) * panel_w)
    py2 = int((ry2 - by) / max(1, crop_h) * panel_h)

    # Expand bbox slightly for better inpainting
    margin = max(3, int(0.1 * max(px2 - px1, py2 - py1)))
    px1 = max(0, px1 - margin)
    py1 = max(0, py1 - margin)
    px2 = min(panel_w, px2 + margin)
    py2 = min(panel_h, py2 + margin)

    if px2 <= px1 or py2 <= py1:
        return crop
    if px1 >= panel_w or py1 >= panel_h or px2 <= 0 or py2 <= 0:
        return crop

    mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    mask[py1:py2, px1:px2] = 255
    return cv2.inpaint(crop, mask, 5, cv2.INPAINT_TELEA)


def generate_composite(video_path, det, poses, shot_idx, draw_skel=True,
                       racket_data=None):
    """Generate a filmstrip composite for one shot.

    racket_data: optional racket detection dict with 'frames' list.
    Returns (composite_image, shot_info_dict) or (None, None).
    """
    detections = det.get("detections", [])
    if shot_idx >= len(detections):
        return None, None

    d = detections[shot_idx]
    raw_contact = int(d.get("frame", 0))
    shot_type = d.get("shot_type", "unknown")
    fps = det.get("fps", 60.0)

    pose_frames = poses.get("frames", [])

    # Refine contact frame to peak wrist speed (better proxy for actual contact)
    contact_frame = raw_contact
    search_range = int(0.3 * fps)  # search ±0.3s around detected frame
    best_speed = 0
    for fi in range(max(1, raw_contact - search_range), min(len(pose_frames) - 1, raw_contact + search_range)):
        lms_prev = get_landmarks(pose_frames, fi - 1)
        lms_curr = get_landmarks(pose_frames, fi)
        if lms_prev and lms_curr and len(lms_prev) > 16 and len(lms_curr) > 16:
            wx0, wy0, wv0 = lms_prev[16]
            wx1, wy1, wv1 = lms_curr[16]
            if wv0 > 0.3 and wv1 > 0.3:
                speed = ((wx1 - wx0)**2 + (wy1 - wy0)**2) ** 0.5
                if speed > best_speed:
                    best_speed = speed
                    contact_frame = fi

    # Compute frame indices to sample
    before_frames = int(BEFORE_SEC * fps)
    after_frames = int(AFTER_SEC * fps)
    total_window = before_frames + after_frames
    step = max(1, total_window // (NUM_FRAMES - 1))

    frame_indices = []
    for i in range(NUM_FRAMES):
        fi = contact_frame - before_frames + i * step
        frame_indices.append(max(0, fi))
    # Ensure contact frame is included
    contact_panel_idx = None
    closest = min(range(len(frame_indices)), key=lambda i: abs(frame_indices[i] - contact_frame))
    frame_indices[closest] = contact_frame
    contact_panel_idx = closest

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None
    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Crop: bbox of ALL landmarks at the CONTACT frame, padded to show
    # full body + racket. Same crop for all 7 panels.
    # Fall back to union of all frames if contact frame has poor pose.
    contact_xs, contact_ys = [], []
    all_xs, all_ys = [], []

    for fi in frame_indices:
        lms = get_landmarks(pose_frames, fi)
        if not lms:
            continue
        for x, y, v in lms:
            if v > 0.3:
                px, py = x * img_w, y * img_h
                all_xs.append(px)
                all_ys.append(py)
                if fi == contact_frame:
                    contact_xs.append(px)
                    contact_ys.append(py)

    if len(all_xs) < 30:
        cap.release()
        return None, None

    # Union of all frames — guarantees full body visible in every panel
    all_xs.sort()
    all_ys.sort()
    x_min, x_max = min(all_xs), max(all_xs)
    y_min, y_max = min(all_ys), max(all_ys)
    player_w = x_max - x_min
    player_h = y_max - y_min

    pad = 0.25
    crop_w = int(player_w + 2 * pad * player_w)
    crop_h = int(player_h + 2 * pad * player_h)

    # Enforce 3:4 aspect
    ratio = 3.0 / 4.0
    if crop_w / max(1, crop_h) > ratio:
        crop_h = int(crop_w / ratio)
    else:
        crop_w = int(crop_h * ratio)

    crop_w = max(crop_w, 200)
    crop_h = max(crop_h, 200)

    # Center on TORSO (shoulders+hips), not overall landmark midpoint.
    # This keeps the person centered, not the racket.
    torso_xs, torso_ys = [], []
    for fi in frame_indices:
        lms = get_landmarks(pose_frames, fi)
        if not lms or len(lms) < 25:
            continue
        for idx in [11, 12, 23, 24]:  # shoulders + hips
            x, y, v = lms[idx]
            if v > 0.3:
                torso_xs.append(x * img_w)
                torso_ys.append(y * img_h)

    if torso_xs:
        cx = (min(torso_xs) + max(torso_xs)) / 2
        cy = (min(torso_ys) + max(torso_ys)) / 2
    else:
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2

    bx = int(cx - crop_w / 2)
    by = int(cy - crop_h / 2)
    bx = max(0, min(bx, img_w - crop_w))
    by = max(0, min(by, img_h - crop_h))
    bx2 = min(bx + crop_w, img_w)
    by2 = min(by + crop_h, img_h)
    crop_w = bx2 - bx
    crop_h = by2 - by

    if crop_w < 10 or crop_h < 10:
        cap.release()
        return None, None

    # Panel dimensions
    panel_h = PANEL_HEIGHT
    panel_w = int(panel_h * crop_w / max(1, crop_h))

    panels = []
    wrist_trail = []

    for panel_idx, fi in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            panels.append(np.zeros((panel_h, panel_w, 3), dtype=np.uint8))
            continue

        crop = frame[by:by2, bx:bx2].copy()
        if crop.size == 0:
            panels.append(np.zeros((panel_h, panel_w, 3), dtype=np.uint8))
            continue

        # Resize to panel size
        crop = cv2.resize(crop, (panel_w, panel_h), interpolation=cv2.INTER_LANCZOS4)

        # Racket removal via inpainting
        if racket_data:
            racket_frames = {f["frame"]: f for f in racket_data.get("frames", [])
                             if f.get("detected")}
            rf = racket_frames.get(fi)
            if rf:
                crop = remove_racket_from_crop(
                    crop, rf["bbox"], bx, by, bx2, by2, panel_w, panel_h, crop_w, crop_h
                )

        # Track wrist position for trail
        lms = get_landmarks(pose_frames, fi)
        if lms and len(lms) > 16:
            wx, wy, wv = lms[16]  # right wrist
            if wv > 0.3:
                trail_x = int((wx * img_w - bx) / max(1, crop_w) * panel_w)
                trail_y = int((wy * img_h - by) / max(1, crop_h) * panel_h)
                wrist_trail.append((panel_idx, trail_x, trail_y))

        # Draw skeleton
        if draw_skel and lms:
            for i, j in SKELETON_PAIRS:
                if i < len(lms) and j < len(lms):
                    x1, y1, v1 = lms[i]
                    x2, y2, v2 = lms[j]
                    if v1 > 0.3 and v2 > 0.3:
                        p1 = (int((x1 * img_w - bx) / max(1, crop_w) * panel_w),
                              int((y1 * img_h - by) / max(1, crop_h) * panel_h))
                        p2 = (int((x2 * img_w - bx) / max(1, crop_w) * panel_w),
                              int((y2 * img_h - by) / max(1, crop_h) * panel_h))
                        cv2.line(crop, p1, p2, SKELETON_COLOR, 2, cv2.LINE_AA)

        # Orange border on contact frame
        if panel_idx == contact_panel_idx:
            cv2.rectangle(crop, (0, 0), (panel_w - 1, panel_h - 1),
                          CONTACT_FRAME_BORDER, 3)

        panels.append(crop)

    cap.release()

    if not panels:
        return None, None

    # Draw wrist trail across panels
    if len(wrist_trail) >= 2:
        for k in range(1, len(wrist_trail)):
            pi1, x1, y1 = wrist_trail[k - 1]
            pi2, x2, y2 = wrist_trail[k]
            # Offset x by panel position
            x1_abs = x1 + pi1 * panel_w
            x2_abs = x2 + pi2 * panel_w
            # We'll draw the trail on the final composite (below)

    # Compose side-by-side with 2px gap
    gap = 2
    total_w = len(panels) * panel_w + (len(panels) - 1) * gap
    composite = np.zeros((panel_h, total_w, 3), dtype=np.uint8)
    for i, panel in enumerate(panels):
        x_off = i * (panel_w + gap)
        composite[:, x_off:x_off + panel_w] = panel

    # Draw wrist trail on composite (skeleton version only)
    if draw_skel and len(wrist_trail) >= 2:
        for k in range(1, len(wrist_trail)):
            pi1, x1, y1 = wrist_trail[k - 1]
            pi2, x2, y2 = wrist_trail[k]
            pt1 = (x1 + pi1 * (panel_w + gap), y1)
            pt2 = (x2 + pi2 * (panel_w + gap), y2)
            cv2.line(composite, pt1, pt2, WRIST_TRAIL_COLOR, 2, cv2.LINE_AA)
        for pi, x, y in wrist_trail:
            pt = (x + pi * (panel_w + gap), y)
            cv2.circle(composite, pt, 4, WRIST_TRAIL_COLOR, -1, cv2.LINE_AA)

    # No baked-in label bar — the gallery HTML shows labels separately

    info = {
        "shot_idx": shot_idx,
        "shot_type": shot_type,
        "contact_frame": contact_frame,
        "timestamp": contact_frame / fps,
        "num_panels": len(panels),
    }
    return composite, info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Video ID (e.g. IMG_1141)")
    parser.add_argument("--shot", type=int, help="Specific shot index (default: all)")
    parser.add_argument("--no-skeleton", action="store_true")
    parser.add_argument("--racket-removal", action="store_true",
                        help="Remove racket from composites via inpainting")
    parser.add_argument("--upload", action="store_true", help="Upload to R2")
    parser.add_argument("--max-shots", type=int, default=0, help="Limit number of shots")
    args = parser.parse_args()

    det, poses, video_path = load_data(args.video)
    if not det:
        return

    # Load racket detections if available
    racket_data = None
    if args.racket_removal:
        racket_path = PROJECT_ROOT / "racket_detections" / f"{args.video}.json"
        if racket_path.exists():
            with open(racket_path) as f:
                racket_data = json.load(f)
            print(f"Loaded racket detections: {racket_data.get('frames_with_racket', 0)} frames")
        else:
            print(f"[WARN] No racket detections at {racket_path}, generating without removal")

    detections = det.get("detections", [])
    out_dir = PROJECT_ROOT / "exports" / args.video / "sequences"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.shot is not None:
        shots = [args.shot]
    else:
        shots = list(range(len(detections)))
    if args.max_shots > 0:
        shots = shots[:args.max_shots]

    print(f"Generating swing composites for {args.video}: {len(shots)} shots")
    generated = []

    for idx in shots:
        d = detections[idx]
        st = d.get("shot_type", "unknown")
        if st in ("practice", "offscreen", "not_shot", "unknown_shot"):
            continue

        # Generate clean version (no skeleton) — always
        comp_clean, info = generate_composite(
            video_path, det, poses, idx, draw_skel=False,
            racket_data=racket_data
        )
        if comp_clean is None:
            continue

        suffix = "_noracket" if racket_data else ""
        fname_clean = f"shot_{idx:03d}_{st}{suffix}.jpg"
        out_clean = out_dir / fname_clean
        cv2.imwrite(str(out_clean), comp_clean, [cv2.IMWRITE_JPEG_QUALITY, 92])
        generated.append((str(out_clean), info))

        # Generate skeleton version too (unless --no-skeleton)
        if not args.no_skeleton:
            comp_skel, _ = generate_composite(
                video_path, det, poses, idx, draw_skel=True,
                racket_data=racket_data
            )
            if comp_skel is not None:
                fname_skel = f"shot_{idx:03d}_{st}{suffix}_skel.jpg"
                out_skel = out_dir / fname_skel
                cv2.imwrite(str(out_skel), comp_skel, [cv2.IMWRITE_JPEG_QUALITY, 92])
                generated.append((str(out_skel), info))

        print(f"  [{idx + 1}/{len(detections)}] {st}: {fname_clean}", flush=True)

    print(f"\nGenerated {len(generated)} composites in {out_dir}")

    if args.upload and generated:
        from dotenv import load_dotenv
        load_dotenv()
        from storage.r2_client import R2Client
        r2 = R2Client()
        for path, info in generated:
            key = f"highlights/{args.video}/sequences/{os.path.basename(path)}"
            r2.upload(path, key, content_type="image/jpeg")
        print(f"Uploaded {len(generated)} composites to R2")


if __name__ == "__main__":
    main()
