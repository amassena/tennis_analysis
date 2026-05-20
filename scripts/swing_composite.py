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
AUDIO_HITS_DIR = PROJECT_ROOT / "audio_hits"

# Time offsets (seconds) of each panel relative to raw_contact.
# Non-uniform on purpose: dense sampling around the contact moment so the
# strike is always captured in at least one panel even when the detector's
# raw_contact is off by ±100ms; wide outer panels keep full swing context.
# Center panel (index 5) is raw_contact and gets the orange border.
PANEL_OFFSETS_SEC = [
    -0.700, -0.350, -0.150,           # backswing (wide)
    -0.075, -0.037,  0.000,  0.037, 0.075,   # contact zone (dense, ~37ms step)
     0.150,  0.350,  0.700,           # follow-through (wide)
]
NUM_FRAMES = len(PANEL_OFFSETS_SEC)
CONTACT_PANEL_IDX = PANEL_OFFSETS_SEC.index(0.0)
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


def extract_audio_peaks(video_path, sample_rate=16000, window_ms=10,
                        threshold_pct=95, min_gap_ms=200):
    """Find strike-like audio peaks in a video. Returns dict with peaks list.

    Each peak is {"frame": int, "amp": float, "rel_amp": float} where rel_amp
    is amplitude relative to the percentile threshold (>= 1.0 by definition,
    >> 1.0 means clearly louder than background).
    """
    import os
    import subprocess
    import tempfile
    import wave
    import numpy as np

    # Extract mono PCM via ffmpeg
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-vn", "-ac", "1",
             "-ar", str(sample_rate), "-f", "wav", tmp.name],
            capture_output=True, check=True,
        )
        with wave.open(tmp.name, "rb") as w:
            audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
            audio = audio.astype(np.float32) / 32768.0
    finally:
        os.unlink(tmp.name)

    # Source video fps (audio peaks are emitted in source-frame coords)
    out = subprocess.run(
        ["ffprobe", "-v", "0", "-of", "csv=p=0", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate", video_path],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    num, den = out.split("/")
    fps = float(num) / float(den)

    # RMS envelope
    ws = int(sample_rate * window_ms / 1000)
    hop = ws // 2
    n_win = (len(audio) - ws) // hop
    envelope = np.empty(n_win, dtype=np.float32)
    for i in range(n_win):
        chunk = audio[i * hop:i * hop + ws]
        envelope[i] = float(np.sqrt(np.mean(chunk * chunk)))

    threshold = float(np.percentile(envelope, threshold_pct))
    if threshold <= 0:
        return {"fps": fps, "threshold": 0.0, "peaks": []}

    min_gap_windows = int(min_gap_ms / 1000 / (hop / sample_rate))
    peaks = []
    last_peak_i = -10**9
    for i in range(1, len(envelope) - 1):
        v = float(envelope[i])
        if (v > threshold and v > float(envelope[i - 1]) and v >= float(envelope[i + 1])
                and i - last_peak_i >= min_gap_windows):
            t_sec = (i * hop + ws / 2) / sample_rate
            frame = int(round(t_sec * fps))
            peaks.append({"frame": frame, "amp": v, "rel_amp": v / threshold})
            last_peak_i = i

    return {"fps": float(fps), "threshold": float(threshold), "peaks": peaks}


def load_or_compute_audio_peaks(vid, video_path):
    """Return cached audio peaks for video, computing + caching if missing."""
    cache = AUDIO_HITS_DIR / f"{vid}.json"
    if cache.exists():
        with open(cache) as f:
            return json.load(f)
    print(f"Extracting audio peaks for {vid} (one-time, ~5-10s)...", flush=True)
    data = extract_audio_peaks(video_path)
    AUDIO_HITS_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache, "w") as f:
        json.dump(data, f)
    print(f"  -> {len(data['peaks'])} peaks (threshold={data['threshold']:.4f})", flush=True)
    return data


def snap_to_audio_peak(raw_contact, fps, peaks_data, search_sec=0.3,
                       min_rel_amp=1.5):
    """Snap raw_contact to the loudest audio peak within ±search_sec.

    min_rel_amp gates against ambient noise: a peak must be at least this
    multiple of the per-video amplitude threshold to count as a strike-like
    transient. Returns (frame, snapped: bool).
    """
    if not peaks_data or not peaks_data.get("peaks"):
        return raw_contact, False
    search = int(search_sec * fps)
    lo, hi = raw_contact - search, raw_contact + search
    in_window = [p for p in peaks_data["peaks"]
                 if lo <= p["frame"] <= hi and p["rel_amp"] >= min_rel_amp]
    if not in_window:
        return raw_contact, False
    best = max(in_window, key=lambda p: p["amp"])
    return int(best["frame"]), True


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
                       racket_data=None, audio_peaks=None):
    """Generate a filmstrip composite for one shot.

    Sampling is non-uniform: dense ~37ms-step panels around contact_frame
    plus wide outer panels for swing context. When audio_peaks is provided,
    the detector's raw_contact gets snapped to the nearest loud audio peak
    within ±300ms — fixes shots where the detector is off by 200-500ms.

    racket_data: optional racket detection dict with 'frames' list.
    audio_peaks: optional dict from load_or_compute_audio_peaks().
    Returns (composite_image, shot_info_dict) or (None, None).
    """
    detections = det.get("detections", [])
    if shot_idx >= len(detections):
        return None, None

    d = detections[shot_idx]
    raw_contact = int(d.get("frame", 0))
    shot_type = d.get("shot_type", "unknown")
    fps = det.get("fps", 60.0)

    contact_frame, snapped = snap_to_audio_peak(raw_contact, fps, audio_peaks)
    contact_method = "audio" if snapped else "raw"

    pose_frames = poses.get("frames", [])

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None
    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_indices = [
        max(0, contact_frame + int(round(off * fps)))
        for off in PANEL_OFFSETS_SEC
    ]
    contact_panel_idx = CONTACT_PANEL_IDX

    # PER-FRAME centering: each panel centered on that frame's torso.
    # Crop SIZE is uniform across panels so the framing looks consistent.
    # Crop POSITION moves with the player.
    per_frame_torso = {}  # fi -> (torso_cx, torso_cy)
    # Size_xs/size_ys: landmarks ONLY from panels near contact, so the crop is
    # sized to fit the SWING (arm fully extended, racket up) — NOT the run-up,
    # which can be wide and would zoom the panel out unnecessarily.
    size_xs, size_ys = [], []
    size_panel_lo = max(0, contact_panel_idx - 2)
    size_panel_hi = min(NUM_FRAMES, contact_panel_idx + 3)
    size_frames = set(frame_indices[size_panel_lo:size_panel_hi])

    for fi in frame_indices:
        lms = get_landmarks(pose_frames, fi)
        if not lms:
            continue
        if fi in size_frames:
            for x, y, v in lms:
                if v > 0.3:
                    size_xs.append(x * img_w)
                    size_ys.append(y * img_h)
        if len(lms) >= 25:
            tx, ty = [], []
            for idx in [11, 12, 23, 24]:
                x, y, v = lms[idx]
                if v > 0.3:
                    tx.append(x * img_w)
                    ty.append(y * img_h)
            if len(tx) >= 2:
                per_frame_torso[fi] = (sum(tx) / len(tx), sum(ty) / len(ty))

    if len(size_xs) < 30:
        cap.release()
        return None, None

    # Crop SIZE from landmarks at/near contact only.
    player_w = max(size_xs) - min(size_xs)
    player_h = max(size_ys) - min(size_ys)

    pad = 0.3
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
    crop_w = min(crop_w, img_w)
    crop_h = min(crop_h, img_h)

    # Fallback torso position (mean across frames that have one)
    if per_frame_torso:
        avg_cx = sum(p[0] for p in per_frame_torso.values()) / len(per_frame_torso)
        avg_cy = sum(p[1] for p in per_frame_torso.values()) / len(per_frame_torso)
    else:
        avg_cx = (min(size_xs) + max(size_xs)) / 2
        avg_cy = (min(size_ys) + max(size_ys)) / 2

    # Compute per-frame crop positions
    per_frame_crop = {}
    for fi in frame_indices:
        cx, cy = per_frame_torso.get(fi, (avg_cx, avg_cy))
        bx = int(cx - crop_w / 2)
        by = int(cy - crop_h / 2)
        bx = max(0, min(bx, img_w - crop_w))
        by = max(0, min(by, img_h - crop_h))
        per_frame_crop[fi] = (bx, by, bx + crop_w, by + crop_h)

    # Initial bx/by for the contact frame (used by racket removal coords)
    bx, by, bx2, by2 = per_frame_crop.get(contact_frame, (0, 0, crop_w, crop_h))

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

        # Per-frame crop position (player centered in each panel)
        bx, by, bx2, by2 = per_frame_crop.get(fi, (bx, by, bx2, by2))
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
        "raw_contact": raw_contact,
        "contact_method": contact_method,
        "snap_shift_ms": round((contact_frame - raw_contact) * 1000 / fps, 0),
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
    parser.add_argument("--no-audio-snap", action="store_true",
                        help="Skip audio-peak contact-frame snap (use raw detector frame)")
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

    # Audio peaks for contact-frame snap (one-time per video, cached)
    audio_peaks = None
    if not args.no_audio_snap:
        try:
            audio_peaks = load_or_compute_audio_peaks(args.video, video_path)
        except Exception as e:
            print(f"[WARN] Audio-peak extraction failed: {e}. Using raw detector frame.")

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
            racket_data=racket_data, audio_peaks=audio_peaks
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
                racket_data=racket_data, audio_peaks=audio_peaks
            )
            if comp_skel is not None:
                fname_skel = f"shot_{idx:03d}_{st}{suffix}_skel.jpg"
                out_skel = out_dir / fname_skel
                cv2.imwrite(str(out_skel), comp_skel, [cv2.IMWRITE_JPEG_QUALITY, 92])
                generated.append((str(out_skel), info))

        tag = f"[{info['contact_method']:>5}]"
        shift = f"snap {info['snap_shift_ms']:+.0f}ms" if info['contact_method'] == 'audio' else ""
        print(f"  [{idx + 1}/{len(detections)}] {st}: {fname_clean}  {tag} {shift}", flush=True)

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
