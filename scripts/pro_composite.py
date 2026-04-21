#!/usr/bin/env python3
"""Generate side-by-side pro comparison composites.

Takes a user's shot filmstrip and places it next to a matching pro
shot (same type: forehand, backhand, serve). User shot on top,
pro shot on bottom.

Pro clips live in pros/{player}/ with metadata in pros/index.json.

Usage:
    python scripts/pro_composite.py --video IMG_1141 --shot 5 --pro djokovic
    python scripts/pro_composite.py --video IMG_1141 --pro alcaraz --upload
    python scripts/pro_composite.py --video IMG_1141  # auto-picks best pro match
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
PROS_DIR = PROJECT_ROOT / "pros"
DETECTIONS_DIR = PROJECT_ROOT / "detections"
POSES_DIR = PROJECT_ROOT / "poses_full_videos"
PREPROCESSED_DIR = PROJECT_ROOT / "preprocessed"

from scripts.swing_composite import generate_composite, load_data, NUM_FRAMES, PANEL_HEIGHT


def load_pro_index():
    idx_path = PROS_DIR / "index.json"
    if not idx_path.exists():
        return None
    with open(idx_path) as f:
        return json.load(f)


def get_pro_clip(player, shot_type, pro_index):
    """Find a matching pro clip for the given shot type."""
    players = pro_index.get("players", {})
    if player not in players:
        return None
    for clip in players[player].get("clips", []):
        if clip.get("type") == shot_type:
            clip_path = PROS_DIR / player / clip["file"]
            if clip_path.exists():
                return clip, str(clip_path)
    return None


def generate_pro_frames(clip_path, clip_meta, num_frames=NUM_FRAMES):
    """Extract key frames from a pro clip around the contact frame."""
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    contact = clip_meta.get("contact_frame", total // 2)

    # Sample frames centered on contact
    before = int(total * 0.3)
    after = total - before
    step = max(1, (before + after) // (num_frames - 1))

    frames = []
    for i in range(num_frames):
        fi = max(0, min(total - 1, contact - before + i * step))
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if ret:
            # Resize to consistent panel height
            h, w = frame.shape[:2]
            new_w = int(PANEL_HEIGHT * w / h)
            frame = cv2.resize(frame, (new_w, PANEL_HEIGHT), interpolation=cv2.INTER_LANCZOS4)
            frames.append(frame)
    cap.release()
    return frames


def create_comparison(user_composite, pro_frames, user_info, pro_name, pro_clip):
    """Stack user composite on top, pro filmstrip on bottom."""
    if not pro_frames:
        return None

    # Make pro filmstrip same width as user composite
    user_h, user_w = user_composite.shape[:2]
    gap = 2
    num_panels = len(pro_frames)
    panel_w = (user_w - (num_panels - 1) * gap) // num_panels

    pro_strip = np.zeros((PANEL_HEIGHT, user_w, 3), dtype=np.uint8)
    for i, frame in enumerate(pro_frames):
        # Resize to panel width maintaining aspect
        fh, fw = frame.shape[:2]
        new_h = PANEL_HEIGHT
        new_w = int(new_h * fw / fh)
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        # Center-crop to panel_w
        if new_w > panel_w:
            offset = (new_w - panel_w) // 2
            frame = frame[:, offset:offset + panel_w]
        x_off = i * (panel_w + gap)
        w = min(panel_w, frame.shape[1])
        pro_strip[:, x_off:x_off + w] = frame[:, :w]

    # Label bars
    user_label = np.zeros((28, user_w, 3), dtype=np.uint8)
    cv2.putText(user_label, f"YOU — {user_info.get('shot_type', '').upper()} #{user_info.get('shot_idx', 0) + 1}",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

    pro_label = np.zeros((28, user_w, 3), dtype=np.uint8)
    pro_display = pro_name.upper()
    clip_type = pro_clip.get("type", "").upper()
    cv2.putText(pro_label, f"{pro_display} — {clip_type}",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 200, 255), 1, cv2.LINE_AA)

    # Separator line
    sep = np.zeros((4, user_w, 3), dtype=np.uint8)
    sep[:, :] = (0, 140, 255)  # orange

    # Stack: user_label + user_composite + sep + pro_label + pro_strip
    comparison = np.vstack([user_label, user_composite, sep, pro_label, pro_strip])
    return comparison


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--shot", type=int, help="Specific shot index")
    parser.add_argument("--pro", default="djokovic", help="Pro player name")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--max-shots", type=int, default=5)
    args = parser.parse_args()

    pro_index = load_pro_index()
    if not pro_index:
        print("[ERROR] No pros/index.json found"); return

    det, poses, video_path = load_data(args.video)
    if not det:
        return

    detections = det.get("detections", [])
    out_dir = PROJECT_ROOT / "exports" / args.video / "comparisons"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.shot is not None:
        shots = [args.shot]
    else:
        shots = list(range(min(args.max_shots, len(detections))))

    generated = []
    for idx in shots:
        d = detections[idx]
        st = d.get("shot_type", "unknown")
        if st in ("practice", "offscreen", "not_shot", "unknown_shot"):
            continue

        # Generate user composite
        user_comp, info = generate_composite(video_path, det, poses, idx, draw_skel=False)
        if user_comp is None:
            continue

        # Find matching pro clip
        result = get_pro_clip(args.pro, st, pro_index)
        if not result:
            # Try any pro that has this shot type
            for player in pro_index.get("players", {}):
                result = get_pro_clip(player, st, pro_index)
                if result:
                    args.pro = player
                    break
        if not result:
            print(f"  No pro clip for {st}, skipping")
            continue

        clip_meta, clip_path = result
        pro_frames = generate_pro_frames(clip_path, clip_meta)

        comparison = create_comparison(user_comp, pro_frames, info, args.pro, clip_meta)
        if comparison is None:
            continue

        fname = f"compare_{idx:03d}_{st}_vs_{args.pro}.jpg"
        out_path = out_dir / fname
        cv2.imwrite(str(out_path), comparison, [cv2.IMWRITE_JPEG_QUALITY, 92])
        size_kb = os.path.getsize(out_path) / 1024
        print(f"  Shot #{idx + 1} {st} vs {args.pro}: {fname} ({size_kb:.0f} KB)")
        generated.append((str(out_path), info))

    print(f"\nGenerated {len(generated)} comparisons in {out_dir}")

    if args.upload and generated:
        from dotenv import load_dotenv; load_dotenv()
        from storage.r2_client import R2Client
        r2 = R2Client()
        for path, info in generated:
            key = f"highlights/{args.video}/comparisons/{os.path.basename(path)}"
            r2.upload(path, key, content_type="image/jpeg")
        print(f"Uploaded {len(generated)} comparisons to R2")


if __name__ == "__main__":
    main()
