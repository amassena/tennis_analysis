#!/usr/bin/env python3
"""Side-by-side filmstrip comparison: user shot on top, matched pro on bottom.

Reuses scripts/swing_composite.generate_composite() so both filmstrips
share the same look — skeleton overlay, auto-zoom to player bbox, wrist
trail, contact-refined frame selection.

Pro clips are 240 frames at 60 fps with contact at frame 120 by
convention (see scripts/curate_pro_clips.CONTACT_FRAME).

Usage:
    .venv/bin/python scripts/compare_filmstrip.py --user IMG_0999 --shot 2 --pro sinner
    .venv/bin/python scripts/compare_filmstrip.py --user IMG_0999 --shot 2  # auto-match pro
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Reuse the swing_composite renderer (skeleton + auto-zoom + wrist trail)
from scripts.swing_composite import generate_composite  # noqa: E402

# User data lives in the main tennis_analysis tree, not this worktree
USER_REPO = Path.home() / "tennis_analysis"
USER_PREPROCESSED = USER_REPO / "preprocessed"
USER_POSES = USER_REPO / "poses_full_videos"
USER_DETECTIONS = USER_REPO / "detections"

PROS_DIR = REPO_ROOT / "pros"
INDEX_PATH = PROS_DIR / "index.json"

PRO_CONTACT_FRAME = 120
PRO_FPS = 60


def load_user_data(vid: str):
    """Return (video_path, det, poses) for a user video."""
    video_path = USER_PREPROCESSED / f"{vid}.mp4"
    if not video_path.exists():
        raise FileNotFoundError(f"User video not found: {video_path}")
    for name in (f"{vid}_fused.json", f"{vid}_fused_detections.json"):
        det_path = USER_DETECTIONS / name
        if det_path.exists():
            with det_path.open() as f:
                det = json.load(f)
            break
    else:
        raise FileNotFoundError(f"User det not found for {vid}")
    pose_path = USER_POSES / f"{vid}.json"
    if not pose_path.exists():
        raise FileNotFoundError(f"User pose not found: {pose_path}")
    with pose_path.open() as f:
        poses = json.load(f)
    return video_path, det, poses


def find_user_shot_idx(det: dict, shot_n: int, shot_type: str | None = None) -> int:
    """Find the shot_n'th shot (optionally filtered by type)."""
    detections = det.get("detections", [])
    matching = [i for i, d in enumerate(detections)
                if shot_type is None or d.get("shot_type") == shot_type]
    if not matching:
        raise ValueError(f"No shots of type {shot_type!r} in user detections")
    if shot_n >= len(matching):
        raise ValueError(f"--shot {shot_n} out of range ({len(matching)} matching)")
    return matching[shot_n]


def match_pro_clip(shot_type: str, preferred_slug: str | None = None,
                   user_backhand_style: str = "two-handed") -> tuple[str, str]:
    """Return (slug, filename) of a pro clip of the right type.

    Picks first matching from preferred_slug if specified, else first
    on-disk clip from any pro. For shot_type == 'backhand', hard-filters
    on backhand_style match (1HBH vs 2HBH is meaningless to compare).
    """
    with INDEX_PATH.open() as f:
        index = json.load(f)
    slugs = [preferred_slug] if preferred_slug else list(index["players"].keys())
    user_bh_style = (user_backhand_style or "").lower()
    for slug in slugs:
        player = index["players"].get(slug, {})
        # Hard filter on backhand_style for backhand comparisons
        if shot_type == "backhand" and user_bh_style:
            pro_bh_style = (player.get("backhand_style") or "").lower()
            if pro_bh_style and pro_bh_style != user_bh_style:
                continue
        for clip in player.get("clips", []):
            if clip.get("type") != shot_type:
                continue
            local = PROS_DIR / slug / clip["file"]
            if local.exists():
                return slug, clip["file"]
    raise FileNotFoundError(
        f"No on-disk pro clip of type {shot_type!r} "
        f"(backhand_style filter: {user_bh_style!r})"
    )


def load_pro_data(slug: str, filename: str):
    """Return (video_path, fake_det, poses) for a pro clip.

    The fake det has a single detection at frame PRO_CONTACT_FRAME so
    generate_composite() treats it as one shot.
    """
    video_path = PROS_DIR / slug / filename
    pose_path = video_path.with_suffix(".pose.json")
    if not pose_path.exists():
        raise FileNotFoundError(f"Pro pose not found: {pose_path}. "
                                f"Run extract_pro_clip_poses.py on the GPU machine first.")
    with pose_path.open() as f:
        poses = json.load(f)
    # Look up type from index.json so the synth det matches
    with INDEX_PATH.open() as f:
        index = json.load(f)
    shot_type = "forehand"
    for clip in index["players"][slug].get("clips", []):
        if clip["file"] == filename:
            shot_type = clip.get("type", "forehand")
            break
    fake_det = {
        "fps": PRO_FPS,
        "detections": [{
            "frame": PRO_CONTACT_FRAME,
            "timestamp": PRO_CONTACT_FRAME / PRO_FPS,
            "shot_type": shot_type,
            "confidence": 1.0,
        }],
    }
    return video_path, fake_det, poses


def add_label_band(img: np.ndarray, label: str, height: int = 40) -> np.ndarray:
    """Add a black band above the image with the label text."""
    band = np.zeros((height, img.shape[1], 3), dtype=np.uint8)
    cv2.putText(band, label, (12, height - 12), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return np.vstack([band, img])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--user", required=True, help="User video ID (e.g. IMG_0999)")
    ap.add_argument("--shot", type=int, default=0, help="Nth shot of --shot-type (0-indexed)")
    ap.add_argument("--shot-type", choices=("forehand", "backhand", "serve"),
                    help="Shot type (default: pick any shot at index --shot)")
    ap.add_argument("--pro", help="Preferred pro slug (default: auto)")
    ap.add_argument("--user-backhand-style", default="two-handed",
                    choices=("one-handed", "two-handed"),
                    help="User's backhand style (default two-handed). Hard-filter for backhand matches.")
    ap.add_argument("--output", help="Output PNG path (default: /tmp/<user>_<shot>_vs_<pro>.png)")
    ap.add_argument("--no-skeleton", action="store_true")
    args = ap.parse_args()

    print(f"Loading user data for {args.user}…")
    user_video, user_det, user_poses = load_user_data(args.user)
    user_shot_idx = find_user_shot_idx(user_det, args.shot, args.shot_type)
    user_shot = user_det["detections"][user_shot_idx]
    shot_type = user_shot["shot_type"]
    print(f"  user shot #{user_shot_idx}: {shot_type} @ frame {user_shot.get('frame')}")

    print(f"Generating user filmstrip…")
    user_strip, user_info = generate_composite(
        user_video, user_det, user_poses, user_shot_idx,
        draw_skel=not args.no_skeleton,
    )
    if user_strip is None:
        print("[ERROR] user filmstrip generation failed")
        return 1

    print(f"Matching pro clip of type {shot_type!r}…")
    slug, filename = match_pro_clip(shot_type, args.pro, args.user_backhand_style)
    print(f"  -> {slug}/{filename}")
    pro_video, pro_det, pro_poses = load_pro_data(slug, filename)

    print(f"Generating pro filmstrip…")
    pro_strip, pro_info = generate_composite(
        pro_video, pro_det, pro_poses, 0,
        draw_skel=not args.no_skeleton,
    )
    if pro_strip is None:
        print("[ERROR] pro filmstrip generation failed")
        return 1

    # Resize to same width (the panel heights are the same; widths can differ
    # if aspect ratios differ).
    target_w = max(user_strip.shape[1], pro_strip.shape[1])
    def pad_to_width(img, w):
        if img.shape[1] == w:
            return img
        pad = np.zeros((img.shape[0], w - img.shape[1], 3), dtype=np.uint8)
        return np.hstack([img, pad])
    user_strip = pad_to_width(user_strip, target_w)
    pro_strip = pad_to_width(pro_strip, target_w)

    # Labels above each row
    user_strip = add_label_band(user_strip, f"YOU - {args.user} shot {args.shot} ({shot_type})")
    pro_strip = add_label_band(pro_strip, f"PRO - {slug} ({filename})")

    # Stack
    stacked = np.vstack([user_strip, pro_strip])

    output = args.output or f"/tmp/{args.user}_{args.shot}_{shot_type}_vs_{slug}.png"
    cv2.imwrite(output, stacked)
    print(f"Saved: {output} ({stacked.shape[1]}x{stacked.shape[0]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
