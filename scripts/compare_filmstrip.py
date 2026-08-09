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


DEFAULT_PREFERRED_PROS = ("murray",)  # see feedback_preferred_comparison_pros.md


def _coarse_angle(a: str | None) -> str:
    """Collapse the angle taxonomy to a family so the user's coarse tag
    ('behind') matches the clips' fine tags ('behind-player'). front-broadcast
    ->front, side-ad/side-deuce->side, behind-player->behind."""
    return (a or "").lower().split("-")[0]


def match_pro_clip(shot_type: str, preferred_slug: str | None = None,
                   user_backhand_style: str = "two-handed",
                   target_angle: str | None = None,
                   rotate: int = 0) -> tuple[str, str]:
    """Return (slug, filename) of a pro clip of the right type.

    Builds the full pool of eligible clips (preferred pros first) and returns
    pool[rotate % len(pool)] so successive shots cycle through different clips
    instead of always taking the first match (the "every forehand is the same
    Murray clip" bug). Filters:
      - shot_type (forehand/backhand/serve)
      - For backhand: hard-filter on backhand_style match
      - Coarse-angle match to the user (e.g. 'behind' ~ 'behind-player'); if no
        clip shares the user's angle family, relax and rotate over all matches.
    """
    with INDEX_PATH.open() as f:
        index = json.load(f)
    if preferred_slug:
        slugs = [preferred_slug]
    else:
        all_slugs = list(index["players"].keys())
        ordered = [s for s in DEFAULT_PREFERRED_PROS if s in all_slugs]
        ordered += [s for s in all_slugs if s not in ordered]
        slugs = ordered
    user_bh_style = (user_backhand_style or "").lower()
    want_angle = _coarse_angle(target_angle)

    def _pool(require_angle: bool) -> list[tuple[str, str]]:
        out = []
        for slug in slugs:
            player = index["players"].get(slug, {})
            if shot_type == "backhand" and user_bh_style:
                pro_bh_style = (player.get("backhand_style") or "").lower()
                if pro_bh_style and pro_bh_style != user_bh_style:
                    continue
            for clip in player.get("clips", []):
                if clip.get("type") != shot_type:
                    continue
                if clip.get("quality_ok") is False:   # scored junk (score_pro_clips)
                    continue
                if require_angle and want_angle:
                    clip_angle = _coarse_angle(clip.get("detected_angle")
                                               or clip.get("angle"))
                    if clip_angle != want_angle:
                        continue
                if (PROS_DIR / slug / clip["file"]).exists():
                    out.append((slug, clip["file"]))
        return out

    # Angle-matched pool first (consistent viewpoint); relax only if empty.
    pool = _pool(require_angle=True) or _pool(require_angle=False)
    if not pool:
        raise FileNotFoundError(
            f"No on-disk pro clip of type {shot_type!r} "
            f"(backhand_style={user_bh_style!r}, target_angle={target_angle!r})"
        )
    return pool[rotate % len(pool)]


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
    # Look up type + measured contact frame from index.json so the synth det
    # matches. contact_frame is per-clip (detect_pro_contact.py); falls back to
    # the legacy midpoint only for clips that predate detection.
    with INDEX_PATH.open() as f:
        index = json.load(f)
    shot_type = "forehand"
    contact = PRO_CONTACT_FRAME
    for clip in index["players"][slug].get("clips", []):
        if clip["file"] == filename:
            shot_type = clip.get("type", "forehand")
            contact = int(clip.get("contact_frame", PRO_CONTACT_FRAME))
            break
    fake_det = {
        "fps": PRO_FPS,
        "detections": [{
            "frame": contact,
            "timestamp": contact / PRO_FPS,
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
    ap.add_argument("--global-shot", type=int, default=None,
                    help="Global detection index (matches comparison_shot_NNN.mp4 naming). "
                         "Overrides --shot/--shot-type.")
    ap.add_argument("--pro-only", action="store_true",
                    help="Output only the matched pro's filmstrip (the user's is "
                         "already shown in the /inspect shot card).")
    ap.add_argument("--shot-type", choices=("forehand", "backhand", "serve"),
                    help="Shot type (default: pick any shot at index --shot)")
    ap.add_argument("--pro", help="Preferred pro slug (default: auto)")
    ap.add_argument("--pro-clip", help="Specific clip filename (e.g. backhand_005.mp4) — bypasses matcher")
    ap.add_argument("--rotate", type=int, default=0,
                    help="Rotation index into the matched-clip pool, so batch "
                         "generation cycles different pros/clips per shot.")
    ap.add_argument("--user-backhand-style", default="two-handed",
                    choices=("one-handed", "two-handed"),
                    help="User's backhand style (default two-handed). Hard-filter for backhand matches.")
    ap.add_argument("--user-angle",
                    choices=("behind-player", "side-deuce", "side-ad",
                             "front-broadcast", "side", "behind"),
                    help="Override user camera angle. Prefer the 4-value taxonomy "
                         "(behind-player / side-deuce / side-ad / front-broadcast); "
                         "legacy values 'side' / 'behind' kept for compatibility.")
    ap.add_argument("--output", help="Output PNG path (default: /tmp/<user>_<shot>_vs_<pro>.png)")
    ap.add_argument("--no-skeleton", action="store_true")
    ap.add_argument("--mirror-pro", action="store_true",
                    help="Horizontally flip the pro filmstrip. ONLY useful for "
                         "lefty-vs-righty handedness conversion (e.g., comparing a "
                         "right-handed user to a Nadal clip). Does NOT compensate "
                         "for camera-angle differences — that's a 3D problem "
                         "image-flipping cannot solve.")
    args = ap.parse_args()

    print(f"Loading user data for {args.user}…")
    user_video, user_det, user_poses = load_user_data(args.user)
    # --global-shot maps directly to detections[N] (same indexing the per-shot
    # comparison VIDEOS use), so a batch generator can pair compare_shot_NNN.jpg
    # with comparison_shot_NNN.mp4. Otherwise fall back to Nth-of-type.
    if args.global_shot is not None:
        user_shot_idx = args.global_shot
    else:
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

    if args.pro_clip:
        if not args.pro:
            print("[ERROR] --pro-clip requires --pro")
            return 1
        slug, filename = args.pro, args.pro_clip
        print(f"Using explicit pro clip: {slug}/{filename}")
    else:
        # Resolve user's camera angle for matching
        target_angle = args.user_angle
        if not target_angle:
            target_angle = (user_det.get("camera_angle")
                            or user_det.get("metadata", {}).get("camera_angle")
                            or "behind")  # default: 90% of user footage is behind
            target_angle = target_angle.lower() if isinstance(target_angle, str) else "behind"
        print(f"Matching pro clip of type {shot_type!r} (angle={target_angle})…")
        slug, filename = match_pro_clip(shot_type, args.pro, args.user_backhand_style,
                                        target_angle=target_angle, rotate=args.rotate)
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

    if args.mirror_pro:
        # Only legitimate use is handedness conversion (lefty-vs-righty).
        # Horizontal flip does NOT correct camera-angle differences.
        pro_strip = cv2.flip(pro_strip, 1)
        print(f"  Mirrored pro filmstrip horizontally (handedness conversion)")

    # PRO-ONLY: the user's own filmstrip is already shown in the /inspect shot
    # card, so the comparison only needs the PRO strip below it (avoids showing
    # the user's swing twice). Same panel layout → aligns with the card strip.
    if args.pro_only:
        pro_strip = add_label_band(pro_strip, f"PRO - {slug} ({shot_type})")
        output = args.output or f"/tmp/{args.user}_{user_shot_idx}_pro_{slug}.png"
        cv2.imwrite(output, pro_strip)
        print(f"Saved (pro-only): {output} ({pro_strip.shape[1]}x{pro_strip.shape[0]})")
        return 0

    # Both strips have the SAME panel count (NUM_FRAMES) with contact at the
    # same panel index. Scaling both to equal width therefore aligns every
    # panel column — contact-on-contact — so you compare the same swing moment
    # in the same column. (Old code black-padded the narrower strip, which left
    # a black bar AND mis-aligned the columns.)
    target_w = max(user_strip.shape[1], pro_strip.shape[1])
    def resize_to_width(img, w):
        if img.shape[1] == w:
            return img
        h = int(round(img.shape[0] * w / img.shape[1]))
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    user_strip = resize_to_width(user_strip, target_w)
    pro_strip = resize_to_width(pro_strip, target_w)

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
