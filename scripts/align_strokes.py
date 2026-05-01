#!/usr/bin/env python3
"""Side-by-side stroke alignment: your shot vs a pro reference.

Aligns at contact frame, renders synchronized slow-mo MP4.
Lets you eyeball backswing timing, contact position, follow-through.

Usage:
    # Single shot vs Djokovic
    python scripts/align_strokes.py --video IMG_1195 --shot 5 --pro djokovic

    # Auto-pick a pro clip matching your shot type
    python scripts/align_strokes.py --video IMG_1195 --shot 5
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
PREPROCESSED = PROJECT_ROOT / "preprocessed"
DETECTIONS = PROJECT_ROOT / "detections"
PROS = PROJECT_ROOT / "pros"
EXPORTS = PROJECT_ROOT / "exports"

WINDOW_BEFORE = 1.5   # seconds before contact
WINDOW_AFTER = 1.0    # seconds after contact
SPEED = 0.25          # 4x slow-motion
PANEL_W = 720
PANEL_H = 1080
LABEL_BAR_H = 60


def load_user_shot(video_id: str, shot_idx: int):
    """Return (video_path, contact_time_sec, fps, shot_type)."""
    det_path = DETECTIONS / f"{video_id}_fused_detections.json"
    if not det_path.exists():
        det_path = DETECTIONS / f"{video_id}_fused.json"
    if not det_path.exists():
        sys.exit(f"[ERR] no detections for {video_id}")

    det = json.loads(det_path.read_text())
    detections = det.get("detections", [])
    if shot_idx >= len(detections):
        sys.exit(f"[ERR] shot {shot_idx} out of range ({len(detections)} shots)")

    d = detections[shot_idx]
    fps = det.get("fps", 60.0)
    contact_t = d.get("timestamp")
    if contact_t is None:
        contact_t = d.get("frame", 0) / fps

    video_path = PREPROCESSED / f"{video_id}.mp4"
    if not video_path.exists():
        sys.exit(f"[ERR] no preprocessed video at {video_path}")

    return str(video_path), float(contact_t), fps, d.get("shot_type", "unknown")


def pick_pro_clip(shot_type: str, player: str | None):
    """Find a pro clip for the requested shot_type. Returns (path, contact_t, fps, player_name)."""
    idx = json.loads((PROS / "index.json").read_text())
    players = idx.get("players", {})

    # Filter to requested player or try all
    candidates = [(p, pdata) for p, pdata in players.items()
                  if player is None or p == player]
    if player and player not in players:
        sys.exit(f"[ERR] no pro {player} in index.json. Have: {list(players)}")

    for pname, pdata in candidates:
        for clip in pdata.get("clips", []):
            if clip.get("type") == shot_type:
                clip_path = PROS / pname / clip["file"]
                if clip_path.exists():
                    contact_frame = clip.get("contact_frame", 0)
                    fps = clip.get("fps", 60.0)
                    return str(clip_path), contact_frame / fps, fps, pdata.get("name", pname)

    sys.exit(f"[ERR] no pro clip for shot_type={shot_type}")


def render_aligned(user_path: str, user_contact_t: float, user_label: str,
                   pro_path: str, pro_contact_t: float, pro_label: str,
                   output_path: Path) -> bool:
    """Render side-by-side aligned MP4.

    Both clips: ±WINDOW_BEFORE..WINDOW_AFTER around contact, played back at SPEED.
    Aligned by trimming each clip so contact lands at the same output time.
    """
    user_start = max(0.0, user_contact_t - WINDOW_BEFORE)
    user_end = user_contact_t + WINDOW_AFTER
    pro_start = max(0.0, pro_contact_t - WINDOW_BEFORE)
    pro_end = pro_contact_t + WINDOW_AFTER

    # ffmpeg filter graph: trim each, scale, slow, pad with label, hstack.
    # The setpts=PTS/SPEED slows playback. Both clips are trimmed to the same
    # window length so contact is at the same offset in each output stream.
    # No drawtext (not in stock macOS ffmpeg). Color-code panels via padcolor.
    filt = (
        f"[0:v]trim=start={user_start:.3f}:end={user_end:.3f},setpts=(PTS-STARTPTS)/{SPEED},"
        f"scale={PANEL_W}:{PANEL_H}:force_original_aspect_ratio=decrease,"
        f"pad={PANEL_W}:{PANEL_H}:(ow-iw)/2:(oh-ih)/2:0x301010[u];"
        f"[1:v]trim=start={pro_start:.3f}:end={pro_end:.3f},setpts=(PTS-STARTPTS)/{SPEED},"
        f"scale={PANEL_W}:{PANEL_H}:force_original_aspect_ratio=decrease,"
        f"pad={PANEL_W}:{PANEL_H}:(ow-iw)/2:(oh-ih)/2:0x101030[p];"
        f"[u][p]hstack=inputs=2[v]"
    )

    cmd = [
        "ffmpeg", "-y",
        "-i", user_path,
        "-i", pro_path,
        "-filter_complex", filt,
        "-map", "[v]",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-an",
        "-movflags", "+faststart",
        str(output_path),
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  rendering {output_path.name}...")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  [ERR] ffmpeg failed: {r.stderr[-500:]}")
        return False

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  done: {output_path} ({size_mb:.1f} MB)")
    return True


def escape_text(s: str) -> str:
    """Escape text for ffmpeg drawtext."""
    return (s.replace("\\", "\\\\")
             .replace(":", r"\:")
             .replace("'", r"\'")
             .replace(",", r"\,"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True, help="User video ID e.g. IMG_1195")
    p.add_argument("--shot", type=int, required=True, help="User shot index")
    p.add_argument("--pro", help="Pro player name (default: auto-pick by shot type)")
    p.add_argument("--output", help="Output path (default: exports/<vid>/comparisons/align_<shot>_vs_<pro>.mp4)")
    p.add_argument("--upload", action="store_true", help="Upload to R2 after rendering")
    args = p.parse_args()

    user_path, user_t, user_fps, shot_type = load_user_shot(args.video, args.shot)
    pro_path, pro_t, pro_fps, pro_name = pick_pro_clip(shot_type, args.pro)

    print(f"  user: {Path(user_path).name} contact={user_t:.2f}s shot_type={shot_type}")
    print(f"  pro:  {Path(pro_path).name} contact={pro_t:.2f}s ({pro_name})")

    pro_slug = Path(pro_path).parent.name
    out = Path(args.output) if args.output else (
        EXPORTS / args.video / "comparisons" /
        f"align_{args.shot:03d}_{shot_type}_vs_{pro_slug}.mp4"
    )

    user_label = f"YOU - {shot_type.upper()} #{args.shot + 1}"
    pro_label = f"{pro_name.upper()} - {shot_type.upper()}"

    ok = render_aligned(user_path, user_t, user_label,
                        pro_path, pro_t, pro_label, out)
    if not ok:
        sys.exit(1)

    if args.upload:
        sys.path.insert(0, str(PROJECT_ROOT))
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env")
        from storage.r2_client import R2Client
        c = R2Client()
        key = f"highlights/{args.video}/comparisons/{out.name}"
        c.upload(str(out), key, content_type="video/mp4")
        print(f"  uploaded: https://tennis.playfullife.com/{args.video}/comparisons/{out.name}")


if __name__ == "__main__":
    main()
