#!/usr/bin/env python3
"""Dynamic player tracking — smooth per-frame pan/zoom following the player.

Reads pose data to compute per-frame bounding box, applies exponential moving
average for cinematic smooth camera motion, and outputs a cropped/zoomed video
via frame-by-frame OpenCV processing piped to ffmpeg for encoding.

Usage:
    python scripts/dynamic_track.py preprocessed/IMG_1027.mp4
    python scripts/dynamic_track.py preprocessed/IMG_1027.mp4 --padding 0.4 --smoothing 0.92
    python scripts/dynamic_track.py --video IMG_1027 --start 10 --end 30
"""

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
POSES_DIR = PROJECT_ROOT / "poses_full_videos"
DETECTIONS_DIR = PROJECT_ROOT / "detections"

# Windows GPU paths
if sys.platform == "win32":
    PROJECT_ROOT = Path(r"C:\Users\amass\tennis_analysis")
    POSES_DIR = PROJECT_ROOT / "poses"
    DETECTIONS_DIR = PROJECT_ROOT / "detections"


def get_frame_bbox(frame_data, vis_threshold=0.3):
    """Extract player bounding box from a single pose frame.
    Returns (cx, cy, w, h) in normalized coords, or None if no pose."""
    if not frame_data.get("detected") or not frame_data.get("landmarks"):
        return None

    xs, ys = [], []
    for lm in frame_data["landmarks"]:
        if isinstance(lm, dict):
            v = lm.get("visibility", 0)
            if v < vis_threshold:
                continue
            xs.append(lm["x"])
            ys.append(lm["y"])
        else:
            v = lm[3] if len(lm) >= 4 else 1.0
            if v < vis_threshold:
                continue
            xs.append(lm[0])
            ys.append(lm[1])

    if len(xs) < 5:
        return None

    xs.sort()
    ys.sort()
    lo = max(0, int(len(xs) * 0.05))
    hi = min(len(xs) - 1, int(len(xs) * 0.95))
    x_min, x_max = xs[lo], xs[hi]

    lo_y = max(0, int(len(ys) * 0.05))
    hi_y = min(len(ys) - 1, int(len(ys) * 0.95))
    y_min, y_max = ys[lo_y], ys[hi_y]

    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    w = x_max - x_min
    h = y_max - y_min

    return (cx, cy, w, h)


def compute_smooth_track(poses_data, start_frame, end_frame, video_w, video_h,
                         padding=0.35, min_ratio=0.35, smoothing=0.92, target_aspect=16/9):
    """Compute per-frame crop rectangles with EMA smoothing.

    Args:
        smoothing: EMA alpha for position/size. Higher = smoother (0.95 = very smooth, 0.85 = responsive).
        padding: fraction of player size added as margin.
        min_ratio: minimum crop dimension as fraction of full frame.
        target_aspect: output aspect ratio.

    Returns list of (crop_x, crop_y, crop_w, crop_h) per frame, all in pixels with even values.
    """
    frames = poses_data.get("frames", [])
    n_frames = min(end_frame, len(frames)) - start_frame
    if n_frames <= 0:
        return []

    ema_cx, ema_cy, ema_w, ema_h = None, None, None, None
    crops = []

    for i in range(start_frame, min(end_frame, len(frames))):
        bbox = get_frame_bbox(frames[i]) if i < len(frames) else None

        if bbox is not None:
            cx, cy, w, h = bbox
            w_padded = w + 2 * padding * max(w, 0.1)
            h_padded = h + 2 * padding * max(h, 0.1)
            w_padded = max(w_padded, min_ratio)
            h_padded = max(h_padded, min_ratio)

            current_aspect = (w_padded * video_w) / (h_padded * video_h)
            if current_aspect < target_aspect:
                w_padded = (h_padded * video_h * target_aspect) / video_w
            else:
                h_padded = (w_padded * video_w / target_aspect) / video_h

            w_padded = min(w_padded, 1.0)
            h_padded = min(h_padded, 1.0)

            if ema_cx is None:
                ema_cx, ema_cy = cx, cy
                ema_w, ema_h = w_padded, h_padded
            else:
                ema_cx = smoothing * ema_cx + (1 - smoothing) * cx
                ema_cy = smoothing * ema_cy + (1 - smoothing) * cy
                ema_w = smoothing * ema_w + (1 - smoothing) * w_padded
                ema_h = smoothing * ema_h + (1 - smoothing) * h_padded

        if ema_cx is None:
            crops.append(None)
            continue

        x_start = max(0.0, min(1.0 - ema_w, ema_cx - ema_w / 2))
        y_start = max(0.0, min(1.0 - ema_h, ema_cy - ema_h / 2))

        crop_w = int(ema_w * video_w)
        crop_h = int(ema_h * video_h)
        crop_w -= crop_w % 2
        crop_h -= crop_h % 2
        crop_x = int(x_start * video_w)
        crop_y = int(y_start * video_h)
        crop_x -= crop_x % 2
        crop_y -= crop_y % 2

        crops.append((crop_x, crop_y, crop_w, crop_h))

    return crops


def render_dynamic_track(video_path, poses_data, output_path, start_sec=None, end_sec=None,
                         padding=0.35, smoothing=0.92, output_w=1920, output_h=1080, use_nvenc=True):
    """Render a dynamically-tracked video using OpenCV + ffmpeg pipe."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Cannot open {video_path}")
        return False

    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        fps = 60.0

    start_frame = int(start_sec * fps) if start_sec else 0
    end_frame = int(end_sec * fps) if end_sec else total_frames

    print(f"  Video: {vid_w}x{vid_h} @ {fps:.1f}fps, frames {start_frame}-{end_frame}")
    print(f"  Computing smooth tracking (padding={padding}, smoothing={smoothing})...")

    crops = compute_smooth_track(
        poses_data, start_frame, end_frame, vid_w, vid_h,
        padding=padding, smoothing=smoothing
    )

    if not crops or all(c is None for c in crops):
        print("  [WARN] No pose data for tracking, falling back to full frame")
        cap.release()
        return False

    valid_crops = [c for c in crops if c is not None]
    if not valid_crops:
        cap.release()
        return False

    fallback_crop = valid_crops[0]
    first_valid_idx = next(i for i, c in enumerate(crops) if c is not None)
    for i in range(first_valid_idx):
        crops[i] = crops[first_valid_idx]
    for i in range(len(crops)):
        if crops[i] is None:
            crops[i] = crops[i - 1] if i > 0 else fallback_crop

    output_w -= output_w % 2
    output_h -= output_h % 2

    if use_nvenc:
        encoder_args = ['-c:v', 'h264_nvenc', '-preset', 'p4', '-cq', '22', '-b:v', '0']
    else:
        encoder_args = ['-c:v', 'libx264', '-preset', 'fast', '-crf', '22']

    ffmpeg_cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo', '-pix_fmt', 'bgr24',
        '-s', f'{output_w}x{output_h}',
        '-r', str(fps),
        '-i', 'pipe:0',
        *encoder_args,
        '-an',
        '-movflags', '+faststart',
        str(output_path)
    ]

    print(f"  Rendering {len(crops)} frames → {output_path}")
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE)

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    rendered = 0
    for i, crop in enumerate(crops):
        ret, frame = cap.read()
        if not ret:
            break

        cx, cy, cw, ch = crop
        cx = max(0, min(cx, vid_w - cw))
        cy = max(0, min(cy, vid_h - ch))
        cw = min(cw, vid_w - cx)
        ch = min(ch, vid_h - cy)

        cropped = frame[cy:cy+ch, cx:cx+cw]
        resized = cv2.resize(cropped, (output_w, output_h), interpolation=cv2.INTER_LANCZOS4)
        proc.stdin.write(resized.tobytes())
        rendered += 1

        if rendered % 500 == 0:
            print(f"    {rendered}/{len(crops)} frames...", flush=True)

    cap.release()
    proc.stdin.close()
    proc.wait()

    if proc.returncode != 0:
        stderr = proc.stderr.read().decode()
        if "nvenc" in stderr.lower() or "h264_nvenc" in stderr.lower():
            print("  NVENC not available, retrying with libx264...")
            return render_dynamic_track(
                video_path, poses_data, output_path, start_sec, end_sec,
                padding, smoothing, output_w, output_h, use_nvenc=False
            )
        print(f"  [ERROR] ffmpeg failed: {stderr[-500:]}")
        return False

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Done: {rendered} frames, {size_mb:.1f} MB")
    return True


def main():
    parser = argparse.ArgumentParser(description="Dynamic player tracking with smooth pan/zoom")
    parser.add_argument("video", nargs="?", help="Path to preprocessed video")
    parser.add_argument("--video", dest="video_id", help="Video ID (e.g. IMG_1027)")
    parser.add_argument("--start", type=float, help="Start time in seconds")
    parser.add_argument("--end", type=float, help="End time in seconds")
    parser.add_argument("--padding", type=float, default=0.35, help="Padding around player (0.2-0.5)")
    parser.add_argument("--smoothing", type=float, default=0.92, help="EMA smoothing (0.8-0.98)")
    parser.add_argument("--output", help="Output path")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--cpu", action="store_true", help="Use libx264 instead of NVENC")
    parser.add_argument("--all", action="store_true", help="Process all videos with detections")
    parser.add_argument("--upload", action="store_true", help="Upload to R2 after rendering")
    args = parser.parse_args()

    if args.all:
        det_files = sorted(DETECTIONS_DIR.glob("*_fused_detections.json"))
        videos = []
        for df in det_files:
            vid = df.stem.replace("_fused_detections", "")
            vid_path = PROJECT_ROOT / "preprocessed" / f"{vid}.mp4"
            if vid_path.exists():
                videos.append((vid, str(vid_path)))
        print(f"Dynamic tracking batch: {len(videos)} videos")
        for vid, vpath in videos:
            out_dir = PROJECT_ROOT / "exports" / vid
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{vid}_tracked.mp4"
            if out_path.exists():
                print(f"  {vid}: already exists, skip")
                continue
            pose_path = POSES_DIR / f"{vid}.json"
            if not pose_path.exists():
                print(f"  {vid}: no poses, skip")
                continue
            print(f"\n[{vid}]")
            with open(pose_path) as f:
                poses = json.load(f)
            success = render_dynamic_track(
                vpath, poses, str(out_path),
                padding=args.padding, smoothing=args.smoothing,
                output_w=args.width, output_h=args.height,
                use_nvenc=not args.cpu
            )
            if success and args.upload:
                upload_tracked(vid, str(out_path))
        return

    if args.video:
        video_path = str(args.video)
        vid = Path(video_path).stem
    elif args.video_id:
        vid = args.video_id
        video_path = str(PROJECT_ROOT / "preprocessed" / f"{vid}.mp4")
    else:
        parser.error("Provide a video path or --video ID")
        return

    pose_path = POSES_DIR / f"{vid}.json"
    if not pose_path.exists():
        print(f"[ERROR] No poses at {pose_path}")
        return

    with open(pose_path) as f:
        poses = json.load(f)

    out_dir = PROJECT_ROOT / "exports" / vid
    out_dir.mkdir(parents=True, exist_ok=True)
    output = args.output or str(out_dir / f"{vid}_tracked.mp4")

    print(f"[{vid}] Dynamic player tracking")
    success = render_dynamic_track(
        video_path, poses, output,
        start_sec=args.start, end_sec=args.end,
        padding=args.padding, smoothing=args.smoothing,
        output_w=args.width, output_h=args.height,
        use_nvenc=not args.cpu
    )

    if success and args.upload:
        upload_tracked(vid, output)


def upload_tracked(vid, path):
    """Upload tracked video to R2."""
    sys.path.insert(0, str(PROJECT_ROOT))
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
    from storage.r2_client import R2Client
    r2 = R2Client()
    key = f"highlights/{vid}/{os.path.basename(path)}"
    r2.upload(path, key, content_type="video/mp4")
    print(f"  Uploaded to R2: {key}")


if __name__ == "__main__":
    main()
