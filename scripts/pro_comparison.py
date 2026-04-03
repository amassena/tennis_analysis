#!/usr/bin/env python3
"""Generate side-by-side comparison videos of user shots vs professional players.

For each detected shot, finds a matching pro clip (same shot type), extracts the
user's clip around the contact point, and creates a side-by-side video aligned
at the contact frame.

Usage:
    python scripts/pro_comparison.py preprocessed/IMG_1027.mp4
    python scripts/pro_comparison.py preprocessed/IMG_1027.mp4 --player federer
    python scripts/pro_comparison.py preprocessed/IMG_1027.mp4 --shot-type forehand --max-clips 5
    python scripts/pro_comparison.py preprocessed/IMG_1027.mp4 --upload
"""

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

PROJECT_ROOT = Path(__file__).parent.parent
PROS_DIR = PROJECT_ROOT / "pros"
DETECTIONS_DIR = PROJECT_ROOT / "detections"
EXPORTS_DIR = PROJECT_ROOT / "exports"

# Output resolution for each half of the comparison
HALF_WIDTH = 640
HALF_HEIGHT = 480
OUTPUT_FPS = 60

# Timing around contact point
CLIP_BEFORE = 2.0   # seconds before contact
CLIP_AFTER = 2.0    # seconds after contact
SLOWMO_SPEED = 0.25  # 4x slow motion

# Shot types eligible for comparison
COMPARABLE_TYPES = {'forehand', 'backhand', 'serve'}

# Label colors per shot type (same as export_videos.py)
SHOT_COLORS = {
    'serve': (255, 140, 0),
    'forehand': (155, 89, 182),
    'backhand': (39, 174, 96),
}


def load_pro_library():
    """Load the pro clip library catalog from pros/index.json."""
    index_path = PROS_DIR / "index.json"
    if not index_path.exists():
        print(f"[WARN] Pro library not found: {index_path}")
        return {}

    with open(index_path) as f:
        return json.load(f)


def load_detections(video_name):
    """Load detection JSON for a video."""
    for name in [f"{video_name}_fused.json", f"{video_name}_fused_detections.json"]:
        path = DETECTIONS_DIR / name
        if path.exists():
            with open(path) as f:
                return json.load(f)
    return None


def match_pro_clip(shot_type, library, preferred_player=None, preferred_angle=None):
    """Find the best matching pro clip for a given shot type.

    Returns (player_name, clip_info) or (None, None) if no match.
    """
    candidates = []
    players = library.get("players", {})

    for player_id, player_data in players.items():
        name = player_data.get("name", player_id)
        for clip in player_data.get("clips", []):
            if clip.get("type") == shot_type:
                score = 1.0
                if preferred_player and player_id == preferred_player:
                    score += 2.0
                if preferred_angle and clip.get("angle") == preferred_angle:
                    score += 1.0
                candidates.append((score, name, player_id, clip))

    if not candidates:
        return None, None

    # Sort by score descending, pick randomly among top candidates
    candidates.sort(key=lambda x: -x[0])
    top_score = candidates[0][0]
    top_candidates = [c for c in candidates if c[0] == top_score]
    chosen = random.choice(top_candidates)
    return chosen[1], chosen[3]  # (player_name, clip_info)


def get_pro_clip_path(player_name, clip_info, library):
    """Get local path to a pro clip, downloading from R2 if needed."""
    # Find player_id from name
    player_id = None
    for pid, pdata in library.get("players", {}).items():
        if pdata.get("name") == player_name:
            player_id = pid
            break
    if not player_id:
        return None

    filename = clip_info["file"]
    local_path = PROS_DIR / player_id / filename

    if local_path.exists():
        return local_path

    # Try downloading from R2
    remote_key = f"pros/{player_id}/{filename}"
    try:
        from storage.r2_client import R2Client
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env")
        client = R2Client()
        if client.exists(remote_key):
            client.download(remote_key, str(local_path))
            return local_path
    except Exception as e:
        print(f"  [WARN] Could not download pro clip: {e}")

    return None


def extract_user_clip(video_path, timestamp, output_path, speed=SLOWMO_SPEED):
    """Extract a clip from the user's video centered on the contact point."""
    start = max(0, timestamp - CLIP_BEFORE)
    duration = CLIP_BEFORE + CLIP_AFTER

    # Scale to half-width, pad to exact dimensions, apply slow-mo
    vf = (
        f"setpts=PTS/{speed:.4f},"
        f"scale={HALF_WIDTH}:-1,"
        f"pad={HALF_WIDTH}:{HALF_HEIGHT}:(ow-iw)/2:(oh-ih)/2:black"
    )

    output_duration = duration / speed

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", str(video_path),
        "-t", str(output_duration),
        "-vf", vf,
        "-r", str(OUTPUT_FPS),
        "-c:v", "libx264", "-preset", "fast", "-crf", "22",
        "-an",  # drop audio for slow-mo
        "-movflags", "+faststart",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def extract_pro_clip(clip_path, clip_info, output_path, target_duration):
    """Extract and normalize a pro clip, aligned at contact frame.

    The pro clip is padded/trimmed to match target_duration, with contact
    frame aligned to CLIP_BEFORE / SLOWMO_SPEED into the output.
    """
    contact_frame = clip_info.get("contact_frame", 0)
    clip_fps = clip_info.get("fps", 60)
    contact_time = contact_frame / clip_fps

    # Start so that contact aligns with CLIP_BEFORE/speed point in output
    start = max(0, contact_time - CLIP_BEFORE)

    # Read the entire clip, scale, pad, and slow down
    vf = (
        f"setpts=PTS/{SLOWMO_SPEED:.4f},"
        f"scale={HALF_WIDTH}:-1,"
        f"pad={HALF_WIDTH}:{HALF_HEIGHT}:(ow-iw)/2:(oh-ih)/2:black"
    )

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", str(clip_path),
        "-t", str(target_duration),
        "-vf", vf,
        "-r", str(OUTPUT_FPS),
        "-c:v", "libx264", "-preset", "fast", "-crf", "22",
        "-an",
        "-movflags", "+faststart",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def create_label_overlay(text, width, height, output_path):
    """Create a semi-transparent label image using PIL."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return False

    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    font = None
    for fp in [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]:
        try:
            font = ImageFont.truetype(fp, 32)
            break
        except (OSError, IOError):
            continue
    if font is None:
        font = ImageFont.load_default()

    try:
        bbox = font.getbbox(text)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
    except AttributeError:
        text_w = len(text) * 18
        text_h = 28

    pad = 12
    box_w = text_w + pad * 2
    box_h = text_h + pad * 2
    x = 10
    y = 10

    draw.rounded_rectangle(
        [x, y, x + box_w, y + box_h],
        radius=8,
        fill=(0, 0, 0, 160),
    )
    draw.text((x + pad, y + pad), text, fill=(255, 255, 255, 255), font=font)
    img.save(output_path)
    return True


def hstack_with_labels(left_path, right_path, output_path,
                        left_label="YOU", right_label="PRO"):
    """Combine two clips side by side with label overlays."""
    tmpdir = tempfile.mkdtemp(prefix="tennis_compare_")

    try:
        # Create label images
        left_label_img = os.path.join(tmpdir, "label_left.png")
        right_label_img = os.path.join(tmpdir, "label_right.png")

        has_labels = (
            create_label_overlay(left_label, HALF_WIDTH, HALF_HEIGHT, left_label_img)
            and create_label_overlay(right_label, HALF_WIDTH, HALF_HEIGHT, right_label_img)
        )

        if has_labels:
            # hstack with overlaid labels
            cmd = [
                "ffmpeg", "-y",
                "-i", str(left_path),
                "-i", str(right_path),
                "-i", left_label_img,
                "-i", right_label_img,
                "-filter_complex",
                "[0:v][1:v]hstack=inputs=2[stacked];"
                f"[2:v]scale={HALF_WIDTH}:{HALF_HEIGHT}[ll];"
                f"[3:v]scale={HALF_WIDTH}:{HALF_HEIGHT}[rl];"
                "[stacked][ll]overlay=0:0:enable='between(t,0,3)'[tmp];"
                f"[tmp][rl]overlay={HALF_WIDTH}:0:enable='between(t,0,3)'[out]",
                "-map", "[out]",
                "-c:v", "libx264", "-preset", "fast", "-crf", "22",
                "-an",
                "-movflags", "+faststart",
                str(output_path),
            ]
        else:
            # Simple hstack without labels
            cmd = [
                "ffmpeg", "-y",
                "-i", str(left_path),
                "-i", str(right_path),
                "-filter_complex", "[0:v][1:v]hstack=inputs=2",
                "-c:v", "libx264", "-preset", "fast", "-crf", "22",
                "-an",
                "-movflags", "+faststart",
                str(output_path),
            ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def generate_comparisons(video_path, output_dir=None, player=None,
                          shot_type_filter=None, max_clips=None, upload=False):
    """Generate comparison clips for all eligible shots in a video.

    Args:
        video_path: Path to preprocessed video
        output_dir: Output directory (default: exports/{video_name}/)
        player: Preferred pro player ID (e.g., "federer")
        shot_type_filter: Only generate for this shot type
        max_clips: Maximum number of comparison clips to generate
        upload: Upload results to R2

    Returns:
        List of generated comparison file paths
    """
    video_name = Path(video_path).stem

    # Load detection data
    det_data = load_detections(video_name)
    if not det_data:
        print(f"[ERROR] No detections found for {video_name}")
        return []

    # Load pro library
    library = load_pro_library()
    if not library.get("players"):
        print("[ERROR] Pro clip library is empty — add clips to pros/index.json")
        return []

    # Set up output
    if output_dir is None:
        output_dir = EXPORTS_DIR / video_name
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    detections = det_data.get("detections", [])
    eligible = [
        d for d in detections
        if d.get("shot_type") in COMPARABLE_TYPES
        and (shot_type_filter is None or d.get("shot_type") == shot_type_filter)
    ]

    if max_clips:
        eligible = eligible[:max_clips]

    print(f"\n{'='*60}")
    print(f"Pro Comparison: {video_name}")
    print(f"  {len(eligible)} eligible shots (of {len(detections)} total)")
    print(f"  Pro library: {len(library.get('players', {}))} players")

    generated = []
    tmpdir = tempfile.mkdtemp(prefix="tennis_procomp_")

    try:
        # Track which pro clips we've used to add variety
        used_clips = {}  # shot_type -> set of clip files used

        for i, det in enumerate(eligible):
            shot_type = det["shot_type"]
            timestamp = det["timestamp"]
            idx = i + 1

            # Find matching pro clip (avoiding repeats when possible)
            pro_name, pro_clip = match_pro_clip(shot_type, library, player)
            if not pro_clip:
                print(f"  [{idx}/{len(eligible)}] {shot_type} @ {timestamp:.1f}s — no pro clip available")
                continue

            # Get local path to pro clip
            clip_path = get_pro_clip_path(pro_name, pro_clip, library)
            if not clip_path:
                print(f"  [{idx}/{len(eligible)}] {shot_type} @ {timestamp:.1f}s — pro clip file not found")
                continue

            print(f"  [{idx}/{len(eligible)}] {shot_type} @ {timestamp:.1f}s vs {pro_name}")

            # Extract user clip
            user_clip = os.path.join(tmpdir, f"user_{i:03d}.mp4")
            if not extract_user_clip(video_path, timestamp, user_clip):
                print(f"    [ERROR] Failed to extract user clip")
                continue

            # Get user clip duration for alignment
            target_duration = (CLIP_BEFORE + CLIP_AFTER) / SLOWMO_SPEED

            # Extract pro clip
            pro_clip_out = os.path.join(tmpdir, f"pro_{i:03d}.mp4")
            if not extract_pro_clip(clip_path, pro_clip, pro_clip_out, target_duration):
                print(f"    [ERROR] Failed to extract pro clip")
                continue

            # Combine side by side
            output_name = f"{video_name}_comparison_{shot_type}_{idx:02d}.mp4"
            output_path = str(output_dir / output_name)

            # Format pro label
            pro_label = pro_name.upper().split()[0]  # First name only

            if hstack_with_labels(user_clip, pro_clip_out, output_path,
                                   left_label="YOU", right_label=pro_label):
                size_mb = os.path.getsize(output_path) / (1024 * 1024)
                print(f"    Saved: {output_name} ({size_mb:.1f}MB)")
                generated.append(output_path)
            else:
                print(f"    [ERROR] Failed to create comparison")

        # Compile all comparisons into one video
        if len(generated) > 1:
            compiled_path = str(output_dir / f"{video_name}_comparisons.mp4")
            concat_file = os.path.join(tmpdir, "concat.txt")
            with open(concat_file, "w") as f:
                for path in generated:
                    f.write(f"file '{os.path.abspath(path)}'\n")

            result = subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-f", "concat", "-safe", "0",
                    "-i", concat_file,
                    "-c", "copy",
                    "-movflags", "+faststart",
                    compiled_path,
                ],
                capture_output=True, text=True,
            )
            if result.returncode == 0:
                size_mb = os.path.getsize(compiled_path) / (1024 * 1024)
                print(f"\n  Compiled: {os.path.basename(compiled_path)} ({size_mb:.1f}MB)")
                generated.append(compiled_path)

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    # Upload to R2 if requested
    if upload and generated:
        print(f"\nUploading {len(generated)} comparison files to R2...")
        try:
            from scripts.export_videos import upload_to_r2
            for filepath in generated:
                filename = os.path.basename(filepath)
                remote_key = f"highlights/{video_name}/{filename}"
                upload_to_r2(filepath, remote_key)
        except Exception as e:
            print(f"  [ERROR] R2 upload failed: {e}")

    print(f"\nDone! Generated {len(generated)} comparison clips.")
    return generated


def main():
    parser = argparse.ArgumentParser(description="Generate pro comparison videos")
    parser.add_argument("video", help="Path to preprocessed video")
    parser.add_argument("--player", default=None,
                        help="Preferred pro player (e.g., federer, djokovic)")
    parser.add_argument("--shot-type", default=None,
                        choices=["forehand", "backhand", "serve"],
                        help="Only generate for this shot type")
    parser.add_argument("--max-clips", type=int, default=None,
                        help="Maximum number of comparison clips")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: exports/{video}/)")
    parser.add_argument("--upload", action="store_true",
                        help="Upload to R2 after generating")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"[ERROR] Video not found: {args.video}")
        sys.exit(1)

    generate_comparisons(
        args.video,
        output_dir=args.output_dir,
        player=args.player,
        shot_type_filter=args.shot_type,
        max_clips=args.max_clips,
        upload=args.upload,
    )


if __name__ == "__main__":
    main()
