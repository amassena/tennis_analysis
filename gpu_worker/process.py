#!/usr/bin/env python3
"""Stateless GPU worker for tennis video processing.

Runs on local GPU machines (windows/tmassena). Downloads video from R2,
processes it, uploads results to R2. Exit code indicates success/failure.

Design principles:
- STATELESS: No local state, all data flows through R2
- FAIL-LOUD: Non-zero exit code on any failure
- OBSERVABLE: Progress written to stdout in parseable format
- IDEMPOTENT: Safe to retry - checks for existing outputs

Usage:
    python process.py --video-key raw/IMG_1234.mov --video-name IMG_1234

Exit codes:
    0: Success
    1: Invalid arguments
    2: Download failed
    3: Preprocessing failed
    4: Pose extraction failed
    5: Shot detection failed
    6: Clip extraction failed
    7: Upload failed
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent

# Exit codes
EXIT_SUCCESS = 0
EXIT_INVALID_ARGS = 1
EXIT_DOWNLOAD_FAILED = 2
EXIT_PREPROCESS_FAILED = 3
EXIT_POSE_FAILED = 4
EXIT_DETECT_FAILED = 5
EXIT_CLIPS_FAILED = 6
EXIT_UPLOAD_FAILED = 7


# ─────────────────────────────────────────────────────────────
# Progress Reporting
# ─────────────────────────────────────────────────────────────

def progress(stage: str, percent: int, message: str = ""):
    """Output progress in parseable format."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"PROGRESS|{ts}|{stage}|{percent}|{message}", flush=True)


def log(level: str, msg: str):
    """Log message to stdout."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [{level}] {msg}", flush=True)


def log_info(msg: str):
    log("INFO", msg)


def log_error(msg: str):
    log("ERROR", msg)


# ─────────────────────────────────────────────────────────────
# R2 Storage
# ─────────────────────────────────────────────────────────────

def load_env() -> dict:
    """Load environment from .env file."""
    env_paths = [
        PROJECT_ROOT / ".env",
        Path("/opt/tennis/.env"),
    ]

    env = {}
    for env_path in env_paths:
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        env[key.strip()] = value.strip().strip('"').strip("'")
            break

    return env


def get_r2_client(env: dict):
    """Get boto3 client for R2."""
    import boto3
    from botocore.config import Config

    account_id = env.get("CF_ACCOUNT_ID")
    access_key = env.get("CF_R2_ACCESS_KEY_ID")
    secret_key = env.get("CF_R2_SECRET_ACCESS_KEY")

    if not all([account_id, access_key, secret_key]):
        raise RuntimeError("R2 credentials not configured in .env")

    return boto3.client(
        "s3",
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(signature_version="s3v4"),
    )


def download_from_r2(client, key: str, local_path: str, bucket: str = "tennis-videos"):
    """Download file from R2."""
    log_info(f"Downloading: {key}")
    client.download_file(bucket, key, local_path)
    size_mb = os.path.getsize(local_path) / (1024 * 1024)
    log_info(f"Downloaded: {size_mb:.1f} MB")


def upload_to_r2(client, local_path: str, key: str, bucket: str = "tennis-videos",
                 content_type: str = None):
    """Upload file to R2."""
    log_info(f"Uploading: {key}")
    extra_args = {}
    if content_type:
        extra_args["ContentType"] = content_type
    client.upload_file(local_path, bucket, key, ExtraArgs=extra_args or None)
    size_mb = os.path.getsize(local_path) / (1024 * 1024)
    log_info(f"Uploaded: {size_mb:.1f} MB")


def r2_key_exists(client, key: str, bucket: str = "tennis-videos") -> bool:
    """Check if key exists in R2."""
    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────
# Pipeline Steps
# ─────────────────────────────────────────────────────────────

def run_cmd(cmd: list, cwd: Path = None, timeout: int = 3600) -> tuple:
    """Run command, return (success, stdout, stderr)."""
    log_info(f"Running: {' '.join(str(c) for c in cmd[:5])}...")

    try:
        result = subprocess.run(
            cmd,
            cwd=cwd or PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", f"Timeout after {timeout}s"
    except Exception as e:
        return False, "", str(e)


def preprocess_video(input_path: Path, output_path: Path) -> bool:
    """Preprocess video (VFR -> CFR, NVENC encoding)."""
    progress("preprocess", 0, "Starting NVENC preprocessing")

    if output_path.exists():
        log_info(f"Preprocessed file already exists: {output_path}")
        progress("preprocess", 100, "Using cached")
        return True

    # Use NVENC if available, fall back to libx264
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-r", "60",  # 60fps CFR
        "-c:v", "h264_nvenc",  # Try NVENC first
        "-preset", "p4",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-an",  # No audio
        str(output_path),
    ]

    success, stdout, stderr = run_cmd(cmd)

    if not success:
        # Fall back to libx264
        log_info("NVENC failed, falling back to libx264")
        cmd[6] = "libx264"
        cmd[7] = "medium"
        success, stdout, stderr = run_cmd(cmd)

    if success:
        progress("preprocess", 100, "Complete")
    else:
        log_error(f"Preprocess failed: {stderr[:500]}")

    return success


def extract_poses(video_path: Path, output_path: Path) -> bool:
    """Extract poses using MediaPipe or YOLO."""
    progress("poses", 0, "Starting pose extraction")

    if output_path.exists():
        log_info(f"Poses file already exists: {output_path}")
        progress("poses", 100, "Using cached")
        return True

    python = sys.executable
    cmd = [
        python,
        str(PROJECT_ROOT / "scripts" / "extract_poses.py"),
        str(video_path),
        "--skip-dead",
    ]

    success, stdout, stderr = run_cmd(cmd, timeout=7200)

    # Check for output file even if returncode is non-zero
    if output_path.exists():
        success = True

    if success:
        progress("poses", 100, "Complete")
    else:
        log_error(f"Pose extraction failed: {stderr[:500]}")

    return success


def detect_shots(poses_path: Path, output_path: Path) -> bool:
    """Run shot detection model on poses."""
    progress("detect", 0, "Running shot detection")

    if output_path.exists():
        log_info(f"Detections file already exists: {output_path}")
        progress("detect", 100, "Using cached")
        return True

    python = sys.executable
    cmd = [
        python,
        str(PROJECT_ROOT / "scripts" / "detect_shots.py"),
        str(poses_path),
        "-o", str(output_path),
    ]

    success, stdout, stderr = run_cmd(cmd)

    if success:
        progress("detect", 100, "Complete")
    else:
        log_error(f"Shot detection failed: {stderr[:500]}")

    return success


def extract_clips(detections_path: Path, video_path: Path,
                  highlights_dir: Path, video_name: str) -> Path:
    """Extract clips and compile highlights."""
    progress("clips", 0, "Extracting clips")

    # Check for existing highlights
    highlight_pattern = f"{video_name}*highlights*.mp4"
    existing = list(highlights_dir.glob(highlight_pattern))
    if existing:
        log_info(f"Highlights already exist: {existing[0]}")
        progress("clips", 100, "Using cached")
        return existing[0]

    python = sys.executable
    cmd = [
        python,
        str(PROJECT_ROOT / "scripts" / "extract_clips.py"),
        "-i", str(detections_path),
        "-v", str(video_path),
        "--highlights",
        "--group-by-type",
    ]

    success, stdout, stderr = run_cmd(cmd, timeout=3600)

    if not success:
        log_error(f"Clip extraction failed: {stderr[:500]}")
        return None

    # Find the highlight file
    highlights = list(highlights_dir.glob(highlight_pattern))
    if highlights:
        progress("clips", 100, "Complete")
        return highlights[0]

    log_error(f"No highlight file found matching {highlight_pattern}")
    return None


def generate_thumbnail(video_path: Path, output_path: Path) -> bool:
    """Generate thumbnail from video."""
    if output_path.exists():
        return True

    # Get duration
    probe_cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]

    result = subprocess.run(probe_cmd, capture_output=True, text=True)
    try:
        duration = float(result.stdout.strip())
        seek_time = duration * 0.1
    except (ValueError, TypeError):
        seek_time = 5

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(seek_time),
        "-i", str(video_path),
        "-vframes", "1",
        "-vf", "scale=480:-1",
        "-q:v", "2",
        str(output_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Stateless GPU video processor")
    parser.add_argument("--video-key", required=True, help="R2 key for input video")
    parser.add_argument("--video-name", required=True, help="Video name (without extension)")
    parser.add_argument("--bucket", default="tennis-videos", help="R2 bucket name")
    parser.add_argument("--work-dir", help="Working directory (default: temp)")
    args = parser.parse_args()

    video_key = args.video_key
    video_name = args.video_name
    bucket = args.bucket

    log_info(f"Processing: {video_name}")
    log_info(f"Input key: {video_key}")

    # Load environment
    env = load_env()
    if not env.get("CF_ACCOUNT_ID"):
        log_error("R2 credentials not found")
        sys.exit(EXIT_INVALID_ARGS)

    # Get R2 client
    try:
        r2_client = get_r2_client(env)
    except Exception as e:
        log_error(f"R2 client init failed: {e}")
        sys.exit(EXIT_DOWNLOAD_FAILED)

    # Check if highlights already exist in R2
    highlights_key = f"highlights/{video_name}_highlights.mp4"
    if r2_key_exists(r2_client, highlights_key, bucket):
        log_info(f"Highlights already exist in R2: {highlights_key}")
        progress("done", 100, "Already processed")
        sys.exit(EXIT_SUCCESS)

    # Work directory
    if args.work_dir:
        work_dir = Path(args.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        cleanup_work_dir = False
    else:
        work_dir = Path(tempfile.mkdtemp(prefix="tennis_"))
        cleanup_work_dir = True

    log_info(f"Work directory: {work_dir}")

    try:
        # === Step 1: Download from R2 ===
        progress("download", 0, "Downloading from R2")
        raw_video = work_dir / f"{video_name}.mov"

        try:
            download_from_r2(r2_client, video_key, str(raw_video), bucket)
            progress("download", 100, "Complete")
        except Exception as e:
            log_error(f"Download failed: {e}")
            sys.exit(EXIT_DOWNLOAD_FAILED)

        # === Step 2: Preprocess ===
        preprocessed = work_dir / f"{video_name}.mp4"
        if not preprocess_video(raw_video, preprocessed):
            sys.exit(EXIT_PREPROCESS_FAILED)

        # === Step 3: Extract poses ===
        # Poses go to project poses dir
        poses_dir = PROJECT_ROOT / "poses"
        poses_dir.mkdir(exist_ok=True)
        poses_file = poses_dir / f"{video_name}.json"

        if not extract_poses(preprocessed, poses_file):
            sys.exit(EXIT_POSE_FAILED)

        # === Step 4: Detect shots ===
        detections_file = work_dir / f"shots_detected_{video_name}.json"
        if not detect_shots(poses_file, detections_file):
            sys.exit(EXIT_DETECT_FAILED)

        # === Step 5: Extract clips and highlights ===
        highlights_dir = PROJECT_ROOT / "highlights"
        highlights_dir.mkdir(exist_ok=True)

        highlights_path = extract_clips(detections_file, preprocessed,
                                        highlights_dir, video_name)
        if not highlights_path:
            sys.exit(EXIT_CLIPS_FAILED)

        # === Step 6: Generate thumbnail ===
        thumb_path = work_dir / f"{video_name}_thumb.jpg"
        generate_thumbnail(preprocessed, thumb_path)

        # === Step 7: Upload results to R2 ===
        progress("upload", 0, "Uploading results")

        try:
            # Upload poses
            poses_key = f"poses/{video_name}.json"
            upload_to_r2(r2_client, str(poses_file), poses_key, bucket)

            # Upload detections
            detections_key = f"shots/{video_name}_detected.json"
            upload_to_r2(r2_client, str(detections_file), detections_key, bucket)

            # Upload highlights
            upload_to_r2(r2_client, str(highlights_path), highlights_key, bucket,
                         content_type="video/mp4")

            # Upload thumbnail
            if thumb_path.exists():
                thumb_key = f"thumbs/{video_name}.jpg"
                upload_to_r2(r2_client, str(thumb_path), thumb_key, bucket,
                             content_type="image/jpeg")

            progress("upload", 100, "Complete")

        except Exception as e:
            log_error(f"Upload failed: {e}")
            sys.exit(EXIT_UPLOAD_FAILED)

        # === Done ===
        progress("done", 100, f"Highlights: {highlights_key}")
        log_info(f"Processing complete: {highlights_key}")
        sys.exit(EXIT_SUCCESS)

    finally:
        # Cleanup
        if cleanup_work_dir:
            import shutil
            try:
                shutil.rmtree(work_dir)
            except Exception:
                pass


if __name__ == "__main__":
    main()
