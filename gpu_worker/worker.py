"""GPU Worker - polls coordinator for jobs and processes videos locally.

Each GPU machine runs this worker. It:
1. Polls the coordinator API for pending jobs
2. Claims a job
3. Downloads the video from iCloud
4. Runs the full pipeline (preprocess -> poses -> detect -> clips -> highlights)
5. Uploads to YouTube
6. Reports completion to coordinator
"""

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path


# Configuration
COORDINATOR_URL = os.environ.get("COORDINATOR_URL", "http://localhost:8080")
POLL_INTERVAL = int(os.environ.get("POLL_INTERVAL", "60"))  # seconds
WORKER_ID = os.environ.get("WORKER_ID", socket.gethostname())

# Project paths (auto-detect)
PROJECT_ROOT = Path(__file__).parent.parent
RAW_DIR = PROJECT_ROOT / "raw"
PREPROCESSED_DIR = PROJECT_ROOT / "preprocessed"
POSES_DIR = PROJECT_ROOT / "poses"
CLIPS_DIR = PROJECT_ROOT / "clips"
HIGHLIGHTS_DIR = PROJECT_ROOT / "highlights"
THUMBS_DIR = PROJECT_ROOT / "thumbs"


def log(msg: str, level: str = "INFO"):
    """Log with timestamp."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [{level}] {msg}", flush=True)


def report_stage(video_id: str, stage: str, progress: float = None, message: str = None):
    """Report current processing stage to coordinator."""
    try:
        data = {"stage": stage}
        if progress is not None:
            data["progress"] = progress
        if message:
            data["message"] = message
        api_request("POST", f"/jobs/{video_id}/stage?worker_id={WORKER_ID}", data)
    except Exception as e:
        log(f"Failed to report stage: {e}", "WARN")


def api_request(method: str, endpoint: str, data: dict = None) -> dict:
    """Make HTTP request to coordinator API."""
    url = f"{COORDINATOR_URL}{endpoint}"

    if data:
        body = json.dumps(data).encode()
        headers = {"Content-Type": "application/json"}
    else:
        body = None
        headers = {}

    req = urllib.request.Request(url, data=body, headers=headers, method=method)

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        error_body = e.read().decode() if e.fp else ""
        log(f"API error: {e.code} {error_body}", "ERROR")
        raise
    except urllib.error.URLError as e:
        log(f"Connection error: {e.reason}", "ERROR")
        raise


def claim_job() -> dict:
    """Try to claim a pending job."""
    # Get pending jobs
    try:
        pending = api_request("GET", "/jobs/pending")
    except Exception:
        return None

    if not pending.get("jobs"):
        return None

    # Try to claim first available
    for job in pending["jobs"]:
        video_id = job["video_id"]
        try:
            result = api_request("POST", f"/jobs/{video_id}/claim?worker_id={WORKER_ID}")
            if result.get("success"):
                log(f"Claimed job {video_id}: {job['filename']}")
                return result["job"]
        except Exception:
            continue

    return None


def download_from_icloud(icloud_asset_id: str, filename: str) -> Path:
    """Download video from iCloud to raw directory."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Track downloaded assets to handle same-name files with different content
    asset_map_file = PROJECT_ROOT / "config" / "downloaded_assets.json"
    asset_map_file.parent.mkdir(parents=True, exist_ok=True)

    asset_map = {}
    if asset_map_file.exists():
        try:
            with open(asset_map_file) as f:
                asset_map = json.load(f)
        except Exception:
            asset_map = {}

    # Check if this specific asset was already downloaded
    if icloud_asset_id in asset_map:
        local_path = Path(asset_map[icloud_asset_id])
        if local_path.exists():
            log(f"Asset already downloaded: {local_path}")
            return local_path

    # Check for existing file with same name (case-insensitive)
    existing_file = None
    for existing in RAW_DIR.iterdir():
        if existing.name.lower() == filename.lower():
            existing_file = existing
            break

    if existing_file:
        # Check if this file is tracked to a DIFFERENT asset
        tracked_assets = {v: k for k, v in asset_map.items()}  # path -> asset_id
        existing_asset = tracked_assets.get(str(existing_file))

        if existing_asset is None:
            # Legacy file, not tracked - assume it's this asset
            log(f"File already exists (legacy): {existing_file}")
            asset_map[icloud_asset_id] = str(existing_file)
            with open(asset_map_file, "w") as f:
                json.dump(asset_map, f, indent=2)
            return existing_file
        elif existing_asset == icloud_asset_id:
            # Same asset, use it
            log(f"File already exists: {existing_file}")
            return existing_file
        else:
            # Different asset with same filename! Need unique name
            stem = Path(filename).stem
            ext = Path(filename).suffix
            unique_name = f"{stem}_{icloud_asset_id[:8]}{ext}"
            log(f"Name collision - using unique name: {unique_name}")
            filename = unique_name

    # Import iCloud library (pyicloud)
    try:
        from pyicloud import PyiCloudService
    except ImportError:
        log("pyicloud not installed. Install with: pip install pyicloud", "ERROR")
        raise

    # Load credentials
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        raise RuntimeError(".env file not found with iCloud credentials")

    # Parse .env file
    env_vars = {}
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                key, value = line.split("=", 1)
                env_vars[key.strip()] = value.strip().strip('"').strip("'")

    apple_id = env_vars.get("ICLOUD_USER") or env_vars.get("ICLOUD_USERNAME")
    password = env_vars.get("ICLOUD_PASS") or env_vars.get("ICLOUD_PASSWORD")

    if not apple_id or not password:
        raise RuntimeError("ICLOUD_USER(NAME) and ICLOUD_PASS(WORD) must be in .env")

    # Use shared session directory for cookie persistence
    cookie_dir = PROJECT_ROOT / "config" / "icloud_session"
    cookie_dir.mkdir(parents=True, exist_ok=True)

    log(f"Connecting to iCloud as {apple_id}")
    api = PyiCloudService(apple_id, password, cookie_directory=str(cookie_dir))

    if api.requires_2fa:
        raise RuntimeError("iCloud requires 2FA - authenticate manually first")

    # Find the asset by searching all photos
    log(f"Looking for asset {icloud_asset_id}")
    photo = None

    # Search through all photos (more reliable than album iteration)
    log("Searching all photos...")
    for p in api.photos.all:
        if p.id == icloud_asset_id:
            photo = p
            log(f"Found: {p.filename}")
            break

    if not photo:
        raise RuntimeError(f"Asset {icloud_asset_id} not found in iCloud")

    # Download
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RAW_DIR / filename

    log(f"Downloading to {output_path}")
    download = photo.download()

    with open(output_path, "wb") as f:
        # Handle both Response object (.content) and raw bytes
        if hasattr(download, 'content'):
            f.write(download.content)
        else:
            f.write(download)

    log(f"Downloaded {output_path.stat().st_size / 1024 / 1024:.1f} MB")

    # Track the downloaded asset
    asset_map[icloud_asset_id] = str(output_path)
    with open(asset_map_file, "w") as f:
        json.dump(asset_map, f, indent=2)

    return output_path


def generate_thumbnail(video_path: Path, output_path: Path = None) -> Path:
    """Generate a thumbnail from a video at ~10% into the video."""
    THUMBS_DIR.mkdir(parents=True, exist_ok=True)

    if output_path is None:
        output_path = THUMBS_DIR / f"{video_path.stem}.jpg"

    if output_path.exists():
        log(f"Thumbnail already exists: {output_path}")
        return output_path

    # Get video duration
    probe_cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path)
    ]
    result = subprocess.run(probe_cmd, capture_output=True, text=True)

    try:
        duration = float(result.stdout.strip())
        # Seek to 10% into video for a good frame
        seek_time = duration * 0.1
    except (ValueError, TypeError):
        seek_time = 5  # Default to 5 seconds if duration unknown

    # Extract frame
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(seek_time),
        "-i", str(video_path),
        "-vframes", "1",
        "-vf", "scale=480:-1",  # 480px wide, maintain aspect ratio
        "-q:v", "2",  # High quality JPEG
        str(output_path)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log(f"Thumbnail generation failed: {result.stderr}", "WARN")
        return None

    log(f"Generated thumbnail: {output_path}")
    return output_path


def upload_thumbnail_to_r2(thumb_path: Path, video_name: str) -> bool:
    """Upload thumbnail to R2 storage."""
    try:
        import boto3
        from botocore.config import Config
    except ImportError:
        log("boto3 not installed, skipping R2 upload", "WARN")
        return False

    # Load R2 credentials from .env
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        log("No .env file for R2 credentials", "WARN")
        return False

    env_vars = {}
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                key, value = line.split("=", 1)
                env_vars[key.strip()] = value.strip().strip('"').strip("'")

    account_id = env_vars.get("CF_ACCOUNT_ID")
    access_key = env_vars.get("CF_R2_ACCESS_KEY_ID")
    secret_key = env_vars.get("CF_R2_SECRET_ACCESS_KEY")
    bucket = env_vars.get("R2_BUCKET", "tennis-videos")

    if not all([account_id, access_key, secret_key]):
        log("R2 credentials not configured in .env", "WARN")
        return False

    try:
        client = boto3.client(
            "s3",
            endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
            region_name="us-east-1",
        )

        key = f"thumbs/{video_name}.jpg"
        client.upload_file(str(thumb_path), bucket, key, ExtraArgs={"ContentType": "image/jpeg"})
        log(f"Uploaded thumbnail to R2: {key}")
        return True
    except Exception as e:
        log(f"R2 upload failed: {e}", "WARN")
        return False


def run_pipeline(video_path: Path) -> Path:
    """Run the full processing pipeline on a video (no stage reporting)."""
    return run_pipeline_with_stages(video_path, None)


def run_pipeline_with_stages(video_path: Path, video_id: str = None) -> Path:
    """Run the full processing pipeline on a video with stage reporting."""
    video_name = video_path.stem
    python = sys.executable

    def stage(name: str, progress: float = 0, msg: str = None):
        if video_id:
            report_stage(video_id, name, progress, msg)

    # Step 1: Preprocess (VFR -> CFR)
    log("Step 1: Preprocessing video")
    stage("preprocessing", 0, "Starting NVENC preprocessing")
    preprocessed = PREPROCESSED_DIR / f"{video_name}.mp4"

    if not preprocessed.exists():
        result = subprocess.run(
            [python, str(PROJECT_ROOT / "preprocess_nvenc.py"), video_path.name],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Preprocess failed: {result.stderr}")
    stage("preprocessing", 100, "Preprocessing complete")

    # Step 2: Generate and upload thumbnail
    log("Step 2: Generating thumbnail")
    stage("thumbnail", 0, "Generating thumbnail")
    thumb_path = generate_thumbnail(preprocessed)
    if thumb_path:
        upload_thumbnail_to_r2(thumb_path, video_name)
    stage("thumbnail", 100, "Thumbnail complete")

    # Step 3: Pre-scan for dead sections
    log("Step 3: Pre-scanning for dead sections")
    stage("prescan", 0, "Scanning for dead sections")
    # Pre-scan is part of pose extraction with --skip-dead

    # Step 4: Extract poses
    log("Step 4: Extracting poses")
    stage("poses", 0, "Starting pose extraction")
    poses_file = POSES_DIR / f"{video_name}.json"

    if not poses_file.exists():
        result = subprocess.run(
            [python, str(PROJECT_ROOT / "scripts" / "extract_poses.py"),
             str(preprocessed), "--skip-dead"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Pose extraction failed: {result.stderr}")
    stage("poses", 100, "Pose extraction complete")

    # Step 5: Detect shots
    log("Step 5: Detecting shots")
    stage("detection", 0, "Running shot detection model")
    shots_file = PROJECT_ROOT / f"shots_detected_{video_name}.json"

    if not shots_file.exists():
        result = subprocess.run(
            [
                python,
                str(PROJECT_ROOT / "scripts" / "detect_shots.py"),
                str(poses_file),
                "-o", str(shots_file),
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Shot detection failed: {result.stderr}")
    stage("detection", 100, "Shot detection complete")

    # Step 6: Extract clips and compile highlights
    log("Step 6: Extracting clips and highlights")
    stage("clips", 0, "Extracting clips")

    # Check if highlights already exist
    highlight_pattern = f"{video_name}*highlights*.mp4"
    existing_highlights = list(HIGHLIGHTS_DIR.glob(highlight_pattern))

    if existing_highlights:
        log(f"Highlights already exist: {existing_highlights[0].name}")
        stage("clips", 100, "Using existing highlights")
        return existing_highlights[0]

    result = subprocess.run(
        [
            python,
            str(PROJECT_ROOT / "scripts" / "extract_clips.py"),
            "-i", str(shots_file),
            "-v", str(preprocessed),
            "--highlights",
            "--group-by-type",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Clip extraction failed: {result.stderr}")
    stage("clips", 100, "Clip extraction complete")

    # Find the highlight file
    highlights = list(HIGHLIGHTS_DIR.glob(highlight_pattern))

    if not highlights:
        raise RuntimeError(f"No highlight file found matching {highlight_pattern}")

    return highlights[0]


def upload_to_youtube(video_path: Path, title: str = None, dry_run: bool = False) -> str:
    """Upload video to YouTube and return URL.

    If dry_run=True, simulates the upload without actually uploading.
    """
    if not title:
        date_str = datetime.now().strftime("%Y-%m-%d")
        title = f"Tennis Practice {date_str}"

    python = sys.executable

    cmd = [
        python,
        str(PROJECT_ROOT / "scripts" / "upload.py"),
        str(video_path),
        "--title", title,
        "--youtube",  # Upload to YouTube (unlisted by default)
    ]

    if dry_run:
        cmd.append("--dry-run")
        log("YouTube upload: DRY-RUN mode")

    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"YouTube upload failed: {result.stderr}")

    # Extract URL from output - look for https://youtu.be/... or https://youtube.com/...
    for line in result.stdout.split("\n"):
        for word in line.split():
            word = word.strip()
            if word.startswith("https://youtu"):
                return word

    raise RuntimeError("Could not find YouTube URL in output")


def process_job(job: dict, skip_youtube: bool = False, youtube_dry_run: bool = False) -> str:
    """Process a single job. Returns YouTube URL or highlight path on success.

    Args:
        job: Job dict with video_id, filename, icloud_asset_id
        skip_youtube: If True, skip YouTube upload entirely
        youtube_dry_run: If True, simulate YouTube upload without actually uploading
    """
    video_id = job["video_id"]
    filename = job["filename"]
    icloud_asset_id = job["icloud_asset_id"]

    log(f"Processing job {video_id}: {filename}")

    # Update status to processing
    api_request("POST", f"/jobs/{video_id}/processing?worker_id={WORKER_ID}")

    # Download from iCloud
    report_stage(video_id, "downloading", 0, f"Downloading {filename}")
    video_path = download_from_icloud(icloud_asset_id, filename)
    report_stage(video_id, "downloading", 100, "Download complete")

    # Run pipeline
    highlight_path = run_pipeline_with_stages(video_path, video_id)

    # Upload to YouTube
    if skip_youtube:
        report_stage(video_id, "done", 100, "Complete (YouTube skipped)")
        log(f"Completed: {highlight_path} (YouTube upload skipped)")
        return f"file://{highlight_path}"

    report_stage(video_id, "uploading", 0, "Uploading to YouTube")
    youtube_url = upload_to_youtube(highlight_path, dry_run=youtube_dry_run)
    report_stage(video_id, "done", 100, f"Complete: {youtube_url}")
    log(f"Completed: {youtube_url}")
    return youtube_url


def worker_loop(coordinator_url: str, worker_id: str, poll_interval: int,
                once: bool = False, skip_youtube: bool = False, youtube_dry_run: bool = False):
    """Main worker loop.

    Args:
        coordinator_url: URL of the coordinator API
        worker_id: Identifier for this worker
        poll_interval: Seconds between polling for jobs
        once: If True, process one job and exit
        skip_youtube: If True, skip YouTube upload entirely
        youtube_dry_run: If True, simulate YouTube upload without actually uploading
    """
    # Update globals for other functions
    global COORDINATOR_URL, WORKER_ID, POLL_INTERVAL
    COORDINATOR_URL = coordinator_url
    WORKER_ID = worker_id
    POLL_INTERVAL = poll_interval

    log(f"Starting GPU worker: {WORKER_ID}")
    log(f"Coordinator: {COORDINATOR_URL}")
    log(f"Poll interval: {POLL_INTERVAL}s")
    if skip_youtube:
        log("YouTube upload: DISABLED")
    elif youtube_dry_run:
        log("YouTube upload: DRY-RUN MODE (no actual uploads)")

    while True:
        try:
            job = claim_job()

            if job:
                try:
                    youtube_url = process_job(job, skip_youtube=skip_youtube, youtube_dry_run=youtube_dry_run)
                    api_request(
                        "POST",
                        f"/jobs/{job['video_id']}/complete?worker_id={WORKER_ID}",
                        {"success": True, "youtube_url": youtube_url},
                    )
                except Exception as e:
                    log(f"Job failed: {e}", "ERROR")
                    api_request(
                        "POST",
                        f"/jobs/{job['video_id']}/complete?worker_id={WORKER_ID}",
                        {"success": False, "error_message": str(e)},
                    )
            else:
                log("No pending jobs", "DEBUG")

        except Exception as e:
            log(f"Error in worker loop: {e}", "ERROR")

        if once:
            break

        time.sleep(POLL_INTERVAL)


def main():
    parser = argparse.ArgumentParser(description="GPU Worker for tennis pipeline")
    parser.add_argument(
        "--coordinator",
        default=None,
        help="Coordinator API URL",
    )
    parser.add_argument(
        "--worker-id",
        default=None,
        help="Worker identifier (default: hostname)",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=None,
        help="Seconds between polls",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Process one job and exit",
    )
    parser.add_argument(
        "--skip-youtube",
        action="store_true",
        help="Skip YouTube upload entirely (just generate highlights)",
    )
    parser.add_argument(
        "--youtube-dry-run",
        action="store_true",
        help="Simulate YouTube upload without actually uploading (for testing)",
    )
    args = parser.parse_args()

    coordinator_url = args.coordinator or COORDINATOR_URL
    worker_id = args.worker_id or WORKER_ID
    poll_interval = args.poll_interval or POLL_INTERVAL
    skip_youtube = args.skip_youtube
    youtube_dry_run = args.youtube_dry_run

    worker_loop(coordinator_url, worker_id, poll_interval,
                once=args.once, skip_youtube=skip_youtube, youtube_dry_run=youtube_dry_run)


if __name__ == "__main__":
    main()
