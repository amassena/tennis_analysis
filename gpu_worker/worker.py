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


def log(msg: str, level: str = "INFO"):
    """Log with timestamp."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [{level}] {msg}", flush=True)


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
    # Check if file already exists locally
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RAW_DIR / filename

    # Also check for case-insensitive match
    for existing in RAW_DIR.iterdir():
        if existing.name.lower() == filename.lower():
            log(f"File already exists: {existing}")
            return existing

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
        f.write(download.content)

    log(f"Downloaded {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    return output_path


def run_pipeline(video_path: Path) -> Path:
    """Run the full processing pipeline on a video."""
    video_name = video_path.stem
    python = sys.executable

    # Step 1: Preprocess (VFR -> CFR)
    log("Step 1: Preprocessing video")
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

    # Step 2: Extract poses
    log("Step 2: Extracting poses")
    poses_file = POSES_DIR / f"{video_name}.json"

    if not poses_file.exists():
        result = subprocess.run(
            [python, str(PROJECT_ROOT / "scripts" / "extract_poses.py"), str(preprocessed)],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Pose extraction failed: {result.stderr}")

    # Step 3: Detect shots
    log("Step 3: Detecting shots")
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

    # Step 4: Extract clips and compile highlights
    log("Step 4: Extracting clips and highlights")

    # Check if highlights already exist
    highlight_pattern = f"{video_name}*highlights*.mp4"
    existing_highlights = list(HIGHLIGHTS_DIR.glob(highlight_pattern))

    if existing_highlights:
        log(f"Highlights already exist: {existing_highlights[0].name}")
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

    # Find the highlight file
    highlights = list(HIGHLIGHTS_DIR.glob(highlight_pattern))

    if not highlights:
        raise RuntimeError(f"No highlight file found matching {highlight_pattern}")

    return highlights[0]


def upload_to_youtube(video_path: Path, title: str = None) -> str:
    """Upload video to YouTube and return URL."""
    if not title:
        date_str = datetime.now().strftime("%Y-%m-%d")
        title = f"Tennis Practice {date_str}"

    python = sys.executable

    result = subprocess.run(
        [
            python,
            str(PROJECT_ROOT / "scripts" / "upload.py"),
            str(video_path),
            "--title", title,
            "--youtube",  # Upload to YouTube (unlisted by default)
        ],
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


def process_job(job: dict) -> str:
    """Process a single job. Returns YouTube URL on success."""
    video_id = job["video_id"]
    filename = job["filename"]
    icloud_asset_id = job["icloud_asset_id"]

    log(f"Processing job {video_id}: {filename}")

    # Update status to processing
    api_request("POST", f"/jobs/{video_id}/processing?worker_id={WORKER_ID}")

    # Download from iCloud
    video_path = download_from_icloud(icloud_asset_id, filename)

    # Run pipeline
    highlight_path = run_pipeline(video_path)

    # Upload to YouTube
    youtube_url = upload_to_youtube(highlight_path)

    log(f"Completed: {youtube_url}")
    return youtube_url


def worker_loop(coordinator_url: str, worker_id: str, poll_interval: int, once: bool = False):
    """Main worker loop."""
    # Update globals for other functions
    global COORDINATOR_URL, WORKER_ID, POLL_INTERVAL
    COORDINATOR_URL = coordinator_url
    WORKER_ID = worker_id
    POLL_INTERVAL = poll_interval

    log(f"Starting GPU worker: {WORKER_ID}")
    log(f"Coordinator: {COORDINATOR_URL}")
    log(f"Poll interval: {POLL_INTERVAL}s")

    while True:
        try:
            job = claim_job()

            if job:
                try:
                    youtube_url = process_job(job)
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
    args = parser.parse_args()

    coordinator_url = args.coordinator or COORDINATOR_URL
    worker_id = args.worker_id or WORKER_ID
    poll_interval = args.poll_interval or POLL_INTERVAL

    worker_loop(coordinator_url, worker_id, poll_interval, once=args.once)


if __name__ == "__main__":
    main()
