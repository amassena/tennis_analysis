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
WEBSITE_URL = os.environ.get("WEBSITE_URL", "https://tennis.playfullife.com")
UPLOAD_PASSWORD = os.environ.get("UPLOAD_PASSWORD", "")

# Project paths (auto-detect)
PROJECT_ROOT = Path(__file__).parent.parent
RAW_DIR = PROJECT_ROOT / "raw"
PREPROCESSED_DIR = PROJECT_ROOT / "preprocessed"
POSES_DIR = PROJECT_ROOT / "poses_full_videos"
DETECTIONS_DIR = PROJECT_ROOT / "detections"
EXPORTS_DIR = PROJECT_ROOT / "exports"
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


def report_queue_status(upload_id: str, status: str = None, stage: str = None,
                        progress: int = None, error: str = None, video_url: str = None):
    """Report processing status to the website queue (/api/status/:id/update).

    Args:
        upload_id: The upload ID from the website (e.g. 'mnnlku4e_0za6yz')
        status: Overall status: pending, processing, complete, failed
        stage: Current stage: downloading, preprocessing, extracting_poses,
               detecting_shots, exporting, uploading_results
        progress: Percentage 0-100
        error: Error message (for failed status)
        video_url: Link to finished video (for complete status)
    """
    if not upload_id or not UPLOAD_PASSWORD:
        return

    data = {"password": UPLOAD_PASSWORD}
    if status:
        data["status"] = status
    if stage:
        data["stage"] = stage
    if progress is not None:
        data["progress"] = progress
    if error:
        data["error"] = error
    if video_url:
        data["video_url"] = video_url

    try:
        body = json.dumps(data).encode()
        url = f"{WEBSITE_URL}/api/status/{upload_id}/update"
        req = urllib.request.Request(
            url, data=body,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            resp.read()
        log(f"Queue status: {stage or status} {progress or ''}%")
    except Exception as e:
        log(f"Queue status update failed: {e}", "WARN")


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

    # Find the asset — try smart albums first (much faster), then full scan
    log(f"Looking for asset {icloud_asset_id}")
    photo = None

    # Try Slo-mo smart album first (most pipeline videos are slo-mo)
    for album_name in ["Slo-mo", "Videos", "Recents"]:
        try:
            album = api.photos.albums.get(album_name)
            if not album:
                continue
            log(f"Searching '{album_name}' album...")
            for p in album:
                if p.id == icloud_asset_id:
                    photo = p
                    log(f"Found in '{album_name}': {p.filename}")
                    break
            if photo:
                break
        except Exception as e:
            log(f"Error searching '{album_name}': {e}")

    # Fallback: full library scan
    if not photo:
        log("Searching all photos (this may take a while)...")
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


def _get_icloud_created(video_name: str) -> str:
    """Query iCloud for a video's creation date. Returns ISO timestamp or empty string."""
    try:
        from pyicloud import PyiCloudService
        env_path = PROJECT_ROOT / ".env"
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
            return ""
        cookie_dir = PROJECT_ROOT / "config" / "icloud_session"
        cookie_dir.mkdir(parents=True, exist_ok=True)
        api = PyiCloudService(apple_id, password, cookie_directory=str(cookie_dir))
        if api.requires_2fa:
            return ""
        target = f"{video_name}.MOV"
        for album_name in ["Slo-mo", "Videos"]:
            try:
                album = api.photos.albums.get(album_name)
                if not album:
                    continue
                for asset in album:
                    if asset.filename and asset.filename.upper() == target.upper():
                        dt = asset.created
                        if dt:
                            log(f"iCloud creation date for {video_name}: {dt}")
                            return dt.isoformat()
            except Exception:
                continue
    except Exception as e:
        log(f"iCloud date lookup failed for {video_name}: {e}", "WARN")
    return ""


def _get_r2_client():
    """Get boto3 R2 client and bucket name. Returns (client, bucket) or (None, None)."""
    try:
        import boto3
        from botocore.config import Config
    except ImportError:
        log("boto3 not installed", "WARN")
        return None, None

    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return None, None

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
        return None, None

    client = boto3.client(
        "s3",
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
        region_name="us-east-1",
    )
    return client, bucket


def _upload_file_to_r2(local_path: str, r2_key: str, content_type: str = "application/octet-stream") -> bool:
    """Upload a file to R2. Returns True on success."""
    client, bucket = _get_r2_client()
    if not client:
        return False
    try:
        client.upload_file(str(local_path), bucket, r2_key, ExtraArgs={"ContentType": content_type})
        return True
    except Exception as e:
        log(f"R2 upload failed ({r2_key}): {e}", "WARN")
        return False


def upload_thumbnail_to_r2(thumb_path: Path, video_name: str) -> bool:
    """Upload thumbnail to R2 storage."""
    key = f"thumbs/{video_name}.jpg"
    if _upload_file_to_r2(str(thumb_path), key, "image/jpeg"):
        log(f"Uploaded thumbnail to R2: {key}")
        return True
    return False


def run_pipeline(video_path: Path) -> Path:
    """Run the full processing pipeline on a video (no stage reporting)."""
    return run_pipeline_with_stages(video_path, None)


def run_pipeline_with_stages(video_path: Path, video_id: str = None,
                             upload_id: str = None) -> Path:
    """Run the full processing pipeline on a video with stage reporting.

    Args:
        video_path: Path to raw video file
        video_id: Coordinator job ID (for Hetzner coordinator)
        upload_id: Website upload ID (for gallery queue status)
    """
    video_name = video_path.stem
    python = sys.executable

    def stage(name: str, progress: float = 0, msg: str = None):
        if video_id:
            report_stage(video_id, name, progress, msg)
        if upload_id:
            # Map internal stage names to website queue stages
            queue_stage_map = {
                "preprocessing": "preprocessing",
                "thumbnail": "preprocessing",
                "prescan": "preprocessing",
                "poses": "extracting_poses",
                "detection": "detecting_shots",
                "clips": "exporting",
                "uploading": "uploading_results",
            }
            queue_stage = queue_stage_map.get(name, name)
            report_queue_status(upload_id, status="processing",
                                stage=queue_stage, progress=int(progress))

    # Step 1: Preprocess (VFR -> CFR)
    log("Step 1: Preprocessing video")
    stage("preprocessing", 0, "Starting NVENC preprocessing")
    preprocessed = PREPROCESSED_DIR / f"{video_name}.mp4"

    if not preprocessed.exists():
        result = subprocess.run(
            [python, str(PROJECT_ROOT / "scripts" / "preprocess_nvenc.py"), video_path.name],
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

    # Step 5: Detect shots (sequence CNN)
    log("Step 5: Detecting shots (sequence CNN)")
    stage("detection", 0, "Running sequence CNN detection")
    DETECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    det_file = DETECTIONS_DIR / f"{video_name}_fused_detections.json"

    if not det_file.exists():
        result = subprocess.run(
            [
                python,
                str(PROJECT_ROOT / "scripts" / "detect_shots_sequence.py"),
                str(preprocessed),
                "--threshold", "0.90",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Shot detection failed: {result.stderr}")
    stage("detection", 100, "Shot detection complete")

    # Step 5b: Upload video metadata to R2 (so gallery index works from any machine)
    try:
        det_path = DETECTIONS_DIR / f"{video_name}_fused_detections.json"
        if not det_path.exists():
            det_path = DETECTIONS_DIR / f"{video_name}_fused.json"
        if det_path.exists():
            import json as _json
            with open(det_path) as _f:
                det_data = _json.load(_f)
            # Extract creation date — try raw video first, then iCloud
            created = ""
            raw_path = RAW_DIR / f"{video_name}.MOV"
            if not raw_path.exists():
                raw_path = RAW_DIR / f"{video_name}.mov"
            if raw_path.exists():
                try:
                    r = subprocess.run(
                        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(raw_path)],
                        capture_output=True, text=True, timeout=10)
                    tags = _json.loads(r.stdout).get("format", {}).get("tags", {})
                    created = tags.get("com.apple.quicktime.creationdate", tags.get("creation_time", ""))
                except Exception:
                    pass
            # Fallback: query iCloud for the asset creation date
            if not created:
                created = _get_icloud_created(video_name)
            meta = {
                "duration": det_data.get("duration", 0),
                "shots": len(det_data.get("detections", [])),
                "breakdown": {},
                "created": created or det_data.get("created", ""),
            }
            for d in det_data.get("detections", []):
                st = d.get("shot_type", "unknown")
                meta["breakdown"][st] = meta["breakdown"].get(st, 0) + 1
            meta_json = _json.dumps(meta).encode()
            # Upload to R2
            from pathlib import Path as _P
            import tempfile as _tf
            with _tf.NamedTemporaryFile(suffix=".json", delete=False, mode="wb") as mf:
                mf.write(meta_json)
                meta_tmp = mf.name
            _upload_file_to_r2(meta_tmp, f"highlights/{video_name}/meta.json", "application/json")
            os.unlink(meta_tmp)
            log(f"Uploaded metadata to R2: highlights/{video_name}/meta.json")
    except Exception as e:
        log(f"Metadata upload failed (non-fatal): {e}", "WARN")

    # Step 6: Export videos and upload to R2
    log("Step 6: Exporting videos to R2")
    stage("clips", 0, "Exporting video formats + uploading to R2")
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [
            python,
            str(PROJECT_ROOT / "scripts" / "export_videos.py"),
            str(preprocessed),
            "--types", "timeline", "rally", "grouped",
            "--slow-motion",
            "--upload",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        log(f"Export output: {result.stdout[-500:]}", "WARN")
        log(f"Export stderr: {result.stderr[-500:]}", "WARN")
        # Don't fail entirely — partial exports are still useful
    else:
        log("Export and R2 upload complete")
    stage("clips", 100, "Export complete")

    # Step 7: Update R2 gallery index
    log("Step 7: Updating gallery index")
    stage("uploading", 0, "Updating gallery index")
    result = subprocess.run(
        [python, str(PROJECT_ROOT / "scripts" / "update_r2_index.py")],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        log(f"Index update failed: {result.stderr[-300:]}", "WARN")
    else:
        log("Gallery index updated")
    stage("uploading", 100, "Gallery index updated")

    # Return best export file for YouTube upload (prefer grouped)
    export_dir = EXPORTS_DIR / video_name
    for candidate in [
        export_dir / f"{video_name}_grouped.mp4",
        export_dir / f"{video_name}_highlights.mp4",
        export_dir / f"{video_name}_rally.mp4",
    ]:
        if candidate.exists():
            return candidate
    return preprocessed


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


def _get_detection_stats(video_name: str) -> dict:
    """Read detection JSON and return stats for email notification."""
    det_file = DETECTIONS_DIR / f"{video_name}_fused_detections.json"
    if not det_file.exists():
        return {"total_clips": 0, "shot_counts": {}, "duration_seconds": 0}

    with open(det_file) as f:
        data = json.load(f)

    detections = data.get("detections", [])
    shot_counts = {}
    for d in detections:
        st = d.get("shot_type", "unknown")
        shot_counts[st] = shot_counts.get(st, 0) + 1

    return {
        "total_clips": len(detections),
        "shot_counts": shot_counts,
        "duration_seconds": data.get("duration", 0),
    }


def _send_notification(func_name: str, *args, **kwargs):
    """Import and call an email_notify function. Fails silently."""
    try:
        # Lazy import to avoid hard dependency
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        mod_path = PROJECT_ROOT / "scripts" / "email_notify.py"
        if not mod_path.exists():
            return
        import importlib.util
        spec = importlib.util.spec_from_file_location("email_notify", str(mod_path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        fn = getattr(mod, func_name)
        fn(*args, **kwargs)
    except Exception as e:
        log(f"Notification ({func_name}) failed: {e}", "WARN")


def process_job(job: dict, skip_youtube: bool = False, youtube_dry_run: bool = False,
                upload_id: str = None) -> str:
    """Process a single job. Returns YouTube URL or highlight path on success.

    Args:
        job: Job dict with video_id, filename, icloud_asset_id
        skip_youtube: If True, skip YouTube upload entirely
        youtube_dry_run: If True, simulate YouTube upload without actually uploading
        upload_id: Website upload ID for queue status reporting
    """
    video_id = job["video_id"]
    filename = job["filename"]
    icloud_asset_id = job["icloud_asset_id"]
    video_name = Path(filename).stem

    log(f"Processing job {video_id}: {filename}")

    # Report to website queue if applicable
    if upload_id:
        report_queue_status(upload_id, status="processing", stage="downloading", progress=0)

    # Notifications disabled for intermediate steps — only notify on completion

    # Update status to processing
    api_request("POST", f"/jobs/{video_id}/processing?worker_id={WORKER_ID}")

    # Download from iCloud
    report_stage(video_id, "downloading", 0, f"Downloading {filename}")
    video_path = download_from_icloud(icloud_asset_id, filename)
    report_stage(video_id, "downloading", 100, "Download complete")
    if upload_id:
        report_queue_status(upload_id, stage="downloading", progress=100)

    # Run pipeline
    highlight_path = run_pipeline_with_stages(video_path, video_id, upload_id=upload_id)

    # Upload to YouTube
    if skip_youtube:
        report_stage(video_id, "done", 100, "Complete (YouTube skipped)")
        log(f"Completed: {highlight_path} (YouTube upload skipped)")
        return f"file://{highlight_path}"

    report_stage(video_id, "uploading", 0, "Uploading to YouTube")
    highlights_url = upload_to_youtube(highlight_path, dry_run=youtube_dry_run)
    report_stage(video_id, "done", 100, f"Complete: {highlights_url}")
    log(f"Completed: {highlights_url}")
    return highlights_url


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
                video_name = Path(job.get("filename", "")).stem
                # Check if this job has a linked website upload ID
                job_upload_id = job.get("upload_id")
                try:
                    highlights_url = process_job(
                        job, skip_youtube=skip_youtube,
                        youtube_dry_run=youtube_dry_run,
                        upload_id=job_upload_id,
                    )
                    api_request(
                        "POST",
                        f"/jobs/{job['video_id']}/complete?worker_id={WORKER_ID}",
                        {"success": True, "highlights_url": highlights_url},
                    )
                    # Notify upload complete with direct video links
                    gallery_url = f"{WEBSITE_URL}/"
                    base_url = f"{WEBSITE_URL}/{video_name}"
                    video_links = {
                        "Timeline": f"{base_url}/{video_name}_timeline.mp4",
                        "Rally": f"{base_url}/{video_name}_rally.mp4",
                        "Rally (Slow-Mo)": f"{base_url}/{video_name}_rally_slowmo.mp4",
                        "Grouped": f"{base_url}/{video_name}_grouped.mp4",
                        "Grouped (Slow-Mo)": f"{base_url}/{video_name}_grouped_slowmo.mp4",
                    }
                    stats = _get_detection_stats(video_name)
                    _send_notification(
                        "notify_upload_complete",
                        video_name, video_links, stats,
                    )
                    # Mark website queue as complete
                    if job_upload_id:
                        report_queue_status(
                            job_upload_id, status="complete", stage="complete",
                            progress=100, video_url=gallery_url,
                        )
                except Exception as e:
                    log(f"Job failed: {e}", "ERROR")
                    api_request(
                        "POST",
                        f"/jobs/{job['video_id']}/complete?worker_id={WORKER_ID}",
                        {"success": False, "error_message": str(e)},
                    )
                    # Notify failure
                    _send_notification(
                        "notify_processing_failed",
                        video_name, WORKER_ID, str(e),
                    )
                    # Mark website queue as failed
                    if job_upload_id:
                        report_queue_status(
                            job_upload_id, status="failed",
                            error=str(e)[:200],
                        )
            else:
                log("No pending jobs", "DEBUG")

        except Exception as e:
            log(f"Error in worker loop: {e}", "ERROR")

        if once:
            break

        time.sleep(POLL_INTERVAL)


def process_local_video(video_path: str, upload_id: str = None):
    """Process a local video file through the full pipeline.

    This is for processing videos that were uploaded via the website
    or added manually, without needing the coordinator.

    Args:
        video_path: Path to the video file (raw or preprocessed)
        upload_id: Website upload ID for queue status updates
    """
    path = Path(video_path)
    if not path.exists():
        log(f"Video not found: {path}", "ERROR")
        return

    video_name = path.stem
    log(f"Processing local video: {video_name}")

    if upload_id:
        report_queue_status(upload_id, status="processing", stage="preprocessing", progress=0)

    try:
        highlight_path = run_pipeline_with_stages(path, upload_id=upload_id)

        # Send email notification
        gallery_url = f"{WEBSITE_URL}/"
        base_url = f"{WEBSITE_URL}/{video_name}"
        video_links = {
            "Timeline": f"{base_url}/{video_name}_timeline.mp4",
            "Rally": f"{base_url}/{video_name}_rally.mp4",
            "Rally (Slow-Mo)": f"{base_url}/{video_name}_rally_slowmo.mp4",
            "By Shot Type": f"{base_url}/{video_name}_grouped.mp4",
            "By Shot Type (Slow-Mo)": f"{base_url}/{video_name}_grouped_slowmo.mp4",
        }
        stats = _get_detection_stats(video_name)
        _send_notification("notify_upload_complete", video_name, video_links, stats)

        if upload_id:
            report_queue_status(
                upload_id, status="complete", stage="complete",
                progress=100, video_url=gallery_url,
            )

        log(f"Done: {highlight_path}")
    except Exception as e:
        log(f"Processing failed: {e}", "ERROR")
        _send_notification("notify_processing_failed", video_name, WORKER_ID, str(e))
        if upload_id:
            report_queue_status(upload_id, status="failed", error=str(e)[:200])
        raise


def main():
    parser = argparse.ArgumentParser(description="GPU Worker for tennis pipeline")
    subparsers = parser.add_subparsers(dest="command")

    # 'run' subcommand — process a local video
    run_parser = subparsers.add_parser("run", help="Process a local video file")
    run_parser.add_argument("video", help="Path to video file")
    run_parser.add_argument("--upload-id", help="Website upload ID for queue status")

    # 'poll' subcommand (default) — coordinator polling loop
    poll_parser = subparsers.add_parser("poll", help="Poll coordinator for jobs")
    poll_parser.add_argument("--coordinator", default=None, help="Coordinator API URL")
    poll_parser.add_argument("--worker-id", default=None, help="Worker identifier")
    poll_parser.add_argument("--poll-interval", type=int, default=None)
    poll_parser.add_argument("--once", action="store_true", help="Process one job and exit")
    poll_parser.add_argument("--skip-youtube", action="store_true")
    poll_parser.add_argument("--youtube-dry-run", action="store_true")

    # Legacy: no subcommand = poll mode (backward compat)
    parser.add_argument("--coordinator", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-id", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--poll-interval", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--once", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--skip-youtube", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--youtube-dry-run", action="store_true", help=argparse.SUPPRESS)

    args = parser.parse_args()

    if args.command == "run":
        # Load password from .env if not in environment
        global UPLOAD_PASSWORD
        if not UPLOAD_PASSWORD:
            env_path = PROJECT_ROOT / ".env"
            if env_path.exists():
                with open(env_path) as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("UPLOAD_PASSWORD="):
                            UPLOAD_PASSWORD = line.split("=", 1)[1].strip().strip('"')
        process_local_video(args.video, upload_id=args.upload_id)
    else:
        # Poll mode (default)
        coordinator_url = args.coordinator or COORDINATOR_URL
        worker_id = args.worker_id or WORKER_ID
        poll_interval = args.poll_interval or POLL_INTERVAL
        skip_youtube = args.skip_youtube
        youtube_dry_run = args.youtube_dry_run

        worker_loop(coordinator_url, worker_id, poll_interval,
                    once=args.once, skip_youtube=skip_youtube,
                    youtube_dry_run=youtube_dry_run)


if __name__ == "__main__":
    main()
