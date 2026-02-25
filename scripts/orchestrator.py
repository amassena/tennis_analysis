#!/usr/bin/env python3
"""Orchestrator for tennis video processing pipeline.

Runs on Hetzner VM (5.78.96.237). Coordinates processing across local GPU machines
via Tailscale SSH. No cloud GPU = no billing leaks.

Architecture:
    Cloud VM (this script)          Local GPU Machines
    ┌─────────────────────┐        ┌─────────────────────┐
    │  orchestrator.py    │──SSH──>│  process.py         │
    │  - Poll iCloud      │        │  - Stateless        │
    │  - Track jobs SQLite│        │  - Video in         │
    │  - Alert on failure │        │  - Highlights out   │
    │  - Upload YouTube   │        │  - Exit code=status │
    └─────────────────────┘        └─────────────────────┘

Design principles:
- LOCAL GPU ONLY: No cloud GPU = no billing leaks
- FAIL-LOUD: Never silent failures, errors logged clearly
- OBSERVABLE: SQLite state, JSON logs
- IDEMPOTENT: Safe to retry any operation
"""

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────

# Paths - adjust for deployment
PROJECT_ROOT = Path(__file__).parent.parent
ENV_FILE = os.environ.get("ENV_FILE", "/opt/tennis/.env")
DB_PATH = os.environ.get("DB_PATH", "/opt/tennis/jobs.db")
STATE_FILE = os.environ.get("STATE_FILE", "/opt/tennis/watcher_state.json")

# iCloud
ALBUMS = ["Tennis Videos", "Tennis Videos Group By Shot Type"]
POLL_INTERVAL = int(os.environ.get("POLL_INTERVAL", "300"))  # 5 minutes

# GPU machines (Tailscale IPs)
GPU_MACHINES = [
    {"name": "windows", "host": os.environ.get("WINDOWS_HOST", "windows")},
    {"name": "tmassena", "host": os.environ.get("TMASSENA_HOST", "tmassena")},
]


# Timeouts
SSH_TIMEOUT = 30
JOB_TIMEOUT_HOURS = 2
HEARTBEAT_INTERVAL = 30


# ─────────────────────────────────────────────────────────────
# Job States
# ─────────────────────────────────────────────────────────────

class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# ─────────────────────────────────────────────────────────────
# Logging - JSON format for observability
# ─────────────────────────────────────────────────────────────

def log(level: str, msg: str, **kwargs):
    """Log in JSON format for observability."""
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "level": level,
        "msg": msg,
        **kwargs,
    }
    print(json.dumps(entry), flush=True)


def log_info(msg: str, **kwargs):
    log("INFO", msg, **kwargs)


def log_error(msg: str, **kwargs):
    log("ERROR", msg, **kwargs)


def log_warn(msg: str, **kwargs):
    log("WARN", msg, **kwargs)



# ─────────────────────────────────────────────────────────────
# Database - SQLite for job tracking
# ─────────────────────────────────────────────────────────────

def init_db(db_path: str) -> sqlite3.Connection:
    """Initialize SQLite database with job tracking schema."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    conn.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            video_id TEXT PRIMARY KEY,
            icloud_asset_id TEXT UNIQUE,
            filename TEXT NOT NULL,
            album_name TEXT,
            status TEXT DEFAULT 'pending',
            worker TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            started_at TEXT,
            completed_at TEXT,
            last_heartbeat TEXT,
            highlights_url TEXT,
            error_message TEXT,
            retry_count INTEGER DEFAULT 0
        )
    """)

    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)
    """)

    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_jobs_icloud_asset ON jobs(icloud_asset_id)
    """)

    conn.commit()
    return conn


def add_job(conn: sqlite3.Connection, icloud_asset_id: str, filename: str,
            album_name: str) -> Optional[str]:
    """Add a new job. Returns video_id if created, None if already exists."""
    video_id = filename.rsplit(".", 1)[0]  # Remove extension

    try:
        conn.execute("""
            INSERT INTO jobs (video_id, icloud_asset_id, filename, album_name)
            VALUES (?, ?, ?, ?)
        """, (video_id, icloud_asset_id, filename, album_name))
        conn.commit()
        log_info(f"Created job", video_id=video_id, filename=filename)
        return video_id
    except sqlite3.IntegrityError:
        # Already exists
        return None


def get_pending_jobs(conn: sqlite3.Connection) -> list:
    """Get all pending jobs."""
    cursor = conn.execute("""
        SELECT * FROM jobs WHERE status = 'pending'
        ORDER BY created_at ASC
    """)
    return [dict(row) for row in cursor.fetchall()]


def claim_job(conn: sqlite3.Connection, video_id: str, worker: str) -> bool:
    """Atomically claim a pending job."""
    cursor = conn.execute("""
        UPDATE jobs
        SET status = 'processing', worker = ?, started_at = datetime('now')
        WHERE video_id = ? AND status = 'pending'
    """, (worker, video_id))
    conn.commit()
    return cursor.rowcount > 0


def update_heartbeat(conn: sqlite3.Connection, video_id: str):
    """Update job heartbeat."""
    conn.execute("""
        UPDATE jobs SET last_heartbeat = datetime('now')
        WHERE video_id = ?
    """, (video_id,))
    conn.commit()


def complete_job(conn: sqlite3.Connection, video_id: str,
                 highlights_url: str = None, error: str = None):
    """Mark job as completed or failed."""
    if error:
        conn.execute("""
            UPDATE jobs
            SET status = 'failed',
                completed_at = datetime('now'),
                error_message = ?,
                retry_count = retry_count + 1
            WHERE video_id = ?
        """, (error[:500], video_id))  # Limit error length
        log_error(f"Job failed", video_id=video_id, error=error[:200])
    else:
        conn.execute("""
            UPDATE jobs
            SET status = 'completed',
                completed_at = datetime('now'),
                highlights_url = ?
            WHERE video_id = ?
        """, (highlights_url, video_id))
        log_info(f"Job completed", video_id=video_id, highlights_url=highlights_url)

    conn.commit()


def get_stuck_jobs(conn: sqlite3.Connection, max_hours: float = 2.0) -> list:
    """Find jobs stuck in processing state."""
    cursor = conn.execute("""
        SELECT * FROM jobs
        WHERE status = 'processing'
        AND (
            julianday('now') - julianday(started_at)
        ) * 24 > ?
    """, (max_hours,))
    return [dict(row) for row in cursor.fetchall()]


def reset_stuck_jobs(conn: sqlite3.Connection, max_hours: float = 2.0) -> int:
    """Reset stuck jobs back to pending."""
    cursor = conn.execute("""
        UPDATE jobs
        SET status = 'pending', worker = NULL, started_at = NULL
        WHERE status = 'processing'
        AND (
            julianday('now') - julianday(started_at)
        ) * 24 > ?
    """, (max_hours,))
    conn.commit()
    return cursor.rowcount


def job_exists(conn: sqlite3.Connection, icloud_asset_id: str) -> bool:
    """Check if job already exists for this iCloud asset."""
    cursor = conn.execute("""
        SELECT 1 FROM jobs WHERE icloud_asset_id = ?
    """, (icloud_asset_id,))
    return cursor.fetchone() is not None


# ─────────────────────────────────────────────────────────────
# Environment Loading
# ─────────────────────────────────────────────────────────────

def load_env(env_file: str) -> dict:
    """Load environment variables from .env file."""
    env = {}
    if os.path.exists(env_file):
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    env[key.strip()] = value.strip().strip('"').strip("'")
    return env


# ─────────────────────────────────────────────────────────────
# iCloud Integration
# ─────────────────────────────────────────────────────────────

def authenticate_icloud(env: dict, cookie_dir: str):
    """Authenticate to iCloud. Returns API object."""
    try:
        from pyicloud import PyiCloudService
    except ImportError:
        log_error("pyicloud not installed")
        raise RuntimeError("pip install pyicloud")

    username = env.get("ICLOUD_USERNAME")
    password = env.get("ICLOUD_PASSWORD")

    if not username or not password:
        raise RuntimeError("ICLOUD_USERNAME and ICLOUD_PASSWORD required in .env")

    os.makedirs(cookie_dir, exist_ok=True)

    log_info(f"Authenticating to iCloud", username=username)
    api = PyiCloudService(username, password, cookie_directory=cookie_dir)

    if api.requires_2fa:
        raise RuntimeError("iCloud requires 2FA - authenticate manually first")

    log_info("iCloud authentication successful")
    return api


def scan_album(api, album_name: str, conn: sqlite3.Connection) -> int:
    """Scan an iCloud album for new videos. Returns count of new jobs created."""
    try:
        album = api.photos.albums[album_name]
    except KeyError:
        log_warn(f"Album not found", album=album_name)
        return 0

    new_count = 0
    for asset in album:
        if asset.item_type != "movie":
            continue

        asset_id = str(asset.id)

        # Skip if already processed
        if job_exists(conn, asset_id):
            continue

        filename = asset.filename
        video_id = add_job(conn, asset_id, filename, album_name)

        if video_id:
            new_count += 1
            log_info(f"New video found", video_id=video_id, album=album_name)

    return new_count


# ─────────────────────────────────────────────────────────────
# GPU Machine Management
# ─────────────────────────────────────────────────────────────

def check_gpu_available(host: str) -> bool:
    """Check if GPU machine is reachable via SSH."""
    try:
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes",
             host, "echo ok"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0 and "ok" in result.stdout
    except Exception:
        return False


def get_available_gpu() -> Optional[dict]:
    """Find an available GPU machine."""
    for machine in GPU_MACHINES:
        if check_gpu_available(machine["host"]):
            log_info(f"GPU available", machine=machine["name"])
            return machine
    return None


# ─────────────────────────────────────────────────────────────
# R2 Storage
# ─────────────────────────────────────────────────────────────

def get_r2_client(env: dict):
    """Get boto3 client for R2."""
    try:
        import boto3
        from botocore.config import Config
    except ImportError:
        raise RuntimeError("pip install boto3")

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


def upload_to_r2(client, local_path: str, r2_key: str, bucket: str = "tennis-videos"):
    """Upload file to R2."""
    log_info(f"Uploading to R2", key=r2_key)
    client.upload_file(local_path, bucket, r2_key)
    log_info(f"Upload complete", key=r2_key)


def download_from_r2(client, r2_key: str, local_path: str, bucket: str = "tennis-videos"):
    """Download file from R2."""
    log_info(f"Downloading from R2", key=r2_key)
    client.download_file(bucket, r2_key, local_path)
    log_info(f"Download complete", key=r2_key)


# ─────────────────────────────────────────────────────────────
# Video Processing
# ─────────────────────────────────────────────────────────────

def download_from_icloud(api, icloud_asset_id: str, filename: str,
                         output_dir: str) -> str:
    """Download video from iCloud. Returns local path."""
    # Find the asset
    asset = None
    for p in api.photos.all:
        if p.id == icloud_asset_id:
            asset = p
            break

    if not asset:
        raise RuntimeError(f"Asset {icloud_asset_id} not found in iCloud")

    local_path = os.path.join(output_dir, filename)

    log_info(f"Downloading from iCloud", filename=filename)
    download = asset.download()

    with open(local_path, "wb") as f:
        if hasattr(download, "content"):
            f.write(download.content)
        else:
            f.write(download)

    size_mb = os.path.getsize(local_path) / (1024 * 1024)
    log_info(f"Download complete", filename=filename, size_mb=round(size_mb, 1))
    return local_path


def run_on_gpu(host: str, command: str, timeout: int = 7200) -> tuple:
    """Run command on GPU machine via SSH. Returns (returncode, stdout, stderr)."""
    cmd = [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=3",
        host,
        command,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired as e:
        return -1, "", f"Timeout after {timeout}s"
    except Exception as e:
        return -1, "", str(e)


def process_video(job: dict, api, env: dict, conn: sqlite3.Connection) -> str:
    """Process a video job. Returns highlights URL on success."""
    video_id = job["video_id"]
    filename = job["filename"]
    icloud_asset_id = job["icloud_asset_id"]

    log_info(f"Processing job", video_id=video_id, filename=filename)

    # Find available GPU
    gpu = get_available_gpu()
    if not gpu:
        raise RuntimeError("No GPU machines available")

    host = gpu["host"]
    log_info(f"Using GPU", machine=gpu["name"], host=host)

    # Get R2 client
    r2_client = get_r2_client(env)
    bucket = "tennis-videos"

    with tempfile.TemporaryDirectory() as tmpdir:
        # Step 1: Download from iCloud
        local_video = download_from_icloud(api, icloud_asset_id, filename, tmpdir)

        # Step 2: Upload raw video to R2
        r2_raw_key = f"raw/{filename}"
        upload_to_r2(r2_client, local_video, r2_raw_key, bucket)

        # Step 3: Run processing on GPU machine
        # The process.py script is stateless - downloads from R2, processes, uploads results
        video_name = filename.rsplit(".", 1)[0]

        cmd = f"cd C:/Users/amass/tennis_analysis && python gpu_worker/process.py --video-key {r2_raw_key} --video-name {video_name}"

        log_info(f"Running on GPU", host=host, command=cmd[:100])
        returncode, stdout, stderr = run_on_gpu(host, cmd)

        # Log output
        for line in stdout.split("\n"):
            if line.strip():
                log_info(f"GPU: {line.strip()}")

        if returncode != 0:
            for line in stderr.split("\n"):
                if line.strip():
                    log_error(f"GPU: {line.strip()}")
            raise RuntimeError(f"GPU processing failed: {stderr[:500]}")

        # Step 4: Get highlights URL from R2
        highlights_key = f"highlights/{video_name}_highlights.mp4"

        # Verify the highlights exist
        try:
            r2_client.head_object(Bucket=bucket, Key=highlights_key)
        except Exception as e:
            raise RuntimeError(f"Highlights not found in R2: {highlights_key}")

        # Generate presigned URL for YouTube upload
        highlights_url = r2_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": highlights_key},
            ExpiresIn=3600,  # 1 hour
        )

        log_info(f"Highlights ready", key=highlights_key)
        return highlights_url


# ─────────────────────────────────────────────────────────────
# Main Loop
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Tennis Pipeline Orchestrator")
    parser.add_argument("--once", action="store_true", help="Single pass then exit")
    parser.add_argument("--env", default=ENV_FILE, help="Path to .env file")
    parser.add_argument("--db", default=DB_PATH, help="Path to SQLite database")
    parser.add_argument("--poll-interval", type=int, default=POLL_INTERVAL,
                        help="Seconds between iCloud polls")
    args = parser.parse_args()

    log_info("Orchestrator starting", poll_interval=args.poll_interval)

    # Load environment
    env = load_env(args.env)
    if not env:
        log_error(f"No .env file found at {args.env}")
        sys.exit(1)

    # Initialize database
    conn = init_db(args.db)
    log_info(f"Database initialized", path=args.db)

    # Cookie directory for iCloud session
    cookie_dir = os.path.dirname(args.env) + "/icloud_cookies"

    while True:
        try:
            # Step 1: Check for stuck jobs
            stuck = get_stuck_jobs(conn, max_hours=JOB_TIMEOUT_HOURS)
            for job in stuck:
                log_warn(f"Job stuck", video_id=job["video_id"], hours=JOB_TIMEOUT_HOURS)

            reset_count = reset_stuck_jobs(conn, max_hours=JOB_TIMEOUT_HOURS)
            if reset_count > 0:
                log_warn(f"Reset stuck jobs", count=reset_count)

            # Step 2: Authenticate to iCloud
            api = authenticate_icloud(env, cookie_dir)

            # Step 3: Scan albums for new videos
            for album in ALBUMS:
                try:
                    new_count = scan_album(api, album, conn)
                    if new_count > 0:
                        log_info(f"Found new videos", album=album, count=new_count)
                except Exception as e:
                    log_error(f"Error scanning album", album=album, error=str(e))

            # Step 4: Process pending jobs
            pending = get_pending_jobs(conn)

            if pending:
                log_info(f"Pending jobs", count=len(pending))

                for job in pending:
                    video_id = job["video_id"]

                    # Claim the job
                    if not claim_job(conn, video_id, "orchestrator"):
                        continue

                    try:
                        highlights_url = process_video(job, api, env, conn)
                        complete_job(conn, video_id, highlights_url=highlights_url)
                    except Exception as e:
                        log_error(f"Job failed", video_id=video_id, error=str(e))
                        complete_job(conn, video_id, error=str(e))
            else:
                log_info("No pending jobs")

            if args.once:
                break

            log_info(f"Sleeping", seconds=args.poll_interval)
            time.sleep(args.poll_interval)

        except KeyboardInterrupt:
            log_info("Shutting down")
            break
        except Exception as e:
            log_error(f"Error in main loop", error=str(e))
            if args.once:
                sys.exit(1)
            time.sleep(60)  # Back off on errors

    conn.close()


if __name__ == "__main__":
    main()
