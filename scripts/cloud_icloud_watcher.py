#!/usr/bin/env python3
"""iCloud watcher for cloud pipeline.

Runs on Hetzner. Polls iCloud albums for new videos, uploads to R2,
and creates jobs in the coordinator.

Usage:
    python cloud_icloud_watcher.py          # daemon mode (polls every 5 min)
    python cloud_icloud_watcher.py --once   # single pass then exit
    python cloud_icloud_watcher.py --auth   # authenticate only (for 2FA setup)
"""

import argparse
import json
import os
import sys
import time
import tempfile
import requests
from datetime import datetime
from pathlib import Path

# Load environment
from dotenv import load_dotenv
load_dotenv('/opt/tennis/.env')

ICLOUD_USERNAME = os.environ.get('ICLOUD_USERNAME')
ICLOUD_PASSWORD = os.environ.get('ICLOUD_PASSWORD')
CF_ACCOUNT_ID = os.environ.get('CF_ACCOUNT_ID')
CF_R2_ACCESS_KEY_ID = os.environ.get('CF_R2_ACCESS_KEY_ID')
CF_R2_SECRET_ACCESS_KEY = os.environ.get('CF_R2_SECRET_ACCESS_KEY')

COORDINATOR_URL = "http://localhost:8080"
R2_ENDPOINT = f"https://{CF_ACCOUNT_ID}.r2.cloudflarestorage.com"
R2_BUCKET = "tennis-videos"
COOKIE_DIR = "/opt/tennis/icloud_cookies"
STATE_FILE = "/opt/tennis/watcher_state.json"

# Albums to watch
ALBUMS = ["Tennis Videos", "Tennis Videos Group By Shot Type"]
POLL_INTERVAL = 300  # 5 minutes


def log(msg):
    """Log with timestamp."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def load_state():
    """Load processed videos state."""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"processed": {}}


def save_state(state):
    """Save processed videos state."""
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def authenticate(interactive=False):
    """Authenticate to iCloud."""
    from pyicloud import PyiCloudService

    if not ICLOUD_USERNAME or not ICLOUD_PASSWORD:
        log("ERROR: ICLOUD_USERNAME and ICLOUD_PASSWORD must be set in .env")
        sys.exit(1)

    os.makedirs(COOKIE_DIR, exist_ok=True)

    log(f"Authenticating to iCloud as {ICLOUD_USERNAME}...")
    api = PyiCloudService(ICLOUD_USERNAME, ICLOUD_PASSWORD, cookie_directory=COOKIE_DIR)

    if api.requires_2fa:
        if not interactive:
            log("ERROR: 2FA required but running non-interactively.")
            log("Run with --auth flag from a terminal to complete 2FA.")
            return None

        log("Two-factor authentication required.")
        for attempt in range(1, 4):
            code = input(f"  Enter 2FA code (attempt {attempt}/3): ").strip()
            if api.validate_2fa_code(code):
                log("2FA verified.")
                break
            elif attempt == 3:
                log("Failed 2FA after 3 attempts.")
                sys.exit(1)

        if not api.is_trusted_session:
            api.trust_session()

    log("iCloud authentication successful.")
    return api


def get_r2_client():
    """Get boto3 client for R2."""
    import boto3
    from botocore.config import Config

    return boto3.client('s3',
        endpoint_url=R2_ENDPOINT,
        aws_access_key_id=CF_R2_ACCESS_KEY_ID,
        aws_secret_access_key=CF_R2_SECRET_ACCESS_KEY,
        config=Config(signature_version='s3v4'))


def upload_to_r2(local_path, r2_key):
    """Upload a file to R2."""
    client = get_r2_client()
    file_size = os.path.getsize(local_path)

    log(f"Uploading to R2: {r2_key} ({file_size/1024/1024:.1f} MB)")

    with open(local_path, 'rb') as f:
        client.put_object(Bucket=R2_BUCKET, Key=r2_key, Body=f)

    log(f"Upload complete: {r2_key}")


def create_job(asset_id, filename, album_name):
    """Create a job in the coordinator."""
    try:
        response = requests.post(
            f"{COORDINATOR_URL}/jobs",
            json={
                "icloud_asset_id": asset_id,
                "filename": filename,
                "album_name": album_name,
            },
            timeout=10
        )
        data = response.json()
        if data.get("status") == "created":
            log(f"Created job {data.get('video_id')} for {filename}")
            return True
        else:
            log(f"Failed to create job: {data}")
            return False
    except Exception as e:
        log(f"Error creating job: {e}")
        return False


def download_video(asset, temp_dir):
    """Download video from iCloud to temp directory."""
    filename = asset.filename
    dest_path = os.path.join(temp_dir, filename)

    log(f"Downloading from iCloud: {filename}")

    try:
        response = asset.download("original")
        if response is None:
            log(f"No download URL for {filename}")
            return None

        downloaded = 0
        with open(dest_path, "wb") as f:
            if isinstance(response, bytes):
                f.write(response)
                downloaded = len(response)
            else:
                for chunk in response.iter_content(chunk_size=1024*1024):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if downloaded % (50*1024*1024) == 0:  # Log every 50MB
                            log(f"  Downloaded {downloaded/1024/1024:.0f} MB...")

        log(f"Downloaded {filename} ({downloaded/1024/1024:.1f} MB)")
        return dest_path

    except Exception as e:
        log(f"Error downloading {filename}: {e}")
        return None


def process_album(api, album_name, state):
    """Check album for new videos and process them."""
    try:
        album = api.photos.albums[album_name]
    except KeyError:
        log(f"Album '{album_name}' not found")
        return

    new_videos = []
    for asset in album:
        if asset.item_type != "movie":
            continue

        asset_id = str(asset.id)
        if asset_id in state.get("processed", {}):
            continue

        new_videos.append(asset)

    if not new_videos:
        log(f"No new videos in '{album_name}'")
        return

    log(f"Found {len(new_videos)} new videos in '{album_name}'")

    for asset in new_videos:
        asset_id = str(asset.id)
        filename = asset.filename

        with tempfile.TemporaryDirectory() as temp_dir:
            # Download from iCloud
            local_path = download_video(asset, temp_dir)
            if not local_path:
                continue

            # Upload to R2
            r2_key = f"raw/{filename}"
            try:
                upload_to_r2(local_path, r2_key)
            except Exception as e:
                log(f"Error uploading to R2: {e}")
                continue

        # Create job in coordinator
        if create_job(asset_id, filename, album_name):
            # Mark as processed
            state.setdefault("processed", {})[asset_id] = {
                "filename": filename,
                "album": album_name,
                "processed_at": datetime.now().isoformat(),
                "size": asset.size,
            }
            save_state(state)


def main():
    parser = argparse.ArgumentParser(description="iCloud watcher for cloud pipeline")
    parser.add_argument("--once", action="store_true", help="Single pass then exit")
    parser.add_argument("--auth", action="store_true", help="Authenticate only (for 2FA)")
    args = parser.parse_args()

    # Auth-only mode
    if args.auth:
        authenticate(interactive=True)
        return

    log("iCloud watcher starting...")
    log(f"Coordinator: {COORDINATOR_URL}")
    log(f"R2 Bucket: {R2_BUCKET}")
    log(f"Albums: {ALBUMS}")

    while True:
        api = authenticate(interactive=False)
        if not api:
            log("Authentication failed, will retry...")
            time.sleep(POLL_INTERVAL)
            continue

        state = load_state()

        for album_name in ALBUMS:
            try:
                process_album(api, album_name, state)
            except Exception as e:
                log(f"Error processing album '{album_name}': {e}")

        if args.once:
            break

        log(f"Sleeping {POLL_INTERVAL}s...")
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
