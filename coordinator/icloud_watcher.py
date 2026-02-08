"""iCloud watcher - monitors albums for new videos and adds jobs to queue.

Runs alongside the coordinator API (on Pi or cloud).
Polls iCloud albums and pushes new videos to the job queue.
"""

import argparse
import json
import os
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import AUTO_PIPELINE


# Configuration
COORDINATOR_URL = os.environ.get("COORDINATOR_URL", "http://localhost:8080")
POLL_INTERVAL = int(os.environ.get("ICLOUD_POLL_INTERVAL", "300"))  # 5 minutes
ALBUMS = AUTO_PIPELINE.get("albums", ["Tennis Videos", "Tennis Videos Group By Shot Type"])


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
    except Exception as e:
        log(f"API error: {e}", "ERROR")
        raise


def get_icloud_api():
    """Get authenticated iCloud API."""
    try:
        from pyicloud import PyiCloudService
    except ImportError:
        log("pyicloud not installed. Install with: pip install pyicloud", "ERROR")
        raise

    # Load credentials from .env
    project_root = Path(__file__).parent.parent
    env_path = project_root / ".env"

    if not env_path.exists():
        raise RuntimeError(".env file not found with iCloud credentials")

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

    log(f"Connecting to iCloud as {apple_id}")
    api = PyiCloudService(apple_id, password)

    if api.requires_2fa:
        raise RuntimeError("iCloud requires 2FA - authenticate manually first")

    return api


def scan_albums(api) -> list[dict]:
    """Scan iCloud albums for video assets."""
    videos = []

    for album_name in ALBUMS:
        log(f"Scanning album: {album_name}")

        try:
            # Use bracket notation for pyicloud album access
            album = api.photos.albums[album_name]

            count = 0
            for asset in album:
                # Only process videos
                if not hasattr(asset, "filename"):
                    continue

                filename = asset.filename
                if not filename.lower().endswith((".mov", ".mp4")):
                    continue

                videos.append({
                    "icloud_asset_id": asset.id,
                    "filename": filename,
                    "album_name": album_name,
                })
                count += 1

            log(f"Found {count} videos in '{album_name}'")

        except KeyError:
            log(f"Album '{album_name}' not found", "WARN")
        except Exception as e:
            log(f"Error scanning album {album_name}: {e}", "ERROR")

    return videos


def add_jobs(videos: list[dict]) -> int:
    """Add new videos to the job queue. Returns count added."""
    added = 0

    for video in videos:
        try:
            result = api_request("POST", "/jobs", {
                "icloud_asset_id": video["icloud_asset_id"],
                "filename": video["filename"],
                "album_name": video["album_name"],
            })

            if result.get("status") == "created":
                log(f"Added job: {video['filename']}")
                added += 1
            # "exists" status means already in queue, skip silently

        except Exception as e:
            log(f"Failed to add job {video['filename']}: {e}", "ERROR")

    return added


def watcher_loop(coordinator_url: str, poll_interval: int, once: bool = False):
    """Main watcher loop."""
    log("Starting iCloud watcher")
    log(f"Coordinator: {coordinator_url}")
    log(f"Albums: {ALBUMS}")
    log(f"Poll interval: {poll_interval}s")

    # Update global for api_request function
    global COORDINATOR_URL
    COORDINATOR_URL = coordinator_url

    api = None

    while True:
        try:
            # Reconnect to iCloud each poll (session may expire)
            if api is None:
                api = get_icloud_api()

            # Scan for videos
            videos = scan_albums(api)
            log(f"Found {len(videos)} videos in albums")

            # Add to queue
            if videos:
                added = add_jobs(videos)
                if added > 0:
                    log(f"Added {added} new jobs to queue")

        except Exception as e:
            log(f"Error in watcher loop: {e}", "ERROR")
            api = None  # Force reconnect

        if once:
            break

        time.sleep(poll_interval)


def main():
    parser = argparse.ArgumentParser(description="iCloud album watcher")
    parser.add_argument(
        "--coordinator",
        default=None,
        help="Coordinator API URL",
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
        help="Scan once and exit",
    )
    args = parser.parse_args()

    coordinator_url = args.coordinator or COORDINATOR_URL
    poll_interval = args.poll_interval or POLL_INTERVAL

    watcher_loop(coordinator_url, poll_interval, once=args.once)


if __name__ == "__main__":
    main()
