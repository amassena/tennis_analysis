#!/usr/bin/env python3
"""iCloud watcher for cloud pipeline.

Two detection modes:
1. Album-based: any video placed in "Tennis Videos" album → process
2. Slo-mo auto-detect: any new slo-mo video in library → process automatically

Session health: verifies a known-good video is still visible after each scan.
If not, the session is stale → clears cookies, sends alert, stops until re-authed.

Usage:
    python cloud_icloud_watcher.py          # daemon mode
    python cloud_icloud_watcher.py --once   # single pass then exit
    python cloud_icloud_watcher.py --auth   # authenticate only (for 2FA setup)
"""

import argparse
import json
import os
import sys
import time
import requests
from datetime import datetime, timedelta
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

# Albums to watch (manual trigger — any video type)
ALBUMS = ["Tennis Videos", "Tennis Videos Group By Shot Type"]

# Slo-mo detection: assetSubtype 2 = slo-mo in iCloud
SLOMO_SUBTYPE = 2

POLL_INTERVAL = 60  # seconds

# Session health: how often to do full validation (every N polls)
HEALTH_CHECK_INTERVAL = 30  # every 30 minutes (30 polls * 60s)


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


def send_alert(subject, body):
    """Send alert via email/SMS using email_notify if available."""
    try:
        # Try importing email_notify for SMS
        sys.path.insert(0, '/opt/tennis')
        from scripts.email_notify import send_sms, send_email
        send_sms(body)
        send_email(subject, body)
        log(f"Alert sent: {subject}")
    except Exception as e:
        log(f"Alert send failed: {e}")


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


def verify_session_health(api, state):
    """Verify iCloud session is returning real data by checking for a known video.

    Picks the most recently processed slo-mo asset and verifies it's still
    visible in the Slo-mo album. If not, the session is stale.

    Returns True if healthy, False if stale.
    """
    # Find the most recent processed slo-mo asset ID
    processed = state.get("processed", {})
    recent_slomo = []
    for asset_id, info in processed.items():
        if info.get("source") == "auto:slomo" and info.get("processed_at"):
            recent_slomo.append((asset_id, info))
    if not recent_slomo:
        return True  # Nothing to verify against

    # Sort by processed_at, pick the most recent
    recent_slomo.sort(key=lambda x: x[1].get("processed_at", ""), reverse=True)
    check_asset_id, check_info = recent_slomo[0]
    check_filename = check_info.get("filename", "?")

    log(f"Session health check: looking for {check_filename} (asset {check_asset_id[:16]}...)")

    try:
        album = api.photos.albums["Slo-mo"]
        found = False
        count = 0
        for asset in album:
            if asset.item_type != "movie":
                continue
            count += 1
            if str(asset.id) == check_asset_id:
                found = True
                break
            if count > 200:  # Don't scan forever
                break

        if found:
            log(f"Session health: OK (found {check_filename} in {count} items)")
            return True
        else:
            log(f"SESSION STALE: {check_filename} not found after scanning {count} items")
            return False

    except Exception as e:
        log(f"Session health check error: {e}")
        return False


def handle_stale_session():
    """Handle a stale iCloud session: clear cookies and alert."""
    log("Clearing stale iCloud session cookies...")
    for f in Path(COOKIE_DIR).glob("*"):
        f.unlink()
        log(f"  Removed {f.name}")

    send_alert(
        "Tennis Watcher: iCloud session expired",
        "iCloud session is stale. New videos won't be detected. "
        "Re-authenticate: ssh -t devserver 'cd /opt/tennis && venv/bin/python cloud_icloud_watcher.py --auth'"
    )


def create_job(asset_id, filename, source):
    """Create a job in the coordinator."""
    try:
        response = requests.post(
            f"{COORDINATOR_URL}/jobs",
            json={
                "icloud_asset_id": asset_id,
                "filename": filename,
                "album_name": source,
            },
            timeout=10
        )
        data = response.json()
        if data.get("status") == "created":
            log(f"Created job for {filename} (source: {source})")
            return True
        elif data.get("status") == "exists":
            return False  # Already queued, not an error
        else:
            log(f"Failed to create job: {data}")
            return False
    except Exception as e:
        log(f"Error creating job: {e}")
        return False


def process_album(api, album_name, state):
    """Check album for new videos and create jobs."""
    try:
        album = api.photos.albums[album_name]
    except KeyError:
        log(f"Album '{album_name}' not found")
        return 0

    added = 0
    for asset in album:
        if asset.item_type != "movie":
            continue

        asset_id = str(asset.id)
        if asset_id in state.get("processed", {}):
            continue

        if create_job(asset_id, asset.filename, f"album:{album_name}"):
            state.setdefault("processed", {})[asset_id] = {
                "filename": asset.filename,
                "source": f"album:{album_name}",
                "processed_at": datetime.now().isoformat(),
            }
            save_state(state)
            added += 1

    if added == 0:
        log(f"No new videos in '{album_name}'")
    return added


def scan_slomo(api, state):
    """Scan iCloud's built-in Slo-mo smart album for new videos.

    Uses the "Slo-mo" smart album (pre-filtered by iCloud) instead of
    scanning the entire library. Only processes videos from the last 7 days
    that haven't already been processed.
    """
    SLOMO_ALBUM = "Slo-mo"
    LOOKBACK_DAYS = 7

    try:
        album = api.photos.albums[SLOMO_ALBUM]
    except KeyError:
        log(f"Smart album '{SLOMO_ALBUM}' not found — available albums: {list(api.photos.albums.keys())[:10]}")
        return 0

    cutoff = datetime.now() - timedelta(days=LOOKBACK_DAYS)
    added = 0
    checked = 0
    skipped_old = 0

    for asset in album:
        if asset.item_type != "movie":
            continue
        checked += 1

        # Skip videos older than lookback window
        try:
            created = asset.created
            if created and created.replace(tzinfo=None) < cutoff:
                skipped_old += 1
                continue
        except Exception:
            pass  # If we can't read date, process it anyway

        asset_id = str(asset.id)

        # Skip already processed
        if asset_id in state.get("processed", {}):
            continue

        log(f"Slo-mo detected: {asset.filename} (created: {asset.created})")

        if create_job(asset_id, asset.filename, "auto:slomo"):
            state.setdefault("processed", {})[asset_id] = {
                "filename": asset.filename,
                "source": "auto:slomo",
                "processed_at": datetime.now().isoformat(),
                "created": str(asset.created),
            }
            save_state(state)
            added += 1

    if added == 0:
        log(f"No new slo-mo videos (checked {checked} in last {LOOKBACK_DAYS}d, {skipped_old} older skipped)")
    else:
        log(f"Found {added} new slo-mo videos (checked {checked}, {skipped_old} older skipped)")

    return added


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
    log(f"Slo-mo auto-detect: enabled (assetSubtype={SLOMO_SUBTYPE})")

    poll_count = 0
    stale_detected = False

    while True:
        poll_count += 1

        api = authenticate(interactive=False)
        if not api:
            if not stale_detected:
                log("2FA required — session expired.")
                handle_stale_session()
                stale_detected = True
            # Keep retrying every 5 minutes in case someone re-auths
            time.sleep(300)
            continue

        stale_detected = False
        state = load_state()

        # Session health check every HEALTH_CHECK_INTERVAL polls
        if poll_count % HEALTH_CHECK_INTERVAL == 1:
            if not verify_session_health(api, state):
                handle_stale_session()
                time.sleep(300)
                continue

        # Mode 1: Check albums (manual trigger — any video type)
        for album_name in ALBUMS:
            try:
                process_album(api, album_name, state)
            except Exception as e:
                log(f"Error processing album '{album_name}': {e}")

        # Mode 2: Scan for new slo-mo videos (automatic)
        try:
            scan_slomo(api, state)
        except Exception as e:
            log(f"Error scanning for slo-mo: {e}")

        if args.once:
            break

        log(f"Sleeping {POLL_INTERVAL}s...")
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
