#!/usr/bin/env python3
"""Upload highlight reels to YouTube (unlisted) and iCloud Drive."""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import HIGHLIGHTS_DIR, ICLOUD, PROJECT_ROOT

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
YOUTUBE_API_SERVICE = "youtube"
YOUTUBE_API_VERSION = "v3"
CLIENT_SECRETS_FILE = os.path.join(PROJECT_ROOT, "config", "client_secrets.json")
CREDENTIALS_FILE = os.path.join(PROJECT_ROOT, "config", "youtube_credentials.json")


# ── YouTube ─────────────────────────────────────────────────


def get_youtube_service():
    """Authenticate and return a YouTube API service object."""
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build

    creds = None

    if os.path.exists(CREDENTIALS_FILE):
        creds = Credentials.from_authorized_user_file(CREDENTIALS_FILE, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            print("Refreshing YouTube credentials...")
            creds.refresh(Request())
        else:
            if not os.path.exists(CLIENT_SECRETS_FILE):
                print(f"[ERROR] OAuth client secrets not found: {CLIENT_SECRETS_FILE}")
                print("Download from Google Cloud Console > APIs & Services > Credentials")
                sys.exit(1)
            print("Opening browser for YouTube authorization...")
            flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)

        with open(CREDENTIALS_FILE, "w") as f:
            f.write(creds.to_json())
        print("Credentials saved for future uploads.")

    return build(YOUTUBE_API_SERVICE, YOUTUBE_API_VERSION, credentials=creds)


def upload_to_youtube(video_path, title=None, description=None):
    """Upload a video to YouTube as unlisted. Returns the video URL."""
    from googleapiclient.http import MediaFileUpload

    if not os.path.exists(video_path):
        print(f"[ERROR] Video not found: {video_path}")
        return None

    basename = os.path.splitext(os.path.basename(video_path))[0]
    if title is None:
        title = basename.replace("_", " ").title()
    if description is None:
        description = f"Tennis practice highlight reel - {basename}"

    youtube = get_youtube_service()

    body = {
        "snippet": {
            "title": title,
            "description": description,
            "tags": ["tennis", "serve", "practice", "slow motion", "240fps"],
            "categoryId": "17",  # Sports
        },
        "status": {
            "privacyStatus": "unlisted",
        },
    }

    size_mb = os.path.getsize(video_path) / (1024 * 1024)
    print(f"Uploading to YouTube (unlisted): {os.path.basename(video_path)} ({size_mb:.1f} MB)")
    print(f"  Title: {title}")

    media = MediaFileUpload(video_path, mimetype="video/mp4", resumable=True)
    request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)

    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            pct = int(status.progress() * 100)
            print(f"\r  Uploading: {pct}%", end="", flush=True)

    video_id = response["id"]
    url = f"https://youtu.be/{video_id}"
    print(f"\n  Uploaded: {url}")
    return url


# ── iCloud Drive ────────────────────────────────────────────


def _load_dotenv(path):
    """Parse a .env file and set os.environ."""
    if not os.path.exists(path):
        print(f"[ERROR] .env file not found: {path}")
        sys.exit(1)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ[key.strip()] = value.strip()


def get_icloud_api():
    """Authenticate to iCloud, handling 2FA."""
    from pyicloud import PyiCloudService

    _load_dotenv(ICLOUD["env_file"])

    username = os.environ.get("ICLOUD_USERNAME")
    password = os.environ.get("ICLOUD_PASSWORD")
    if not username or not password:
        print("[ERROR] ICLOUD_USERNAME and ICLOUD_PASSWORD must be set in .env")
        sys.exit(1)

    cookie_dir = ICLOUD["cookie_directory"]
    os.makedirs(cookie_dir, exist_ok=True)

    print(f"Authenticating to iCloud as {username}...")
    api = PyiCloudService(username, password, cookie_directory=cookie_dir)

    if api.requires_2fa:
        print("Two-factor authentication required.")
        for attempt in range(1, 4):
            code = input(f"  Enter 2FA code (attempt {attempt}/3): ").strip()
            if api.validate_2fa_code(code):
                print("  2FA verified.")
                break
            elif attempt == 3:
                print("[ERROR] Failed 2FA after 3 attempts.")
                sys.exit(1)
        if not api.is_trusted_session:
            api.trust_session()

    print("iCloud authentication successful.")
    return api


def upload_to_icloud(api, video_path, folder_name="Tennis"):
    """Upload a video to iCloud Drive in the specified folder."""
    if not os.path.exists(video_path):
        print(f"[ERROR] Video not found: {video_path}")
        return False

    basename = os.path.basename(video_path)
    size_mb = os.path.getsize(video_path) / (1024 * 1024)

    drive = api.drive

    try:
        existing = drive.dir()
        if folder_name not in existing:
            print(f"  Creating folder: {folder_name}")
            drive.mkdir(folder_name)
            # Re-fetch drive to pick up the new folder
            drive = api.drive
        folder = drive[folder_name]
    except Exception as e:
        print(f"[ERROR] Cannot access iCloud Drive folder: {e}")
        return False

    print(f"Uploading to iCloud Drive/{folder_name}: {basename} ({size_mb:.1f} MB)")

    try:
        with open(video_path, "rb") as f:
            folder.upload(f)
        print(f"  Uploaded: {basename}")
        return True
    except Exception as e:
        print(f"  [ERROR] Upload failed: {e}")
        return False


# ── Main ────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Upload highlight reels to YouTube and/or iCloud Drive")
    parser.add_argument("video", nargs="?", help="Video file to upload (default: latest in highlights/)")
    parser.add_argument("--youtube", action="store_true", help="Upload to YouTube (unlisted)")
    parser.add_argument("--icloud", action="store_true", help="Upload to iCloud Drive")
    parser.add_argument("--title", type=str, help="YouTube video title")
    parser.add_argument("--description", type=str, help="YouTube video description")
    parser.add_argument("--all", action="store_true", dest="upload_all",
                        help="Upload to both YouTube and iCloud Drive")
    args = parser.parse_args()

    # Default: upload to both
    if not args.youtube and not args.icloud and not args.upload_all:
        args.upload_all = True

    # Find video to upload
    if args.video:
        video_path = args.video
    else:
        os.makedirs(HIGHLIGHTS_DIR, exist_ok=True)
        highlights = sorted([
            os.path.join(HIGHLIGHTS_DIR, f)
            for f in os.listdir(HIGHLIGHTS_DIR)
            if f.endswith(".mp4")
        ], key=os.path.getmtime, reverse=True)
        if not highlights:
            print("[ERROR] No highlight videos found in", HIGHLIGHTS_DIR)
            sys.exit(1)
        video_path = highlights[0]
        print(f"Auto-selected latest highlight: {os.path.basename(video_path)}")

    if not os.path.exists(video_path):
        print(f"[ERROR] File not found: {video_path}")
        sys.exit(1)

    print()
    results = {}

    # YouTube
    if args.youtube or args.upload_all:
        url = upload_to_youtube(video_path, title=args.title, description=args.description)
        results["youtube"] = url

    # iCloud Drive
    if args.icloud or args.upload_all:
        api = get_icloud_api()
        ok = upload_to_icloud(api, video_path)
        results["icloud"] = ok

    # Summary
    print("\n" + "=" * 50)
    print("Upload Summary")
    print("=" * 50)
    if "youtube" in results:
        if results["youtube"]:
            print(f"  YouTube: {results['youtube']}")
        else:
            print("  YouTube: FAILED")
    if "icloud" in results:
        print(f"  iCloud:  {'OK' if results['icloud'] else 'FAILED'}")
    print()


if __name__ == "__main__":
    main()
