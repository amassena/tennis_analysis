#!/usr/bin/env python3
"""Download original-quality videos from iCloud Photos to raw/."""

import os
import sys
import time

# Add project root to path so config is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import ICLOUD


# ── Helpers ──────────────────────────────────────────────────


def _load_dotenv(path):
    """Parse a .env file and set os.environ (no extra dependency)."""
    if not os.path.exists(path):
        print(f"[ERROR] .env file not found: {path}")
        print("Create it with ICLOUD_USERNAME and ICLOUD_PASSWORD.")
        sys.exit(1)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ[key.strip()] = value.strip()


def format_size(num_bytes):
    """Return human-readable file size string."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(num_bytes) < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"


# ── iCloud Authentication ────────────────────────────────────


def authenticate():
    """Login via pyicloud, handle 2FA with up to 3 attempts, persist session."""
    from pyicloud import PyiCloudService

    _load_dotenv(ICLOUD["env_file"])

    username = os.environ.get("ICLOUD_USERNAME")
    password = os.environ.get("ICLOUD_PASSWORD")
    if not username or not password:
        print("[ERROR] ICLOUD_USERNAME and ICLOUD_PASSWORD must be set in .env")
        sys.exit(1)

    cookie_dir = ICLOUD["cookie_directory"]
    os.makedirs(cookie_dir, exist_ok=True)

    print(f"Authenticating as {username}...")
    api = PyiCloudService(username, password, cookie_directory=cookie_dir)

    if api.requires_2fa:
        print("\nTwo-factor authentication required.")
        for attempt in range(1, 4):
            code = input(f"  Enter 2FA code (attempt {attempt}/3): ").strip()
            if api.validate_2fa_code(code):
                print("  2FA verified successfully.")
                break
            else:
                print("  Invalid code.")
                if attempt == 3:
                    print("[ERROR] Failed 2FA after 3 attempts.")
                    sys.exit(1)

        if not api.is_trusted_session:
            api.trust_session()
            print("  Session trusted for future logins.")

    elif api.requires_2sa:
        print("\nTwo-step authentication required.")
        devices = api.trusted_devices
        for i, device in enumerate(devices):
            name = device.get("deviceName", f"Device {i}")
            print(f"  {i}: {name}")
        idx = int(input("  Select device: ").strip())
        device = devices[idx]
        if not api.send_verification_code(device):
            print("[ERROR] Failed to send verification code.")
            sys.exit(1)

        for attempt in range(1, 4):
            code = input(f"  Enter verification code (attempt {attempt}/3): ").strip()
            if api.validate_verification_code(device, code):
                print("  Verification successful.")
                break
            else:
                print("  Invalid code.")
                if attempt == 3:
                    print("[ERROR] Failed verification after 3 attempts.")
                    sys.exit(1)

    print("Authentication successful.\n")
    return api


# ── Video Listing & Selection ────────────────────────────────


def list_videos(api):
    """Fetch videos from the Videos smart album, sorted by date descending."""
    album_name = ICLOUD["video_album"]
    try:
        album = api.photos.albums[album_name]
    except KeyError:
        print(f"[ERROR] Album '{album_name}' not found.")
        print("Available albums:")
        for name in api.photos.albums:
            print(f"  - {name}")
        sys.exit(1)

    print(f"Fetching videos from '{album_name}' album...")
    videos = [asset for asset in album if asset.item_type == "movie"]
    videos.sort(key=lambda v: v.created, reverse=True)
    print(f"Found {len(videos)} video(s).\n")
    return videos


def display_video_list(videos):
    """Print a numbered table of videos."""
    if not videos:
        print("No videos found.")
        return

    print(f"{'#':>4}  {'Date':18}  {'Filename':40}  {'Size':>10}  {'Dimensions'}")
    print("-" * 95)
    for i, v in enumerate(videos, 1):
        date_str = v.created.strftime("%Y-%m-%d %H:%M") if v.created else "Unknown"
        dims = v.dimensions
        dim_str = f"{dims[0]}x{dims[1]}" if dims and dims[0] else "N/A"
        size_str = format_size(v.size) if v.size else "N/A"
        print(f"{i:>4}  {date_str:18}  {v.filename:40}  {size_str:>10}  {dim_str}")
    print()


def select_videos(videos):
    """Interactive prompt: accepts '1,3,5', '1-3', 'all', or 'q'."""
    while True:
        choice = input("Select videos (e.g. 1,3,5 or 1-3 or all, q to quit): ").strip().lower()
        if choice == "q":
            print("Cancelled.")
            sys.exit(0)
        if choice == "all":
            return list(videos)

        selected = []
        try:
            for part in choice.split(","):
                part = part.strip()
                if "-" in part:
                    start, end = part.split("-", 1)
                    for idx in range(int(start), int(end) + 1):
                        selected.append(videos[idx - 1])
                else:
                    selected.append(videos[int(part) - 1])
        except (ValueError, IndexError):
            print(f"  Invalid selection. Enter numbers 1-{len(videos)}.")
            continue

        if selected:
            return selected
        print("  No videos selected.")


# ── Download ─────────────────────────────────────────────────


def download_video(asset, dest_dir, chunk_size, max_retries, retry_delay):
    """Stream original video to dest_dir. Uses .part temp file, skips if exists."""
    filename = asset.filename
    dest_path = os.path.join(dest_dir, filename)
    part_path = dest_path + ".part"

    # Check if already downloaded (filename + size match)
    expected_size = asset.versions.get("original", {}).get("size") or asset.size
    if os.path.exists(dest_path):
        existing_size = os.path.getsize(dest_path)
        if expected_size and existing_size == expected_size:
            print(f"  [SKIP] {filename} (already exists, {format_size(existing_size)})")
            return True

    os.makedirs(dest_dir, exist_ok=True)

    for attempt in range(1, max_retries + 1):
        try:
            response = asset.download("original")
            if response is None:
                print(f"  [ERROR] No download URL for {filename}")
                return False

            total = expected_size or 0
            downloaded = 0

            with open(part_path, "wb") as f:
                # Handle both bytes (pyicloud >=2.3) and Response objects (pyicloud <2.3)
                if isinstance(response, bytes):
                    f.write(response)
                    downloaded = len(response)
                    if total:
                        pct = downloaded / total * 100
                        print(
                            f"  Downloading {filename}: "
                            f"{format_size(downloaded)} / {format_size(total)} "
                            f"({pct:.1f}%)",
                            flush=True,
                        )
                    else:
                        print(
                            f"  Downloading {filename}: {format_size(downloaded)}",
                            flush=True,
                        )
                else:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total:
                                pct = downloaded / total * 100
                                print(
                                    f"\r  Downloading {filename}: "
                                    f"{format_size(downloaded)} / {format_size(total)} "
                                    f"({pct:.1f}%)",
                                    end="",
                                    flush=True,
                                )
                            else:
                                print(
                                    f"\r  Downloading {filename}: {format_size(downloaded)}",
                                    end="",
                                    flush=True,
                                )

            print()  # newline after progress

            # Rename .part to final on success
            os.replace(part_path, dest_path)
            print(f"  [OK] {filename} ({format_size(downloaded)})")
            return True

        except Exception as e:
            print(f"\n  [RETRY {attempt}/{max_retries}] {filename}: {e}")
            # Clean up partial file on failure
            if os.path.exists(part_path):
                os.remove(part_path)
            if attempt < max_retries:
                time.sleep(retry_delay)

    print(f"  [FAILED] {filename} after {max_retries} attempts.")
    return False


# ── Main ─────────────────────────────────────────────────────


def parse_selection(choice, videos):
    """Parse a selection string like '1,3,5' or '1-3' or 'all'."""
    choice = choice.strip().lower()
    if choice == "all":
        return list(videos)
    selected = []
    for part in choice.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            for idx in range(int(start), int(end) + 1):
                selected.append(videos[idx - 1])
        else:
            selected.append(videos[int(part) - 1])
    return selected


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Download videos from iCloud Photos")
    parser.add_argument("--list", action="store_true", dest="list_only",
                        help="List available videos and exit")
    parser.add_argument("--select", type=str,
                        help="Non-interactive selection (e.g. '1', '1,3', '1-3', 'all')")
    args = parser.parse_args()

    api = authenticate()
    videos = list_videos(api)

    if not videos:
        print("No videos to download.")
        return

    display_video_list(videos)

    if args.list_only:
        return

    if args.select:
        try:
            selected = parse_selection(args.select, videos)
        except (ValueError, IndexError):
            print(f"[ERROR] Invalid selection: {args.select}")
            sys.exit(1)
    else:
        selected = select_videos(videos)

    print(f"\nDownloading {len(selected)} video(s) to {ICLOUD['download_directory']}/\n")

    results = {"ok": [], "skip": [], "fail": []}
    for asset in selected:
        dest_path = os.path.join(ICLOUD["download_directory"], asset.filename)
        expected_size = asset.versions.get("original", {}).get("size") or asset.size

        # Pre-check skip to categorize correctly
        if os.path.exists(dest_path) and expected_size:
            if os.path.getsize(dest_path) == expected_size:
                results["skip"].append(asset.filename)
                print(f"  [SKIP] {asset.filename} (already exists, {format_size(expected_size)})")
                continue

        success = download_video(
            asset,
            ICLOUD["download_directory"],
            ICLOUD["chunk_size"],
            ICLOUD["max_retries"],
            ICLOUD["retry_delay"],
        )
        if success:
            results["ok"].append(asset.filename)
        else:
            results["fail"].append(asset.filename)

    # Summary
    print("\n" + "=" * 50)
    print("Download Summary")
    print("=" * 50)
    print(f"  Downloaded: {len(results['ok'])}")
    print(f"  Skipped:    {len(results['skip'])}")
    print(f"  Failed:     {len(results['fail'])}")
    if results["fail"]:
        print("  Failed files:")
        for name in results["fail"]:
            print(f"    - {name}")
    print()


if __name__ == "__main__":
    main()
