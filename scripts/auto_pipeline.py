#!/usr/bin/env python3
"""Automated tennis pipeline: poll iCloud -> GPU processing -> YouTube upload.

Downloads iPhone videos from the "tennis_training" iCloud album, processes
them on GPU machines (preprocess, pose extraction, shot detection, clip
extraction), compiles combined normal + 0.25x slow-mo highlights, and
uploads to YouTube.  Supports multiple GPU machines with parallel dispatch.

Usage:
    python scripts/auto_pipeline.py          # daemon mode (polls every 5 min)
    python scripts/auto_pipeline.py --once   # single pass then exit
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import (
    RAW_DIR, HIGHLIGHTS_DIR, ICLOUD, AUTO_PIPELINE, PROJECT_ROOT,
)

# ── Logging ─────────────────────────────────────────────────

log = logging.getLogger("auto_pipeline")

def setup_logging(debug=False):
    log.setLevel(logging.DEBUG if debug else logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    # Console
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    log.addHandler(ch)
    # File
    fh = logging.FileHandler(os.path.join(PROJECT_ROOT, "pipeline.log"))
    fh.setFormatter(fmt)
    log.addHandler(fh)


# ── State Management ────────────────────────────────────────

def load_state():
    path = AUTO_PIPELINE["state_file"]
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"processed": {}, "daily_counts": {}}


def save_state(state):
    path = AUTO_PIPELINE["state_file"]
    with open(path, "w") as f:
        json.dump(state, f, indent=2)


# ── iCloud Authentication ──────────────────────────────────

def _load_dotenv(path):
    if not os.path.exists(path):
        log.error(".env file not found: %s", path)
        sys.exit(1)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ[key.strip()] = value.strip()


def authenticate():
    from pyicloud import PyiCloudService

    _load_dotenv(ICLOUD["env_file"])
    username = os.environ.get("ICLOUD_USERNAME")
    password = os.environ.get("ICLOUD_PASSWORD")
    if not username or not password:
        log.error("ICLOUD_USERNAME and ICLOUD_PASSWORD must be set in .env")
        sys.exit(1)

    cookie_dir = ICLOUD["cookie_directory"]
    os.makedirs(cookie_dir, exist_ok=True)

    log.info("Authenticating to iCloud as %s...", username)
    api = PyiCloudService(username, password, cookie_directory=cookie_dir)

    if api.requires_2fa:
        log.info("Two-factor authentication required.")
        for attempt in range(1, 4):
            code = input(f"  Enter 2FA code (attempt {attempt}/3): ").strip()
            if api.validate_2fa_code(code):
                log.info("2FA verified.")
                break
            elif attempt == 3:
                log.error("Failed 2FA after 3 attempts.")
                sys.exit(1)
        if not api.is_trusted_session:
            api.trust_session()

    log.info("iCloud authentication successful.")
    return api


# ── Poll iCloud ────────────────────────────────────────────

def poll_icloud(api, album_name, state):
    """Check the specified iCloud album for new unprocessed videos."""
    try:
        album = api.photos.albums[album_name]
    except KeyError:
        try:
            available = list(api.photos.albums)
        except Exception:
            available = "(unable to list)"
        log.error("Album '%s' not found. Available: %s", album_name, available)
        return []

    new_videos = []
    for asset in album:
        if asset.item_type != "movie":
            continue

        asset_id = str(asset.id)
        if asset_id in state.get("processed", {}):
            continue

        log.info("Found new video: %s", asset.filename)
        new_videos.append(asset)

    return new_videos


# ── Download ───────────────────────────────────────────────

def format_size(num_bytes):
    for unit in ("B", "KB", "MB", "GB"):
        if abs(num_bytes) < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"


def download_video(asset, raw_dir):
    """Download original-quality video from iCloud to raw_dir."""
    filename = asset.filename
    dest_path = os.path.join(raw_dir, filename)

    expected_size = asset.versions.get("original", {}).get("size") or asset.size
    if os.path.exists(dest_path) and expected_size:
        if os.path.getsize(dest_path) == expected_size:
            log.info("Skip download (exists): %s", filename)
            return dest_path

    os.makedirs(raw_dir, exist_ok=True)
    part_path = dest_path + ".part"

    for attempt in range(1, ICLOUD["max_retries"] + 1):
        try:
            response = asset.download("original")
            if response is None:
                log.error("No download URL for %s", filename)
                return None

            downloaded = 0
            with open(part_path, "wb") as f:
                if isinstance(response, bytes):
                    f.write(response)
                    downloaded = len(response)
                else:
                    for chunk in response.iter_content(chunk_size=ICLOUD["chunk_size"]):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)

            os.replace(part_path, dest_path)
            log.info("Downloaded %s (%s)", filename, format_size(downloaded))
            return dest_path

        except Exception as e:
            log.warning("Download attempt %d/%d failed for %s: %s",
                        attempt, ICLOUD["max_retries"], filename, e)
            if os.path.exists(part_path):
                os.remove(part_path)
            if attempt < ICLOUD["max_retries"]:
                time.sleep(ICLOUD["retry_delay"])

    log.error("Download failed after %d attempts: %s", ICLOUD["max_retries"], filename)
    return None


# ── SSH/SCP Helpers ────────────────────────────────────────

def _run_ssh(host, cmd, timeout=3600):
    """Run a command on a remote machine via SSH. Returns (success, stdout)."""
    full_cmd = ["ssh", host, cmd]
    log.info("SSH [%s]: %s", host, cmd)
    result = subprocess.run(full_cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        log.error("SSH [%s] failed (rc=%d): %s", host, result.returncode, result.stderr.strip())
        return False, result.stderr
    return True, result.stdout


def _scp_to(host, project, local_path, remote_relative):
    """Copy a file to a remote machine via SCP."""
    remote_path = f"{host}:{project}/{remote_relative}"
    log.info("SCP to %s: %s -> %s", host, os.path.basename(local_path), remote_relative)
    result = subprocess.run(
        ["scp", local_path, remote_path],
        capture_output=True, text=True, timeout=600,
    )
    return result.returncode == 0


def _scp_from(host, project, remote_relative, local_path):
    """Copy a file from a remote machine via SCP."""
    remote_path = f"{host}:{project}/{remote_relative}"
    log.info("SCP from %s: %s -> %s", host, remote_relative, os.path.basename(local_path))
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    result = subprocess.run(
        ["scp", remote_path, local_path],
        capture_output=True, text=True, timeout=600,
    )
    return result.returncode == 0


# ── GPU Processing ─────────────────────────────────────────

def process_on_gpu(filename, machine):
    """Run the full pipeline on a GPU machine via SSH: preprocess, poses, detect, clips."""
    base = os.path.splitext(filename)[0]
    host = machine["host"]
    project = machine["project"]

    # 1. Transfer raw video
    local_raw = os.path.join(RAW_DIR, filename)
    if not _scp_to(host, project, local_raw, f"raw/{filename}"):
        log.error("Failed to transfer %s to %s", filename, host)
        return False

    # 2. Preprocess (NVENC, 60fps CFR)
    ok, _ = _run_ssh(
        host,
        f"cd {project} && python preprocess_nvenc.py {filename}",
        timeout=1800,
    )
    if not ok:
        log.error("Preprocess failed for %s on %s", filename, host)
        return False

    # 3. Extract poses
    ok, _ = _run_ssh(
        host,
        f"cd {project} && python scripts/extract_poses.py preprocessed/{base}.mp4",
        timeout=3600,
    )
    if not ok:
        log.error("Pose extraction failed for %s on %s", filename, host)
        return False

    # 4. Detect shots
    shots_file = f"shots_detected_{base}.json"
    ok, _ = _run_ssh(
        host,
        f"cd {project} && python scripts/detect_shots.py poses/{base}.json -o {shots_file}",
        timeout=1800,
    )
    if not ok:
        log.error("Shot detection failed for %s on %s", filename, host)
        return False

    # 5. Extract clips + highlights
    ok, _ = _run_ssh(
        host,
        f"cd {project} && python scripts/extract_clips.py -i {shots_file} --highlights",
        timeout=3600,
    )
    if not ok:
        log.error("Clip extraction failed for %s on %s", filename, host)
        return False

    return True


# ── Combined Video (Normal + Slow-Mo) ─────────────────────

def compile_combined_on_gpu(filename, machine):
    """Compile the combined normal + slow-mo highlight on a GPU machine.

    Runs the slow-mo extraction from the raw 240fps source, then
    concatenates with the normal-speed highlights already produced.
    """
    base = os.path.splitext(filename)[0]
    host = machine["host"]
    project = machine["project"]
    slowmo_factor = AUTO_PIPELINE["slowmo_factor"]
    slowmo_fps = AUTO_PIPELINE["slowmo_output_fps"]
    shots_file = f"shots_detected_{base}.json"

    # Build and run a Python one-liner that calls the functions
    # in extract_clips.py to produce slow-mo + combined video
    script = (
        "import json, sys; "
        "sys.path.insert(0, '.'); "
        "from scripts.extract_clips import compile_slowmo_highlights, compile_combined_video; "
        "from config.settings import HIGHLIGHTS_DIR, RAW_DIR; "
        "import os; "
        f"data = json.load(open('{shots_file}')); "
        f"raw_path = os.path.join(RAW_DIR, '{filename}'); "
        f"slowmo_path = os.path.join(HIGHLIGHTS_DIR, '{base}_slowmo_highlights.mp4'); "
        f"normal_path = os.path.join(HIGHLIGHTS_DIR, '{base}_all_highlights.mp4'); "
        f"combined_path = os.path.join(HIGHLIGHTS_DIR, '{base}_combined.mp4'); "
        f"ok1 = compile_slowmo_highlights(raw_path, data['segments'], slowmo_path, {slowmo_factor}, {slowmo_fps}); "
        "print('Slow-mo:', ok1); "
        "ok2 = compile_combined_video(normal_path, slowmo_path, combined_path) if ok1 else False; "
        "print('Combined:', ok2); "
        "sys.exit(0 if ok2 else 1)"
    )

    ok, output = _run_ssh(
        host,
        f'cd {project} && python -c "{script}"',
        timeout=3600,
    )
    if not ok:
        # Fall back: if slow-mo fails, just use the normal highlights
        log.warning("Combined video failed on %s, will use normal highlights only for %s", host, base)
        return False

    log.info("Combined video compiled for %s on %s", base, host)
    return True


def transfer_to_mac(filename, machine):
    """Transfer the combined (or fallback normal) highlight from a GPU machine to Mac."""
    base = os.path.splitext(filename)[0]
    host = machine["host"]
    project = machine["project"]
    combined_name = f"{base}_combined.mp4"
    normal_name = f"{base}_all_highlights.mp4"

    local_combined = os.path.join(HIGHLIGHTS_DIR, combined_name)
    os.makedirs(HIGHLIGHTS_DIR, exist_ok=True)

    # Try combined first, fall back to normal highlights
    if _scp_from(host, project, f"highlights/{combined_name}", local_combined):
        return local_combined

    log.warning("Combined not found on %s, trying normal highlights for %s", host, base)
    local_normal = os.path.join(HIGHLIGHTS_DIR, normal_name)
    if _scp_from(host, project, f"highlights/{normal_name}", local_normal):
        return local_normal

    log.error("No highlight video found on %s for %s", host, base)
    return None


# ── YouTube Upload ─────────────────────────────────────────

def upload_to_youtube(video_path, creation_date, video_number):
    """Upload the highlight video to YouTube with formatted title."""
    from scripts.upload import upload_to_youtube as _yt_upload

    date_str = creation_date.strftime("%Y-%m-%d") if creation_date else "unknown"
    title = AUTO_PIPELINE["youtube_title_format"].format(
        date=date_str, n=video_number,
    )
    description = (
        f"Tennis training session highlights - {date_str}\n"
        f"Normal speed + 0.25x slow motion\n\n"
        f"Auto-generated by tennis_analysis pipeline"
    )

    log.info("Uploading to YouTube: %s", title)
    url = _yt_upload(video_path, title=title, description=description)
    if url:
        log.info("YouTube upload complete: %s", url)
    else:
        log.error("YouTube upload failed for %s", video_path)
    return url


# ── Main Loop ──────────────────────────────────────────────

def process_single_video(asset, state, machine):
    """Process one video through the full pipeline. Returns youtube URL or None."""
    filename = asset.filename
    base = os.path.splitext(filename)[0]
    asset_id = str(asset.id)
    host = machine["host"]

    log.info("=" * 60)
    log.info("Processing: %s on %s", filename, host)
    log.info("=" * 60)

    # 1. Download from iCloud
    local_path = download_video(asset, RAW_DIR)
    if not local_path:
        return None

    # 2. GPU pipeline: preprocess -> poses -> detect -> clips
    if not process_on_gpu(filename, machine):
        return None

    # 3. Compile combined normal + slow-mo highlight on GPU
    compile_combined_on_gpu(filename, machine)

    # 4. Transfer highlight back to Mac
    highlight_path = transfer_to_mac(filename, machine)
    if not highlight_path:
        return None

    # 5. Upload to YouTube
    creation_date = asset.asset_date if hasattr(asset, "asset_date") else asset.created
    date_str = creation_date.strftime("%Y-%m-%d") if creation_date else "unknown"
    daily_counts = state.get("daily_counts", {})
    count = daily_counts.get(date_str, 0) + 1

    youtube_url = upload_to_youtube(highlight_path, creation_date, count)

    # 6. Update state
    state.setdefault("processed", {})[asset_id] = {
        "filename": filename,
        "date": date_str,
        "youtube_url": youtube_url,
        "processed_at": datetime.now().isoformat(),
        "machine": host,
    }
    state.setdefault("daily_counts", {})[date_str] = count
    save_state(state)

    log.info("Finished processing %s on %s", filename, host)
    return youtube_url


def main_loop(once=False, debug=False):
    """Main daemon loop: poll iCloud, process new videos, upload."""
    setup_logging(debug=debug)
    album_name = AUTO_PIPELINE["album"]
    poll_interval = AUTO_PIPELINE["poll_interval"]
    machines = AUTO_PIPELINE["gpu_machines"]

    log.info("Auto pipeline started (album=%s, poll=%ds, once=%s, machines=%s)",
             album_name, poll_interval, once,
             [m["host"] for m in machines])

    api = authenticate()
    state = load_state()

    while True:
        try:
            new_videos = poll_icloud(api, album_name, state)

            if new_videos:
                log.info("Found %d new video(s) to process across %d machine(s)",
                         len(new_videos), len(machines))

                if len(new_videos) > 1 and len(machines) > 1:
                    # Parallel: distribute videos across machines
                    log.info("Dispatching %d videos in parallel", len(new_videos))
                    with ThreadPoolExecutor(max_workers=len(machines)) as pool:
                        futures = {}
                        for i, asset in enumerate(new_videos):
                            machine = machines[i % len(machines)]
                            fut = pool.submit(process_single_video, asset, state, machine)
                            futures[fut] = (asset, machine)
                        for fut in as_completed(futures):
                            asset, machine = futures[fut]
                            try:
                                url = fut.result()
                                if url:
                                    log.info("Completed %s on %s -> %s",
                                             asset.filename, machine["host"], url)
                            except Exception as e:
                                log.error("Failed to process %s on %s: %s",
                                          asset.filename, machine["host"], e,
                                          exc_info=True)
                else:
                    # Sequential: single video or single machine
                    for i, asset in enumerate(new_videos):
                        machine = machines[i % len(machines)]
                        try:
                            process_single_video(asset, state, machine)
                        except Exception as e:
                            log.error("Failed to process %s on %s: %s",
                                      asset.filename, machine["host"], e,
                                      exc_info=True)
            else:
                log.info("No new videos found.")

        except Exception as e:
            log.error("Poll cycle error: %s", e, exc_info=True)

        if once:
            log.info("Single pass complete, exiting.")
            break

        log.info("Sleeping %d seconds until next poll...", poll_interval)
        time.sleep(poll_interval)


# ── CLI ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Automated tennis pipeline: iCloud -> GPU -> YouTube",
    )
    parser.add_argument("--once", action="store_true",
                        help="Single pass: process any new videos and exit")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging (dumps iCloud field names)")
    args = parser.parse_args()

    main_loop(once=args.once, debug=args.debug)


if __name__ == "__main__":
    main()
