#!/usr/bin/env python3
"""Watch Photos for new 240fps slo-mo videos, download and process them.

Pipeline: download original → preprocess (60fps CFR) → extract poses →
CNN shot detection → export (timeline + rally + rally slowmo + grouped) → upload to R2.

Usage:
    .venv/bin/python scripts/watch_and_process.py          # check once and process
    .venv/bin/python scripts/watch_and_process.py --dry-run # show what would be processed
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import PROJECT_ROOT, RAW_DIR, PREPROCESSED_DIR, POSES_DIR

STATE_FILE = os.path.join(PROJECT_ROOT, "watcher_state.json")
LOG_FILE = os.path.join(PROJECT_ROOT, "watcher.log")
PYTHON = os.path.join(PROJECT_ROOT, ".venv", "bin", "python")


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"processed": []}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def find_new_slomo_videos(already_processed):
    """Query Photos for 240fps slo-mo videos not yet processed."""
    from Photos import PHAsset, PHAssetResource, PHAssetMediaTypeVideo
    from Foundation import NSURL

    results = PHAsset.fetchAssetsWithMediaType_options_(PHAssetMediaTypeVideo, None)
    new_videos = []

    for i in range(results.count()):
        asset = results.objectAtIndex_(i)
        resources = PHAssetResource.assetResourcesForAsset_(asset)

        # Get filename from original resource (type 2)
        name = None
        for r in resources:
            if r.type() == 2:
                name = str(r.originalFilename())
                break
        if not name:
            continue

        stem = os.path.splitext(name)[0]
        if stem in already_processed:
            continue

        # Check if this is a slo-mo video (has edited version = type 6)
        is_slomo = any(r.type() == 6 for r in resources)
        if not is_slomo:
            continue

        # Get the original resource (type 2) for downloading
        original = None
        for r in resources:
            if r.type() == 2:
                original = r
                break
        if not original:
            continue

        # Check file size to estimate frame rate — slo-mo originals are large
        size = int(original.valueForKey_("fileSize") or 0)
        new_videos.append((asset, name, stem, original, size, list(resources)))

    return sorted(new_videos, key=lambda x: x[1])


def download_original(resource, dest_path):
    """Download a PHAssetResource to disk."""
    from Photos import PHAssetResourceManager, PHAssetResourceRequestOptions
    from Foundation import NSURL

    manager = PHAssetResourceManager.defaultManager()
    options = PHAssetResourceRequestOptions.alloc().init()
    options.setNetworkAccessAllowed_(True)

    dest_url = NSURL.fileURLWithPath_(dest_path)
    done = {"finished": False, "error": None}

    def completion_handler(error):
        if error:
            done["error"] = str(error)
        done["finished"] = True

    if os.path.exists(dest_path):
        os.remove(dest_path)

    size_mb = int(resource.valueForKey_("fileSize") or 0) / (1024 * 1024)

    manager.writeDataForAssetResource_toFile_options_completionHandler_(
        resource, dest_url, options, completion_handler
    )

    start = time.time()
    while not done["finished"]:
        time.sleep(2)
        elapsed = time.time() - start
        if os.path.exists(dest_path):
            cur = os.path.getsize(dest_path) / (1024 * 1024)
            pct = (cur / size_mb * 100) if size_mb > 0 else 0
            print(f"\r  Downloading: {cur:.0f}/{size_mb:.0f} MB ({pct:.0f}%)", end="", flush=True)
        if elapsed > 7200:
            print("\n  TIMEOUT!")
            return False

    print()
    if done["error"]:
        log(f"  Download error: {done['error']}")
        return False
    return True


def verify_fps(filepath):
    """Check frame rate of a video file."""
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-select_streams", "v",
         "-show_entries", "stream=r_frame_rate,nb_frames",
         "-show_entries", "format=duration",
         "-of", "json", filepath],
        capture_output=True, text=True, timeout=10
    )
    data = json.loads(result.stdout)
    stream = data.get("streams", [{}])[0]
    fps_str = stream.get("r_frame_rate", "0/1")
    if "/" in fps_str:
        num, den = fps_str.split("/")
        fps = int(num) / int(den)
    else:
        fps = float(fps_str)
    return fps


def run_step(description, cmd, timeout=7200):
    """Run a pipeline step, return True on success."""
    log(f"  {description}...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        log(f"  FAILED: {description}")
        if result.stderr:
            log(f"  stderr: {result.stderr[-500:]}")
        return False
    return True


def process_video(stem, raw_path):
    """Run the full pipeline on a downloaded video."""
    preprocessed = os.path.join(PREPROCESSED_DIR, f"{stem}.mp4")
    poses_file = os.path.join(POSES_DIR, f"{stem}.json")
    det_file = os.path.join(PROJECT_ROOT, "detections", f"{stem}_fused_detections.json")

    # 1. Preprocess (240fps → 60fps CFR)
    if os.path.exists(preprocessed) and os.path.getsize(preprocessed) > 0:
        log(f"  Preprocessed file exists, skipping")
    else:
        if not run_step("Preprocessing to 60fps CFR",
                        [PYTHON, "scripts/preprocess_nvenc.py", raw_path],
                        timeout=3600):
            return False

    # 2. Extract poses (MediaPipe)
    if os.path.exists(poses_file) and os.path.getsize(poses_file) > 0:
        log(f"  Poses file exists, skipping")
    else:
        if not run_step("Extracting MediaPipe poses",
                        [PYTHON, "scripts/extract_poses.py", preprocessed],
                        timeout=7200):
            return False

    # 3. CNN shot detection
    if not run_step("Running CNN shot detection",
                    [PYTHON, "scripts/detect_shots_sequence.py", preprocessed,
                     "--threshold", "0.92"],
                    timeout=600):
        return False

    # 4. Export videos:
    #    - timeline (regular speed full video)
    #    - rally (normal speed)
    #    - rally (slow motion)
    #    - grouped (shots by type)
    log("  Exporting: timeline (regular speed)...")
    subprocess.run(
        [PYTHON, "scripts/export_videos.py", preprocessed,
         "--types", "timeline", "--upload"],
        capture_output=True, text=True, timeout=3600
    )

    log("  Exporting: rally (normal + slow-mo)...")
    subprocess.run(
        [PYTHON, "scripts/export_videos.py", preprocessed,
         "--types", "rally", "--slow-motion", "--upload"],
        capture_output=True, text=True, timeout=3600
    )

    log("  Exporting: grouped by shot type...")
    subprocess.run(
        [PYTHON, "scripts/export_videos.py", preprocessed,
         "--types", "grouped", "--upload"],
        capture_output=True, text=True, timeout=3600
    )

    return True


def update_r2_index():
    """Regenerate the browsable index.html on R2 with thumbnails and metadata."""
    try:
        from scripts.update_r2_index import update_index
        update_index()
        log("  Updated R2 index.html")
    except Exception as e:
        log(f"  Warning: failed to update R2 index: {e}")


def download_icloud_share(share_url, dest_path):
    """Download a video from an iCloud share link."""
    import re
    import requests

    # Extract token from URL (e.g. https://share.icloud.com/photos/XXXXX)
    match = re.search(r'(?:photos|iclouddrive)/(.+?)(?:\?|#|$)', share_url)
    if not match:
        raise ValueError(f"Cannot parse iCloud share URL: {share_url}")
    token = match.group(1).strip('/')

    # Resolve partition server by following redirect
    log(f"  Resolving iCloud share link...")
    r = requests.head(share_url, allow_redirects=True, timeout=15)
    host_match = re.search(r'(p\d+-sharedstreams\.icloud\.com)', r.url)
    host = host_match.group(1) if host_match else 'p63-sharedstreams.icloud.com'
    base = f"https://{host}/{token}"

    # Get stream metadata
    r = requests.post(f"{base}/sharedstreams/webstream",
                      json={"streamCtag": None}, timeout=15)
    r.raise_for_status()
    stream = r.json()

    photos = stream.get("photos", [])
    if not photos:
        raise Exception("No media found in share link")

    # Collect all asset GUIDs
    guids = [p["photoGuid"] for p in photos]

    # Get download URLs
    r = requests.post(f"{base}/sharedstreams/webasseturls",
                      json={"photoGuids": guids}, timeout=15)
    r.raise_for_status()
    items = r.json().get("items", {})

    if not items:
        raise Exception("No download URLs returned from iCloud")

    # Pick the largest file (the original video)
    best_url = None
    best_size = 0
    for checksum, info in items.items():
        size = int(info.get("fileSize", 0))
        if size > best_size:
            best_size = size
            loc = info.get("url_location", "")
            path = info.get("url_path", "")
            if loc and path:
                best_url = f"https://{loc}{path}"
                best_size = size

    if not best_url:
        raise Exception("Could not resolve download URL from iCloud")

    log(f"  Downloading {best_size/(1024*1024):.0f} MB from iCloud...")
    with requests.get(best_url, stream=True, timeout=30) as r:
        r.raise_for_status()
        total = int(r.headers.get('content-length', 0)) or best_size
        downloaded = 0
        with open(dest_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r  Downloading: {downloaded/(1024*1024):.0f}/{total/(1024*1024):.0f} MB ({pct:.0f}%)", end="", flush=True)
    print()
    return dest_path


def poll_r2_uploads():
    """Check R2 uploads/ prefix for videos uploaded via the web form."""
    import tempfile
    from dotenv import load_dotenv
    load_dotenv(os.path.join(PROJECT_ROOT, '.env'))
    import importlib, config.settings
    importlib.reload(config.settings)
    from storage.r2_client import R2Client

    client = R2Client()
    keys = client.list(prefix='uploads/', max_keys=100)

    meta_keys = [k for k in keys if k.endswith('.json')]
    if not meta_keys:
        return []

    processed = []
    for meta_key in meta_keys:
        tmp_meta = None
        try:
            # Download and parse metadata
            tmp_meta = tempfile.NamedTemporaryFile(suffix='.json', delete=False).name
            client.download(meta_key, tmp_meta)
            with open(tmp_meta) as f:
                meta = json.load(f)

            if meta.get('status') != 'pending':
                continue

            upload_id = meta['id']

            # Update metadata to processing
            meta['status'] = 'processing'
            with open(tmp_meta, 'w') as f:
                json.dump(meta, f)
            client.upload(tmp_meta, meta_key, content_type='application/json')

            if meta.get('type') == 'icloud_link':
                # iCloud link — download directly from iCloud
                icloud_url = meta['url']
                log(f"Found iCloud link upload (id={upload_id})")
                # Use a temp filename until we know the real one
                raw_path = os.path.join(RAW_DIR, f"icloud_{upload_id}.mov")
                download_icloud_share(icloud_url, raw_path)
                stem = f"icloud_{upload_id}"
                video_key = None  # No R2 video file to clean up
            else:
                # Direct file upload — download from R2
                filename = meta['filename']
                stem = os.path.splitext(filename)[0]
                video_key = meta.get('key', f'uploads/{upload_id}.mov')
                log(f"Found uploaded video: {filename} (id={upload_id})")
                raw_path = os.path.join(RAW_DIR, filename)
                log(f"  Downloading from R2...")
                client.download(video_key, raw_path)

            fps = verify_fps(raw_path)
            log(f"  Downloaded: {fps:.0f}fps, {os.path.getsize(raw_path)/(1024*1024):.0f} MB")

            # Run full pipeline
            success = process_video(stem, raw_path)

            if success:
                log(f"  COMPLETE: {stem}")
                if video_key:
                    client.delete(video_key)
                client.delete(meta_key)
                update_r2_index()
                processed.append(stem)
            else:
                log(f"  FAILED: {stem}")
                meta['status'] = 'failed'
                with open(tmp_meta, 'w') as f:
                    json.dump(meta, f)
                client.upload(tmp_meta, meta_key, content_type='application/json')

        except Exception as e:
            log(f"  Error processing upload {meta_key}: {e}")
        finally:
            if tmp_meta and os.path.exists(tmp_meta):
                os.unlink(tmp_meta)

    return processed


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Watch for new 240fps videos and process them")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be processed")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PREPROCESSED_DIR, exist_ok=True)
    os.makedirs(POSES_DIR, exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, "detections"), exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, "exports"), exist_ok=True)

    # Poll R2 uploads first (user-initiated, higher priority)
    log("Checking R2 for uploaded videos...")
    try:
        r2_processed = poll_r2_uploads()
        if r2_processed:
            log(f"Processed {len(r2_processed)} R2 upload(s): {', '.join(r2_processed)}")
        else:
            log("No pending R2 uploads.")
    except Exception as e:
        log(f"R2 poll error (continuing to Photos scan): {e}")

    state = load_state()
    already = set(state["processed"])

    log("Scanning Photos for new 240fps slo-mo videos...")
    new_videos = find_new_slomo_videos(already)

    if not new_videos:
        log("No new slo-mo videos found.")
        return

    log(f"Found {len(new_videos)} new video(s):")
    for _, name, stem, _, size, _ in new_videos:
        log(f"  {name} ({size / (1024*1024):.0f} MB)")

    if args.dry_run:
        return

    for asset, name, stem, original, size, resources in new_videos:
        log(f"\n{'='*60}")
        log(f"Processing: {name}")

        # Download original 240fps
        raw_path = os.path.join(RAW_DIR, name)
        if os.path.exists(raw_path):
            fps = verify_fps(raw_path)
            if fps > 60:
                log(f"  Already downloaded ({fps:.0f}fps)")
            else:
                log(f"  Existing file is {fps:.0f}fps (edited version), re-downloading original")
                if not download_original(original, raw_path):
                    log(f"  FAILED to download {name}, skipping")
                    continue
        else:
            if not download_original(original, raw_path):
                log(f"  FAILED to download {name}, skipping")
                continue

        fps = verify_fps(raw_path)
        log(f"  Downloaded: {fps:.0f}fps, {os.path.getsize(raw_path)/(1024*1024):.0f} MB")

        if fps <= 60:
            log(f"  Not a 240fps video ({fps:.0f}fps), skipping pipeline")
            state["processed"].append(stem)
            save_state(state)
            continue

        # Run full pipeline
        success = process_video(stem, raw_path)
        if success:
            log(f"  COMPLETE: {stem}")
            update_r2_index()
        else:
            log(f"  FAILED: {stem} (partial output may exist)")

        # Mark as processed either way (don't retry failures automatically)
        state["processed"].append(stem)
        save_state(state)

    log(f"\nAll done. Processed {len(new_videos)} video(s).")


if __name__ == "__main__":
    main()
