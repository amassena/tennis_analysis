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
    """Regenerate the browsable index.html on R2."""
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(PROJECT_ROOT, '.env'))
        import importlib, config.settings
        importlib.reload(config.settings)
        from storage.r2_client import R2Client
        import tempfile

        c = R2Client()
        keys = c.list(prefix='highlights/', max_keys=1000)

        videos = {}
        for k in keys:
            if k.endswith('index.html'):
                continue
            parts = k.split('/')
            if len(parts) == 3:
                videos.setdefault(parts[1], []).append(parts[2])

        rows = ''
        for vid in sorted(videos.keys(), reverse=True):
            links = []
            for f in sorted(videos[vid]):
                label = f.replace(vid + '_', '').replace('.mp4', '')
                url = f'https://media.playfullife.com/highlights/{vid}/{f}'
                links.append(f'<a href="{url}">{label}</a>')
            rows += f'<tr><td><strong>{vid}</strong></td><td>{" &middot; ".join(links)}</td></tr>\n'

        html = f'''<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Tennis Highlights</title>
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif; max-width: 800px; margin: 40px auto; padding: 0 20px; background: #111; color: #eee; }}
  h1 {{ color: #FF8C00; }}
  table {{ width: 100%; border-collapse: collapse; }}
  td {{ padding: 10px 8px; border-bottom: 1px solid #333; }}
  a {{ color: #5dade2; text-decoration: none; margin: 0 2px; }}
  a:hover {{ text-decoration: underline; }}
</style>
</head><body>
<h1>Tennis Highlights</h1>
<table>{rows}</table>
</body></html>
'''
        tmp = tempfile.NamedTemporaryFile(suffix='.html', delete=False, mode='w')
        tmp.write(html)
        tmp.close()
        c.upload(tmp.name, 'highlights/index.html', content_type='text/html')
        os.unlink(tmp.name)
        log("  Updated R2 index.html")
    except Exception as e:
        log(f"  Warning: failed to update R2 index: {e}")


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
