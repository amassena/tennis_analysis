#!/usr/bin/env python3
"""Download original (unedited) slo-mo videos from Photos via PhotoKit.

When iPhone records slo-mo, Photos stores:
  - Type 2 (video): Original 240fps camera recording
  - Type 6 (fullSizeVideo): Edited/flattened 30fps version with slo-mo baked in
  - Type 7 (adjustmentData): Edit instructions (Adjustments.plist)
  - Type 16: AAE sidecar

This script fetches the Type 2 ORIGINAL using PHAssetResourceManager.
"""
import os
import sys
import time
import subprocess
from Photos import (
    PHAsset,
    PHAssetResource,
    PHAssetResourceManager,
    PHAssetMediaTypeVideo,
    PHAssetResourceRequestOptions,
)
from Foundation import NSURL

RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "raw")


def find_matching_assets(name_filters=None):
    """Find video assets matching name filters."""
    results = PHAsset.fetchAssetsWithMediaType_options_(PHAssetMediaTypeVideo, None)
    assets = []
    seen = set()

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
            for r in resources:
                name = str(r.originalFilename())
                break
        if not name:
            continue

        # Filter by name
        if name_filters:
            if not any(f in name for f in name_filters):
                continue

        # Only take slo-mo versions (subtypes & 0x20000, which have adjustments)
        subtypes = asset.mediaSubtypes()
        has_adjustments = any(r.type() == 6 for r in resources)

        # Dedup: prefer the version with adjustments (the slo-mo one)
        if name in seen and not has_adjustments:
            continue
        if name in seen and has_adjustments:
            # Replace the previous entry
            assets = [(a, n, r) for a, n, r in assets if n != name]
        seen.add(name)

        assets.append((asset, name, list(resources)))

    return sorted(assets, key=lambda x: x[1])


def download_resource(resource, dest_path):
    """Download a PHAssetResource to disk."""
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

    size = int(resource.valueForKey_("fileSize") or 0)
    size_mb = size / (1024 * 1024)

    manager.writeDataForAssetResource_toFile_options_completionHandler_(
        resource, dest_url, options, completion_handler
    )

    start = time.time()
    while not done["finished"]:
        time.sleep(1)
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
        print(f"  ERROR: {done['error']}")
        return False
    return True


def verify_fps(filepath):
    """Check frame rate of a video file."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-select_streams", "v",
             "-show_entries", "stream=r_frame_rate,nb_frames",
             "-show_entries", "format=duration",
             "-of", "json", filepath],
            capture_output=True, text=True, timeout=10
        )
        import json
        data = json.loads(result.stdout)
        stream = data.get("streams", [{}])[0]
        fps_str = stream.get("r_frame_rate", "0/1")
        if "/" in fps_str:
            num, den = fps_str.split("/")
            fps = int(num) / int(den)
        else:
            fps = float(fps_str)
        nb_frames = stream.get("nb_frames", "?")
        duration = data.get("format", {}).get("duration", "?")
        return fps, nb_frames, duration
    except Exception:
        return 0, "?", "?"


def main():
    os.makedirs(RAW_DIR, exist_ok=True)

    name_filters = sys.argv[1:] if len(sys.argv) > 1 else None

    print("Querying Photos library...")
    assets = find_matching_assets(name_filters)
    print(f"Found {len(assets)} matching video(s)\n")

    for asset, name, resources in assets:
        stem = os.path.splitext(name)[0]
        dest = os.path.join(RAW_DIR, name)
        is_slomo = any(r.type() == 6 for r in resources)

        print(f"[{name}] {'SLO-MO' if is_slomo else 'normal'}")

        # Show resources
        type_names = {1: "photo", 2: "video(original)", 3: "audio",
                      5: "adjustmentBase", 6: "fullSizeVideo(edited)",
                      7: "adjustmentData", 9: "adjustmentBasePairedVideo",
                      10: "fullSizePairedVideo", 16: "AAE"}
        for r in resources:
            rtype = r.type()
            fname = str(r.originalFilename())
            sz = int(r.valueForKey_("fileSize") or 0) // (1024 * 1024)
            tname = type_names.get(rtype, f"type_{rtype}")
            print(f"  {tname}: {fname} ({sz}MB)")

        # Check if already downloaded correctly
        if os.path.exists(dest):
            fps, nb, dur = verify_fps(dest)
            if is_slomo and fps > 60:
                print(f"  [SKIP] Already have original ({fps:.0f}fps, {nb} frames)\n")
                continue
            elif not is_slomo and fps > 0:
                print(f"  [SKIP] Already have file ({fps:.0f}fps)\n")
                continue
            else:
                print(f"  Replacing edited version ({fps:.0f}fps) with original")

        # Download the original (type 2)
        original = None
        for r in resources:
            if r.type() == 2:
                original = r
                break

        if original is None:
            print("  [ERROR] No original resource found!\n")
            continue

        success = download_resource(original, dest)
        if success:
            fps, nb, dur = verify_fps(dest)
            print(f"  Result: {fps:.0f}fps, {nb} frames, {dur}s")
        print()


if __name__ == "__main__":
    main()
