#!/usr/bin/env python3
"""Upload slo-mo videos from local Photos library to the tennis pipeline.

Replaces the Hetzner pyicloud watcher and the iOS Shortcut path. Runs on Mac
where iCloud Photos is signed in and synced — every tennis video the user
records as slo-mo is automatically a candidate for upload.

How it works:
  1. Enumerate Photos library, filter to assets with the slo-mo (high frame
     rate) subtype.
  2. Use each asset's localIdentifier as the deterministic dedup key.
  3. Hit the Worker `/api/upload/iphone/check` endpoint to ask whether this
     asset has already been uploaded. If yes, skip.
  4. Otherwise, fetch the original (type-2 resource) via PHAssetResourceManager
     to a temp file, POST it to `/api/upload/iphone`, delete the temp file.
  5. The Worker writes the source MOV to R2 and a marker JSON;
     tennis-iphone-poller.service on Hetzner picks up the marker and creates
     a coordinator job. The GPU pipeline takes it from there.

Ergonomics:
  - One-shot run by default. Use `--watch N` to loop forever, scanning every
    N seconds.
  - `--limit N` caps how many uploads happen in one run (useful first time
    when there are a lot to backfill).
  - `--dry-run` lists what would be uploaded without actually uploading.

Auth: token comes from /tmp/iphone_upload_token.txt (or override with --token).

Run:
    .venv/bin/python scripts/upload_tennis_album.py
    .venv/bin/python scripts/upload_tennis_album.py --dry-run
    .venv/bin/python scripts/upload_tennis_album.py --watch 60
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from Photos import (
    PHAsset,
    PHAssetMediaTypeVideo,
    PHAssetResource,
    PHAssetResourceManager,
    PHAssetResourceRequestOptions,
)
from Foundation import NSURL

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TOKEN_PATH = Path("/tmp/iphone_upload_token.txt")
WORKER_BASE = "https://tennis.playfullife.com"


def load_token(token_arg: str | None) -> str:
    if token_arg:
        return token_arg.strip()
    if DEFAULT_TOKEN_PATH.exists():
        return DEFAULT_TOKEN_PATH.read_text().strip()
    raise SystemExit(
        f"Token not found at {DEFAULT_TOKEN_PATH} and --token not provided. "
        "Generate one with: python -c \"import secrets; print(secrets.token_urlsafe(32))\""
    )


def find_slomo_assets(since_epoch: float | None = None):
    """Return list of (asset, original_filename, resources) for every slo-mo video.

    Detection: a slo-mo asset has a resource of type 6 (the edited "flat" 30fps
    version that Photos auto-generates from the 240fps original). This is more
    reliable than mediaSubtypes flag, which sometimes doesn't survive iCloud
    sync from iPhone → Mac.

    If since_epoch is provided, only assets with creationDate >= since_epoch
    are returned. Used to skip non-tennis slo-mo videos from years past.
    """
    results = PHAsset.fetchAssetsWithMediaType_options_(PHAssetMediaTypeVideo, None)
    out = []
    for i in range(results.count()):
        asset = results.objectAtIndex_(i)
        resources = list(PHAssetResource.assetResourcesForAsset_(asset))
        if not any(r.type() == 6 for r in resources):
            continue  # not slo-mo
        if since_epoch is not None:
            d = asset.creationDate()
            if d is None or float(d.timeIntervalSince1970()) < since_epoch:
                continue
        # Find original filename (type 2 = original video resource)
        name = None
        for r in resources:
            if r.type() == 2:
                name = str(r.originalFilename())
                break
        if not name and resources:
            name = str(resources[0].originalFilename())
        if not name:
            continue
        out.append((asset, name, resources))
    # Sort by creation date (newest first) — not strictly required but helps
    # the first --limit N run prioritize recent recordings.
    out.sort(
        key=lambda t: (t[0].creationDate() or 0).timeIntervalSince1970()
        if t[0].creationDate() is not None else 0,
        reverse=True,
    )
    return out


UA = "tennis-upload/1.0 (Mac; PhotoKit)"


def already_uploaded(token: str, asset_id: str) -> bool:
    """Hit /api/upload/iphone/check; return True if Worker says it's uploaded."""
    url = f"{WORKER_BASE}/api/upload/iphone/check?asset_id={urllib.parse.quote(asset_id, safe='')}"
    req = urllib.request.Request(
        url,
        method="GET",
        headers={"Authorization": f"Bearer {token}", "User-Agent": UA},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read().decode()
    except Exception as e:
        print(f"  [WARN] /check failed for {asset_id[:32]}…: {e}", file=sys.stderr)
        return False
    import json

    try:
        return bool(json.loads(data).get("uploaded", False))
    except json.JSONDecodeError:
        return False


def write_resource_to_tempfile(resource) -> Path:
    """Use PHAssetResourceManager to materialize an asset resource to a temp file."""
    manager = PHAssetResourceManager.defaultManager()
    options = PHAssetResourceRequestOptions.alloc().init()
    options.setNetworkAccessAllowed_(True)

    suffix = Path(str(resource.originalFilename())).suffix or ".mov"
    fd, tmp_path = tempfile.mkstemp(suffix=suffix, prefix="tennis_upload_")
    os.close(fd)
    os.unlink(tmp_path)  # PhotoKit refuses to overwrite — start clean.

    dest_url = NSURL.fileURLWithPath_(tmp_path)
    state = {"finished": False, "error": None}

    def completion(error):
        if error:
            state["error"] = str(error)
        state["finished"] = True

    size_bytes = int(resource.valueForKey_("fileSize") or 0)
    size_mb = size_bytes / (1024 * 1024)

    manager.writeDataForAssetResource_toFile_options_completionHandler_(
        resource, dest_url, options, completion
    )

    start = time.time()
    while not state["finished"]:
        time.sleep(1)
        elapsed = time.time() - start
        if os.path.exists(tmp_path):
            cur = os.path.getsize(tmp_path) / (1024 * 1024)
            pct = (cur / size_mb * 100) if size_mb > 0 else 0
            print(f"\r    pulling: {cur:.0f}/{size_mb:.0f} MB ({pct:.0f}%)", end="", flush=True)
        if elapsed > 7200:
            print("\n    [TIMEOUT] resource pull > 2h")
            raise TimeoutError("PhotoKit resource pull timed out")
    print()
    if state["error"]:
        raise RuntimeError(f"PhotoKit error: {state['error']}")
    return Path(tmp_path)


def upload_to_worker(token: str, asset_id: str, filename: str,
                     created_at: str, mov_path: Path) -> dict:
    """POST mov bytes to /api/upload/iphone; return parsed JSON response."""
    url = f"{WORKER_BASE}/api/upload/iphone"
    with mov_path.open("rb") as f:
        body = f.read()
    req = urllib.request.Request(
        url,
        method="POST",
        data=body,
        headers={
            "Authorization": f"Bearer {token}",
            "X-Asset-Id": asset_id,
            "X-Filename": filename,
            "X-Created-At": created_at,
            "Content-Type": "video/quicktime",
            "User-Agent": UA,
        },
    )
    import json

    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body_text = e.read().decode(errors="replace")[:500]
        return {"error": f"HTTP {e.code}", "body": body_text}


def iso_creation_date(asset) -> str:
    d = asset.creationDate()
    if d is None:
        return ""
    # NSDate → epoch → ISO 8601 UTC
    epoch = float(d.timeIntervalSince1970())
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def scan_once(token: str, dry_run: bool, limit: int | None,
              since_epoch: float | None = None) -> int:
    print("Querying Photos library for slo-mo videos…")
    assets = find_slomo_assets(since_epoch)
    if since_epoch is not None:
        since_str = datetime.fromtimestamp(since_epoch, tz=timezone.utc).date().isoformat()
        print(f"  found {len(assets)} slo-mo asset(s) since {since_str}")
    else:
        print(f"  found {len(assets)} slo-mo asset(s) total")

    uploaded = 0
    skipped = 0
    failed = 0

    for asset, name, resources in assets:
        if limit is not None and uploaded >= limit:
            print(f"  --limit {limit} reached; stopping.")
            break

        asset_id = str(asset.localIdentifier())
        created = iso_creation_date(asset)
        print(f"\n[{name}] asset_id={asset_id[:24]}… created={created}")

        if already_uploaded(token, asset_id):
            print("  already uploaded, skipping.")
            skipped += 1
            continue

        # Find the original resource (type 2). For slo-mo this is the 240fps source.
        original = None
        for r in resources:
            if r.type() == 2:
                original = r
                break
        if original is None and resources:
            original = resources[0]
        if original is None:
            print("  [ERROR] no resource available — skipping.")
            failed += 1
            continue

        if dry_run:
            sz = int(original.valueForKey_("fileSize") or 0) // (1024 * 1024)
            print(f"  [DRY-RUN] would upload {sz} MB")
            continue

        # Pull bytes locally so we can stream them with a known length.
        try:
            tmp_path = write_resource_to_tempfile(original)
        except Exception as e:
            print(f"  [ERROR] PhotoKit pull failed: {e}")
            failed += 1
            continue

        try:
            print(f"  uploading {tmp_path.stat().st_size / (1024*1024):.0f} MB to Worker…")
            resp = upload_to_worker(token, asset_id, name, created, tmp_path)
            if "video_id" in resp:
                print(f"  → {resp['status']} (video_id={resp['video_id']})")
                uploaded += 1
            else:
                print(f"  [FAIL] worker said: {resp}")
                failed += 1
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

    print(f"\nsummary: uploaded={uploaded} skipped={skipped} failed={failed}")
    return uploaded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--token", help="Bearer token (default: read /tmp/iphone_upload_token.txt)")
    parser.add_argument("--dry-run", action="store_true", help="List what would be uploaded; don't upload")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap number of uploads in this run")
    parser.add_argument("--watch", type=int, default=0,
                        help="Daemon mode: scan every N seconds (default 0 = single scan)")
    parser.add_argument("--since", default="2026-04-01",
                        help="Only consider videos recorded on/after this date (YYYY-MM-DD). "
                             "Defaults to 2026-04-01 to skip non-tennis slo-mo from years past. "
                             "Pass --since 1970-01-01 to include everything.")
    args = parser.parse_args()

    token = load_token(args.token)

    since_epoch = None
    if args.since:
        try:
            since_dt = datetime.strptime(args.since, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            since_epoch = since_dt.timestamp()
        except ValueError:
            raise SystemExit(f"--since must be YYYY-MM-DD, got {args.since!r}")

    if args.watch > 0:
        print(f"watch mode: scanning every {args.watch}s. Ctrl-C to stop.")
        while True:
            try:
                scan_once(token, args.dry_run, args.limit, since_epoch)
            except Exception as e:
                print(f"[scan failed: {e}] continuing.")
            time.sleep(args.watch)
    else:
        scan_once(token, args.dry_run, args.limit, since_epoch)


if __name__ == "__main__":
    main()
