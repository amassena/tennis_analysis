"""iPhone Shortcut ingest poller.

Runs on Hetzner alongside the coordinator. Polls R2 for upload markers
written by the Cloudflare Worker (`uploads/iphone_*.json`), creates jobs in
the coordinator state, and flips the marker status so we don't re-create.

This exists because Cloudflare Workers can't fetch the coordinator's bare
origin IP directly (CF error 1003). The Worker writes a marker file in R2;
this poller is the bridge into the coordinator.

Usage:
    python -m coordinator.iphone_upload_poller [--once] [--interval=30]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

# Reuse the project's state backend so we operate on the same SQLite as
# the coordinator service.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from coordinator.state import VideoJob, VideoStatus  # noqa: E402
from coordinator.state_sqlite import SQLiteStateBackend  # noqa: E402

LOG = logging.getLogger("iphone_upload_poller")

DB_PATH = os.environ.get("COORDINATOR_DB", "/opt/tennis/coordinator.db")
R2_BUCKET = os.environ.get("R2_BUCKET", "tennis-videos")
ENV_PATH = Path(os.environ.get("TENNIS_ENV_FILE", "/opt/tennis/.env"))


def _load_env() -> dict[str, str]:
    out: dict[str, str] = {}
    if not ENV_PATH.exists():
        return out
    for line in ENV_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def _r2_client():
    env = _load_env()
    account_id = env.get("CF_ACCOUNT_ID")
    access_key = env.get("CF_R2_ACCESS_KEY_ID")
    secret_key = env.get("CF_R2_SECRET_ACCESS_KEY")
    if not all([account_id, access_key, secret_key]):
        raise RuntimeError(
            "Missing CF_ACCOUNT_ID / CF_R2_ACCESS_KEY_ID / CF_R2_SECRET_ACCESS_KEY in env"
        )
    return boto3.client(
        "s3",
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
        region_name="us-east-1",
    )


async def _scan_once(state: SQLiteStateBackend, r2) -> int:
    """One scan pass. Returns number of newly-registered jobs."""
    paginator = r2.get_paginator("list_objects_v2")
    new_jobs = 0
    for page in paginator.paginate(Bucket=R2_BUCKET, Prefix="uploads/iphone_"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".json"):
                continue
            try:
                resp = r2.get_object(Bucket=R2_BUCKET, Key=key)
                marker = json.loads(resp["Body"].read())
            except (ClientError, json.JSONDecodeError) as e:
                LOG.warning("skipping %s: %s", key, e)
                continue

            if marker.get("status") != "awaiting_coordinator":
                continue

            video_id = marker.get("video_id")
            asset_id = marker.get("asset_id")
            filename = marker.get("filename")
            if not (video_id and asset_id and filename):
                LOG.warning("marker %s missing required fields, skipping", key)
                continue

            # Create the job — idempotent via icloud_asset_id dedup in coordinator.
            existing = await state.job_exists(asset_id)
            if existing:
                LOG.info(
                    "marker %s: job already exists for asset_id=%s, marking registered",
                    key,
                    asset_id,
                )
            else:
                job = VideoJob(
                    video_id=video_id,
                    icloud_asset_id=asset_id,
                    filename=filename,
                    status=VideoStatus.PENDING,
                    album_name=marker.get("source", "iphone_shortcut"),
                )
                await state.add_job(job)
                new_jobs += 1
                LOG.info("created job %s (file=%s) from marker %s", video_id, filename, key)

            # Flip marker status so we don't reprocess.
            marker["status"] = "coordinator_registered"
            marker["coordinator_registered_at"] = datetime.now(timezone.utc).isoformat()
            r2.put_object(
                Bucket=R2_BUCKET,
                Key=key,
                Body=json.dumps(marker).encode(),
                ContentType="application/json",
            )

    return new_jobs


async def _main(once: bool, interval: int) -> None:
    state = SQLiteStateBackend(DB_PATH)
    await state.init()
    r2 = _r2_client()
    LOG.info("iphone_upload_poller starting (db=%s, bucket=%s, interval=%ds, once=%s)",
             DB_PATH, R2_BUCKET, interval, once)
    try:
        while True:
            try:
                n = await _scan_once(state, r2)
                if n:
                    LOG.info("scan complete: %d new job(s)", n)
            except Exception:
                LOG.exception("scan failed; will retry")
            if once:
                break
            await asyncio.sleep(interval)
    finally:
        await state.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true",
                        help="Single scan then exit (cron mode)")
    parser.add_argument("--interval", type=int, default=30,
                        help="Seconds between scans in daemon mode (default 30)")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    asyncio.run(_main(args.once, args.interval))


if __name__ == "__main__":
    main()
