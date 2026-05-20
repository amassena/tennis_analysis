#!/usr/bin/env python3
"""Upload curated pro clips from pros/<slug>/*.mp4 to R2 under pros/<slug>/<file>.

Production consumers (scripts/pro_comparison.py, gallery) resolve clips
by key `pros/<slug>/<filename>`. This script puts every locally-curated
clip there. Idempotent — skips clips already in R2 (head_object check).

Usage:
    .venv/bin/python scripts/upload_pro_clips_to_r2.py
    .venv/bin/python scripts/upload_pro_clips_to_r2.py --players sinner,swiatek
    .venv/bin/python scripts/upload_pro_clips_to_r2.py --dry-run
    .venv/bin/python scripts/upload_pro_clips_to_r2.py --force  # re-upload even if exists
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv
# CF_R2_* credentials live in the main tennis_analysis repo's .env (worktrees
# each have their own working dir but share the same external secrets).
for candidate in (REPO_ROOT / ".env", Path.home() / "tennis_analysis" / ".env"):
    if candidate.exists():
        load_dotenv(candidate)
        break

from storage.r2_client import R2Client  # noqa: E402

PROS_DIR = REPO_ROOT / "pros"


def discover_clips(players: list[str] | None) -> list[tuple[str, Path]]:
    out = []
    for slug_dir in sorted(PROS_DIR.iterdir()):
        if not slug_dir.is_dir() or slug_dir.name == "_raw":
            continue
        if players and slug_dir.name not in players:
            continue
        for mp4 in sorted(slug_dir.glob("*.mp4")):
            out.append((slug_dir.name, mp4))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--players", help="Comma-separated slugs (default: all on disk)")
    ap.add_argument("--dry-run", action="store_true", help="List what would upload, don't push")
    ap.add_argument("--force", action="store_true", help="Re-upload even if remote exists")
    args = ap.parse_args()

    players = None
    if args.players:
        players = [s.strip() for s in args.players.split(",") if s.strip()]

    clips = discover_clips(players)
    if not clips:
        print("No clips found.")
        return 0

    print(f"Found {len(clips)} clips across {len({s for s, _ in clips})} pros")

    if args.dry_run:
        for slug, mp4 in clips:
            print(f"  pros/{slug}/{mp4.name}")
        return 0

    client = R2Client()
    skip = 0
    ok = 0
    fail = 0
    total_bytes = 0
    for i, (slug, mp4) in enumerate(clips, 1):
        remote_key = f"pros/{slug}/{mp4.name}"
        if not args.force and client.exists(remote_key):
            skip += 1
            if i % 50 == 0:
                print(f"  [{i}/{len(clips)}] (skip-exists, running total: {ok} ok / {skip} skip / {fail} fail)")
            continue
        size = mp4.stat().st_size
        try:
            client.upload(str(mp4), remote_key)
            ok += 1
            total_bytes += size
            print(f"  [{i}/{len(clips)}] uploaded {remote_key} ({size//1024} KB)")
        except Exception as e:
            fail += 1
            print(f"  [{i}/{len(clips)}] FAIL {remote_key}: {e}", file=sys.stderr)

    print(f"\n=== {ok} uploaded, {skip} skipped, {fail} failed; "
          f"{total_bytes/(1024*1024):.1f} MB total ===")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
