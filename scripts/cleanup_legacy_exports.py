#!/usr/bin/env python3
"""Phase 4a back-catalog cleanup: delete the per-type / *_slowmo mp4s
that the new playlist-filter player no longer needs.

After Phase 2 the GPU pipeline stopped emitting these for *new* videos,
but ~100 existing videos still have all of them sitting in R2:

    highlights/<u_hash>/<vid>/<vid>_forehands.mp4         ← legacy
    highlights/<u_hash>/<vid>/<vid>_forehands_slowmo.mp4  ← legacy
    highlights/<u_hash>/<vid>/<vid>_backhands.mp4         ← legacy
    highlights/<u_hash>/<vid>/<vid>_backhands_slowmo.mp4  ← legacy
    highlights/<u_hash>/<vid>/<vid>_serves.mp4            ← legacy
    highlights/<u_hash>/<vid>/<vid>_serves_slowmo.mp4     ← legacy
    highlights/<u_hash>/<vid>/<vid>_volleys.mp4           ← legacy
    highlights/<u_hash>/<vid>/<vid>_volleys_slowmo.mp4    ← legacy
    highlights/<u_hash>/<vid>/<vid>_timeline_slowmo.mp4   ← legacy
    highlights/<u_hash>/<vid>/<vid>_rally_slowmo.mp4      ← legacy
    highlights/<u_hash>/<vid>/<vid>_grouped*.mp4          ← legacy
    highlights/<u_hash>/<vid>/<vid>_highlights*.mp4       ← legacy

Kept (still in active use):
    highlights/<u_hash>/<vid>/<vid>_timeline.mp4      ← source of truth
    highlights/<u_hash>/<vid>/<vid>_rally.mp4         ← until Rally chip ships
    highlights/<u_hash>/<vid>/<vid>_tracked.mp4       ← optional feature
    highlights/<u_hash>/<vid>/meta.json
    highlights/<u_hash>/<vid>/shots.json
    highlights/<u_hash>/<vid>/coaching.json
    highlights/<u_hash>/<vid>/sequences/...
    highlights/<u_hash>/thumbs/<vid>.jpg

Usage:
    # 1. Always start with a dry run to see the impact
    python scripts/cleanup_legacy_exports.py --user u_ae629639

    # 2. Once you've reviewed, execute
    python scripts/cleanup_legacy_exports.py --user u_ae629639 --execute

    # All users in one go (dry-run by default)
    python scripts/cleanup_legacy_exports.py --all-users

The script never deletes timeline / rally / tracked / meta / shots /
coaching / thumbs — those are the live UX surface. Anything else under
highlights/<u_hash>/<vid>/ ending in `.mp4` is treated as legacy.
"""

from __future__ import annotations
import argparse
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / '.env')

from storage.r2_client import R2Client  # noqa: E402

# Variants we keep. Anything else under highlights/<hash>/<vid>/ ending
# in .mp4 is fair game for the cleanup.
KEEP_VARIANTS = {'timeline', 'rally', 'tracked'}

USER_HASH_RE = re.compile(r'^u_[a-f0-9]{8}$')


def list_legacy_keys(c: R2Client, user_hash: str) -> list[tuple[str, int]]:
    """Return [(key, size_bytes)] for every legacy .mp4 under this user.

    Path layout: highlights/<user_hash>/<vid>/<vid>_<variant>.mp4
    We use the parent directory as the canonical <vid> (since the video
    ID itself can contain underscores like `IMG_0991` or `iphone_9ca0a615`)
    and strip that prefix from the filename to extract the variant.
    """
    prefix = f'highlights/{user_hash}/'
    out: list[tuple[str, int]] = []
    paginator = c.client.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=c.bucket_name, Prefix=prefix):
        for obj in page.get('Contents', []):
            key = obj['Key']
            size = obj['Size']
            if not key.endswith('.mp4'):
                continue
            rel = key[len(prefix):]
            parts = rel.split('/')
            # Skip nested folders (sequences/, comparisons/) and the
            # bare-prefix thumbs/ directory.
            if len(parts) != 2:
                continue
            vid, fname = parts
            stem = fname[:-4]  # strip .mp4
            expected_prefix = f'{vid}_'
            if not stem.startswith(expected_prefix):
                # Defensive: file doesn't follow <vid>_<variant>.mp4 — leave it alone.
                continue
            variant = stem[len(expected_prefix):]
            if variant in KEEP_VARIANTS:
                continue  # timeline / rally / tracked stay
            out.append((key, size))
    return out


def fmt_bytes(n: int) -> str:
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if n < 1024:
            return f'{n:.1f} {unit}'
        n /= 1024
    return f'{n:.1f} PB'


def discover_user_hashes(c: R2Client) -> list[str]:
    """Find every u_* prefix under highlights/."""
    paginator = c.client.get_paginator('list_objects_v2')
    seen: set[str] = set()
    for page in paginator.paginate(
        Bucket=c.bucket_name, Prefix='highlights/', Delimiter='/'
    ):
        for cp in page.get('CommonPrefixes', []):
            p = cp['Prefix']
            # p looks like 'highlights/u_xxxxxxxx/'
            sub = p[len('highlights/'):].rstrip('/')
            if USER_HASH_RE.match(sub):
                seen.add(sub)
    return sorted(seen)


def cleanup_one_user(
    c: R2Client, user_hash: str, *, execute: bool, verbose: bool = True
) -> tuple[int, int]:
    """Returns (count_deleted_or_listed, total_bytes)."""
    items = list_legacy_keys(c, user_hash)
    if not items:
        if verbose:
            print(f'  {user_hash}: nothing to clean')
        return 0, 0
    total = sum(s for _, s in items)
    # Group by variant for the summary line so we know what's going.
    by_variant: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for k, s in items:
        # Key: highlights/<u_hash>/<vid>/<vid>_<variant>.mp4
        parts = k.split('/')
        vid = parts[-2]
        stem = parts[-1][:-4]
        variant = stem[len(vid) + 1:] if stem.startswith(vid + '_') else stem
        by_variant[variant].append((k, s))
    if verbose:
        head = f'  {user_hash}: {len(items)} files / {fmt_bytes(total)}'
        print(head)
        for variant in sorted(by_variant):
            v_items = by_variant[variant]
            v_size = sum(s for _, s in v_items)
            print(f'    - {variant}: {len(v_items)} files / {fmt_bytes(v_size)}')
    if execute:
        # Batch delete (R2 supports up to 1000 keys per delete_objects call).
        for i in range(0, len(items), 1000):
            chunk = items[i:i + 1000]
            c.client.delete_objects(
                Bucket=c.bucket_name,
                Delete={'Objects': [{'Key': k} for k, _ in chunk]},
            )
        if verbose:
            print(f'    ✓ deleted')
    return len(items), total


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--user', help='Single user hash (e.g. u_ae629639)')
    g.add_argument('--all-users', action='store_true',
                   help='Discover and clean all u_* prefixes under highlights/')
    ap.add_argument('--execute', action='store_true',
                    help='Actually delete. Without this flag, runs as a dry run.')
    args = ap.parse_args()

    c = R2Client()

    if args.user:
        users = [args.user]
    else:
        users = discover_user_hashes(c)
        print(f'Discovered {len(users)} user prefix(es): {users}')

    print()
    print(f'Mode: {"EXECUTE (deleting)" if args.execute else "DRY RUN (no deletes)"}')
    print()

    grand_count = 0
    grand_bytes = 0
    for u in users:
        count, total = cleanup_one_user(c, u, execute=args.execute)
        grand_count += count
        grand_bytes += total
        print()

    print('=' * 60)
    print(f'Total: {grand_count} files / {fmt_bytes(grand_bytes)} '
          f'{"deleted" if args.execute else "would be deleted"}')
    if not args.execute:
        print()
        print('To execute, re-run with --execute')


if __name__ == '__main__':
    main()
