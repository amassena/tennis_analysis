#!/usr/bin/env python3
"""Harvest raw YouTube highlight reels for pros with empty `clips: []` in pros/index.json.

Stages downloaded material at `pros/_raw/<slug>/<video_id>.mp4` (Mac-local, gitignored).
Phase 2 (shot detection) and Phase 3 (curation + ingest to R2) consume this raw pool.

Usage:
    # Dry-run for a few pros: show candidates without downloading
    .venv/bin/python scripts/fetch_pro_highlights.py --players sinner,wawrinka,sabalenka --dry-run

    # Actual download
    .venv/bin/python scripts/fetch_pro_highlights.py --players sinner,wawrinka,sabalenka

    # All pros with empty clips
    .venv/bin/python scripts/fetch_pro_highlights.py --all-empty

License framing: single-user personal/research use of public ATP/WTA highlights via yt-dlp.
Preferred channels are official tour and slow-motion analysis accounts (see PREFERRED_CHANNELS).
See CLAUDE.md for the documented rationale.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PROS_DIR = REPO_ROOT / "pros"
RAW_DIR = PROS_DIR / "_raw"
INDEX_PATH = PROS_DIR / "index.json"

YT_DLP = shutil.which("yt-dlp") or "/opt/homebrew/bin/yt-dlp"

# Channels we trust as source quality: official tour streams + slow-mo analysis.
# Lowercased substring match against the candidate's `channel` / `uploader` field.
PREFERRED_CHANNELS = [
    "atp tour",
    "wta",
    "tennis tv",
    "tennistv",
    "us open tennis",
    "roland-garros",
    "wimbledon",
    "australian open",
    "slowmotennis",
    "slow-mo tennis",
    "slow motion tennis",
    "essentialtennis",
    "top tennis training",
    "intuitive tennis",
    "online tennis instruction",
    "tennisfiles",
]

# Search hit constraints
MIN_UPLOAD_YEAR = 2022       # last ~3 years of style
CANDIDATES_PER_QUERY = 12    # how many ytsearch results to inspect
RESULTS_PER_PRO = 2          # download this many highlight reels per pro
SLEEP_BETWEEN_PROS_S = 4     # be polite to YouTube

# Per-style query suffix + duration window. Slow-motion content is typically
# shorter (technique edits, single-shot focus) and biases toward single-player
# material — useful when match-highlight reels mix two players.
QUERY_STYLES = {
    "highlights": {
        "suffix": "tennis highlights",
        "min_duration_s": 5 * 60,
        "max_duration_s": 30 * 60,
    },
    "slow-motion": {
        "suffix": "slow motion",
        "min_duration_s": 60,
        "max_duration_s": 15 * 60,
    },
}


def load_index() -> dict:
    with INDEX_PATH.open() as f:
        return json.load(f)


def empty_clip_pros(index: dict) -> list[tuple[str, dict]]:
    """Return list of (slug, player_data) for pros with no clips yet."""
    return [
        (slug, data)
        for slug, data in index["players"].items()
        if not data.get("clips")
    ]


def query_candidates(pro_name: str, style: str, n: int = CANDIDATES_PER_QUERY) -> list[dict]:
    """Run yt-dlp ytsearch and return parsed JSON for n candidates."""
    suffix = QUERY_STYLES[style]["suffix"]
    query = f"ytsearch{n}:{pro_name} {suffix}"
    cmd = [
        YT_DLP,
        query,
        "--skip-download",
        "--print", "%(.{id,title,channel,uploader,duration,upload_date,view_count,webpage_url})j",
        "--no-warnings",
        "--quiet",
        "--ignore-errors",
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120, check=False
        )
    except subprocess.TimeoutExpired:
        print(f"  [warn] ytsearch timeout for {pro_name!r}", file=sys.stderr)
        return []
    out: list[dict] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def score_candidate(c: dict, style: str, pro_name: str = "") -> tuple[int, int]:
    """Return (preference_tier, view_count) — higher tuple ranks higher.

    Tier 4: preferred channel + recent + duration window + pro name in title (best — likely single-player content)
    Tier 3: (preferred channel + recent + duration) OR (recent + duration + name in title)
    Tier 2: recent + duration (no preferred, no name)
    Tier 1: duration only (older)
    Tier 0: out of duration window — won't be picked
    """
    spec = QUERY_STYLES[style]
    duration = c.get("duration") or 0
    if not (spec["min_duration_s"] <= duration <= spec["max_duration_s"]):
        return (0, c.get("view_count") or 0)
    upload_date = c.get("upload_date") or ""
    try:
        year = int(upload_date[:4]) if upload_date else 0
    except ValueError:
        year = 0
    in_date = year >= MIN_UPLOAD_YEAR
    channel = ((c.get("channel") or "") + " " + (c.get("uploader") or "")).lower()
    is_preferred = any(p in channel for p in PREFERRED_CHANNELS)
    title = (c.get("title") or "").lower()
    # Name match: any token from the pro's name (length > 2 to skip articles)
    name_tokens = [t.lower() for t in pro_name.split() if len(t) > 2]
    name_in_title = any(tok in title for tok in name_tokens) if name_tokens else False

    if is_preferred and in_date and name_in_title:
        tier = 4
    elif (is_preferred and in_date) or (in_date and name_in_title):
        tier = 3
    elif in_date:
        tier = 2
    else:
        tier = 1
    return (tier, c.get("view_count") or 0)


def pick_top(candidates: list[dict], style: str, pro_name: str, n: int) -> list[dict]:
    return sorted(candidates, key=lambda c: score_candidate(c, style, pro_name), reverse=True)[:n]


def already_have(slug_dir: Path, video_id: str) -> bool:
    return any(slug_dir.glob(f"{video_id}.*"))


def download_one(video_url: str, slug_dir: Path) -> Path | None:
    """Download a single highlight reel as MP4. Returns the downloaded path or None on failure."""
    slug_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        YT_DLP,
        video_url,
        "-o", str(slug_dir / "%(id)s.%(ext)s"),
        "-f", "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080][ext=mp4]/best",
        "--merge-output-format", "mp4",
        "--no-warnings",
        "--quiet",
        "--no-overwrites",
    ]
    try:
        subprocess.run(cmd, check=True, timeout=900)
    except subprocess.CalledProcessError as e:
        print(f"  [fail] download exit {e.returncode}: {video_url}", file=sys.stderr)
        return None
    except subprocess.TimeoutExpired:
        print(f"  [fail] download timeout: {video_url}", file=sys.stderr)
        return None
    # yt-dlp writes <id>.<ext>; we know the URL but not necessarily the id without parsing
    # Use the most-recently-modified file under slug_dir as a heuristic
    files = sorted(slug_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def update_manifest(slug_dir: Path, entry: dict) -> None:
    manifest_path = slug_dir / "_manifest.json"
    if manifest_path.exists():
        with manifest_path.open() as f:
            data = json.load(f)
    else:
        data = {"downloads": []}
    # de-dup by id
    existing_ids = {d.get("id") for d in data["downloads"]}
    if entry["id"] not in existing_ids:
        data["downloads"].append(entry)
        with manifest_path.open("w") as f:
            json.dump(data, f, indent=2)


def fetch_for_pro(slug: str, player_data: dict, style: str, dry_run: bool) -> None:
    name = player_data["name"]
    print(f"\n=== {slug} ({name})  [style={style}] ===")
    candidates = query_candidates(name, style)
    if not candidates:
        print(f"  [skip] no search results")
        return
    picks = pick_top(candidates, style, name, RESULTS_PER_PRO)
    if not picks:
        print(f"  [skip] no candidates after filtering")
        return

    slug_dir = RAW_DIR / slug
    for pick in picks:
        tier, views = score_candidate(pick, style, name)
        marker = "★" if tier >= 3 else ("·" if tier >= 1 else "?")
        duration_min = (pick.get("duration") or 0) / 60
        print(
            f"  {marker} tier={tier} "
            f"{pick.get('upload_date', '????????')[:4]} "
            f"{duration_min:5.1f}min  "
            f"[{pick.get('channel') or pick.get('uploader') or '?'}] "
            f"{pick.get('title', '')[:80]}"
        )
        if dry_run:
            continue
        vid_id = pick.get("id")
        if not vid_id:
            continue
        if already_have(slug_dir, vid_id):
            print(f"    [skip] already downloaded")
            continue
        path = download_one(pick["webpage_url"], slug_dir)
        if path is None:
            continue
        update_manifest(slug_dir, {
            "id": vid_id,
            "title": pick.get("title"),
            "channel": pick.get("channel") or pick.get("uploader"),
            "upload_date": pick.get("upload_date"),
            "duration_s": pick.get("duration"),
            "view_count": pick.get("view_count"),
            "url": pick.get("webpage_url"),
            "query_style": style,
            "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "file": path.name,
        })
        print(f"    [ok] {path.name}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--players", help="Comma-separated slugs to fetch for (overrides --all-empty)")
    ap.add_argument("--all-empty", action="store_true", help="Fetch for all pros with empty clips:[] in index.json")
    ap.add_argument("--exclude", default="", help="Comma-separated slugs to skip (useful with --all-empty)")
    ap.add_argument("--query-style", choices=list(QUERY_STYLES), default="highlights",
                    help="Search query bias. 'highlights' = match reels (default); 'slow-motion' = single-player technique edits")
    ap.add_argument("--dry-run", action="store_true", help="Show candidates but don't download")
    args = ap.parse_args()

    if not args.players and not args.all_empty:
        ap.error("specify --players or --all-empty")

    excluded = {s.strip() for s in args.exclude.split(",") if s.strip()}

    index = load_index()
    if args.players:
        slugs = [s.strip() for s in args.players.split(",") if s.strip()]
        targets = []
        for slug in slugs:
            if slug not in index["players"]:
                print(f"[warn] unknown slug: {slug}", file=sys.stderr)
                continue
            targets.append((slug, index["players"][slug]))
    else:
        targets = empty_clip_pros(index)

    if excluded:
        targets = [(s, d) for (s, d) in targets if s not in excluded]

    if not targets:
        print("Nothing to do.")
        return 0

    print(f"Targets ({len(targets)}, style={args.query_style}): {[t[0] for t in targets]}")
    if excluded:
        print(f"Excluded: {sorted(excluded)}")
    if args.dry_run:
        print("(dry-run — no downloads will occur)")

    for i, (slug, data) in enumerate(targets):
        fetch_for_pro(slug, data, args.query_style, args.dry_run)
        if i < len(targets) - 1 and not args.dry_run:
            time.sleep(SLEEP_BETWEEN_PROS_S)

    return 0


if __name__ == "__main__":
    sys.exit(main())
