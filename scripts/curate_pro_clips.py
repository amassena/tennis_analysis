#!/usr/bin/env python3
"""Auto-curate top-N candidate shots per pro per shot type and extract as
standardized clips.

Reads Phase 2 outputs (pros/_raw/<slug>/<id>_shots.json), picks top-N detections
per shot type by model confidence, ffmpeg-trims to 4-sec windows centered on
the detected contact moment, normalizes to 60 fps / 640x480 / no-audio.

Output: pros/<slug>/<type>_NN.mp4 (1-indexed, zero-padded).
Also updates pros/index.json clips:[] for each pro.

For lefty pros (handedness=left), also writes <type>_NN_mirrored.mp4 (hflip)
so the matcher can compare against right-handed users too.

Usage:
    # Default: top 8 of each type for every pro with empty clips:[]
    .venv/bin/python scripts/curate_pro_clips.py

    # Just sinner, top 6 each
    .venv/bin/python scripts/curate_pro_clips.py --players sinner --top-n 6

    # Dry-run to see what would be picked
    .venv/bin/python scripts/curate_pro_clips.py --players sinner --dry-run
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PROS_DIR = REPO_ROOT / "pros"
RAW_DIR = PROS_DIR / "_raw"
INDEX_PATH = PROS_DIR / "index.json"

# Output spec (matches existing alcaraz/djokovic/federer/nadal clips)
TARGET_FPS = 60
TARGET_W = 640
TARGET_H = 480
CLIP_DURATION_S = 4.0
CONTACT_OFFSET_S = 2.0      # lead-in before contact
CONTACT_FRAME = int(CONTACT_OFFSET_S * TARGET_FPS)  # = 120

SHOT_TYPES = ("forehand", "backhand", "serve")
DEFAULT_TOP_N = 8

FFMPEG = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
FFPROBE = shutil.which("ffprobe") or "/opt/homebrew/bin/ffprobe"


def probe_dims(mp4: Path) -> tuple[int, int]:
    """Return (width, height) of a video via ffprobe."""
    cmd = [
        FFPROBE, "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0:s=x", str(mp4),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    w, h = result.stdout.strip().split("x")
    return int(w), int(h)


def compute_crop(src_w: int, src_h: int, target_w: int, target_h: int) -> tuple[int, int, int, int]:
    """Return (crop_w, crop_h, crop_x, crop_y) to center-crop src to target aspect ratio.

    If source matches target aspect already, the crop is the full frame (no-op).
    Aspect determined by target_w:target_h. We crop the wider dimension.
    """
    target_aspect = target_w / target_h
    src_aspect = src_w / src_h
    if src_aspect > target_aspect:
        # source is wider — crop horizontally
        crop_h = src_h
        crop_w = int(round(src_h * target_aspect))
        crop_x = (src_w - crop_w) // 2
        crop_y = 0
    else:
        # source is taller — crop vertically
        crop_w = src_w
        crop_h = int(round(src_w / target_aspect))
        crop_x = 0
        crop_y = (src_h - crop_h) // 2
    return crop_w, crop_h, crop_x, crop_y


def load_index() -> dict:
    with INDEX_PATH.open() as f:
        return json.load(f)


def save_index(index: dict) -> None:
    with INDEX_PATH.open("w") as f:
        json.dump(index, f, indent=2)
        f.write("\n")


def load_manifest(slug: str) -> dict:
    manifest_path = RAW_DIR / slug / "_manifest.json"
    if not manifest_path.exists():
        return {"downloads": []}
    with manifest_path.open() as f:
        return json.load(f)


def candidates_for_slug(slug: str) -> list[dict]:
    """Return list of detection dicts annotated with source reel info.

    Each entry:
        {
          'timestamp': float,
          'shot_type': str,
          'confidence': float,
          'source_id': str (youtube video id),
          'source_mp4': Path,
          'fps': float,
          'duration': float,
          'channel': str,
        }
    """
    slug_dir = RAW_DIR / slug
    if not slug_dir.is_dir():
        return []

    manifest = load_manifest(slug)
    channel_by_id = {d["id"]: d.get("channel", "") for d in manifest.get("downloads", [])}

    out = []
    dims_cache: dict[Path, tuple[int, int]] = {}
    for shots_path in sorted(slug_dir.glob("*_shots.json")):
        with shots_path.open() as f:
            data = json.load(f)
        source_id = data.get("source_video") or shots_path.stem.replace("_shots", "")
        source_mp4 = slug_dir / f"{source_id}.mp4"
        if not source_mp4.exists():
            continue
        fps = data.get("fps") or 60.0
        duration = data.get("duration") or 0.0
        if source_mp4 not in dims_cache:
            try:
                dims_cache[source_mp4] = probe_dims(source_mp4)
            except Exception as e:
                print(f"  [warn] ffprobe failed for {source_mp4.name}: {e}", file=sys.stderr)
                continue
        src_w, src_h = dims_cache[source_mp4]
        for det in data.get("detections", []):
            t = det.get("timestamp", 0.0)
            # Skip detections too close to the start/end to fit the 4-sec window
            if t < CONTACT_OFFSET_S or t > duration - CONTACT_OFFSET_S:
                continue
            shot_type = det.get("shot_type")
            if shot_type not in SHOT_TYPES:
                continue
            out.append({
                "timestamp": t,
                "shot_type": shot_type,
                "confidence": det.get("confidence", 0.0),
                "source_id": source_id,
                "source_mp4": source_mp4,
                "src_w": src_w,
                "src_h": src_h,
                "fps": fps,
                "duration": duration,
                "channel": channel_by_id.get(source_id, ""),
            })
    return out


def pick_top_per_type(candidates: list[dict], top_n: int) -> dict[str, list[dict]]:
    by_type: dict[str, list[dict]] = defaultdict(list)
    for c in candidates:
        by_type[c["shot_type"]].append(c)
    out = {}
    for shot_type, items in by_type.items():
        items.sort(key=lambda c: -c["confidence"])
        out[shot_type] = items[:top_n]
    return out


def trim_clip(source_mp4: Path, timestamp: float, output: Path,
              src_dims: tuple[int, int], mirrored: bool = False) -> bool:
    """ffmpeg-trim a 4-sec window centered on contact, normalize to 60fps 640x480.

    Center-crops source to 4:3 aspect (matches target), then scales to 640x480.
    Crop dims are computed in Python (no embedded ffmpeg expressions, which
    have nasty comma-escaping issues for nested min()).
    """
    start = max(0.0, timestamp - CONTACT_OFFSET_S)
    src_w, src_h = src_dims
    cw, ch, cx, cy = compute_crop(src_w, src_h, TARGET_W, TARGET_H)
    filters = [f"fps={TARGET_FPS}", f"crop={cw}:{ch}:{cx}:{cy}", f"scale={TARGET_W}:{TARGET_H}"]
    if mirrored:
        filters.append("hflip")
    vf = ",".join(filters)
    cmd = [
        FFMPEG, "-y",
        "-ss", f"{start:.3f}",
        "-i", str(source_mp4),
        "-t", f"{CLIP_DURATION_S:.3f}",
        "-vf", vf,
        "-an",                       # strip audio
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-loglevel", "error",
        str(output),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [ffmpeg fail] {output.name}: {result.stderr.strip()[:200]}", file=sys.stderr)
        return False
    return True


def curate_pro(slug: str, player_data: dict, top_n: int, dry_run: bool) -> dict:
    """Returns a summary dict; updates pros/<slug>/ on disk if not dry-run."""
    print(f"\n=== {slug} ({player_data['name']}) ===")
    candidates = candidates_for_slug(slug)
    if not candidates:
        print(f"  [skip] no candidates")
        return {"slug": slug, "clips": []}

    picks_by_type = pick_top_per_type(candidates, top_n)
    is_lefty = player_data.get("handedness") == "left"

    for shot_type in SHOT_TYPES:
        picks = picks_by_type.get(shot_type, [])
        if picks:
            confs = [p["confidence"] for p in picks]
            print(f"  {shot_type:10s} picks: {len(picks)}  confs={[round(c,2) for c in confs]}")
        else:
            print(f"  {shot_type:10s} picks: 0  (no candidates)")

    if dry_run:
        return {"slug": slug, "dry_run": True}

    out_dir = PROS_DIR / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    clip_entries = []

    for shot_type in SHOT_TYPES:
        picks = picks_by_type.get(shot_type, [])
        for i, pick in enumerate(picks, 1):
            base = f"{shot_type}_{i:03d}"
            out_path = out_dir / f"{base}.mp4"
            ok = trim_clip(pick["source_mp4"], pick["timestamp"], out_path,
                           src_dims=(pick["src_w"], pick["src_h"]), mirrored=False)
            if not ok:
                continue
            entry = {
                "file": out_path.name,
                "type": shot_type,
                "contact_frame": CONTACT_FRAME,
                "fps": TARGET_FPS,
                "source": f"youtube:{pick['channel']}" if pick["channel"] else "youtube",
                "source_video_id": pick["source_id"],
                "source_timestamp": round(pick["timestamp"], 2),
                "confidence": round(pick["confidence"], 3),
                # Slow-motion technique videos are predominantly filmed from the
                # side. Court-level practice reels are wider but still mostly
                # side. Default to "side" so the matcher's angle preference
                # doesn't push these picks below the existing 4 pros with
                # explicit angle tags. Manual override per clip is fine.
                "angle": "side",
                "mirrored": False,
            }
            clip_entries.append(entry)

            if is_lefty:
                mirrored_path = out_dir / f"{base}_mirrored.mp4"
                ok = trim_clip(pick["source_mp4"], pick["timestamp"], mirrored_path,
                               src_dims=(pick["src_w"], pick["src_h"]), mirrored=True)
                if ok:
                    clip_entries.append({**entry,
                                         "file": mirrored_path.name,
                                         "mirrored": True})

    print(f"  -> wrote {len(clip_entries)} clip entries to {out_dir}")
    return {"slug": slug, "clips": clip_entries}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--players", help="Comma-separated slugs to curate")
    ap.add_argument("--all-empty", action="store_true",
                    help="Curate every pro currently with clips:[] in index.json (default)")
    ap.add_argument("--top-n", type=int, default=DEFAULT_TOP_N,
                    help=f"Picks per shot type (default {DEFAULT_TOP_N})")
    ap.add_argument("--dry-run", action="store_true", help="Show picks without trimming")
    args = ap.parse_args()

    index = load_index()
    if args.players:
        slugs = [s.strip() for s in args.players.split(",") if s.strip()]
    else:
        # Default to --all-empty behavior
        slugs = [s for s, d in index["players"].items() if not d.get("clips")]

    if not slugs:
        print("Nothing to do.")
        return 0

    print(f"Curating {len(slugs)} pro(s): {slugs}")
    if args.dry_run:
        print("(dry-run — no trims will occur)")

    results = []
    for slug in slugs:
        if slug not in index["players"]:
            print(f"[warn] unknown slug: {slug}", file=sys.stderr)
            continue
        result = curate_pro(slug, index["players"][slug], args.top_n, args.dry_run)
        results.append(result)
        if not args.dry_run and result.get("clips"):
            index["players"][slug]["clips"] = result["clips"]

    if not args.dry_run:
        save_index(index)
        print(f"\nUpdated {INDEX_PATH}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
