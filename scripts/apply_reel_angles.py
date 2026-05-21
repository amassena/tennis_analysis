#!/usr/bin/env python3
"""Apply visually-classified per-reel camera angles to pros/index.json.

Reads a hardcoded dict mapping source video IDs to camera angle (classified
by hand from per-reel thumbnails — see /tmp/reel_grid/big_*.png for the
classification source). Looks up each clip's source_video_id in this map
and writes the matching angle back to the clip entry.

This replaces the lazy `angle="side"` default that scripts/curate_pro_clips.py
applied during auto-curation. Pose-based auto-detection (sh_z_max heuristic)
proved unreliable on the calibration set — per-reel visual classification
is more accurate.

Usage:
    .venv/bin/python scripts/apply_reel_angles.py
    .venv/bin/python scripts/apply_reel_angles.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
INDEX_PATH = REPO_ROOT / "pros" / "index.json"

# Visually classified 2026-05-20 from /tmp/reel_grid/big_*.png thumbnails.
# Convention:
#   "side"   — camera perpendicular to swing direction, player visible
#              from their side (slow-mo technique reels, sideline practice)
#   "behind" — camera behind the player (court-level practice) OR behind
#              the baseline (broadcast TV wide). Both group together because
#              they're visually distinct from side angles.
REEL_ANGLES: dict[str, str] = {
    "8ioYzl_dVg8": "side",       # deminaur (Wgb_Lh8Z8dM is BEHIND)
    "Wgb_Lh8Z8dM": "behind",     # deminaur
    "5DttMwAwD0U": "side",       # fritz
    "gbVDhUjRfYg": "side",       # fritz
    "kzuOY_UQ2e0": "behind",     # fritz (aerial)
    "oK5E1Imk6HM": "behind",     # fritz
    "y03GdaSvz_w": "side",       # fritz
    "0bP6ulUv6oc": "behind",     # gauff (player back to camera)
    "MEXEUdA4uVs": "behind",     # gauff (broadcast)
    "tPkfjRYMpkw": "side",       # gauff
    "tr-ztoxmc2I": "behind",     # gauff/pegula (same reel id appears in both — it's a multi-player compilation)
    "8quG5dlAUPM": "behind",     # henin (clay match)
    "CSeFN7oHzYs": "behind",     # henin
    "KBioEMX2IdM": "side",       # henin
    "LdDwMj3_WMA": "side",       # henin
    "OTXfZt3Q4kw": "side",       # henin
    "A5SZ0wC44A0": "side",       # hurkacz
    "qkQmT4fXKD8": "behind",     # hurkacz
    "WsAleiZg5ig": "side",       # jabeur
    "o14zEDsIQG8": "behind",     # jabeur (Wimbledon broadcast)
    "1uPetW4Uuig": "behind",     # murray
    "5H_kb2abbvc": "behind",     # murray
    "tqPe-IdUUa8": "side",       # murray
    "xBLRe3hrmzo": "side",       # murray (verified)
    "0xT0G31GUo4": "behind",     # pegula (broadcast)
    "S5IgXi2HASE": "behind",     # pegula
    "wcmemO2-nhg": "behind",     # pegula
    "4SSkTPYqxpk": "behind",     # rublev (broadcast aerial)
    "CxttdybdLLI": "side",       # rublev
    "fhn2ANDE4kA": "side",       # rublev/sinner (slow-mo serve comp)
    "928wJjWeVyk": "side",       # rune
    "fAqpv9jv1jk": "side",       # rune
    "u3Uf-J6gKd0": "behind",     # rune
    "IBL2jPkZuow": "behind",     # rybakina (broadcast)
    "W0uSDf3AaY8": "side",       # rybakina
    "XEshCsHrvcc": "side",       # rybakina
    "H52VQbqUIF0": "behind",     # sabalenka (Roland-Garros)
    "eGM-ecFcPUc": "side",       # sabalenka
    "nBMEzugQKJk": "side",       # sabalenka
    "wDSk5zYeys4": "behind",     # sabalenka (Wimbledon broadcast)
    "3_TwJpz96VE": "side",       # serena
    "9F3UYGMsqA0": "side",       # serena
    "VpObWdU4ab4": "side",       # serena
    "dZ473rEvgtM": "side",       # serena
    "oOsmGn2_piE": "side",       # serena
    "sDynWX27zIk": "side",       # serena
    "-7V1paGhbiY": "behind",     # sinner (broadcast)
    "AWgTZBADqoo": "side",       # sinner
    "Roc4Yao6iqE": "behind",     # sinner (COURT LEVEL TENNIS — verified)
    "U111uQCv6yc": "behind",     # sinner (broadcast)
    "ZU0tVNMI1qo": "side",       # sinner
    "_wut0HlifLQ": "side",       # sinner
    "WUEUPYv9_oc": "side",       # swiatek
    "hmcrCasO7V4": "side",       # swiatek
    "XdGt3c81RbI": "side",       # tsitsipas
    "aEodOp29ZiA": "behind",     # tsitsipas
    "5M52JoEDtwY": "side",       # venus
    "IDuHO20EiQM": "behind",     # venus
    "Wtrk-y24hNY": "side",       # venus
    "bRCQwLgEs9M": "side",       # venus
    "mXy0jJl8Pnc": "side",       # venus
    "3ZCk3LgZD4o": "side",       # wawrinka
    "4147sgz1QHI": "behind",     # wawrinka (KIA broadcast)
    "Pd1SytrCAsE": "side",       # wawrinka
    "SlgMvQQrYhg": "side",       # wawrinka
    "_cohjbquvwc": "side",       # wawrinka
    "_xmJO_ZUtWU": "side",       # wawrinka
    "nPP8T9xfSj4": "side",       # wawrinka
    "Etaq8Rzo_5A": "side",       # zverev
    "IhjO2ac9gfo": "side",       # zverev
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    with INDEX_PATH.open() as f:
        index = json.load(f)

    updates = 0
    no_match = 0
    by_angle = {"side": 0, "behind": 0}

    for slug, player in index["players"].items():
        for clip in player.get("clips", []):
            vid_id = clip.get("source_video_id")
            if not vid_id:
                continue
            new_angle = REEL_ANGLES.get(vid_id)
            if new_angle is None:
                no_match += 1
                print(f"  [no-map] {slug}/{clip['file']} <- src {vid_id}", file=sys.stderr)
                continue
            by_angle[new_angle] = by_angle.get(new_angle, 0) + 1
            if clip.get("angle") != new_angle:
                if not args.dry_run:
                    clip["angle"] = new_angle
                updates += 1

    print(f"\n=== {updates} angle updates "
          f"({by_angle.get('side', 0)} side, {by_angle.get('behind', 0)} behind), "
          f"{no_match} no-map ===")

    if not args.dry_run and updates:
        with INDEX_PATH.open("w") as f:
            json.dump(index, f, indent=2)
            f.write("\n")
        print(f"Wrote {INDEX_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
