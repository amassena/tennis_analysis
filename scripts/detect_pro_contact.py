#!/usr/bin/env python3
"""Detect the true contact frame in a pro clip from its pose, replacing the
hardcoded midpoint (PRO_CONTACT_FRAME=120) used by compare_filmstrip.

Why: pro comparison strips align the user's (audio-snapped, accurate) contact
against the pro's. The pro contact was never measured — it assumed curation
trimmed each clip so impact lands on frame 120. It often doesn't (observed up
to ~300ms off), so the pro half of every strip is misaligned.

Signal (no ball tracking available on highlight footage, audio is music/crowd):
  1. PEAK WRIST SPEED locates the swing robustly — the racket-hand wrist is the
     fastest joint through the hitting zone. Pick whichever wrist (L=15, R=16)
     has the higher peak; that's the racket hand (handedness/mirroring-agnostic).
  2. Peak hand speed LAGS contact (the hand keeps accelerating into the
     follow-through wrap — observed +4f on a forehand, +9f on a 2H backhand).
     Refine to the frame of MAX WRIST REACH (wrist→hip-center distance) within a
     tight window just before the speed peak: the arm is fully extended toward
     the ball at contact and folds back during follow-through. The window
     excludes the backswing (also an extended-arm pose, but >200ms earlier).

Validated: alcaraz backhand_001 -> 128 (visual contact ~130); the assumed 120
was in the backswing.

Usage:
    python scripts/detect_pro_contact.py                 # all clips in index.json
    python scripts/detect_pro_contact.py --player alcaraz
    python scripts/detect_pro_contact.py --out /tmp/pro_contacts.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PROS_DIR = REPO_ROOT / "pros"
INDEX_PATH = PROS_DIR / "index.json"

L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
MIN_VIS = 0.2
# Search the middle of the clip for the swing (avoid run-up / recovery at edges).
SEARCH_LO, SEARCH_HI = 0.20, 0.85
# Contact sits at or just before the wrist-speed peak; the backswing (also an
# extended-arm pose) is well before it. 12f ≈ 200ms back, 3f ≈ 50ms forward.
REACH_BACK, REACH_FWD = 12, 3


def _xy(frames, idx):
    out = [None] * len(frames)
    for i, f in enumerate(frames):
        lm = f.get("landmarks")
        if f.get("detected") and lm and len(lm) > idx:
            x, y, _z, v = lm[idx]
            if v > MIN_VIS:
                out[i] = (x, y)
    return out


def _smooth(s, k=2):
    n = len(s)
    out = [0.0] * n
    for i in range(n):
        lo, hi = max(0, i - k), min(n, i + k + 1)
        out[i] = sum(s[lo:hi]) / (hi - lo)
    return out


def _speed(s):
    n = len(s)
    sp = [0.0] * n
    for i in range(1, n - 1):
        a, b = s[i - 1], s[i + 1]
        if a and b:
            sp[i] = ((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5
    return sp


def _dist(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5 if a and b else 0.0


def detect_contact(pose: dict, default: int = 120) -> dict:
    """Return {contact_frame, confidence, wrist, peak_frame, method}.

    confidence is the peak wrist speed (normalized px/frame); ~<0.01 means a
    weak/ambiguous swing (far/elevated angle) — caller may keep the default.
    """
    frames = pose.get("frames", [])
    n = len(frames)
    if n < 30:
        return {"contact_frame": default, "confidence": 0.0, "method": "fallback:short"}
    lo, hi = int(SEARCH_LO * n), int(SEARCH_HI * n)
    if hi - lo < 5:
        return {"contact_frame": default, "confidence": 0.0, "method": "fallback:short"}

    # Racket wrist = whichever has the higher peak speed in the search window.
    best = None
    for wi in (L_WRIST, R_WRIST):
        w = _xy(frames, wi)
        sp = _smooth(_speed(w))
        pk = max(range(lo, hi), key=lambda i: sp[i])
        if best is None or sp[pk] > best[1]:
            best = (wi, sp[pk], pk, w)
    wrist, peak_v, peak_f, w = best

    # Hip center for the reach measure.
    lh, rh = _xy(frames, L_HIP), _xy(frames, R_HIP)
    hc = [((lh[i][0] + rh[i][0]) / 2, (lh[i][1] + rh[i][1]) / 2)
          if lh[i] and rh[i] else None for i in range(n)]
    reach = _smooth([_dist(w[i], hc[i]) for i in range(n)])

    a, b = max(lo, peak_f - REACH_BACK), min(hi, peak_f + REACH_FWD + 1)
    contact = max(range(a, b), key=lambda i: reach[i]) if b > a else peak_f
    method = "reach"
    # If reach maxed at the window edge, the swing geometry is ambiguous (often
    # a far/elevated clip) — fall back to the peak-speed frame, which at least
    # sits in the contact/follow-through zone, not the backswing.
    if contact in (a, b - 1):
        contact = peak_f
        method = "peak"
    return {"contact_frame": int(contact), "confidence": round(peak_v, 5),
            "wrist": wrist, "peak_frame": int(peak_f), "method": method}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--player", default=None, help="only this slug")
    ap.add_argument("--out", default="/tmp/pro_contacts.json")
    ap.add_argument("--min-confidence", type=float, default=0.008,
                    help="below this peak speed, keep the existing contact_frame")
    args = ap.parse_args()

    index = json.loads(INDEX_PATH.read_text())
    players = index["players"]
    slugs = [args.player] if args.player else list(players)

    results = {}
    n_clips = n_done = n_weak = n_missing = 0
    for slug in slugs:
        for clip in players[slug].get("clips", []):
            n_clips += 1
            fn = clip["file"]
            pose_path = PROS_DIR / slug / Path(fn).with_suffix(".pose.json")
            key = f"{slug}/{fn}"
            if not pose_path.exists():
                n_missing += 1
                results[key] = {"error": "no_pose", "contact_frame": clip.get("contact_frame", 120)}
                continue
            try:
                pose = json.loads(pose_path.read_text())
            except Exception as e:
                results[key] = {"error": str(e)[:80], "contact_frame": clip.get("contact_frame", 120)}
                continue
            r = detect_contact(pose, default=clip.get("contact_frame", 120))
            r["old"] = clip.get("contact_frame", 120)
            r["delta"] = r["contact_frame"] - r["old"]
            r["type"] = clip.get("type")
            r["n_frames"] = len(pose.get("frames", []))
            if r.get("confidence", 0) < args.min_confidence:
                r["contact_frame"] = r["old"]
                r["method"] = "fallback:low_conf"
                n_weak += 1
            else:
                n_done += 1
            results[key] = r

    Path(args.out).write_text(json.dumps(results, indent=1))
    deltas = [abs(r["delta"]) for r in results.values()
              if "delta" in r and r.get("method", "").startswith(("reach", "peak"))]
    big = sum(1 for d in deltas if d > 6)
    print(f"clips={n_clips} detected={n_done} weak/kept={n_weak} no_pose={n_missing}")
    if deltas:
        deltas_sorted = sorted(deltas)
        med = deltas_sorted[len(deltas_sorted) // 2]
        print(f"|delta| from old(120): median={med}f  max={max(deltas)}f  "
              f">6f(>100ms)={big}/{len(deltas)}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
