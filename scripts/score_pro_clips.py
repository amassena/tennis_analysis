#!/usr/bin/env python3
"""Quality-gate the pro clip library from pose alone — the self-reinforcement
loop for comparison quality. Every failure mode the user flagged is measurable
without a human eye:

  - no real stroke (a clip of someone walking): peak wrist speed too low
  - untrackable (sparse pose): low detection rate / joint visibility
  - bad trim / contact off: contact lands at a clip edge, or the two
    independent contact estimators (peak wrist speed vs max wrist reach)
    disagree badly (low confidence the center panel is really contact)
  - bad framing: player bbox at contact too large (zoomed in) or too small (far)

Writes quality fields back into pros/index.json per clip (quality_ok + reasons +
metrics) so match_pro_clip can skip the failures. Run, exclude, regenerate,
re-score — loop until the pool is clean.

Usage:
    python scripts/score_pro_clips.py            # score all, write index.json
    python scripts/score_pro_clips.py --dry-run  # print scorecard, no write
    python scripts/score_pro_clips.py --player wawrinka
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from detect_pro_contact import (_xy, _smooth, _speed, _dist, detect_contact,
                                 L_WRIST, R_WRIST, L_HIP, R_HIP, SEARCH_LO, SEARCH_HI)

REPO_ROOT = Path(__file__).resolve().parent.parent
PROS_DIR = REPO_ROOT / "pros"
INDEX_PATH = PROS_DIR / "index.json"

# Gates. Tuned against known good/bad clips; --dry-run to see the distribution.
# Swing PROMINENCE (peak speed / median speed in the window) is scale- and
# fps-invariant: a real stroke spikes far above the player's baseline motion,
# walking/no-contact stays flat. Raw speed conflates "small/far player" with
# "no swing", so we don't gate on it.
MIN_SWING_PROMINENCE = 3.5  # below this the wrist never spikes (normal-speed swing)
MIN_WRIST_SWEEP = 1.2       # ...AND wrist sweeps < ~1.2 body-heights => no stroke
MIN_POSE_RATE = 0.80        # fraction of frames in search window with a pose
MIN_JOINT_VIS = 0.45        # median visibility of racket wrist + shoulders
EDGE_MARGIN = 0.15          # contact must be inside [margin, 1-margin] of clip
MAX_AGREE_FRAMES = 12       # speed-peak vs reach-peak must agree within this
# Framing is advisory only (crop is recomputed at strip time; bbox can exceed 1
# from MediaPipe extrapolating occluded joints) — recorded, not a hard fail.
ZOOM_MAX_BBOX = 0.95
FAR_MIN_BBOX = 0.14


def _bbox_h(frames, fi):
    pf = {f.get("frame_idx", i): f for i, f in enumerate(frames)}.get(fi)
    if not pf or not pf.get("detected"):
        return 0.0
    lm = pf.get("hitter_landmarks") or pf.get("landmarks") or []
    ys = [l[1] for l in lm if len(l) >= 4 and l[3] > 0.2]
    return (max(ys) - min(ys)) if len(ys) >= 2 else 0.0


def score_clip(pose: dict) -> dict:
    frames = pose.get("frames", [])
    n = len(frames)
    if n < 30:
        return {"quality_ok": False, "reasons": ["too_short"], "metrics": {"n": n}}
    lo, hi = int(SEARCH_LO * n), int(SEARCH_HI * n)

    # racket wrist = higher peak speed
    best = None
    for wi in (L_WRIST, R_WRIST):
        w = _xy(frames, wi)
        sp = _smooth(_speed(w))
        pk = max(range(lo, hi), key=lambda i: sp[i])
        if best is None or sp[pk] > best[1]:
            best = (wi, sp[pk], pk, w, sp)
    wrist, swing_speed, speed_pk, w, sp = best
    win = [sp[i] for i in range(lo, hi) if sp[i] > 0]
    median_sp = sorted(win)[len(win) // 2] if win else 0.0
    prominence = swing_speed / (median_sp + 1e-6)

    # Wrist SWEEP: how far the racket wrist travels across the clip, normalized
    # by body size. Speed-independent, so slow-motion strokes (low prominence)
    # still register a big sweep, while walking/footwork keeps the wrist near
    # the body. Used together with prominence so neither alone misfires.
    wpts = [w[i] for i in range(lo, hi) if w[i]]
    bbs = [_bbox_h(frames, fi) for fi in range(lo, hi)]
    bbs = [b for b in bbs if b > 0.05]
    med_bbox = sorted(bbs)[len(bbs) // 2] if bbs else 0.0
    if len(wpts) >= 2 and med_bbox:
        xr = max(p[0] for p in wpts) - min(p[0] for p in wpts)
        yr = max(p[1] for p in wpts) - min(p[1] for p in wpts)
        wrist_sweep = (xr + yr) / med_bbox
    else:
        wrist_sweep = 0.0

    # pose quality over the search window
    det = sum(1 for i in range(lo, hi) if frames[i].get("detected"))
    pose_rate = det / max(1, hi - lo)
    vis = []
    for i in range(lo, hi):
        lm = frames[i].get("hitter_landmarks") or frames[i].get("landmarks") or []
        for j in (wrist, L_HIP, R_HIP):
            if len(lm) > j and len(lm[j]) >= 4:
                vis.append(lm[j][3])
    joint_vis = sorted(vis)[len(vis) // 2] if vis else 0.0

    # contact + the two-estimator agreement
    c = detect_contact(pose)
    contact = c["contact_frame"]
    reach_pk = contact if c["method"] == "reach" else speed_pk
    agree = abs(speed_pk - reach_pk)

    bbox_at_contact = _bbox_h(frames, contact)

    reasons = []        # hard fails -> excluded from rotation
    warnings = []       # advisory -> recorded, still usable
    # No real stroke only if the wrist NEITHER spikes (prominence) NOR sweeps a
    # wide arc (sweep). Either one alone => keep (sharp normal swing, or slow-mo).
    if prominence < MIN_SWING_PROMINENCE and wrist_sweep < MIN_WRIST_SWEEP:
        reasons.append("no_swing")
    if pose_rate < MIN_POSE_RATE or joint_vis < MIN_JOINT_VIS:
        reasons.append("untrackable")
    if not (n * EDGE_MARGIN <= contact <= n * (1 - EDGE_MARGIN)):
        reasons.append("contact_at_edge")
    if agree > MAX_AGREE_FRAMES:
        reasons.append("contact_uncertain")
    if bbox_at_contact > ZOOM_MAX_BBOX:
        warnings.append("zoomed")
    if bbox_at_contact and bbox_at_contact < FAR_MIN_BBOX:
        warnings.append("far")

    return {
        "quality_ok": not reasons,
        "reasons": reasons,
        "warnings": warnings,
        "metrics": {
            "prominence": round(prominence, 2),
            "wrist_sweep": round(wrist_sweep, 2),
            "swing_speed": round(swing_speed, 4),
            "pose_rate": round(pose_rate, 3),
            "joint_vis": round(joint_vis, 3),
            "contact": contact,
            "agree_frames": agree,
            "bbox_h_at_contact": round(bbox_at_contact, 3),
            "n": n,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--player", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    index = json.loads(INDEX_PATH.read_text())
    slugs = [args.player] if args.player else list(index["players"])

    n_ok = n_bad = n_nopose = 0
    from collections import Counter
    reason_counts = Counter()
    fails = []
    for slug in slugs:
        for clip in index["players"][slug].get("clips", []):
            pose_path = PROS_DIR / slug / Path(clip["file"]).with_suffix(".pose.json")
            if not pose_path.exists():
                n_nopose += 1
                clip["quality_ok"] = False
                clip["quality_reasons"] = ["no_pose"]
                continue
            try:
                pose = json.loads(pose_path.read_text())
            except Exception:
                clip["quality_ok"] = False
                clip["quality_reasons"] = ["pose_unreadable"]
                n_bad += 1
                continue
            r = score_clip(pose)
            clip["quality_ok"] = r["quality_ok"]
            clip["quality_reasons"] = r["reasons"]
            clip["quality_warnings"] = r["warnings"]
            clip["quality_metrics"] = r["metrics"]
            if r["quality_ok"]:
                n_ok += 1
            else:
                n_bad += 1
                for x in r["reasons"]:
                    reason_counts[x] += 1
                fails.append((f"{slug}/{clip['file']}", clip.get("type"),
                              r["reasons"], r["metrics"]))

    print(f"OK={n_ok} BAD={n_bad} no_pose={n_nopose}")
    print(f"reasons: {dict(reason_counts)}")
    print("--- failures ---")
    for key, typ, reasons, m in sorted(fails):
        print(f"  {key:32s} {typ:9s} {','.join(reasons):28s} "
              f"prom={m.get('prominence')} pose={m.get('pose_rate')} "
              f"agree={m.get('agree_frames')} vis={m.get('joint_vis')}")

    if not args.dry_run:
        INDEX_PATH.write_text(json.dumps(index, indent=1))
        print(f"wrote {INDEX_PATH}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
