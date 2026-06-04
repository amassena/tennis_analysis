#!/usr/bin/env python3
"""Stream B of precise contact detection (docs/contact_detection.md): derive the
SUB-FRAME contact instant from a ball track, robust to the ball being occluded
at impact.

Core idea: contact is where the INCOMING and OUTGOING ball trajectories meet.
The ball's *position* is continuous at contact; only its *velocity* flips. So we
fit a low-order curve to the visible incoming points and another to the visible
outgoing points, then find the time where the two curves are closest. That time
is generally between two integer frames (sub-frame), and — crucially — it needs
NO detection at the contact frame itself: the occluded/blurred frames around
impact are exactly the ones we fit *through*.

This module is pure geometry: it takes a ball track (from the Stream A detector)
and returns a contact estimate with an explicit occlusion/uncertainty window. It
has no CV dependencies beyond numpy and is unit-tested on synthetic arcs:

    python scripts/contact_trajectory.py --selftest
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class ContactEstimate:
    contact_frame: float          # sub-frame (e.g. 122.6)
    contact_time: float           # seconds
    confidence: float             # 0..1
    uncertainty_ms: float         # 1-sigma-ish timing uncertainty
    occluded_from: int | None     # first missing frame in the impact gap
    occluded_to: int | None       # last missing frame in the impact gap
    occluded_ms: float            # duration the ball is unseen across impact
    method: str                   # 'intersection' | 'reversal' | 'insufficient'
    n_in: int                     # incoming points used
    n_out: int                    # outgoing points used


def _segfit(ts: np.ndarray, xs: np.ndarray, ys: np.ndarray, deg: int):
    """Fit x(t) and y(t) polynomials; return (px, py) np.poly1d and rms residual."""
    deg = min(deg, len(ts) - 1)
    px = np.poly1d(np.polyfit(ts, xs, deg))
    py = np.poly1d(np.polyfit(ts, ys, deg))
    res = np.hypot(px(ts) - xs, py(ts) - ys)
    rms = float(np.sqrt(np.mean(res ** 2))) if len(res) else 0.0
    return px, py, rms


def estimate_contact(track, fps,
                     window=None,
                     min_seg=4,
                     fit_deg=2,
                     guard=1):
    """Estimate the contact instant from a ball track.

    track: list of dicts with at least {'frame': int, 'x': float, 'y': float}
           for the frames where the ball is VISIBLE (missing frames = occluded).
           A dict may carry 'visible': False to mark an explicit non-detection;
           such rows are ignored as samples.
    fps:   frames per second (for time + ms).
    window: optional (lo, hi) frame bounds to search for contact; default = full.
    min_seg: minimum visible points required on each side.
    fit_deg: polynomial degree per axis (2 = parabola; gravity/perspective).
    guard:  frames on each side of the candidate excluded from the fits (the
            impact frames are unreliable even if "visible").

    Returns a ContactEstimate.
    """
    pts = [(int(p['frame']), float(p['x']), float(p['y']))
           for p in track if p.get('visible', True) and p.get('x') is not None]
    pts.sort()
    if len(pts) < 2 * min_seg:
        return ContactEstimate(0, 0, 0.0, 0.0, None, None, 0.0, 'insufficient',
                               0, 0)
    fr = np.array([p[0] for p in pts], float)
    xs = np.array([p[1] for p in pts], float)
    ys = np.array([p[2] for p in pts], float)

    lo = fr[0] if window is None else max(fr[0], window[0])
    hi = fr[-1] if window is None else min(fr[-1], window[1])

    # --- locate the candidate contact: the sharpest change in velocity
    # direction along the dominant-motion axis, within [lo, hi]. Works on the
    # visible samples; the true contact may sit in an occlusion gap nearby.
    vx = np.gradient(xs, fr)
    vy = np.gradient(ys, fr)
    # dominant axis = the one with the larger overall velocity range
    axis_v = vx if (np.ptp(vx) >= np.ptp(vy)) else vy
    # turning score = how much the velocity reverses around each interior sample
    best_i, best_score = None, -1.0
    for i in range(1, len(fr) - 1):
        if not (lo <= fr[i] <= hi):
            continue
        # reversal: sign change in velocity, weighted by magnitude drop+recover
        rev = axis_v[i - 1] * axis_v[i + 1]
        score = -rev  # large positive when signs differ (a reversal)
        if score > best_score:
            best_score, best_i = score, i
    if best_i is None:
        best_i = len(fr) // 2
    cand = fr[best_i]

    # --- split into incoming (before) / outgoing (after), excluding a guard
    # band and any frames inside the occlusion gap straddling the candidate.
    in_mask = fr <= cand - guard
    out_mask = fr >= cand + guard
    if in_mask.sum() < min_seg or out_mask.sum() < min_seg:
        # fall back: just report the reversal frame, low confidence
        return ContactEstimate(float(cand), float(cand / fps), 0.3, 1000.0 / fps,
                               None, None, 0.0, 'reversal',
                               int(in_mask.sum()), int(out_mask.sum()))

    px_in, py_in, rms_in = _segfit(fr[in_mask], xs[in_mask], ys[in_mask], fit_deg)
    px_out, py_out, rms_out = _segfit(fr[out_mask], xs[out_mask], ys[out_mask], fit_deg)

    # --- occlusion window: the gap of missing frames straddling the candidate.
    last_in = int(fr[in_mask][-1])
    first_out = int(fr[out_mask][0])
    present = set(int(f) for f in fr)
    gap = [f for f in range(last_in + 1, first_out) if f not in present]
    occ_from = gap[0] if gap else None
    occ_to = gap[-1] if gap else None
    occ_ms = (len(gap) / fps * 1000.0) if gap else 0.0

    # --- sub-frame contact: time where the two extrapolated position curves are
    # closest (the ball is at one place at contact; velocity flips there).
    grid = np.linspace(last_in, first_out, 400)
    d = np.hypot(px_in(grid) - px_out(grid), py_in(grid) - py_out(grid))
    j = int(np.argmin(d))
    contact = float(grid[j])
    dmin = float(d[j])

    # --- confidence + uncertainty. Good fits (low rms) + a sharp, low-distance
    # crossing => high confidence. Uncertainty ~ how wide the near-minimum basin
    # is, in time, plus the fit noise.
    scale = float(np.median(np.hypot(np.diff(xs), np.diff(ys))) + 1e-6)  # px/frame
    near = grid[d < dmin + scale]
    basin_frames = (near.max() - near.min()) if len(near) else 1.0
    unc_frames = 0.5 * basin_frames + (rms_in + rms_out) / (2 * scale)
    uncertainty_ms = float(unc_frames / fps * 1000.0)
    fit_quality = 1.0 / (1.0 + (rms_in + rms_out) / (2 * scale))
    cross_quality = 1.0 / (1.0 + dmin / scale)
    confidence = float(max(0.0, min(1.0, 0.5 * fit_quality + 0.5 * cross_quality)))

    return ContactEstimate(
        contact_frame=round(contact, 3),
        contact_time=round(contact / fps, 5),
        confidence=round(confidence, 3),
        uncertainty_ms=round(uncertainty_ms, 1),
        occluded_from=occ_from, occluded_to=occ_to, occluded_ms=round(occ_ms, 1),
        method='intersection',
        n_in=int(in_mask.sum()), n_out=int(out_mask.sum()),
    )


# ─────────────────────────── self-test ───────────────────────────

def _synth(true_contact, fps=120, n=60, occlude=(0, 0), noise=0.0, seed=0,
           v_in=(6.0, -4.0), v_out=(-3.0, -5.0), g=0.15):
    """Synthesize a ball track: two parabolic arcs meeting at true_contact.
    occlude=(before,after): drop that many frames on each side of contact.
    Returns a track list (visible frames only)."""
    rng = np.random.default_rng(seed)
    # position continuous at contact; choose a contact point
    cx, cy = 100.0, 80.0
    track = []
    for f in range(n):
        dt = f - true_contact
        if dt <= 0:
            x = cx + v_in[0] * dt
            y = cy + v_in[1] * dt + 0.5 * g * dt * dt
        else:
            x = cx + v_out[0] * dt
            y = cy + v_out[1] * dt + 0.5 * g * dt * dt
        if noise:
            x += rng.normal(0, noise); y += rng.normal(0, noise)
        track.append({'frame': f, 'x': x, 'y': y})
    # drop the occlusion window straddling contact
    cf = int(round(true_contact))
    drop = set(range(cf - occlude[0], cf + occlude[1] + 1))
    return [p for p in track if p['frame'] not in drop]


def _selftest():
    fps = 120
    cases = [
        ("clean, integer contact",      30.0, (0, 0), 0.0),
        ("clean, sub-frame contact",    30.4, (0, 0), 0.0),
        ("occluded ±2 at impact",       30.4, (2, 2), 0.0),
        ("occluded ±4 at impact",       30.6, (4, 4), 0.0),
        ("noisy (1px) + occluded ±3",   30.5, (3, 3), 1.0),
        ("noisy (2px) + occluded ±2",   30.5, (2, 2), 2.0),
    ]
    print(f"{'case':32s} {'true':>6} {'est':>7} {'err_ms':>7} {'conf':>5} {'occ_ms':>7} {'±ms':>6}")
    worst = 0.0
    for name, tc, occ, noise in cases:
        tr = _synth(tc, fps=fps, occlude=occ, noise=noise)
        est = estimate_contact(tr, fps)
        err_ms = abs(est.contact_frame - tc) / fps * 1000.0
        worst = max(worst, err_ms)
        print(f"{name:32s} {tc:6.2f} {est.contact_frame:7.2f} {err_ms:7.1f} "
              f"{est.confidence:5.2f} {est.occluded_ms:7.1f} {est.uncertainty_ms:6.1f}")
    # accept if even the noisy/occluded cases land within ~1 frame (8.3ms @120)
    ok = worst < 1000.0 / fps
    print(f"\nworst error: {worst:.1f}ms  ({'PASS' if ok else 'FAIL'} "
          f"@ <{1000.0/fps:.1f}ms = 1 frame)")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--track", help="JSON file: {fps, track:[{frame,x,y,visible?}]}")
    ap.add_argument("--fps", type=float, default=None)
    args = ap.parse_args()
    if args.selftest:
        return _selftest()
    if args.track:
        d = json.loads(open(args.track).read())
        fps = args.fps or d.get("fps", 120)
        est = estimate_contact(d["track"], fps)
        print(json.dumps(asdict(est), indent=2))
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
