#!/usr/bin/env python3
"""Measure contact-detection accuracy using the ball-STRIKE SOUND as truth.

For each detected shot, we have the model's contact frame. The ball-strike is
audible — extract_audio_peaks() (from swing_composite) finds those acoustic
impacts. The signed error between the detected contact frame and the nearest
strong audio peak is an OBJECTIVE, label-free measure of how well we pin the
contact moment — the root cause behind:
  - shot-detection timing feeling early/late
  - filmstrips centering on the wrong frame
  - pro comparisons drifting out of sync

This is read-only: it reads detections + the preprocessed video's audio and
prints/returns a report. No model, no pipeline change. Run anywhere ffmpeg +
the preprocessed mp4 + the detections JSON are available (i.e. a GPU machine,
or the Mac if those files are local).

Usage:
    python scripts/contact_accuracy.py IMG_0867
    python scripts/contact_accuracy.py IMG_0867 --json out.json
    python scripts/contact_accuracy.py --all            # every video with detections
    python scripts/contact_accuracy.py IMG_0867 --match-window-ms 200
"""

import argparse
import glob
import json
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import PROJECT_ROOT, PREPROCESSED_DIR
from scripts.swing_composite import extract_audio_peaks

DETECTIONS_DIR = Path(PROJECT_ROOT) / "detections"


def _load_detections(vid):
    for name in (f"{vid}_fused_detections.json", f"{vid}_fused.json"):
        p = DETECTIONS_DIR / name
        if p.exists():
            return json.load(open(p))
    return None


def _nearest_peak_frame(frame, peaks, max_gap_frames, min_rel_amp):
    """Nearest STRONG audio peak to `frame` within max_gap_frames. Returns
    (peak_frame, gap_frames) or (None, None)."""
    best = None
    for pk in peaks:
        if pk.get("rel_amp", 0) < min_rel_amp:
            continue
        gap = pk["frame"] - frame
        if abs(gap) <= max_gap_frames and (best is None or abs(gap) < abs(best[1])):
            best = (pk["frame"], gap)
    return best if best else (None, None)


def measure_video(vid, match_window_ms=200, min_rel_amp=1.5, video_path=None):
    """Compute per-shot contact error for one video. Returns a report dict."""
    det = _load_detections(vid)
    if not det:
        return {"video": vid, "error": "no detections"}
    fps = det.get("fps") or 60.0
    detections = det.get("detections", [])

    vp = Path(video_path) if video_path else (Path(PREPROCESSED_DIR) / f"{vid}.mp4")
    if not vp.exists():
        return {"video": vid, "error": f"no preprocessed video at {vp}"}

    try:
        audio = extract_audio_peaks(str(vp))
    except Exception as e:
        # No audio track / corrupt audio — can't measure this one, don't crash.
        return {"video": vid, "error": f"audio extraction failed: {str(e)[:80]}"}
    peaks = audio.get("peaks", [])
    afps = audio.get("fps") or fps
    # Audio peaks are in SOURCE-frame coords at afps; detections are at det fps.
    # Convert audio peak frames to seconds → compare in seconds, robust to fps.
    peak_secs = [{"frame": pk["frame"], "t": pk["frame"] / afps,
                  "rel_amp": pk.get("rel_amp", 0)} for pk in peaks]
    max_gap_s = match_window_ms / 1000.0

    per_shot = []
    matched_errors_ms = []          # ALL matches (advisory)
    confident_errors_ms = []        # only high-confidence matches (gate-grade)
    for d in detections:
        det_t = d.get("timestamp")
        if det_t is None and d.get("frame") is not None:
            det_t = d["frame"] / fps
        # Find the nearest strong peak, AND the runner-up, within the window.
        cands = []
        for pk in peak_secs:
            if pk["rel_amp"] < min_rel_amp:
                continue
            gap = pk["t"] - det_t
            if abs(gap) <= max_gap_s:
                cands.append((abs(gap), gap, pk["rel_amp"]))
        cands.sort()
        err_ms = None
        confident = False
        if cands:
            _, gap, amp = cands[0]
            err_ms = round(gap * 1000, 1)
            matched_errors_ms.append(err_ms)
            # Confident match = the peak is strong AND isolated: no OTHER
            # qualifying peak within the window (so we're not guessing between
            # this shot's strike and a neighbouring shot / court noise), and
            # the peak is clearly above threshold.
            isolated = len(cands) == 1 or cands[1][0] - cands[0][0] > (max_gap_s * 0.5)
            if isolated and amp >= max(min_rel_amp, 1.5):
                confident = True
                confident_errors_ms.append(err_ms)
        per_shot.append({
            "shot_type": d.get("shot_type"),
            "timestamp": round(det_t, 3) if det_t is not None else None,
            "contact_error_ms": err_ms,           # signed: + = audio after detection
            "confident": confident,               # gate-grade match?
        })

    n = len(detections)
    # CONFIDENT matches are the gate-grade signal; all-matches are advisory.
    cabs = [abs(e) for e in confident_errors_ms]
    summary = {
        "video": vid,
        "fps": fps,
        "shots": n,
        "matched_to_audio": len(matched_errors_ms),
        "confident_matches": len(confident_errors_ms),
        "confident_coverage": round(len(confident_errors_ms) / n, 3) if n else 0,
        "audio_peaks_found": len(peaks),
        # Headline error numbers are computed from CONFIDENT matches only.
        "median_abs_error_ms": round(statistics.median(cabs), 1) if cabs else None,
        "p90_abs_error_ms": (round(sorted(cabs)[int(len(cabs) * 0.9)], 1)
                             if len(cabs) >= 10 else None),
        "median_signed_error_ms": (round(statistics.median(confident_errors_ms), 1)
                                   if confident_errors_ms else None),
        "per_shot": per_shot,
    }
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="?", help="video id (e.g. IMG_0867)")
    ap.add_argument("--all", action="store_true", help="every video with detections")
    ap.add_argument("--match-window-ms", type=int, default=200,
                    help="max |detected - audio| to count as the same strike")
    ap.add_argument("--min-rel-amp", type=float, default=1.5,
                    help="min audio peak strength (×percentile threshold) to trust")
    ap.add_argument("--json", default=None, help="write full report JSON here")
    args = ap.parse_args()

    if not args.video and not args.all:
        ap.error("give a video id or --all")

    if args.all:
        vids = sorted({Path(p).name.replace("_fused_detections.json", "").replace("_fused.json", "")
                       for p in glob.glob(str(DETECTIONS_DIR / "*_fused*.json"))})
    else:
        vids = [args.video]

    reports = []
    all_abs = []
    for vid in vids:
        r = measure_video(vid, args.match_window_ms, args.min_rel_amp)
        reports.append(r)
        if "error" in r:
            print(f"{vid:18} — {r['error']}")
            continue
        print(f"{vid:18} shots={r['shots']:3} confident={r['confident_matches']:3} "
              f"({int(r['confident_coverage']*100):3}%)  median|err|="
              f"{r['median_abs_error_ms']}ms  p90={r['p90_abs_error_ms']}ms  "
              f"bias={r['median_signed_error_ms']}ms")
        all_abs += [abs(s["contact_error_ms"]) for s in r["per_shot"]
                    if s.get("confident") and s["contact_error_ms"] is not None]

    if len(vids) > 1 and all_abs:
        print("\n=== catalog ===")
        print(f"  shots matched to audio: {len(all_abs)}")
        print(f"  median |contact error|: {round(statistics.median(all_abs),1)}ms")
        print(f"  p90 |contact error|:    {round(sorted(all_abs)[int(len(all_abs)*0.9)],1)}ms")

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"reports": reports}, f, indent=2)
        print(f"\nWrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
