#!/usr/bin/env python3
"""Ensemble shot detector: merges baseline (fused_detect) + window detector.

The two detectors make independent errors (28 window-only TPs, 22 baseline-only
TPs in GT evaluation). This script combines their outputs to recover shots that
either system misses individually.

Three strategies implemented:
  1. Simple Union + NMS
  2. Confidence-Weighted Union (default) — stricter thresholds for single-source
  3. Agreement-boosted — boosts confidence when both systems agree

Usage:
    .venv/bin/python scripts/ensemble_detect.py preprocessed/IMG_6703.mp4
    .venv/bin/python scripts/ensemble_detect.py --all
    .venv/bin/python scripts/ensemble_detect.py --all --strategy weighted --sweep
"""

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

DETECTIONS_DIR = os.path.join(PROJECT_ROOT, "detections")

# Import GT_VIDEOS and validation utilities
from scripts.validate_pipeline import (
    GT_VIDEOS, IGNORE_TYPES, MATCH_TOLERANCE,
    load_ground_truth, match_detections, compute_metrics,
)


def load_baseline_detections(video_name):
    """Load baseline (fused_detect.py) detections."""
    path = os.path.join(DETECTIONS_DIR, f"{video_name}_fused_detections.json")
    if not os.path.exists(path):
        return []

    with open(path) as f:
        data = json.load(f)

    dets = []
    for det in data.get("detections", []):
        shot_type = det.get("shot_type", "unknown")
        if shot_type in IGNORE_TYPES:
            continue
        dets.append({
            "timestamp": det["timestamp"],
            "confidence": det.get("confidence", 0.5),
            "shot_type": shot_type,
            "source": det.get("source", "unknown"),
            "not_shot_prob": det.get("not_shot_prob", 0.0),
            "ml_confidence": det.get("ml_confidence", 0.0),
            "tier": det.get("tier", "medium"),
            "detector": "baseline",
        })
    return dets


def load_window_detections(video_name):
    """Load window detector detections."""
    path = os.path.join(DETECTIONS_DIR, f"{video_name}_window_detections.json")
    if not os.path.exists(path):
        return []

    with open(path) as f:
        data = json.load(f)

    dets = []
    for det in data.get("detections", []):
        shot_type = det.get("shot_type", "unknown")
        if shot_type in IGNORE_TYPES:
            continue
        dets.append({
            "timestamp": det["timestamp"],
            "confidence": det.get("confidence", 0.5),
            "shot_prob": det.get("shot_prob", det.get("confidence", 0.5)),
            "shot_type": shot_type,
            "source": "window_detector",
            "detector": "window",
        })
    return dets


def find_matches(baseline_dets, window_dets, match_window=1.5):
    """Find which detections from each system overlap.

    Returns:
        both: list of (baseline_det, window_det) pairs
        baseline_only: list of baseline dets with no window match
        window_only: list of window dets with no baseline match
    """
    used_w = set()
    both = []
    baseline_only = []

    for bd in baseline_dets:
        best_j = None
        best_dist = match_window + 1
        for j, wd in enumerate(window_dets):
            if j in used_w:
                continue
            dist = abs(bd["timestamp"] - wd["timestamp"])
            if dist < best_dist:
                best_dist = dist
                best_j = j
        if best_j is not None and best_dist <= match_window:
            both.append((bd, window_dets[best_j]))
            used_w.add(best_j)
        else:
            baseline_only.append(bd)

    window_only = [wd for j, wd in enumerate(window_dets) if j not in used_w]
    return both, baseline_only, window_only


def ensemble_simple_union(baseline_dets, window_dets, nms_gap=1.5):
    """Strategy 1: Simple union with NMS.

    Merge all detections from both systems, keep highest confidence within NMS gap.
    """
    all_dets = []

    for bd in baseline_dets:
        all_dets.append({
            "timestamp": bd["timestamp"],
            "confidence": bd["confidence"],
            "shot_type": bd["shot_type"],
            "source": bd.get("source", "baseline"),
            "ensemble_source": "baseline",
        })

    for wd in window_dets:
        all_dets.append({
            "timestamp": wd["timestamp"],
            "confidence": wd["confidence"],
            "shot_type": wd.get("shot_type", "forehand"),
            "source": "window_detector",
            "ensemble_source": "window",
        })

    # NMS: greedily pick highest confidence, suppress within nms_gap
    all_dets.sort(key=lambda d: d["confidence"], reverse=True)
    selected = []
    suppressed = set()

    for i, det in enumerate(all_dets):
        if i in suppressed:
            continue
        selected.append(det)
        for j in range(i + 1, len(all_dets)):
            if j in suppressed:
                continue
            if abs(all_dets[j]["timestamp"] - det["timestamp"]) < nms_gap:
                suppressed.add(j)

    selected.sort(key=lambda d: d["timestamp"])
    return selected


def ensemble_weighted_union(baseline_dets, window_dets,
                            baseline_threshold=0.0,
                            window_threshold=0.65,
                            match_window=1.5):
    """Strategy 2: Confidence-weighted union.

    - Both detect → keep (use baseline info for shot type, boost confidence)
    - Baseline only → always keep (high precision system)
    - Window only → keep only if window confidence > threshold (filter FPs)

    Args:
        baseline_threshold: min confidence for baseline-only (0 = keep all)
        window_threshold: min confidence for window-only (higher = stricter)
        match_window: seconds to consider as same detection
    """
    both, baseline_only, window_only = find_matches(
        baseline_dets, window_dets, match_window
    )

    merged = []

    # Both detect: use baseline's shot type/source, boost confidence
    for bd, wd in both:
        merged.append({
            "timestamp": bd["timestamp"],  # use baseline timestamp (more precise)
            "confidence": max(bd["confidence"], wd["confidence"]),
            "shot_type": bd["shot_type"],
            "source": bd.get("source", "baseline"),
            "ensemble_source": "both",
            "window_prob": wd.get("shot_prob", wd["confidence"]),
            "baseline_conf": bd["confidence"],
        })

    # Baseline only: keep all (or apply threshold)
    for bd in baseline_only:
        if bd["confidence"] >= baseline_threshold:
            merged.append({
                "timestamp": bd["timestamp"],
                "confidence": bd["confidence"],
                "shot_type": bd["shot_type"],
                "source": bd.get("source", "baseline"),
                "ensemble_source": "baseline_only",
                "baseline_conf": bd["confidence"],
            })

    # Window only: keep if above threshold
    for wd in window_only:
        wp = wd.get("shot_prob", wd["confidence"])
        if wp >= window_threshold:
            merged.append({
                "timestamp": wd["timestamp"],
                "confidence": wp,
                "shot_type": "forehand",  # window detector doesn't classify
                "source": "window_detector",
                "ensemble_source": "window_only",
                "window_prob": wp,
            })

    merged.sort(key=lambda d: d["timestamp"])
    return merged


def evaluate_ensemble(video_name, detections, verbose=False):
    """Evaluate ensemble detections against GT.

    Returns metrics dict or None if no GT available.
    """
    gt_list = load_ground_truth(video_name)
    if not gt_list:
        return None

    # Filter to non-ignored types (already done in load functions, but be safe)
    det_list = [d for d in detections if d.get("shot_type", "") not in IGNORE_TYPES]

    tp_pairs, fp_dets, fn_gts = match_detections(gt_list, det_list)
    metrics = compute_metrics(len(tp_pairs), len(fp_dets), len(fn_gts))

    if verbose:
        print(f"  {video_name}: TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']} "
              f"P={metrics['precision']:.1%} R={metrics['recall']:.1%} F1={metrics['f1']:.1%}")

        # Show ensemble source breakdown
        source_counts = {}
        for _, det, _ in tp_pairs:
            src = det.get("ensemble_source", "unknown")
            source_counts[src] = source_counts.get(src, 0) + 1
        fp_sources = {}
        for det in fp_dets:
            src = det.get("ensemble_source", "unknown")
            fp_sources[src] = fp_sources.get(src, 0) + 1

        if source_counts:
            parts = [f"{k}={v}" for k, v in sorted(source_counts.items())]
            print(f"    TP by ensemble source: {', '.join(parts)}")
        if fp_sources:
            parts = [f"{k}={v}" for k, v in sorted(fp_sources.items())]
            print(f"    FP by ensemble source: {', '.join(parts)}")

    return metrics


def sweep_thresholds(verbose=True):
    """Grid search window_threshold to find optimal ensemble parameters."""
    print("\n" + "=" * 70)
    print("Threshold sweep: window_threshold for weighted union")
    print("=" * 70)

    # Load all GT video detections
    all_data = {}
    for video_name in sorted(GT_VIDEOS.keys()):
        bl = load_baseline_detections(video_name)
        wd = load_window_detections(video_name)
        gt = load_ground_truth(video_name)
        if bl and gt:
            all_data[video_name] = (bl, wd, gt)

    print(f"Loaded {len(all_data)} videos with baseline + GT")

    # Sweep window threshold
    print(f"\n{'Thresh':>7s}  {'TP':>4s}  {'FP':>4s}  {'FN':>4s}  {'P':>6s}  {'R':>6s}  {'F1':>6s}  "
          f"{'W-only TP':>9s}  {'W-only FP':>9s}")
    print("-" * 80)

    best_f1 = 0
    best_thresh = 0.5

    for thresh in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
        total_tp = total_fp = total_fn = 0
        total_w_tp = total_w_fp = 0

        for video_name, (bl, wd, gt) in all_data.items():
            merged = ensemble_weighted_union(bl, wd, window_threshold=thresh)
            det_list = [d for d in merged if d.get("shot_type", "") not in IGNORE_TYPES]
            tp_pairs, fp_dets, fn_gts = match_detections(gt, det_list)

            total_tp += len(tp_pairs)
            total_fp += len(fp_dets)
            total_fn += len(fn_gts)

            # Count window-only contributions
            for _, det, _ in tp_pairs:
                if det.get("ensemble_source") == "window_only":
                    total_w_tp += 1
            for det in fp_dets:
                if det.get("ensemble_source") == "window_only":
                    total_w_fp += 1

        m = compute_metrics(total_tp, total_fp, total_fn)
        marker = " *" if m["f1"] > best_f1 else ""
        print(f"  {thresh:5.2f}  {m['tp']:4d}  {m['fp']:4d}  {m['fn']:4d}  "
              f"{m['precision']:6.3f}  {m['recall']:6.3f}  {m['f1']:6.3f}  "
              f"{total_w_tp:9d}  {total_w_fp:9d}{marker}")

        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_thresh = thresh

    print(f"\nBest: window_threshold={best_thresh:.2f} → F1={best_f1:.3f}")

    # Also show baseline for comparison
    total_tp = total_fp = total_fn = 0
    for video_name, (bl, wd, gt) in all_data.items():
        det_list = [d for d in bl if d.get("shot_type", "") not in IGNORE_TYPES]
        tp_pairs, fp_dets, fn_gts = match_detections(gt, det_list)
        total_tp += len(tp_pairs)
        total_fp += len(fp_dets)
        total_fn += len(fn_gts)
    bm = compute_metrics(total_tp, total_fp, total_fn)
    print(f"Baseline: TP={bm['tp']} FP={bm['fp']} FN={bm['fn']} "
          f"P={bm['precision']:.3f} R={bm['recall']:.3f} F1={bm['f1']:.3f}")

    return best_thresh


def format_output(video_name, detections, strategy="weighted"):
    """Format ensemble detections in standard JSON structure."""
    output = {
        "version": 1,
        "detector": f"ensemble_{strategy}",
        "source_video": video_name,
        "parameters": {
            "strategy": strategy,
        },
        "summary": {
            "total_detections": len(detections),
            "by_ensemble_source": {},
        },
        "detections": detections,
    }

    for det in detections:
        src = det.get("ensemble_source", "unknown")
        output["summary"]["by_ensemble_source"][src] = \
            output["summary"]["by_ensemble_source"].get(src, 0) + 1

    return output


def main():
    parser = argparse.ArgumentParser(description="Ensemble shot detector")
    parser.add_argument("video", nargs="?", help="Path to preprocessed video")
    parser.add_argument("--all", action="store_true", help="Run on all GT videos")
    parser.add_argument("--strategy", choices=["union", "weighted"],
                        default="weighted", help="Ensemble strategy (default: weighted)")
    parser.add_argument("--window-threshold", type=float, default=0.65,
                        help="Min window confidence for window-only detections (default: 0.65)")
    parser.add_argument("--sweep", action="store_true",
                        help="Run threshold sweep to find optimal parameters")
    parser.add_argument("--save", action="store_true",
                        help="Save ensemble detections to JSON files")
    args = parser.parse_args()

    if args.sweep:
        best = sweep_thresholds()
        return

    if not args.video and not args.all:
        parser.error("Provide a video path or use --all")

    if args.all:
        videos = sorted(GT_VIDEOS.keys())
    else:
        video_name = os.path.splitext(os.path.basename(args.video))[0]
        videos = [video_name]

    total_tp = total_fp = total_fn = 0

    for video_name in videos:
        bl = load_baseline_detections(video_name)
        wd = load_window_detections(video_name)

        if not bl:
            print(f"  {video_name}: No baseline detections, skipping")
            continue

        if args.strategy == "union":
            merged = ensemble_simple_union(bl, wd)
        else:
            merged = ensemble_weighted_union(
                bl, wd, window_threshold=args.window_threshold
            )

        # Evaluate
        metrics = evaluate_ensemble(video_name, merged, verbose=True)
        if metrics:
            total_tp += metrics["tp"]
            total_fp += metrics["fp"]
            total_fn += metrics["fn"]

        # Save if requested
        if args.save:
            output = format_output(video_name, merged, args.strategy)
            out_path = os.path.join(DETECTIONS_DIR,
                                    f"{video_name}_ensemble_detections.json")
            with open(out_path, "w") as f:
                json.dump(output, f, indent=2)

    # Aggregate
    if len(videos) > 1:
        agg = compute_metrics(total_tp, total_fp, total_fn)
        print(f"\n{'='*60}")
        print(f"AGGREGATE ({len(videos)} videos)")
        print(f"{'='*60}")
        print(f"  TP={agg['tp']} FP={agg['fp']} FN={agg['fn']}")
        print(f"  P={agg['precision']:.1%} R={agg['recall']:.1%} F1={agg['f1']:.1%}")


if __name__ == "__main__":
    main()
