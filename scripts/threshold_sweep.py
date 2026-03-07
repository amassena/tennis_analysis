#!/usr/bin/env python3
"""Sweep pipeline thresholds to find optimal values for a given model.

Runs fused_detect on all GT videos with different threshold offsets,
then validates and reports the best combination.
"""

import json
import os
import subprocess
import sys
import itertools
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# 12 baseline GT videos
GT_VIDEOS = [
    "IMG_0864", "IMG_0865", "IMG_0866", "IMG_0867", "IMG_0868",
    "IMG_0869", "IMG_0870", "IMG_0929", "IMG_6665", "IMG_6703",
    "IMG_6711", "IMG_6713",
]

# GT file mapping (same as validate_pipeline.py)
GT_FILES = {
    "IMG_0864": "detections/IMG_0864_fused.json",
    "IMG_0865": "detections/IMG_0865_fused.json",
    "IMG_0866": "detections/IMG_0866_fused.json",
    "IMG_0867": "detections/IMG_0867_fused.json",
    "IMG_0868": "detections/IMG_0868_fused.json",
    "IMG_0869": "detections/IMG_0869_fused.json",
    "IMG_0870": "detections/IMG_0870_fused.json",
    "IMG_0929": "detections/IMG_0929_fused_v5.json",
    "IMG_6665": "detections/IMG_6665_fused_v5.json",
    "IMG_6703": "detections/IMG_6703_fused_v5.json",
    "IMG_6711": "detections/IMG_6711_fused_v5.json",
    "IMG_6713": "detections/IMG_6713_fused.json",
}

DEFAULT_THRESHOLDS = {
    'ns_permissive': 0.40, 'ns_moderate': 0.35, 'ns_strict': 0.32,
    'ns_first_pass': 0.30, 'ns_weak_jerk': 0.25, 'ns_weak_heuristic': 0.24,
    'mc_strong': 0.50, 'mc_moderate': 0.40, 'mc_weak': 0.30,
    'mc_floor_audio_heuristic': 0.43, 'mc_floor_heuristic_only': 0.55,
    'mc_floor_jerk': 0.60, 'mc_low_pass': 0.50,
    'mc_sliding_window': 0.50, 'mc_rhythm_fill': 0.40,
    'ns_rescue': 0.40, 'ns_rescue_reject': 0.35, 'mc_rescue': 0.62,
    'ns_jerk_high': 0.40, 'mc_jerk_high': 0.40,
    'ns_jerk_low': 0.30, 'mc_jerk_low': 0.50,
    'fc_weak_jerk': 0.45, 'ns_weak_jerk_floor': 0.10,
    'fc_weak_low': 0.52,
}


def apply_offsets(ns_offset, mc_offset, fc_offset=0.0):
    """Apply uniform offsets to threshold groups.

    ns_offset: NEGATIVE = stricter (lower threshold, easier to reject as not_shot)
               POSITIVE = more permissive
    mc_offset: POSITIVE = stricter (higher confidence floor needed)
               NEGATIVE = more permissive
    fc_offset: same as mc_offset direction
    """
    thresholds = dict(DEFAULT_THRESHOLDS)
    for key in thresholds:
        if key.startswith('ns_'):
            thresholds[key] = max(0.05, min(0.95, DEFAULT_THRESHOLDS[key] + ns_offset))
        elif key.startswith('mc_'):
            thresholds[key] = max(0.05, min(0.95, DEFAULT_THRESHOLDS[key] + mc_offset))
        elif key.startswith('fc_'):
            thresholds[key] = max(0.05, min(0.95, DEFAULT_THRESHOLDS[key] + fc_offset))
    return thresholds


def write_thresholds_to_meta(thresholds, meta_path):
    """Write calibrated thresholds to model metadata."""
    with open(meta_path) as f:
        meta = json.load(f)
    meta['calibrated_thresholds'] = thresholds
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)


def clear_thresholds_from_meta(meta_path):
    """Remove calibrated thresholds from model metadata."""
    with open(meta_path) as f:
        meta = json.load(f)
    meta.pop('calibrated_thresholds', None)
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)


def run_detections(videos, poses_3d_dir=None):
    """Run fused_detect on all videos."""
    python = os.path.join(PROJECT_ROOT, ".venv", "bin", "python")
    for video in videos:
        video_path = os.path.join(PROJECT_ROOT, "preprocessed", f"{video}.mp4")
        cmd = [python, os.path.join(PROJECT_ROOT, "scripts", "fused_detect.py"), video_path]
        if poses_3d_dir:
            cmd.extend(["--poses-3d-dir", poses_3d_dir])
        subprocess.run(cmd, capture_output=True, text=True)


def validate(videos):
    """Compute aggregate TP/FP/FN for given videos."""
    total_tp = total_fp = total_fn = 0
    per_video = {}

    for video in videos:
        gt_file = os.path.join(PROJECT_ROOT, GT_FILES[video])
        det_file = os.path.join(PROJECT_ROOT, "detections", f"{video}_fused_detections.json")

        if not os.path.exists(gt_file) or not os.path.exists(det_file):
            continue

        with open(gt_file) as f:
            gt = json.load(f)
        with open(det_file) as f:
            det = json.load(f)

        gt_shots = [s for s in gt.get('shots', gt.get('detections', []))
                     if s.get('shot_type') not in ('practice', 'offscreen', 'unknown_shot')]
        det_shots = [s for s in det.get('shots', det.get('detections', []))
                      if s.get('shot_type') not in ('practice', 'offscreen', 'unknown_shot')]

        gt_times = sorted([s.get('time', s.get('timestamp', 0)) for s in gt_shots])
        det_times = sorted([s.get('time', s.get('timestamp', 0)) for s in det_shots])

        # Match with 1.5s tolerance
        matched_gt = set()
        matched_det = set()
        tolerance = 1.5

        for di, dt in enumerate(det_times):
            best_dist = tolerance + 1
            best_gi = -1
            for gi, gt_t in enumerate(gt_times):
                if gi in matched_gt:
                    continue
                dist = abs(dt - gt_t)
                if dist <= tolerance and dist < best_dist:
                    best_dist = dist
                    best_gi = gi
            if best_gi >= 0:
                matched_gt.add(best_gi)
                matched_det.add(di)

        tp = len(matched_gt)
        fp = len(det_times) - tp
        fn = len(gt_times) - tp

        total_tp += tp
        total_fp += fp
        total_fn += fn
        per_video[video] = {'tp': tp, 'fp': fp, 'fn': fn}

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'tp': total_tp, 'fp': total_fp, 'fn': total_fn,
        'precision': precision, 'recall': recall, 'f1': f1,
        'per_video': per_video,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Sweep pipeline thresholds")
    parser.add_argument("--poses-3d-dir", default=None,
                        help="3D pose directory for dual-pose mode")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: test on 3 key videos only")
    parser.add_argument("--ns-range", nargs=3, type=float, default=[0.0, 0.20, 0.05],
                        metavar=("MIN", "MAX", "STEP"),
                        help="not_shot offset range (default: 0.0 0.20 0.05)")
    parser.add_argument("--mc-range", nargs=3, type=float, default=[0.0, 0.15, 0.05],
                        metavar=("MIN", "MAX", "STEP"),
                        help="ml_conf offset range (default: 0.0 0.15 0.05)")
    args = parser.parse_args()

    meta_path = os.path.join(PROJECT_ROOT, "models", "shot_classifier_meta.json")

    # Backup original meta
    with open(meta_path) as f:
        original_meta = json.load(f)

    videos = GT_VIDEOS
    if args.quick:
        videos = ["IMG_6703", "IMG_6711", "IMG_6713"]
        print(f"Quick mode: testing on {videos}")

    ns_min, ns_max, ns_step = args.ns_range
    mc_min, mc_max, mc_step = args.mc_range

    ns_values = []
    v = ns_min
    while v <= ns_max + 0.001:
        ns_values.append(round(v, 3))
        v += ns_step

    mc_values = []
    v = mc_min
    while v <= mc_max + 0.001:
        mc_values.append(round(v, 3))
        v += mc_step

    combos = list(itertools.product(ns_values, mc_values))
    print(f"Testing {len(combos)} threshold combinations on {len(videos)} videos...")
    print(f"  ns offsets: {ns_values}")
    print(f"  mc offsets: {mc_values}")
    print()

    results = []

    try:
        for i, (ns_off, mc_off) in enumerate(combos):
            thresholds = apply_offsets(ns_off, mc_off)
            write_thresholds_to_meta(thresholds, meta_path)

            print(f"[{i+1}/{len(combos)}] ns={ns_off:+.2f} mc={mc_off:+.2f} ...", end=" ", flush=True)

            # Force model reload by clearing cache
            import scripts.fused_detect as fd
            fd._classifier_loaded = False
            fd._classifier = None
            fd._classifier_meta = None

            run_detections(videos, poses_3d_dir=args.poses_3d_dir)
            result = validate(videos)

            print(f"TP={result['tp']} FP={result['fp']} FN={result['fn']} "
                  f"P={result['precision']:.1%} R={result['recall']:.1%} F1={result['f1']:.1%}")

            results.append({
                'ns_offset': ns_off, 'mc_offset': mc_off,
                **result,
            })

    finally:
        # Restore original meta
        with open(meta_path, 'w') as f:
            json.dump(original_meta, f, indent=2)
        print("\nRestored original model metadata.")

    # Sort by F1
    results.sort(key=lambda r: r['f1'], reverse=True)
    print("\n" + "=" * 70)
    print("TOP 5 RESULTS")
    print("=" * 70)
    for r in results[:5]:
        print(f"  ns={r['ns_offset']:+.2f} mc={r['mc_offset']:+.2f}  "
              f"TP={r['tp']} FP={r['fp']} FN={r['fn']}  "
              f"P={r['precision']:.1%} R={r['recall']:.1%} F1={r['f1']:.1%}")

        # Show per-video for top result
        if r == results[0]:
            print("  Per-video:")
            for v in sorted(r['per_video']):
                pv = r['per_video'][v]
                vf1 = 2*pv['tp']/(2*pv['tp']+pv['fp']+pv['fn']) if (2*pv['tp']+pv['fp']+pv['fn']) > 0 else 0
                if pv['fp'] + pv['fn'] > 0:
                    print(f"    {v}: TP={pv['tp']} FP={pv['fp']} FN={pv['fn']} F1={vf1:.1%}")

    # Save results
    out_path = os.path.join(PROJECT_ROOT, "training", "threshold_sweep_results.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()
