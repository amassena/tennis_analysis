#!/usr/bin/env python3
"""Search for optimal pipeline thresholds for a given model.

Runs fused_detect.py with different threshold combinations on GT videos,
then validates to find the thresholds that maximize aggregate F1.

Uses hierarchical 1D search:
  1. Search ns_offset (shift all ns_* thresholds) with mc fixed
  2. Search mc_offset (shift all mc_* thresholds) with best ns fixed
  3. Fine-tune around the best point

Usage:
    python scripts/search_thresholds.py
    python scripts/search_thresholds.py --videos IMG_0929 IMG_6713 IMG_6703 IMG_6711
    python scripts/search_thresholds.py --full  # Use all 12 GT videos (slower)
"""

import argparse
import json
import os
import subprocess
import sys
import copy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import PROJECT_ROOT, MODELS_DIR

META_PATH = os.path.join(MODELS_DIR, "shot_classifier_meta.json")

# Default thresholds (must match fused_detect.py _DEFAULT_THRESHOLDS)
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

# Representative videos for fast search (covers best/worst/medium)
DEFAULT_SEARCH_VIDEOS = ["IMG_0929", "IMG_6713", "IMG_6703", "IMG_6711"]

# All GT videos for full validation
ALL_GT_VIDEOS = [
    "IMG_0864", "IMG_0865", "IMG_0866", "IMG_0867", "IMG_0868",
    "IMG_0869", "IMG_0870", "IMG_0929", "IMG_6665", "IMG_6703",
    "IMG_6711", "IMG_6713",
]

PYTHON = os.path.join(PROJECT_ROOT, ".venv", "bin", "python")


def write_thresholds(meta, thresholds):
    """Write calibrated thresholds to meta.json."""
    meta_copy = copy.deepcopy(meta)
    meta_copy["calibrated_thresholds"] = thresholds
    with open(META_PATH, "w") as f:
        json.dump(meta_copy, f, indent=2)


def clear_thresholds(meta):
    """Remove calibrated thresholds from meta.json (use defaults)."""
    meta_copy = copy.deepcopy(meta)
    meta_copy.pop("calibrated_thresholds", None)
    with open(META_PATH, "w") as f:
        json.dump(meta_copy, f, indent=2)


def make_thresholds(ns_offset=0.0, mc_offset=0.0):
    """Generate thresholds by offsetting defaults."""
    t = {}
    for key, val in DEFAULT_THRESHOLDS.items():
        if key.startswith('ns_'):
            t[key] = round(val + ns_offset, 4)
        elif key.startswith('mc_'):
            t[key] = round(val + mc_offset, 4)

    # Enforce ns hierarchy: permissive > moderate > strict > first_pass > weak_jerk > weak_heuristic
    ns_keys = ['ns_permissive', 'ns_moderate', 'ns_strict',
               'ns_first_pass', 'ns_weak_jerk', 'ns_weak_heuristic']
    for i in range(1, len(ns_keys)):
        if t[ns_keys[i]] >= t[ns_keys[i - 1]]:
            t[ns_keys[i]] = round(t[ns_keys[i - 1]] - 0.01, 4)

    # Clamp all values to reasonable ranges
    for key in t:
        if key.startswith('ns_'):
            t[key] = max(0.05, min(0.60, t[key]))
        elif key.startswith('mc_'):
            t[key] = max(0.15, min(0.80, t[key]))

    return t


def run_pipeline(videos):
    """Run fused_detect.py on specified videos. Returns True on success."""
    for video in videos:
        video_path = os.path.join(PROJECT_ROOT, "preprocessed", f"{video}.mp4")
        if not os.path.exists(video_path):
            print(f"    WARNING: {video_path} not found, skipping")
            continue
        result = subprocess.run(
            [PYTHON, os.path.join(PROJECT_ROOT, "scripts", "fused_detect.py"), video_path],
            capture_output=True, text=True, timeout=1200
        )
        if result.returncode != 0:
            print(f"    ERROR running {video}: {result.stderr[:200]}")
            return False
    return True


def validate_and_parse():
    """Run validate_pipeline.py and parse aggregate results."""
    result = subprocess.run(
        [PYTHON, os.path.join(PROJECT_ROOT, "scripts", "validate_pipeline.py")],
        capture_output=True, text=True, timeout=120
    )
    if result.returncode != 0:
        return None

    # Parse aggregate line: "  TP=501  FP=25  FN=45"
    for line in result.stdout.split('\n'):
        line = line.strip()
        if line.startswith('TP=') and 'FP=' in line and 'FN=' in line:
            parts = line.split()
            tp = int(parts[0].split('=')[1])
            fp = int(parts[1].split('=')[1])
            fn = int(parts[2].split('=')[1])
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            return {"tp": tp, "fp": fp, "fn": fn, "p": precision, "r": recall, "f1": f1}

    return None


def search_1d(dimension, fixed_offset, values, videos, meta, best_so_far):
    """Search over one dimension (ns or mc) with the other fixed."""
    results = []
    for val in values:
        if dimension == "ns":
            ns_off, mc_off = val, fixed_offset
        else:
            ns_off, mc_off = fixed_offset, val

        thresholds = make_thresholds(ns_offset=ns_off, mc_offset=mc_off)
        write_thresholds(meta, thresholds)

        label = f"ns={ns_off:+.3f} mc={mc_off:+.3f}"
        print(f"  Testing {label} ...", end="", flush=True)

        if not run_pipeline(videos):
            print(" FAILED")
            results.append((val, None))
            continue

        metrics = validate_and_parse()
        if metrics is None:
            print(" PARSE ERROR")
            results.append((val, None))
            continue

        marker = ""
        if metrics["f1"] > best_so_far["f1"]:
            marker = " *** NEW BEST ***"
            best_so_far.update(metrics)
            best_so_far["ns_offset"] = ns_off
            best_so_far["mc_offset"] = mc_off

        print(f" TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']} "
              f"F1={metrics['f1']:.3f}{marker}")
        results.append((val, metrics))

    return results


def main():
    parser = argparse.ArgumentParser(description="Search for optimal pipeline thresholds")
    parser.add_argument("--videos", nargs="+", default=DEFAULT_SEARCH_VIDEOS,
                        help="Videos to use for search (default: 4 representative)")
    parser.add_argument("--full", action="store_true",
                        help="Use all 12 GT videos for search (slower but more accurate)")
    parser.add_argument("--ns-range", type=float, nargs=2, default=[-0.15, 0.10],
                        help="ns_offset search range (default: -0.15 to 0.10)")
    parser.add_argument("--mc-range", type=float, nargs=2, default=[-0.15, 0.15],
                        help="mc_offset search range (default: -0.15 to 0.15)")
    parser.add_argument("--coarse-step", type=float, default=0.03,
                        help="Coarse search step size (default: 0.03)")
    parser.add_argument("--fine-step", type=float, default=0.01,
                        help="Fine search step size (default: 0.01)")
    parser.add_argument("--skip-fine", action="store_true",
                        help="Skip fine-tuning phase")
    args = parser.parse_args()

    if args.full:
        args.videos = ALL_GT_VIDEOS

    # Load original meta
    with open(META_PATH) as f:
        original_meta = json.load(f)

    # Remove any existing calibrated thresholds for clean baseline
    meta = copy.deepcopy(original_meta)
    meta.pop("calibrated_thresholds", None)

    print("=" * 60)
    print("THRESHOLD SEARCH")
    print("=" * 60)
    print(f"Videos: {args.videos}")
    print(f"Model: {meta.get('train_samples', '?')} samples, "
          f"{len(meta.get('feature_names', []))} features")
    print(f"NS range: [{args.ns_range[0]}, {args.ns_range[1]}], step={args.coarse_step}")
    print(f"MC range: [{args.mc_range[0]}, {args.mc_range[1]}], step={args.coarse_step}")

    # Get baseline (default thresholds)
    print(f"\n{'─'*60}")
    print("BASELINE (default thresholds)")
    print(f"{'─'*60}")
    clear_thresholds(meta)
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    run_pipeline(args.videos)
    baseline = validate_and_parse()
    if baseline is None:
        print("ERROR: Could not get baseline metrics")
        # Restore original meta
        with open(META_PATH, "w") as f:
            json.dump(original_meta, f, indent=2)
        sys.exit(1)

    print(f"Baseline: TP={baseline['tp']} FP={baseline['fp']} FN={baseline['fn']} "
          f"F1={baseline['f1']:.3f}")

    best = dict(baseline)
    best["ns_offset"] = 0.0
    best["mc_offset"] = 0.0

    # Phase 1: Coarse ns search (mc fixed at 0)
    print(f"\n{'─'*60}")
    print("PHASE 1: Coarse ns_offset search (mc_offset=0)")
    print(f"{'─'*60}")

    ns_lo, ns_hi = args.ns_range
    ns_step = args.coarse_step
    ns_values = []
    v = ns_lo
    while v <= ns_hi + 1e-6:
        if abs(v) > 1e-6:  # skip 0 (already tested as baseline)
            ns_values.append(round(v, 4))
        v += ns_step

    search_1d("ns", 0.0, ns_values, args.videos, meta, best)
    best_ns = best.get("ns_offset", 0.0)
    print(f"\n  Best ns_offset so far: {best_ns:+.3f} (F1={best['f1']:.3f})")

    # Phase 2: Coarse mc search (ns fixed at best)
    print(f"\n{'─'*60}")
    print(f"PHASE 2: Coarse mc_offset search (ns_offset={best_ns:+.3f})")
    print(f"{'─'*60}")

    mc_lo, mc_hi = args.mc_range
    mc_step = args.coarse_step
    mc_values = []
    v = mc_lo
    while v <= mc_hi + 1e-6:
        if abs(v) > 1e-6:  # skip 0 (already tested)
            mc_values.append(round(v, 4))
        v += mc_step

    search_1d("mc", best_ns, mc_values, args.videos, meta, best)
    best_mc = best.get("mc_offset", 0.0)
    print(f"\n  Best mc_offset so far: {best_mc:+.3f} (F1={best['f1']:.3f})")

    # Phase 3: Fine-tuning around best point
    if not args.skip_fine and (best["f1"] > baseline["f1"] or True):
        print(f"\n{'─'*60}")
        print(f"PHASE 3: Fine-tuning around ns={best_ns:+.3f}, mc={best_mc:+.3f}")
        print(f"{'─'*60}")

        fine_step = args.fine_step

        # Fine ns search around best_ns
        fine_ns = [round(best_ns + d, 4) for d in
                   [-2*fine_step, -fine_step, fine_step, 2*fine_step]
                   if abs(best_ns + d) > 1e-6 or best_ns == 0]
        if fine_ns:
            print(f"  Fine ns search: {fine_ns}")
            search_1d("ns", best_mc, fine_ns, args.videos, meta, best)
            best_ns = best.get("ns_offset", best_ns)

        # Fine mc search around best_mc
        fine_mc = [round(best_mc + d, 4) for d in
                   [-2*fine_step, -fine_step, fine_step, 2*fine_step]
                   if abs(best_mc + d) > 1e-6 or best_mc == 0]
        if fine_mc:
            print(f"  Fine mc search: {fine_mc}")
            search_1d("mc", best_ns, fine_mc, args.videos, meta, best)
            best_mc = best.get("mc_offset", best_mc)

    # Final validation on all videos
    if args.videos != ALL_GT_VIDEOS and best["f1"] > baseline["f1"]:
        print(f"\n{'─'*60}")
        print(f"FINAL VALIDATION (all 12 GT videos)")
        print(f"{'─'*60}")

        best_thresholds = make_thresholds(ns_offset=best_ns, mc_offset=best_mc)
        write_thresholds(meta, best_thresholds)
        print(f"  Running with ns={best_ns:+.3f}, mc={best_mc:+.3f} ...")

        run_pipeline(ALL_GT_VIDEOS)
        full_metrics = validate_and_parse()
        if full_metrics:
            print(f"  Full validation: TP={full_metrics['tp']} FP={full_metrics['fp']} "
                  f"FN={full_metrics['fn']} F1={full_metrics['f1']:.3f}")
            best.update(full_metrics)
            best["ns_offset"] = best_ns
            best["mc_offset"] = best_mc

    # Report
    print(f"\n{'='*60}")
    print(f"SEARCH COMPLETE")
    print(f"{'='*60}")
    print(f"Baseline:  F1={baseline['f1']:.3f} (TP={baseline['tp']} FP={baseline['fp']} FN={baseline['fn']})")
    print(f"Best:      F1={best['f1']:.3f} (TP={best['tp']} FP={best['fp']} FN={best['fn']})")
    print(f"Offsets:   ns={best.get('ns_offset', 0):+.3f}, mc={best.get('mc_offset', 0):+.3f}")

    if best["f1"] > baseline["f1"]:
        delta = best["f1"] - baseline["f1"]
        print(f"IMPROVEMENT: +{delta:.3f} F1")
        best_thresholds = make_thresholds(
            ns_offset=best.get("ns_offset", 0),
            mc_offset=best.get("mc_offset", 0))
        print(f"\nOptimal thresholds:")
        for k, v in sorted(best_thresholds.items()):
            default = DEFAULT_THRESHOLDS[k]
            delta_v = v - default
            print(f"  {k:<28s} = {v:.4f}  (default {default:.4f}, {delta_v:+.4f})")
        print(f"\nTo apply, add to meta.json:")
        print(f'  "calibrated_thresholds": {json.dumps(best_thresholds, indent=4)}')
    else:
        print("No improvement found — default thresholds are optimal for this model.")

    # Restore original meta
    with open(META_PATH, "w") as f:
        json.dump(original_meta, f, indent=2)
    print(f"\nRestored original meta.json")


if __name__ == "__main__":
    main()
