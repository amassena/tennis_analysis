#!/usr/bin/env python3
"""Meta-classifier ensemble: learns when to trust window detector over baseline.

Instead of simple confidence thresholding (which maxes out at F1=93.8%),
trains an RF to decide which disagreements between baseline and window
detector to trust, using features from both systems.
"""

import json
import os
import sys
import numpy as np
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

GT_VIDEOS = [
    "IMG_0864", "IMG_0865", "IMG_0866", "IMG_0867", "IMG_0868",
    "IMG_0869", "IMG_0870", "IMG_0929", "IMG_6665", "IMG_6703",
    "IMG_6711", "IMG_6713",
]

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

MATCH_TOLERANCE = 1.5  # seconds


def load_detections(path):
    """Load detections from JSON file."""
    with open(path) as f:
        d = json.load(f)
    key = 'shots' if 'shots' in d else 'detections'
    shots = d.get(key, [])
    # Filter out practice/offscreen
    return [s for s in shots
            if s.get('shot_type') not in ('practice', 'offscreen', 'unknown_shot')]


def get_time(det):
    """Get timestamp from a detection."""
    return det.get('time', det.get('timestamp', 0))


def match_detections(dets_a, dets_b, tolerance=MATCH_TOLERANCE):
    """Match detections between two lists by timestamp proximity.
    Returns: list of (idx_a or None, idx_b or None) pairs.
    """
    times_a = [(i, get_time(d)) for i, d in enumerate(dets_a)]
    times_b = [(i, get_time(d)) for i, d in enumerate(dets_b)]
    times_a.sort(key=lambda x: x[1])
    times_b.sort(key=lambda x: x[1])

    matched_a = set()
    matched_b = set()
    pairs = []

    # Greedy matching: closest pairs first
    candidates = []
    for ia, ta in times_a:
        for ib, tb in times_b:
            dist = abs(ta - tb)
            if dist <= tolerance:
                candidates.append((dist, ia, ib))
    candidates.sort()

    for dist, ia, ib in candidates:
        if ia not in matched_a and ib not in matched_b:
            pairs.append((ia, ib))
            matched_a.add(ia)
            matched_b.add(ib)

    # Unmatched from A
    for ia, _ in times_a:
        if ia not in matched_a:
            pairs.append((ia, None))

    # Unmatched from B
    for ib, _ in times_b:
        if ib not in matched_b:
            pairs.append((None, ib))

    return pairs


def is_true_positive(det_time, gt_dets, tolerance=MATCH_TOLERANCE):
    """Check if a detection matches any GT shot."""
    for gt in gt_dets:
        if abs(det_time - get_time(gt)) <= tolerance:
            return True
    return False


def build_training_data(videos):
    """Build training data for the meta-classifier.

    For each video:
    1. Load baseline detections, window detections, and GT
    2. Find window-only detections (not in baseline)
    3. For each window-only detection, create features and label (TP or FP)
    """
    features_list = []
    labels = []
    video_ids = []

    for video in videos:
        gt_path = os.path.join(PROJECT_ROOT, GT_FILES[video])
        baseline_path = os.path.join(PROJECT_ROOT, "detections",
                                      f"{video}_fused_detections.json")
        window_path = os.path.join(PROJECT_ROOT, "detections",
                                    f"{video}_window_detections.json")

        if not all(os.path.exists(p) for p in [gt_path, baseline_path, window_path]):
            print(f"  Skipping {video} (missing files)")
            continue

        gt_dets = load_detections(gt_path)
        baseline_dets = load_detections(baseline_path)
        window_dets = load_detections(window_path)

        if not window_dets:
            continue

        # Find window-only detections
        baseline_times = [get_time(d) for d in baseline_dets]

        for wd in window_dets:
            wt = get_time(wd)

            # Check if this detection is already covered by baseline
            min_dist = min((abs(wt - bt) for bt in baseline_times), default=999)
            is_baseline_covered = min_dist <= MATCH_TOLERANCE

            # We only care about window-only detections (not already in baseline)
            # AND baseline detections (to learn what baseline already covers well)
            is_tp = is_true_positive(wt, gt_dets)

            # Extract features
            feats = extract_meta_features(wd, baseline_dets, window_dets, wt)
            features_list.append(feats)

            # Label: 1 = should include, 0 = should reject
            if is_baseline_covered:
                # Baseline already has it — label based on whether baseline is correct
                labels.append(1 if is_tp else 0)
            else:
                # Window-only: should we add it?
                labels.append(1 if is_tp else 0)

            video_ids.append(video)

    return np.array(features_list), np.array(labels), video_ids


def extract_meta_features(window_det, baseline_dets, window_dets, timestamp):
    """Extract features for the meta-classifier decision.

    Features capture:
    - Window detector's confidence
    - Whether baseline has a nearby detection
    - Distance to nearest baseline detection
    - Local density of baseline detections (rally context)
    - Window detector's local density
    - Relative confidence ranking
    """
    wt = timestamp
    w_conf = window_det.get('confidence', window_det.get('shot_prob', 0))

    # Distance to nearest baseline detection
    baseline_times = [get_time(d) for d in baseline_dets]
    if baseline_times:
        baseline_dists = [abs(wt - bt) for bt in baseline_times]
        min_baseline_dist = min(baseline_dists)
        # Number of baseline dets within 5s, 10s, 30s
        baseline_within_5s = sum(1 for d in baseline_dists if d <= 5)
        baseline_within_10s = sum(1 for d in baseline_dists if d <= 10)
        baseline_within_30s = sum(1 for d in baseline_dists if d <= 30)
    else:
        min_baseline_dist = 999
        baseline_within_5s = 0
        baseline_within_10s = 0
        baseline_within_30s = 0

    # Nearest baseline detection's features
    nearest_baseline = None
    if baseline_dets:
        nearest_idx = min(range(len(baseline_dets)),
                         key=lambda i: abs(get_time(baseline_dets[i]) - wt))
        nearest_baseline = baseline_dets[nearest_idx]

    nearest_baseline_conf = nearest_baseline.get('confidence', 0) if nearest_baseline else 0
    nearest_baseline_ml = nearest_baseline.get('ml_confidence', 0) if nearest_baseline else 0
    nearest_baseline_ns = nearest_baseline.get('not_shot_prob', 0.5) if nearest_baseline else 0.5

    # Window detector local context
    window_times = [get_time(d) for d in window_dets]
    window_dists = [abs(wt - t) for t in window_times if abs(wt - t) > 0.1]
    window_within_5s = sum(1 for d in window_dists if d <= 5)
    window_within_10s = sum(1 for d in window_dists if d <= 10)

    # Nearest window neighbor distance
    min_window_neighbor = min(window_dists) if window_dists else 999

    # Is this in a rally cluster? (3+ baseline dets within 15s)
    in_rally = 1 if baseline_within_10s >= 3 else 0

    # Shot type encoding
    shot_type = window_det.get('shot_type', 'unknown')
    is_forehand = 1 if shot_type == 'forehand' else 0
    is_backhand = 1 if shot_type == 'backhand' else 0
    is_serve = 1 if shot_type == 'serve' else 0

    # Tier encoding
    tier = window_det.get('tier', 'low')
    tier_val = {'high': 3, 'medium': 2, 'low': 1}.get(tier, 0)

    # Has baseline within tolerance?
    has_baseline_match = 1 if min_baseline_dist <= MATCH_TOLERANCE else 0

    # Video-level baseline quality — sparse baseline (e.g., side camera)
    # means window detections should be trusted more
    baseline_total = len(baseline_dets)
    window_total = len(window_dets)
    # Ratio: high = baseline is sparse relative to window (trust window more)
    w_to_b_ratio = window_total / max(baseline_total, 1)

    return [
        w_conf,                    # Window confidence
        min_baseline_dist,         # Distance to nearest baseline
        has_baseline_match,        # Binary: baseline covers this?
        baseline_within_5s,        # Rally density (5s)
        baseline_within_10s,       # Rally density (10s)
        baseline_within_30s,       # Rally density (30s)
        nearest_baseline_conf,     # Nearest baseline confidence
        nearest_baseline_ml,       # Nearest baseline ML confidence
        nearest_baseline_ns,       # Nearest baseline not_shot prob
        window_within_5s,          # Window local density (5s)
        window_within_10s,         # Window local density (10s)
        min_window_neighbor,       # Distance to nearest window neighbor
        in_rally,                  # In rally cluster?
        is_forehand,               # Shot type features
        is_backhand,
        is_serve,
        tier_val,                  # Window tier
        baseline_total,            # Video-level: total baseline detections
        w_to_b_ratio,              # Video-level: window/baseline ratio
    ]


FEATURE_NAMES = [
    'w_conf', 'min_baseline_dist', 'has_baseline_match',
    'baseline_within_5s', 'baseline_within_10s', 'baseline_within_30s',
    'nearest_baseline_conf', 'nearest_baseline_ml', 'nearest_baseline_ns',
    'window_within_5s', 'window_within_10s', 'min_window_neighbor',
    'in_rally', 'is_forehand', 'is_backhand', 'is_serve', 'tier_val',
    'baseline_total', 'w_to_b_ratio',
]


def evaluate_ensemble(videos, meta_model=None, threshold=0.5, per_video=False):
    """Evaluate the meta-ensemble on given videos.

    Strategy: Start with baseline detections. For window-only detections,
    use meta-classifier to decide whether to add them.
    """
    total_tp = total_fp = total_fn = 0
    per_video_results = {}

    for video in videos:
        gt_path = os.path.join(PROJECT_ROOT, GT_FILES[video])
        baseline_path = os.path.join(PROJECT_ROOT, "detections",
                                      f"{video}_fused_detections.json")
        window_path = os.path.join(PROJECT_ROOT, "detections",
                                    f"{video}_window_detections.json")

        if not all(os.path.exists(p) for p in [gt_path, baseline_path, window_path]):
            continue

        gt_dets = load_detections(gt_path)
        baseline_dets = load_detections(baseline_path)
        window_dets = load_detections(window_path)

        # Start with baseline
        ensemble_times = [get_time(d) for d in baseline_dets]
        added_window = []

        # Add window-only detections that pass meta-classifier
        baseline_times = [get_time(d) for d in baseline_dets]

        for wd in window_dets:
            wt = get_time(wd)
            min_dist = min((abs(wt - bt) for bt in baseline_times), default=999)

            if min_dist > MATCH_TOLERANCE:
                # Window-only detection — should we add it?
                if meta_model is not None:
                    feats = extract_meta_features(wd, baseline_dets, window_dets, wt)
                    prob = meta_model.predict_proba(np.array([feats]))[0][1]
                    if prob >= threshold:
                        ensemble_times.append(wt)
                        added_window.append((wt, prob, wd.get('shot_type', '?')))
                else:
                    # No meta-model: add based on raw confidence
                    if wd.get('confidence', 0) >= threshold:
                        ensemble_times.append(wt)
                        added_window.append((wt, wd.get('confidence', 0), wd.get('shot_type', '?')))

        # Score against GT
        gt_times = sorted([get_time(g) for g in gt_dets])
        ensemble_times.sort()

        matched_gt = set()
        matched_det = set()
        for di, dt in enumerate(ensemble_times):
            best_dist = MATCH_TOLERANCE + 1
            best_gi = -1
            for gi, gt_t in enumerate(gt_times):
                if gi in matched_gt:
                    continue
                dist = abs(dt - gt_t)
                if dist <= MATCH_TOLERANCE and dist < best_dist:
                    best_dist = dist
                    best_gi = gi
            if best_gi >= 0:
                matched_gt.add(best_gi)
                matched_det.add(di)

        tp = len(matched_gt)
        fp = len(ensemble_times) - tp
        fn = len(gt_times) - tp
        total_tp += tp
        total_fp += fp
        total_fn += fn

        if per_video:
            per_video_results[video] = {
                'tp': tp, 'fp': fp, 'fn': fn, 'added': added_window,
                'gt': len(gt_times), 'det': len(ensemble_times),
            }

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    if per_video:
        return total_tp, total_fp, total_fn, precision, recall, f1, per_video_results
    return total_tp, total_fp, total_fn, precision, recall, f1


def main():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report

    print("Building meta-classifier training data...")
    X, y, video_ids = build_training_data(GT_VIDEOS)
    print(f"  Total samples: {len(X)} (TP={sum(y)}, FP/reject={len(y)-sum(y)})")
    print()

    # LOOCV by video
    unique_videos = sorted(set(video_ids))
    video_ids = np.array(video_ids)

    print("Leave-one-video-out CV:")
    all_preds = np.zeros(len(y))
    all_probs = np.zeros(len(y))

    for fold_video in unique_videos:
        test_mask = video_ids == fold_video
        train_mask = ~test_mask

        if sum(test_mask) == 0 or sum(train_mask) == 0:
            continue

        clf = RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=3,
            class_weight='balanced', random_state=42)
        clf.fit(X[train_mask], y[train_mask])

        preds = clf.predict(X[test_mask])
        probs = clf.predict_proba(X[test_mask])[:, 1]
        all_preds[test_mask] = preds
        all_probs[test_mask] = probs

        acc = np.mean(preds == y[test_mask])
        n_test = sum(test_mask)
        print(f"  {fold_video:12s}: acc={acc:.2f} ({n_test} samples)")

    print()
    print("LOOCV Classification Report:")
    print(classification_report(y, all_preds, target_names=['reject', 'include']))

    # Train final model on all data
    print("\nTraining final meta-classifier on all data...")
    meta_clf = RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_leaf=3,
        class_weight='balanced', random_state=42)
    meta_clf.fit(X, y)

    print("\nTop feature importances:")
    importances = meta_clf.feature_importances_
    for idx in np.argsort(importances)[::-1][:10]:
        print(f"  {FEATURE_NAMES[idx]:30s} {importances[idx]:.4f}")

    # Evaluate at different thresholds
    print("\n" + "=" * 70)
    print("ENSEMBLE EVALUATION (meta-classifier)")
    print("=" * 70)

    # Baseline-only reference
    tp, fp, fn, p, r, f1 = evaluate_ensemble(GT_VIDEOS, meta_model=None, threshold=99)
    print(f"  Baseline only:    TP={tp} FP={fp} FN={fn} P={p:.1%} R={r:.1%} F1={f1:.1%}")

    best_f1 = 0
    best_thresh = 0.5
    for thresh in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        tp, fp, fn, p, r, f1 = evaluate_ensemble(GT_VIDEOS, meta_model=meta_clf, threshold=thresh)
        marker = " <---" if f1 > 0.94 else ""
        print(f"  Meta thresh={thresh:.1f}: TP={tp} FP={fp} FN={fn} P={p:.1%} R={r:.1%} F1={f1:.1%}{marker}")
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    # Per-video breakdown at best threshold
    tp, fp, fn, p, r, f1, pv = evaluate_ensemble(
        GT_VIDEOS, meta_model=meta_clf, threshold=best_thresh, per_video=True)
    print(f"\n  Best threshold: {best_thresh:.1f} (F1={f1:.1%})")
    print(f"\n  Per-video breakdown at threshold={best_thresh:.1f}:")
    for video in GT_VIDEOS:
        if video not in pv:
            continue
        v = pv[video]
        vf1 = 2*v['tp']/(2*v['tp']+v['fp']+v['fn']) if (2*v['tp']+v['fp']+v['fn']) > 0 else 0
        added = len(v['added'])
        note = f" (+{added} window)" if added > 0 else ""
        print(f"    {video}: TP={v['tp']} FP={v['fp']} FN={v['fn']} F1={vf1:.1%}{note}")
        for wt, prob, stype in v['added']:
            # Check if this addition was a TP
            is_tp = is_true_positive(wt, load_detections(
                os.path.join(PROJECT_ROOT, GT_FILES[video])))
            tp_fp = "TP" if is_tp else "FP"
            print(f"      {tp_fp} t={wt:.1f}s type={stype} prob={prob:.2f}")

    # Raw confidence reference (no meta-model)
    print()
    print("  Raw confidence (no meta-model):")
    for thresh in [0.5, 0.6, 0.7, 0.8, 0.9]:
        tp, fp, fn, p, r, f1 = evaluate_ensemble(GT_VIDEOS, meta_model=None, threshold=thresh)
        print(f"    conf>={thresh:.1f}:     TP={tp} FP={fp} FN={fn} P={p:.1%} R={r:.1%} F1={f1:.1%}")

    # Save model
    import pickle
    model_path = os.path.join(PROJECT_ROOT, "models", "meta_ensemble.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(meta_clf, f)
    print(f"\nMeta-classifier saved to {model_path}")

    meta_path = os.path.join(PROJECT_ROOT, "models", "meta_ensemble_meta.json")
    with open(meta_path, 'w') as f:
        json.dump({
            'feature_names': FEATURE_NAMES,
            'n_samples': len(X),
            'n_positive': int(sum(y)),
            'n_negative': int(len(y) - sum(y)),
            'n_estimators': 200,
            'max_depth': 8,
            'best_threshold': best_thresh,
        }, f, indent=2)


def apply_meta_ensemble(video_name, threshold=None):
    """Apply meta-ensemble to merge window detections into baseline for a single video.

    Loads the trained meta-classifier and window detections, adds window-only
    detections that pass the threshold, and updates the baseline detection file.
    """
    import pickle

    model_path = os.path.join(PROJECT_ROOT, "models", "meta_ensemble.pkl")
    meta_path = os.path.join(PROJECT_ROOT, "models", "meta_ensemble_meta.json")

    if not os.path.exists(model_path):
        print(f"Error: meta-ensemble model not found at {model_path}")
        print("Run meta_ensemble.py first to train the model.")
        return False

    with open(model_path, 'rb') as f:
        meta_clf = pickle.load(f)

    with open(meta_path) as f:
        meta_info = json.load(f)

    if threshold is None:
        threshold = meta_info.get('best_threshold', 0.3)

    baseline_path = os.path.join(PROJECT_ROOT, "detections",
                                  f"{video_name}_fused_detections.json")
    window_path = os.path.join(PROJECT_ROOT, "detections",
                                f"{video_name}_window_detections.json")

    if not os.path.exists(baseline_path):
        print(f"Error: baseline detections not found: {baseline_path}")
        return False
    if not os.path.exists(window_path):
        print(f"  No window detections for {video_name}, skipping ensemble merge")
        return True

    with open(baseline_path) as f:
        baseline_data = json.load(f)

    baseline_dets = baseline_data.get('shots', baseline_data.get('detections', []))
    window_dets = load_detections(window_path)

    if not window_dets:
        return True

    baseline_times = [get_time(d) for d in baseline_dets
                      if d.get('shot_type') not in ('practice', 'offscreen', 'unknown_shot')]
    added = 0

    for wd in window_dets:
        wt = get_time(wd)
        min_dist = min((abs(wt - bt) for bt in baseline_times), default=999)

        if min_dist > MATCH_TOLERANCE:
            feats = extract_meta_features(wd, baseline_dets, window_dets, wt)
            prob = meta_clf.predict_proba(np.array([feats]))[0][1]

            if prob >= threshold:
                # Add to baseline detections (match baseline format)
                new_det = {
                    'timestamp': wt,
                    'shot_type': wd.get('shot_type', 'unknown_shot'),
                    'confidence': float(wd.get('confidence', 0)),
                    'source': 'meta_ensemble',
                    'meta_prob': float(prob),
                    'ml_confidence': float(wd.get('confidence', 0)),
                    'not_shot_prob': 0.0,
                }
                baseline_dets.append(new_det)
                baseline_times.append(wt)
                added += 1

    if added > 0:
        # Sort by time
        key = 'shots' if 'shots' in baseline_data else 'detections'
        baseline_data[key] = sorted(baseline_dets, key=lambda d: get_time(d))
        with open(baseline_path, 'w') as f:
            json.dump(baseline_data, f, indent=2)
        print(f"  Meta-ensemble: added {added} window detections to {video_name}")
    else:
        print(f"  Meta-ensemble: no window detections passed threshold for {video_name}")

    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Meta-classifier ensemble")
    parser.add_argument("--apply", nargs="*", metavar="VIDEO",
                        help="Apply trained meta-ensemble to video(s). "
                             "If no videos specified, applies to all GT videos.")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Meta-classifier threshold (default: best from training)")
    args = parser.parse_args()

    if args.apply is not None:
        videos = args.apply if args.apply else GT_VIDEOS
        for v in videos:
            apply_meta_ensemble(v, threshold=args.threshold)
    else:
        main()
