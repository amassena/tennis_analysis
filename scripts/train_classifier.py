#!/usr/bin/env python3
"""Train shot type classifier from extracted features.

Trains a regularized Random Forest classifier on engineered features extracted by
extract_training_features.py. Uses leave-one-video-out cross-validation.

4-class output: serve, forehand, backhand, not_shot

Usage:
    python scripts/train_classifier.py
    python scripts/train_classifier.py --features training/features_v2.json
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import PROJECT_ROOT, TRAINING_DIR, MODELS_DIR

try:
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.metrics import classification_report, confusion_matrix
except ImportError:
    print("scikit-learn not found. Install it:")
    print("  .venv/bin/pip install scikit-learn numpy")
    sys.exit(1)


def video_level_split(samples, val_fraction=0.2, seed=42):
    """Split samples by video to prevent data leakage.

    Groups samples by video, then assigns whole videos to train or val.
    Tries to get close to val_fraction while keeping class balance.
    """
    rng = np.random.RandomState(seed)

    # Group by video
    by_video = {}
    for s in samples:
        vid = s["video"]
        if vid not in by_video:
            by_video[vid] = []
        by_video[vid].append(s)

    videos = list(by_video.keys())
    rng.shuffle(videos)

    # Greedy allocation: add videos to val until we hit target
    total = len(samples)
    target_val = int(total * val_fraction)

    val_videos = set()
    val_count = 0
    for vid in videos:
        if val_count >= target_val:
            break
        val_videos.add(vid)
        val_count += len(by_video[vid])

    train_samples = [s for s in samples if s["video"] not in val_videos]
    val_samples = [s for s in samples if s["video"] in val_videos]

    return train_samples, val_samples, val_videos


def samples_to_arrays(samples, feature_names):
    """Convert sample list to numpy arrays."""
    X = np.array([[s["features"].get(f, 0.0) for f in feature_names] for s in samples],
                 dtype=np.float32)
    y = np.array([s["label"] for s in samples])
    return X, y


def leave_one_video_out_cv(samples, feature_names, clf_params, seed=42, clf_class=None):
    """Leave-one-video-out cross-validation.

    For each unique video, train on all other videos and evaluate on that one.
    Returns per-fold results, aggregate predictions, and per-sample CV predictions
    (used for threshold calibration).
    """
    if clf_class is None:
        clf_class = RandomForestClassifier
    by_video = {}
    for s in samples:
        vid = s["video"]
        if vid not in by_video:
            by_video[vid] = []
        by_video[vid].append(s)

    videos = sorted(by_video.keys())
    all_y_true = []
    all_y_pred = []
    all_y_proba = []
    fold_results = []
    cv_predictions = []

    print(f"\nLeave-one-video-out CV ({len(videos)} folds):")
    print(f"{'─'*70}")

    for fold_idx, held_out_video in enumerate(videos):
        train_samples = [s for s in samples if s["video"] != held_out_video]
        val_samples = by_video[held_out_video]

        if not train_samples or not val_samples:
            continue

        X_train, y_train = samples_to_arrays(train_samples, feature_names)
        X_val, y_val = samples_to_arrays(val_samples, feature_names)

        # Check if train set has all classes
        unique_train = set(y_train)
        if len(unique_train) < 2:
            print(f"  Fold {fold_idx}: {held_out_video} — skipped (train has only {unique_train})")
            continue

        clf = clf_class(**clf_params, random_state=seed)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_val)
        y_proba = clf.predict_proba(X_val)
        acc = np.mean(y_pred == y_val)

        # Collect per-sample predictions for threshold calibration
        classes = list(clf.classes_)
        ns_idx = classes.index("not_shot") if "not_shot" in classes else None
        for i in range(len(y_val)):
            proba = y_proba[i]
            not_shot_prob = float(proba[ns_idx]) if ns_idx is not None else 0.0
            ml_confidence = float(proba.max())
            cv_predictions.append({
                'true_label': str(y_val[i]),
                'predicted_label': str(y_pred[i]),
                'not_shot_prob': not_shot_prob,
                'ml_confidence': ml_confidence,
                'features': {f: float(X_val[i][j]) for j, f in enumerate(feature_names)},
            })

        # Per-class breakdown for this fold
        val_counts = {}
        correct_counts = {}
        for true, pred in zip(y_val, y_pred):
            val_counts[true] = val_counts.get(true, 0) + 1
            if true == pred:
                correct_counts[true] = correct_counts.get(true, 0) + 1

        detail = ", ".join(
            f"{cls}={correct_counts.get(cls, 0)}/{val_counts[cls]}"
            for cls in sorted(val_counts.keys())
        )
        print(f"  Fold {fold_idx:2d}: {held_out_video:<12s} "
              f"acc={acc:.2f} ({len(val_samples):3d} samples) {detail}")

        all_y_true.extend(y_val)
        all_y_pred.extend(y_pred)
        all_y_proba.extend(y_proba.tolist())

        fold_results.append({
            "video": held_out_video,
            "n_samples": len(val_samples),
            "accuracy": round(float(acc), 4),
        })

    return all_y_true, all_y_pred, all_y_proba, fold_results, cv_predictions


def _percentile(sorted_values, p):
    """Compute percentile (0-100 scale) from a pre-sorted list."""
    if not sorted_values:
        return 0.0
    n = len(sorted_values)
    k = (n - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, n - 1)
    d = k - f
    return sorted_values[f] + d * (sorted_values[c] - sorted_values[f])


def _clamp(val, lo, hi):
    return max(lo, min(hi, val))


def compute_calibrated_thresholds(cv_predictions):
    """Derive pipeline thresholds from LOOCV probability distributions.

    cv_predictions: list of dicts with keys:
        true_label, predicted_label, not_shot_prob, ml_confidence, features
    """
    # Filter to true positives (correct non-not_shot predictions)
    tp = [p for p in cv_predictions if p['true_label'] != 'not_shot'
          and p['predicted_label'] == p['true_label']]

    if len(tp) < 10:
        print(f"  WARNING: Only {len(tp)} true positives — insufficient for calibration")
        return None

    ns_values = sorted([p['not_shot_prob'] for p in tp])
    mc_values = sorted([p['ml_confidence'] for p in tp])

    # Source-specific subsets for floor thresholds
    # audio+heuristic: samples that had audio signal
    audio_heuristic_tp = [p for p in tp
                          if p['features'].get('audio_amplitude', 0) > 0.01]
    # heuristic_only / jerk: samples without audio (motion-only detections)
    heuristic_only_tp = [p for p in tp
                         if p['features'].get('audio_amplitude', 0) <= 0.01]

    ah_mc = sorted([p['ml_confidence'] for p in audio_heuristic_tp]) \
        if audio_heuristic_tp else mc_values
    ho_mc = sorted([p['ml_confidence'] for p in heuristic_only_tp]) \
        if heuristic_only_tp else mc_values

    thresholds = {
        'ns_permissive':     _clamp(_percentile(ns_values, 95), 0.30, 0.50),
        'ns_moderate':       _clamp(_percentile(ns_values, 90), 0.25, 0.45),
        'ns_strict':         _clamp(_percentile(ns_values, 85), 0.20, 0.40),
        'ns_first_pass':     _clamp(_percentile(ns_values, 80), 0.15, 0.38),
        'ns_weak_jerk':      _clamp(_percentile(ns_values, 70), 0.15, 0.35),
        'ns_weak_heuristic': _clamp(_percentile(ns_values, 65), 0.12, 0.32),
        'mc_strong':      _clamp(_percentile(mc_values, 30), 0.40, 0.65),
        'mc_moderate':    _clamp(_percentile(mc_values, 20), 0.30, 0.55),
        'mc_weak':        _clamp(_percentile(mc_values, 10), 0.20, 0.45),
        'mc_floor_audio_heuristic': _clamp(_percentile(ah_mc, 15), 0.30, 0.55),
        'mc_floor_heuristic_only':  _clamp(_percentile(ho_mc, 25), 0.40, 0.65),
        'mc_floor_jerk':            _clamp(_percentile(ho_mc, 30), 0.45, 0.70),
        'mc_low_pass':    _clamp(_percentile(mc_values, 30), 0.40, 0.65),
    }

    # Enforce not_shot hierarchy: permissive > moderate > strict > ...
    ns_keys = ['ns_permissive', 'ns_moderate', 'ns_strict',
               'ns_first_pass', 'ns_weak_jerk', 'ns_weak_heuristic']
    for i in range(1, len(ns_keys)):
        if thresholds[ns_keys[i]] >= thresholds[ns_keys[i - 1]]:
            thresholds[ns_keys[i]] = round(thresholds[ns_keys[i - 1]] - 0.01, 4)

    # Round all values for clean JSON output
    thresholds = {k: round(v, 4) for k, v in thresholds.items()}

    return thresholds


def main():
    parser = argparse.ArgumentParser(
        description="Train shot type classifier from extracted features")
    parser.add_argument("--features", default=os.path.join(TRAINING_DIR, "features_v2.json"),
                        help="Path to features JSON")
    parser.add_argument("--output-dir", default=MODELS_DIR,
                        help="Directory to save model files")
    parser.add_argument("--val-fraction", type=float, default=0.2,
                        help="Validation split fraction for final eval (default: 0.2)")
    parser.add_argument("--n-estimators", type=int, default=300,
                        help="Number of trees (default: 300)")
    parser.add_argument("--max-depth", type=int, default=12,
                        help="Max tree depth (default: 12)")
    parser.add_argument("--min-samples-leaf", type=int, default=5,
                        help="Min samples per leaf (default: 5)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--skip-cv", action="store_true",
                        help="Skip leave-one-video-out CV (just train final model)")
    parser.add_argument("--model-type", choices=["rf", "gb"], default="rf",
                        help="Model type: rf=RandomForest, gb=GradientBoosting")
    args = parser.parse_args()

    if not os.path.exists(args.features):
        print(f"Features file not found: {args.features}")
        print("Run extract_training_features.py first.")
        sys.exit(1)

    with open(args.features) as f:
        data = json.load(f)

    samples = data["samples"]
    feature_names = data["feature_names"]
    print(f"Loaded {len(samples)} samples, {len(feature_names)} features")
    print(f"Labels: {data['label_counts']}")

    if args.model_type == "rf":
        clf_params = {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "min_samples_leaf": args.min_samples_leaf,
            "class_weight": "balanced",
            "n_jobs": -1,
        }
        clf_class = RandomForestClassifier
    else:  # gb
        clf_params = {
            "n_estimators": args.n_estimators,
            "max_depth": min(args.max_depth, 6),  # GB needs shallower trees
            "min_samples_leaf": args.min_samples_leaf,
            "learning_rate": 0.1,
            "subsample": 0.8,
        }
        clf_class = GradientBoostingClassifier

    # ── Leave-one-video-out CV ────────────────────────────────────
    cv_predictions = []
    if not args.skip_cv:
        all_y_true, all_y_pred, all_y_proba, fold_results, cv_predictions = leave_one_video_out_cv(
            samples, feature_names, clf_params, seed=args.seed,
            clf_class=clf_class,
        )

        if all_y_true:
            y_true = np.array(all_y_true)
            y_pred = np.array(all_y_pred)
            labels = sorted(set(y_true) | set(y_pred))

            cv_acc = np.mean(y_true == y_pred)
            print(f"\n{'='*50}")
            print(f"LOOCV AGGREGATE RESULTS")
            print(f"{'='*50}")
            print(f"Overall accuracy: {cv_acc:.3f} ({sum(y_true == y_pred)}/{len(y_true)})")

            print(f"\nClassification Report:")
            print(classification_report(y_true, y_pred, labels=labels, zero_division=0))

            print(f"Confusion Matrix (rows=true, cols=predicted):")
            print(f"{'':>12s}  " + "  ".join(f"{l:>10s}" for l in labels))
            cm = confusion_matrix(y_true, y_pred, labels=labels)
            for i, row in enumerate(cm):
                print(f"{labels[i]:>12s}  " + "  ".join(f"{v:>10d}" for v in row))

            # Per-fold summary
            accs = [f["accuracy"] for f in fold_results]
            print(f"\nPer-fold accuracy: min={min(accs):.2f} max={max(accs):.2f} "
                  f"mean={np.mean(accs):.2f} std={np.std(accs):.2f}")
        else:
            print("  No CV results (insufficient data)")

    # ── Calibrate pipeline thresholds from LOOCV ────────────────────
    calibrated_thresholds = None
    if cv_predictions:
        print(f"\n{'='*50}")
        print(f"THRESHOLD CALIBRATION")
        print(f"{'='*50}")
        calibrated_thresholds = compute_calibrated_thresholds(cv_predictions)
        if calibrated_thresholds:
            tp_count = sum(1 for p in cv_predictions
                          if p['true_label'] != 'not_shot'
                          and p['predicted_label'] == p['true_label'])
            print(f"Calibrated from {tp_count} true positives:")
            # not_shot thresholds
            for key in ['ns_permissive', 'ns_moderate', 'ns_strict',
                        'ns_first_pass', 'ns_weak_jerk', 'ns_weak_heuristic']:
                print(f"  {key:<24s} = {calibrated_thresholds[key]:.4f}")
            # ml_confidence thresholds
            for key in ['mc_strong', 'mc_moderate', 'mc_weak',
                        'mc_floor_audio_heuristic', 'mc_floor_heuristic_only',
                        'mc_floor_jerk', 'mc_low_pass']:
                print(f"  {key:<24s} = {calibrated_thresholds[key]:.4f}")
        else:
            print("  Calibration skipped (insufficient true positives)")

    # ── Train/val split for final evaluation ──────────────────────
    print(f"\n{'='*50}")
    print(f"FINAL MODEL (train on {1-args.val_fraction:.0%}, eval on {args.val_fraction:.0%})")
    print(f"{'='*50}")

    train_samples, val_samples, val_videos = video_level_split(
        samples, val_fraction=args.val_fraction, seed=args.seed
    )
    print(f"Video-level split:")
    print(f"  Train: {len(train_samples)} samples")
    print(f"  Val:   {len(val_samples)} samples (videos: {sorted(val_videos)})")

    # Class distribution per split
    for name, samps in [("Train", train_samples), ("Val", val_samples)]:
        counts = {}
        for s in samps:
            counts[s["label"]] = counts.get(s["label"], 0) + 1
        print(f"  {name} distribution: {counts}")

    X_train, y_train = samples_to_arrays(train_samples, feature_names)
    X_val, y_val = samples_to_arrays(val_samples, feature_names)

    # Train on train split for evaluation
    model_name = "Random Forest" if args.model_type == "rf" else "GradientBoosting"
    print(f"\nTraining {model_name} (n_estimators={clf_params['n_estimators']}, "
          f"max_depth={clf_params['max_depth']}, min_samples_leaf={clf_params['min_samples_leaf']})...")
    clf_eval = clf_class(**clf_params, random_state=args.seed)
    clf_eval.fit(X_train, y_train)

    train_acc = clf_eval.score(X_train, y_train)
    val_acc = clf_eval.score(X_val, y_val)
    print(f"\nTrain accuracy: {train_acc:.3f}")
    print(f"Val accuracy:   {val_acc:.3f}")

    y_pred = clf_eval.predict(X_val)
    labels = sorted(clf_eval.classes_)

    print(f"\nValidation Classification Report:")
    print(classification_report(y_val, y_pred, labels=labels, zero_division=0))

    print(f"Confusion Matrix (rows=true, cols=predicted):")
    print(f"{'':>12s}  " + "  ".join(f"{l:>10s}" for l in labels))
    cm = confusion_matrix(y_val, y_pred, labels=labels)
    for i, row in enumerate(cm):
        print(f"{labels[i]:>12s}  " + "  ".join(f"{v:>10d}" for v in row))

    # ── Train final model on ALL data ─────────────────────────────
    print(f"\nTraining FINAL model on all {len(samples)} samples...")
    X_all, y_all = samples_to_arrays(samples, feature_names)
    clf_final = clf_class(**clf_params, random_state=args.seed)
    clf_final.fit(X_all, y_all)
    final_train_acc = clf_final.score(X_all, y_all)
    print(f"Final train accuracy: {final_train_acc:.3f}")

    # Feature importance
    print(f"\nTop 10 feature importances:")
    importances = clf_final.feature_importances_
    indices = np.argsort(importances)[::-1]
    for rank, idx in enumerate(indices[:10], 1):
        print(f"  {rank:2d}. {feature_names[idx]:<30s} {importances[idx]:.4f}")

    # ── Save model ────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, "shot_classifier.pkl")
    meta_path = os.path.join(args.output_dir, "shot_classifier_meta.json")

    with open(model_path, "wb") as f:
        pickle.dump(clf_final, f)

    meta = {
        "version": 2,
        "feature_names": feature_names,
        "classes": list(clf_final.classes_),
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "min_samples_leaf": args.min_samples_leaf,
        "train_samples": len(samples),
        "val_samples": len(val_samples),
        "val_videos": sorted(val_videos),
        "train_accuracy": round(float(final_train_acc), 4),
        "val_accuracy": round(float(val_acc), 4),
        "seed": args.seed,
        "has_not_shot_class": "not_shot" in list(clf_final.classes_),
        "not_shot_rejection_threshold": 0.4,
    }
    if calibrated_thresholds:
        meta["calibrated_thresholds"] = calibrated_thresholds
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nModel saved to {model_path}")
    print(f"Metadata saved to {meta_path}")
    print(f"Classes: {list(clf_final.classes_)}")


if __name__ == "__main__":
    main()
