#!/usr/bin/env python3
"""Train a window-based shot detector model.

Uses leave-one-video-out cross-validation to train a Random Forest
on temporal window features from extract_window_features.py.

Usage:
    .venv/bin/python scripts/train_window_model.py [--features training/window_features.npz]
"""

import argparse
import json
import os
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
TRAINING_DIR = os.path.join(PROJECT_ROOT, "training")


def load_features(path):
    """Load feature matrix from npz file."""
    d = np.load(path, allow_pickle=True)
    return {
        "X": d["X"],
        "labels": d["labels"],
        "videos": d["videos"],
        "timestamps": d["timestamps"],
        "shot_types": d["shot_types"],
        "feature_names": d["feature_names"],
    }


def train_and_evaluate(X, labels, videos, feature_names, model_type="rf"):
    """Leave-one-video-out cross-validation.

    Returns: predictions, probabilities, per-video metrics, overall metrics
    """
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.metrics import classification_report, confusion_matrix

    unique_videos = sorted(set(videos))
    print(f"\nLOOCV across {len(unique_videos)} videos")
    print(f"  Features: {X.shape[1]}, Samples: {X.shape[0]}")

    all_preds = np.empty(len(labels), dtype=object)
    all_probs = np.zeros(len(labels))
    per_video = {}

    for vid in unique_videos:
        test_mask = videos == vid
        train_mask = ~test_mask

        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = labels[train_mask], labels[test_mask]

        if model_type == "rf":
            model = RandomForestClassifier(
                n_estimators=200, max_depth=15, min_samples_leaf=3,
                class_weight="balanced", random_state=42, n_jobs=-1
            )
        elif model_type == "gb":
            model = GradientBoostingClassifier(
                n_estimators=200, max_depth=5, min_samples_leaf=5,
                learning_rate=0.1, random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)

        # Get shot probability (probability of 'shot' class)
        shot_idx = list(model.classes_).index("shot")
        shot_probs = probs[:, shot_idx]

        all_preds[test_mask] = preds
        all_probs[test_mask] = shot_probs

        # Per-video metrics
        n_test = len(y_test)
        n_shot = sum(y_test == "shot")
        n_ns = sum(y_test == "not_shot")
        correct = sum(preds == y_test)
        shot_correct = sum((preds == "shot") & (y_test == "shot"))
        ns_correct = sum((preds == "not_shot") & (y_test == "not_shot"))

        per_video[vid] = {
            "total": n_test,
            "correct": correct,
            "accuracy": correct / n_test if n_test > 0 else 0,
            "shot_total": n_shot,
            "shot_correct": shot_correct,
            "shot_recall": shot_correct / n_shot if n_shot > 0 else 0,
            "not_shot_total": n_ns,
            "not_shot_correct": ns_correct,
            "not_shot_recall": ns_correct / n_ns if n_ns > 0 else 0,
        }

        print(f"  {vid}: {correct}/{n_test} correct ({correct/n_test:.1%}), "
              f"shot_recall={shot_correct}/{n_shot} ({shot_correct/n_shot:.0%}), "
              f"not_shot_recall={ns_correct}/{n_ns} ({ns_correct/n_ns:.0%})")

    # Overall metrics
    overall_acc = sum(all_preds == labels) / len(labels)
    print(f"\nOverall LOOCV accuracy: {overall_acc:.1%}")
    print(f"\nClassification report:")
    print(classification_report(labels, all_preds, digits=3))
    print(f"Confusion matrix:")
    cm = confusion_matrix(labels, all_preds)
    print(cm)

    return all_preds, all_probs, per_video, overall_acc


def train_final_model(X, labels, feature_names, model_type="rf"):
    """Train final model on all data."""
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

    if model_type == "rf":
        model = RandomForestClassifier(
            n_estimators=300, max_depth=15, min_samples_leaf=3,
            class_weight="balanced", random_state=42, n_jobs=-1
        )
    elif model_type == "gb":
        model = GradientBoostingClassifier(
            n_estimators=300, max_depth=5, min_samples_leaf=5,
            learning_rate=0.1, random_state=42
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.fit(X, labels)

    # Feature importances
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    print(f"\nTop 20 features by importance:")
    for i in range(min(20, len(feature_names))):
        idx = sorted_idx[i]
        print(f"  {i+1:2d}. {feature_names[idx]:40s} {importances[idx]:.4f}")

    return model


def find_optimal_threshold(all_probs, labels):
    """Find the threshold that maximizes F1 on shot detection."""
    best_f1 = 0
    best_thresh = 0.5

    for thresh in np.arange(0.1, 0.9, 0.01):
        preds = np.where(all_probs >= thresh, "shot", "not_shot")
        tp = sum((preds == "shot") & (labels == "shot"))
        fp = sum((preds == "shot") & (labels == "not_shot"))
        fn = sum((preds == "not_shot") & (labels == "shot"))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    print(f"\nOptimal threshold: {best_thresh:.2f} (F1={best_f1:.3f})")

    # Show metrics at a few thresholds
    print(f"\nThreshold sweep:")
    print(f"  {'Thresh':>6s}  {'Prec':>6s}  {'Recall':>6s}  {'F1':>6s}  {'TP':>4s}  {'FP':>4s}  {'FN':>4s}")
    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7, best_thresh]:
        preds = np.where(all_probs >= thresh, "shot", "not_shot")
        tp = sum((preds == "shot") & (labels == "shot"))
        fp = sum((preds == "shot") & (labels == "not_shot"))
        fn = sum((preds == "not_shot") & (labels == "shot"))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        marker = " *" if abs(thresh - best_thresh) < 0.005 else ""
        print(f"  {thresh:6.2f}  {precision:6.3f}  {recall:6.3f}  {f1:6.3f}  {tp:4d}  {fp:4d}  {fn:4d}{marker}")

    return best_thresh


def main():
    parser = argparse.ArgumentParser(description="Train window-based shot detector")
    parser.add_argument("--features", default=os.path.join(TRAINING_DIR, "window_features.npz"),
                        help="Path to feature file")
    parser.add_argument("--model-type", choices=["rf", "gb"], default="rf",
                        help="Model type (default: rf)")
    args = parser.parse_args()

    print(f"Loading features from {args.features}...")
    data = load_features(args.features)
    X = data["X"]
    labels = data["labels"]
    videos = data["videos"]
    feature_names = list(data["feature_names"])

    print(f"Loaded {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  shot: {sum(labels == 'shot')}, not_shot: {sum(labels == 'not_shot')}")

    # Handle NaN/Inf
    nan_mask = ~np.isfinite(X)
    if nan_mask.any():
        print(f"  Warning: {nan_mask.sum()} NaN/Inf values, replacing with 0")
        X[nan_mask] = 0

    # LOOCV
    preds, probs, per_video, acc = train_and_evaluate(
        X, labels, videos, feature_names, args.model_type
    )

    # Optimal threshold
    best_thresh = find_optimal_threshold(probs, labels)

    # Train final model
    print(f"\nTraining final model on all data...")
    model = train_final_model(X, labels, feature_names, args.model_type)

    # Save model
    import pickle
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "window_detector.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"\nModel saved to {model_path}")

    # Save metadata
    meta = {
        "model_type": args.model_type,
        "n_features": X.shape[1],
        "n_samples": X.shape[0],
        "n_shot": int(sum(labels == "shot")),
        "n_not_shot": int(sum(labels == "not_shot")),
        "feature_names": feature_names,
        "loocv_accuracy": float(acc),
        "optimal_threshold": float(best_thresh),
        "per_video": {k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else int(vv)
                          for kk, vv in v.items()}
                      for k, v in per_video.items()},
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
    }
    meta_path = os.path.join(MODELS_DIR, "window_detector_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved to {meta_path}")


if __name__ == "__main__":
    main()
