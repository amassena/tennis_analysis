#!/usr/bin/env python3
"""Train the 1D-CNN sequence classifier with leave-one-video-out CV.

Supports two data loading modes:
  1. NPZ file (fast): pre-extracted by prepare_sequence_data.py
  2. Live extraction (slow): loads poses + GT on-the-fly

IMPORTANT: Run this on Windows GPU machine (RTX 5080/4080), not on Mac.

Usage:
    python scripts/train_sequence_model.py --npz training/sequence_data.npz
    python scripts/train_sequence_model.py --npz training/sequence_data.npz --skip-cv
    python scripts/train_sequence_model.py  # fallback: live extraction
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import PROJECT_ROOT, POSES_DIR, MODELS_DIR, TRAINING_DIR

try:
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    print("Required packages not found. Install:")
    print("  pip install torch numpy")
    sys.exit(1)

from scripts.sequence_model import (
    ShotClassifierCNN, extract_pose_sequence, augment_sequence,
    CLASSES, CLASS_TO_IDX, SEQUENCE_LENGTH, FEATURES_PER_FRAME,
)

DETECTIONS_DIR = os.path.join(PROJECT_ROOT, "detections")

# Shot types to train on
TRAINABLE_TYPES = {"forehand", "backhand", "serve"}

# Serve-only videos (skip FH/BH labels as contaminated)
SERVE_ONLY_VIDEOS = {
    "IMG_0864", "IMG_0865", "IMG_0866", "IMG_0867",
    "IMG_0868", "IMG_0869", "IMG_0870",
}


class PoseSequenceDataset(Dataset):
    """Dataset of pose sequences with augmentation.

    When `deltas` is provided, returns (seq, label, delta) for the multi-task
    regression head. delta is in frames; should be 0 for negatives.

    Note: when augmentation `jitter` is non-zero AND deltas are provided, the
    augmentation jitter is folded into the delta target — shifting the seq by
    +k frames means the contact moves to position center-k inside the window,
    so the regression label updates as `delta - k`. (NOT used for negatives.)
    """

    def __init__(self, sequences, labels, videos=None, deltas=None,
                 augment=False, jitter=5, dropout_rate=0.15,
                 not_shot_idx=None):
        self.sequences = sequences   # (N, 90, 99) array
        self.labels = labels         # (N,) array
        self.videos = videos         # (N,) array or None
        self.deltas = deltas         # (N,) array or None — frames offset
        self.augment = augment
        self.jitter = jitter
        self.dropout_rate = dropout_rate
        self.not_shot_idx = not_shot_idx

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        seq = self.sequences[idx].copy()
        label = int(self.labels[idx])
        delta = float(self.deltas[idx]) if self.deltas is not None else 0.0

        if self.augment:
            mirror = np.random.random() < 0.5
            temporal_scale = np.random.uniform(0.85, 1.15) if np.random.random() < 0.5 else None
            noise_std = 0.01 if np.random.random() < 0.5 else 0.0
            # Apply jitter ourselves (not via augment_sequence) so we know the
            # shift and can update delta accordingly. augment_sequence keeps
            # other ops (mirror, temporal_scale, dropout, noise).
            shift = 0
            if self.jitter > 0 and (self.not_shot_idx is None or label != self.not_shot_idx):
                shift = int(np.random.randint(-self.jitter, self.jitter + 1))
                if shift != 0:
                    if shift > 0:
                        seq = np.concatenate([np.zeros((shift, seq.shape[1]),
                                                       dtype=seq.dtype), seq[:-shift]], axis=0)
                    else:
                        s = -shift
                        seq = np.concatenate([seq[s:], np.zeros((s, seq.shape[1]),
                                                                dtype=seq.dtype)], axis=0)
                    delta = delta - shift  # contact moved -shift frames in window
            seq = augment_sequence(
                seq,
                mirror=mirror,
                jitter_frames=0,  # we did jitter ourselves
                dropout_rate=np.random.uniform(0.1, self.dropout_rate),
                temporal_scale=temporal_scale,
                noise_std=noise_std,
            )

        if self.deltas is not None:
            return (torch.from_numpy(seq).float(), label, torch.tensor(delta, dtype=torch.float32))
        return torch.from_numpy(seq).float(), label


def load_from_npz(npz_path):
    """Load pre-extracted data from NPZ file.

    Returns (X, y, videos, deltas). deltas is None if absent (older NPZ).
    """
    data = np.load(npz_path, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    videos = data["videos"]
    deltas = data["delta"] if "delta" in data.files else None
    print(f"Loaded {len(X)} samples from {npz_path}")
    print(f"  Shape: X={X.shape}, y={y.shape}, "
          f"delta={'present' if deltas is not None else 'absent'}")
    return X, y, videos, deltas


def load_all_samples_live(poses_dir, det_dir):
    """Load all labeled samples with their pose sequences (legacy mode).

    Returns (X, y, videos) arrays.
    """
    from scripts.extract_training_features import load_pose_frames

    all_X = []
    all_y = []
    all_videos = []
    pose_cache = {}

    det_files = sorted(
        f for f in os.listdir(det_dir)
        if f.endswith("_fused.json")
        and "_v2" not in f and "_ml" not in f
        and "_pre" not in f and "_baseline" not in f
    )

    for det_file in det_files:
        video_name = det_file.replace("_fused.json", "")
        v5_path = os.path.join(det_dir, f"{video_name}_fused_v5.json")
        det_path = v5_path if os.path.exists(v5_path) else os.path.join(det_dir, det_file)

        with open(det_path) as f:
            det_data = json.load(f)

        is_serve_only = video_name in SERVE_ONLY_VIDEOS

        pose_path = os.path.join(poses_dir, f"{video_name}.json")
        if not os.path.exists(pose_path):
            continue

        if video_name not in pose_cache:
            frames, fps, total_frames = load_pose_frames(pose_path)
            pose_cache[video_name] = frames
        frames = pose_cache[video_name]

        positives = 0
        for det in det_data.get("detections", []):
            shot_type = det.get("shot_type", "")
            if shot_type not in TRAINABLE_TYPES:
                continue
            if is_serve_only and shot_type in ("forehand", "backhand"):
                continue

            frame_idx = det.get("frame", 0)
            seq = extract_pose_sequence(frames, frame_idx)
            if seq is None:
                continue

            label_idx = CLASS_TO_IDX.get(shot_type)
            if label_idx is None:
                continue

            all_X.append(seq)
            all_y.append(label_idx)
            all_videos.append(video_name)
            positives += 1

        # Negatives
        det_times = [d.get("timestamp", 0) for d in det_data.get("detections", [])]
        fps_val = det_data.get("fps", 60.0)
        n_neg = max(5, positives // 3)
        neg_count = 0

        for frame_idx in range(90, len(frames) - 90, int(fps_val * 3)):
            if neg_count >= n_neg:
                break
            t = frame_idx / fps_val
            if any(abs(t - dt) < 3.0 for dt in det_times):
                continue
            seq = extract_pose_sequence(frames, frame_idx)
            if seq is not None and np.sum(np.abs(seq)) > 1e-6:
                all_X.append(seq)
                all_y.append(CLASS_TO_IDX["not_shot"])
                all_videos.append(video_name)
                neg_count += 1

        print(f"  {video_name}: {positives} shots + {neg_count} negatives")

    return np.array(all_X), np.array(all_y), np.array(all_videos)


def _unpack(batch, device):
    """Move batch to device. Returns (x, y, delta_or_None)."""
    if len(batch) == 3:
        x, y, d = batch
        return x.to(device), y.to(device), d.to(device)
    x, y = batch
    return x.to(device), y.to(device), None


def train_epoch(model, loader, optimizer, criterion, device,
                lambda_regr=0.0, not_shot_idx=None, regr_loss_fn=None):
    """Train for one epoch. Returns (avg_loss, accuracy, regr_mae_frames or None)."""
    model.train()
    total_loss = 0.0
    total_regr_loss = 0.0
    n_regr_samples = 0
    correct = 0
    total = 0
    use_regr = lambda_regr > 0 and getattr(model, "with_regression", False)
    if use_regr and regr_loss_fn is None:
        regr_loss_fn = nn.HuberLoss(delta=1.0, reduction="none")

    for batch in loader:
        batch_x, batch_y, batch_delta = _unpack(batch, device)
        optimizer.zero_grad()
        out = model(batch_x)
        if isinstance(out, tuple):
            logits, delta_pred = out
        else:
            logits, delta_pred = out, None

        cls_loss = criterion(logits, batch_y)
        loss = cls_loss

        if use_regr and delta_pred is not None and batch_delta is not None:
            # Mask: only positives contribute to regression loss
            if not_shot_idx is None:
                mask = torch.ones_like(batch_y, dtype=torch.bool)
            else:
                mask = batch_y != not_shot_idx
            if mask.any():
                reg = regr_loss_fn(delta_pred[mask], batch_delta[mask]).mean()
                loss = loss + lambda_regr * reg
                total_regr_loss += reg.item() * mask.sum().item()
                n_regr_samples += int(mask.sum().item())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)
        _, predicted = logits.max(1)
        correct += predicted.eq(batch_y).sum().item()
        total += batch_x.size(0)

    avg_loss = total_loss / total
    acc = correct / total
    regr_mae = (total_regr_loss / n_regr_samples) if n_regr_samples > 0 else None
    return avg_loss, acc, regr_mae


def evaluate(model, loader, device, not_shot_idx=None):
    """Evaluate model. Returns (loss, accuracy, all_preds, all_labels, regr_mae_or_None)."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    abs_regr_err_total = 0.0
    n_regr_samples = 0
    all_preds = []
    all_labels = []

    use_regr = getattr(model, "with_regression", False)

    with torch.no_grad():
        for batch in loader:
            batch_x, batch_y, batch_delta = _unpack(batch, device)
            out = model(batch_x)
            if isinstance(out, tuple):
                logits, delta_pred = out
            else:
                logits, delta_pred = out, None

            loss = criterion(logits, batch_y)
            total_loss += loss.item() * batch_x.size(0)
            _, predicted = logits.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

            if use_regr and delta_pred is not None and batch_delta is not None:
                if not_shot_idx is None:
                    mask = torch.ones_like(batch_y, dtype=torch.bool)
                else:
                    mask = batch_y != not_shot_idx
                if mask.any():
                    abs_err = (delta_pred[mask] - batch_delta[mask]).abs()
                    abs_regr_err_total += abs_err.sum().item()
                    n_regr_samples += int(mask.sum().item())

    total = len(all_labels)
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / total if total > 0 else 0
    avg_loss = total_loss / total if total > 0 else 0
    regr_mae = (abs_regr_err_total / n_regr_samples) if n_regr_samples > 0 else None

    return avg_loss, acc, np.array(all_preds), np.array(all_labels), regr_mae


def make_criterion(train_labels, device, label_smoothing=0.05):
    """Create weighted CrossEntropyLoss with label smoothing."""
    label_counts = defaultdict(int)
    for l in train_labels:
        label_counts[int(l)] += 1
    total = sum(label_counts.values())
    weights = torch.tensor(
        [total / (len(CLASSES) * label_counts.get(i, 1)) for i in range(len(CLASSES))],
        dtype=torch.float32
    ).to(device)
    return nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)


def train_with_early_stopping(model, train_loader, val_loader, optimizer, scheduler,
                              criterion, device, epochs, patience=7,
                              lambda_regr=0.0, not_shot_idx=None):
    """Train with early stopping on validation loss.

    Returns best validation accuracy.
    """
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        train_epoch(model, train_loader, optimizer, criterion, device,
                    lambda_regr=lambda_regr, not_shot_idx=not_shot_idx)
        scheduler.step()

        if val_loader is not None:
            val_loss, val_acc, _, _, _ = evaluate(model, val_loader, device,
                                                   not_shot_idx=not_shot_idx)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    return best_val_loss


def leave_one_video_out_cv(X, y, videos, args, device, deltas=None):
    """Leave-one-video-out cross-validation."""
    unique_videos = sorted(set(videos))
    all_preds = []
    all_labels = []
    fold_results = []
    fold_regr_maes = []

    not_shot_idx = CLASS_TO_IDX["not_shot"]
    use_regr = args.lambda_regr > 0 and deltas is not None

    print(f"\nLeave-one-video-out CV ({len(unique_videos)} folds, "
          f"lambda_regr={args.lambda_regr}, regression={'on' if use_regr else 'off'}):")
    print(f"{'-'*70}")

    for fold_idx, held_out in enumerate(unique_videos):
        train_mask = videos != held_out
        val_mask = videos == held_out

        train_X, train_y = X[train_mask], y[train_mask]
        val_X, val_y = X[val_mask], y[val_mask]
        train_d = deltas[train_mask] if deltas is not None else None
        val_d = deltas[val_mask] if deltas is not None else None

        if len(train_X) == 0 or len(val_X) == 0:
            continue

        train_dataset = PoseSequenceDataset(
            train_X, train_y, deltas=train_d, augment=True,
            not_shot_idx=not_shot_idx)
        val_dataset = PoseSequenceDataset(
            val_X, val_y, deltas=val_d, augment=False,
            not_shot_idx=not_shot_idx)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                                shuffle=False, num_workers=0)

        model = ShotClassifierCNN(num_classes=len(CLASSES),
                                   with_regression=use_regr).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        criterion = make_criterion(train_y, device, label_smoothing=0.05)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        # Train with early stopping
        train_with_early_stopping(
            model, train_loader, val_loader, optimizer, scheduler,
            criterion, device, args.epochs, patience=args.patience,
            lambda_regr=args.lambda_regr, not_shot_idx=not_shot_idx,
        )

        # Final evaluation
        val_loss, val_acc, preds, labels, val_mae = evaluate(
            model, val_loader, device, not_shot_idx=not_shot_idx)

        # Per-class breakdown
        val_counts = defaultdict(int)
        correct_counts = defaultdict(int)
        for p, l in zip(preds, labels):
            val_counts[l] += 1
            if p == l:
                correct_counts[l] += 1

        detail = ", ".join(
            f"{CLASSES[c]}={correct_counts.get(c, 0)}/{val_counts[c]}"
            for c in sorted(val_counts.keys())
        )
        mae_str = f" mae={val_mae:.2f}f" if val_mae is not None else ""

        print(f"  Fold {fold_idx:2d}: {held_out:<12s} "
              f"acc={val_acc:.2f} ({len(val_X):3d} samples){mae_str} {detail}")

        all_preds.extend(preds)
        all_labels.extend(labels)
        fold_record = {
            "video": held_out,
            "n_samples": int(len(val_X)),
            "accuracy": round(float(val_acc), 4),
        }
        if val_mae is not None:
            fold_record["regr_mae_frames"] = round(float(val_mae), 3)
            fold_regr_maes.append(val_mae)
        fold_results.append(fold_record)

    # Aggregate
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    overall_acc = np.mean(all_preds == all_labels)

    print(f"\n{'='*50}")
    print(f"LOOCV AGGREGATE: accuracy={overall_acc:.3f} ({sum(all_preds == all_labels)}/{len(all_labels)})")
    if fold_regr_maes:
        mean_mae = float(np.mean(fold_regr_maes))
        print(f"LOOCV REGR MAE:  {mean_mae:.2f} frames "
              f"(~{mean_mae * 1000 / 60:.0f}ms @60fps) — "
              f"target: <2 frames (~33ms) for event F1 ≥ 0.95")
    print(f"{'='*50}")

    # Confusion matrix
    print(f"\nConfusion Matrix:")
    print(f"{'':>12s}  " + "  ".join(f"{c:>10s}" for c in CLASSES))
    for i, true_class in enumerate(CLASSES):
        row = []
        for j in range(len(CLASSES)):
            count = np.sum((all_labels == i) & (all_preds == j))
            row.append(count)
        print(f"{true_class:>12s}  " + "  ".join(f"{v:>10d}" for v in row))

    # Per-class F1
    print(f"\nPer-class metrics:")
    for i, cls in enumerate(CLASSES):
        tp = np.sum((all_preds == i) & (all_labels == i))
        fp = np.sum((all_preds == i) & (all_labels != i))
        fn = np.sum((all_preds != i) & (all_labels == i))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        print(f"  {cls:>12s}: P={prec:.3f} R={rec:.3f} F1={f1:.3f}")

    return overall_acc, fold_results


def main():
    parser = argparse.ArgumentParser(
        description="Train 1D-CNN sequence classifier")
    parser.add_argument("--npz", default=None,
                        help="Path to pre-extracted NPZ file (from prepare_sequence_data.py)")
    parser.add_argument("--poses-dir", default=POSES_DIR)
    parser.add_argument("--det-dir", default=DETECTIONS_DIR)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=7,
                        help="Early stopping patience (epochs)")
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--skip-cv", action="store_true")
    parser.add_argument("--output", default=os.path.join(MODELS_DIR, "sequence_detector.pt"))
    parser.add_argument("--no-sidecar", action="store_true",
                        help="Skip writing <output>.sidecar.json (default: write)")
    parser.add_argument("--holdout-manifest",
                        default=os.path.join(PROJECT_ROOT, "eval", "holdout", "manifest.json"),
                        help="Path to holdout manifest. If any holdout video appears "
                             "in training data, training is refused. Pass empty string to skip.")
    parser.add_argument("--lambda-regr", type=float, default=0.0,
                        help="Weight on contact-time regression Huber loss "
                             "(default: 0 = classification-only, original behavior). "
                             "Requires NPZ with 'delta' field (jittered training "
                             "samples). Try 0.2 (light) to 0.5 (balanced).")
    args = parser.parse_args()

    # Select device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name()}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Load samples
    t0 = time.time()
    deltas = None
    if args.npz:
        npz_path = args.npz
        if not os.path.isabs(npz_path):
            npz_path = os.path.join(PROJECT_ROOT, npz_path)
        X, y, videos, deltas = load_from_npz(npz_path)
    else:
        print(f"\nLoading samples from {args.poses_dir} (live extraction)...")
        X, y, videos = load_all_samples_live(args.poses_dir, args.det_dir)
        # Live extraction predates the regression head; deltas remain None.
    print(f"Loaded in {time.time() - t0:.1f}s")

    if args.lambda_regr > 0 and deltas is None:
        print("WARNING: --lambda-regr > 0 but NPZ has no 'delta' field. "
              "Re-run prepare_sequence_data.py with --jitter-positives N. "
              "Disabling regression loss.")
        args.lambda_regr = 0.0

    # Holdout-leak assertion. Refuses to train if any video in the
    # eval holdout appears in the training set. Without this gate the
    # 2026-05-02 incident style of "trained-and-evaluated on the same
    # data, deployed broken" is reachable again.
    if args.holdout_manifest:
        if os.path.exists(args.holdout_manifest):
            with open(args.holdout_manifest) as f:
                manifest = json.load(f)
            holdout_ids = {v["video_id"] for v in manifest.get("videos", [])}
            train_ids = set(videos) if videos is not None else set()
            leaked = holdout_ids & train_ids
            if leaked:
                sys.exit(
                    f"FATAL: holdout videos appear in training set: {sorted(leaked)}\n"
                    f"  manifest: {args.holdout_manifest}\n"
                    f"  re-run prepare_sequence_data.py excluding these video_ids,\n"
                    f"  or pass --holdout-manifest '' to override (NOT recommended)."
                )
            print(f"Holdout-leak check passed ({len(holdout_ids)} holdout ids, "
                  f"none in training set)")
        else:
            print(f"WARNING: holdout manifest not found at {args.holdout_manifest} — "
                  f"leak check skipped.")

    use_regr = args.lambda_regr > 0 and deltas is not None
    not_shot_idx = CLASS_TO_IDX["not_shot"]

    # Distribution
    label_counts = defaultdict(int)
    for label in y:
        label_counts[int(label)] += 1
    print(f"\nTotal: {len(X)} samples")
    for idx in sorted(label_counts.keys()):
        print(f"  {CLASSES[idx]}: {label_counts[idx]}")

    # LOOCV
    if not args.skip_cv:
        cv_acc, fold_results = leave_one_video_out_cv(X, y, videos, args, device, deltas=deltas)

        # Save CV results
        cv_path = args.output.replace(".pt", "_cv.json")
        with open(cv_path, "w") as f:
            json.dump({
                "overall_accuracy": round(float(cv_acc), 4),
                "lambda_regr": args.lambda_regr,
                "with_regression": use_regr,
                "folds": fold_results,
            }, f, indent=2)
        print(f"CV results saved to {cv_path}")

    # Train final model on all data
    print(f"\n{'='*50}")
    print(f"TRAINING FINAL MODEL on all {len(X)} samples "
          f"(with_regression={use_regr})")
    print(f"{'='*50}")

    dataset = PoseSequenceDataset(X, y, deltas=deltas, augment=True,
                                  not_shot_idx=not_shot_idx)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = ShotClassifierCNN(num_classes=len(CLASSES),
                               with_regression=use_regr).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = make_criterion(y, device, label_smoothing=args.label_smoothing)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    for epoch in range(args.epochs):
        train_loss, train_acc, regr_mae = train_epoch(
            model, loader, optimizer, criterion, device,
            lambda_regr=args.lambda_regr, not_shot_idx=not_shot_idx,
        )
        scheduler.step()
        if (epoch + 1) % 5 == 0 or epoch == 0:
            mae_str = f" mae={regr_mae:.2f}f" if regr_mae is not None else ""
            print(f"  Epoch {epoch+1:3d}/{args.epochs}: loss={train_loss:.4f} acc={train_acc:.3f}{mae_str}")

    # Save model
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(model.state_dict(), args.output)

    # Save metadata
    meta_path = args.output.replace(".pt", "_meta.json")
    meta = {
        "classes": CLASSES,
        "num_samples": len(X),
        "label_counts": {CLASSES[k]: v for k, v in label_counts.items()},
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "patience": args.patience,
        "label_smoothing": args.label_smoothing,
        "sequence_length": SEQUENCE_LENGTH,
        "features_per_frame": FEATURES_PER_FRAME,
        "normalization": "hip_center + shoulder_width",
        "lambda_regr": args.lambda_regr,
        "with_regression": use_regr,
    }
    if not args.skip_cv:
        meta["loocv_accuracy"] = round(float(cv_acc), 4)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nModel saved to {args.output}")
    print(f"Metadata saved to {meta_path}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Sidecar — self-describing model identity. Always on unless --no-sidecar.
    if not args.no_sidecar:
        write_sidecar(args, meta, X, videos)


def write_sidecar(args, meta, X, videos):
    """Write <output>.sidecar.json — the canonical model identity record."""
    import hashlib, socket, subprocess
    from datetime import datetime, timezone

    model_path = args.output
    sidecar_path = model_path + ".sidecar.json"

    model_sha = hashlib.sha256(open(model_path, "rb").read()).hexdigest()
    train_data_sha = None
    train_data_manifest = None
    if args.npz and os.path.exists(args.npz):
        train_data_sha = hashlib.sha256(open(args.npz, "rb").read()).hexdigest()
    if videos is not None:
        try:
            train_data_manifest = sorted(set(str(v) for v in videos))
        except Exception:
            train_data_manifest = None

    git_commit = None
    try:
        r = subprocess.run(["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
                           capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            git_commit = r.stdout.strip()
    except Exception:
        pass

    sidecar = {
        "schema_version": 1,
        "model_sha256": model_sha,
        "model_path": str(Path(model_path).relative_to(PROJECT_ROOT))
                       if str(Path(model_path)).startswith(str(PROJECT_ROOT))
                       else model_path,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "trained_on": socket.gethostname(),
        "trained_commit": git_commit,
        "trained_command": " ".join(sys.argv),
        "training_data_sha256": train_data_sha,
        "training_data_manifest": train_data_manifest,
        "classes": meta.get("classes"),
        "architecture": (
            f"ShotClassifierCNN(num_classes={len(meta.get('classes', []))}, "
            f"with_regression={meta.get('with_regression', False)}, "
            f"lambda_regr={meta.get('lambda_regr', 0.0)})"
        ),
        "loocv_accuracy": meta.get("loocv_accuracy"),
        "holdout_eval_results": None,
        "holdout_eval_at": None,
        "deploy_status": "candidate",
    }
    with open(sidecar_path, "w") as f:
        json.dump(sidecar, f, indent=2)
    print(f"Sidecar saved to {sidecar_path}")


if __name__ == "__main__":
    main()
