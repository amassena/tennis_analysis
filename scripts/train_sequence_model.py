#!/usr/bin/env python3
"""Train the 1D-CNN sequence classifier with leave-one-video-out CV.

Loads pose data and detection labels, extracts 90-frame sequences,
trains with augmentation, evaluates via LOOCV, and saves the final model.

IMPORTANT: Run this on Windows GPU machine (RTX 5080/4080), not on Mac.

Usage:
    python scripts/train_sequence_model.py
    python scripts/train_sequence_model.py --epochs 30 --batch-size 64
    python scripts/train_sequence_model.py --skip-cv  # just train final model
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
from scripts.extract_training_features import load_pose_frames

DETECTIONS_DIR = os.path.join(PROJECT_ROOT, "detections")

# Shot types to train on
TRAINABLE_TYPES = {"forehand", "backhand", "serve"}

# Serve-only videos (skip FH/BH labels as contaminated)
SERVE_ONLY_VIDEOS = {
    "IMG_0864", "IMG_0865", "IMG_0866", "IMG_0867",
    "IMG_0868", "IMG_0869", "IMG_0870",
}


class PoseSequenceDataset(Dataset):
    """Dataset of pose sequences with augmentation."""

    def __init__(self, samples, augment=False, jitter=5, dropout_rate=0.15):
        self.samples = samples  # list of (sequence_array, label_idx, video_name)
        self.augment = augment
        self.jitter = jitter
        self.dropout_rate = dropout_rate

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq, label, _ = self.samples[idx]

        if self.augment:
            mirror = np.random.random() < 0.5
            seq = augment_sequence(
                seq,
                mirror=mirror,
                jitter_frames=self.jitter,
                dropout_rate=np.random.uniform(0.1, self.dropout_rate),
            )

        return torch.from_numpy(seq).float(), label


def load_all_samples(poses_dir, det_dir):
    """Load all labeled samples with their pose sequences.

    Returns list of (sequence_array, label_idx, video_name) tuples.
    """
    samples = []
    pose_cache = {}

    det_files = sorted(
        f for f in os.listdir(det_dir)
        if f.endswith("_fused.json")
        and "_v2" not in f and "_ml" not in f
        and "_pre" not in f and "_baseline" not in f
    )

    # Also check for v5 versions
    for det_file in det_files:
        video_name = det_file.replace("_fused.json", "")
        v5_path = os.path.join(det_dir, f"{video_name}_fused_v5.json")
        det_path = v5_path if os.path.exists(v5_path) else os.path.join(det_dir, det_file)

        with open(det_path) as f:
            det_data = json.load(f)

        is_serve_only = video_name in SERVE_ONLY_VIDEOS
        dominant_hand = det_data.get("dominant_hand", "right")

        # Load poses
        pose_path = os.path.join(poses_dir, f"{video_name}.json")
        if not os.path.exists(pose_path):
            continue

        if video_name not in pose_cache:
            frames, fps, total_frames = load_pose_frames(pose_path)
            pose_cache[video_name] = frames
        frames = pose_cache[video_name]

        # Extract labeled shots
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

            samples.append((seq, label_idx, video_name))
            positives += 1

        # Extract not_shot negatives: random frames far from detections
        det_times = [d.get("timestamp", 0) for d in det_data.get("detections", [])]
        fps_val = det_data.get("fps", 60.0)
        n_neg = max(5, positives // 3)  # ~1/3 ratio
        neg_count = 0

        for frame_idx in range(90, len(frames) - 90, int(fps_val * 3)):
            if neg_count >= n_neg:
                break
            t = frame_idx / fps_val
            if any(abs(t - dt) < 3.0 for dt in det_times):
                continue

            seq = extract_pose_sequence(frames, frame_idx)
            if seq is not None and np.sum(np.abs(seq)) > 1e-6:
                samples.append((seq, CLASS_TO_IDX["not_shot"], video_name))
                neg_count += 1

        print(f"  {video_name}: {positives} shots + {neg_count} negatives")

    return samples


def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)
        _, predicted = logits.max(1)
        correct += predicted.eq(batch_y).sum().item()
        total += batch_x.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, device):
    """Evaluate model. Returns (loss, accuracy, all_preds, all_labels)."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            total_loss += loss.item() * batch_x.size(0)
            _, predicted = logits.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    total = len(all_labels)
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / total if total > 0 else 0
    avg_loss = total_loss / total if total > 0 else 0

    return avg_loss, acc, np.array(all_preds), np.array(all_labels)


def leave_one_video_out_cv(samples, args, device):
    """Leave-one-video-out cross-validation."""
    by_video = defaultdict(list)
    for seq, label, video in samples:
        by_video[video].append((seq, label, video))

    videos = sorted(by_video.keys())
    all_preds = []
    all_labels = []
    fold_results = []

    print(f"\nLeave-one-video-out CV ({len(videos)} folds):")
    print(f"{'-'*70}")

    for fold_idx, held_out in enumerate(videos):
        train_samples = [s for s in samples if s[2] != held_out]
        val_samples = by_video[held_out]

        if not train_samples or not val_samples:
            continue

        train_dataset = PoseSequenceDataset(train_samples, augment=True)
        val_dataset = PoseSequenceDataset(val_samples, augment=False)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                                shuffle=False, num_workers=0)

        model = ShotClassifierCNN(num_classes=len(CLASSES)).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

        # Class weights for imbalanced data
        label_counts = defaultdict(int)
        for _, l, _ in train_samples:
            label_counts[l] += 1
        total = sum(label_counts.values())
        weights = torch.tensor(
            [total / (len(CLASSES) * label_counts.get(i, 1)) for i in range(len(CLASSES))],
            dtype=torch.float32
        ).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)

        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        best_val_acc = 0
        for epoch in range(args.epochs):
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
            scheduler.step()

        # Final evaluation
        val_loss, val_acc, preds, labels = evaluate(model, val_loader, device)

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

        print(f"  Fold {fold_idx:2d}: {held_out:<12s} "
              f"acc={val_acc:.2f} ({len(val_samples):3d} samples) {detail}")

        all_preds.extend(preds)
        all_labels.extend(labels)
        fold_results.append({
            "video": held_out,
            "n_samples": len(val_samples),
            "accuracy": round(float(val_acc), 4),
        })

    # Aggregate
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    overall_acc = np.mean(all_preds == all_labels)

    print(f"\n{'='*50}")
    print(f"LOOCV AGGREGATE: accuracy={overall_acc:.3f} ({sum(all_preds == all_labels)}/{len(all_labels)})")
    print(f"{'='*50}")

    # Confusion matrix
    print(f"\nConfusion Matrix:")
    print(f"{'':>12s}  " + "  ".join(f"{c:>10s}" for c in CLASSES))
    for i, true_class in enumerate(CLASSES):
        row = []
        for j, pred_class in enumerate(CLASSES):
            count = np.sum((all_labels == i) & (all_preds == j))
            row.append(count)
        print(f"{true_class:>12s}  " + "  ".join(f"{v:>10d}" for v in row))

    return overall_acc, fold_results


def main():
    parser = argparse.ArgumentParser(
        description="Train 1D-CNN sequence classifier")
    parser.add_argument("--poses-dir", default=POSES_DIR)
    parser.add_argument("--det-dir", default=DETECTIONS_DIR)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--skip-cv", action="store_true")
    parser.add_argument("--output", default=os.path.join(MODELS_DIR, "sequence_classifier.pt"))
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
    print(f"\nLoading samples from {args.poses_dir}...")
    samples = load_all_samples(args.poses_dir, args.det_dir)

    # Distribution
    label_counts = defaultdict(int)
    for _, label, _ in samples:
        label_counts[label] += 1
    print(f"\nTotal: {len(samples)} samples")
    for idx in sorted(label_counts.keys()):
        print(f"  {CLASSES[idx]}: {label_counts[idx]}")

    # LOOCV
    if not args.skip_cv:
        cv_acc, fold_results = leave_one_video_out_cv(samples, args, device)

    # Train final model on all data
    print(f"\n{'='*50}")
    print(f"TRAINING FINAL MODEL on all {len(samples)} samples")
    print(f"{'='*50}")

    dataset = PoseSequenceDataset(samples, augment=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = ShotClassifierCNN(num_classes=len(CLASSES)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Class weights
    total = sum(label_counts.values())
    weights = torch.tensor(
        [total / (len(CLASSES) * label_counts.get(i, 1)) for i in range(len(CLASSES))],
        dtype=torch.float32
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, loader, optimizer, criterion, device)
        scheduler.step()
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{args.epochs}: loss={train_loss:.4f} acc={train_acc:.3f}")

    # Save model
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(model.state_dict(), args.output)

    # Save metadata
    meta_path = args.output.replace(".pt", "_meta.json")
    meta = {
        "classes": CLASSES,
        "num_samples": len(samples),
        "label_counts": {CLASSES[k]: v for k, v in label_counts.items()},
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "sequence_length": SEQUENCE_LENGTH,
        "features_per_frame": FEATURES_PER_FRAME,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nModel saved to {args.output}")
    print(f"Metadata saved to {meta_path}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")


if __name__ == "__main__":
    main()
