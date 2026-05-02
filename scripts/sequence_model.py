#!/usr/bin/env python3
"""1D-CNN sequence model for tennis shot classification.

Takes a 90-frame window of 33 MediaPipe landmarks (×3 coords = 99 features per frame)
and classifies the shot type. Designed for ensemble with Random Forest.

Architecture:
    Input: (batch, 90, 99)
    → Conv1d(99→64, k=5) + BN + ReLU
    → Conv1d(64→128, k=7) + BN + ReLU
    → Conv1d(128→128, k=7, dilation=2) + BN + ReLU
    → Conv1d(128→64, k=5, dilation=2) + BN + ReLU
    → AdaptiveAvgPool1d(1) → Dropout(0.3) → Linear(64→4)

~200K parameters. Trains in minutes on GPU.
"""

import json
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import PROJECT_ROOT, POSES_DIR, MODELS_DIR

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None

# Class mapping
CLASSES = ["backhand", "forehand", "not_shot", "serve"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# MediaPipe landmark count
NUM_LANDMARKS = 33
COORDS_PER_LANDMARK = 3  # x, y, z
FEATURES_PER_FRAME = NUM_LANDMARKS * COORDS_PER_LANDMARK  # 99
SEQUENCE_LENGTH = 90  # frames (1.5s at 60fps)

# Hip indices for centering
LEFT_HIP_IDX = 23
RIGHT_HIP_IDX = 24

# Shoulder indices for normalization
LEFT_SHOULDER_IDX = 11
RIGHT_SHOULDER_IDX = 12


# Only define nn.Module subclass if torch is available
_BaseClass = nn.Module if TORCH_AVAILABLE else object


class ShotClassifierCNN(_BaseClass):
    """1D-CNN for shot classification from pose sequences."""

    def __init__(self, num_classes=4, input_features=99, seq_length=90, dropout=0.3):
        super().__init__()

        self.conv1 = nn.Conv1d(input_features, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=7, padding=3)
        self.bn2 = nn.BatchNorm1d(128)

        self.conv3 = nn.Conv1d(128, 128, kernel_size=7, padding=6, dilation=2)
        self.bn3 = nn.BatchNorm1d(128)

        self.conv4 = nn.Conv1d(128, 64, kernel_size=5, padding=4, dilation=2)
        self.bn4 = nn.BatchNorm1d(64)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        """Forward pass.

        Args:
            x: (batch, seq_length, features) — note: time-first input
        Returns:
            logits: (batch, num_classes)
        """
        # Conv1d expects (batch, channels, length)
        x = x.transpose(1, 2)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))

        x = self.pool(x).squeeze(-1)  # (batch, 64)
        x = self.dropout(x)
        return self.fc(x)

    def predict_proba(self, x):
        """Get class probabilities.

        Args:
            x: (batch, seq_length, features)
        Returns:
            probabilities: (batch, num_classes)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=1)


def extract_pose_sequence(frames, center_frame, num_landmarks=33):
    """Extract a hip-centered pose sequence from frame data.

    Args:
        frames: List of pose frame dicts (from load_pose_frames)
        center_frame: Frame index of the detection
        num_landmarks: Number of landmarks per frame (33 for MediaPipe)

    Returns:
        numpy array of shape (SEQUENCE_LENGTH, FEATURES_PER_FRAME) or None
    """
    n = len(frames)
    half = SEQUENCE_LENGTH // 2  # 45
    start = center_frame - half
    end = start + SEQUENCE_LENGTH

    sequence = np.zeros((SEQUENCE_LENGTH, FEATURES_PER_FRAME), dtype=np.float32)

    for seq_idx in range(SEQUENCE_LENGTH):
        frame_idx = start + seq_idx
        if frame_idx < 0 or frame_idx >= n:
            continue  # zero-padded

        frame = frames[frame_idx]
        if not frame:
            continue

        landmarks = frame.get("world_landmarks", [])
        if not landmarks or len(landmarks) < num_landmarks:
            landmarks = frame.get("landmarks", [])
            if not landmarks or len(landmarks) < num_landmarks:
                continue

        # Extract all landmark coordinates
        coords = []
        for lm in landmarks[:num_landmarks]:
            if isinstance(lm, dict):
                coords.extend([lm.get("x", 0), lm.get("y", 0), lm.get("z", 0)])
            elif isinstance(lm, (list, tuple)):
                coords.extend(list(lm[:3]))
                while len(coords) % 3 != 0:
                    coords.append(0.0)
            else:
                coords.extend([0.0, 0.0, 0.0])

        if len(coords) >= FEATURES_PER_FRAME:
            sequence[seq_idx] = coords[:FEATURES_PER_FRAME]

    # Hip-center normalization: subtract hip midpoint from all landmarks per frame
    for i in range(SEQUENCE_LENGTH):
        lh_start = LEFT_HIP_IDX * 3
        rh_start = RIGHT_HIP_IDX * 3

        lh = sequence[i, lh_start:lh_start + 3]
        rh = sequence[i, rh_start:rh_start + 3]

        # Check if frame has data (non-zero)
        if np.sum(np.abs(lh)) < 1e-8 and np.sum(np.abs(rh)) < 1e-8:
            continue

        hip_center = (lh + rh) / 2
        for j in range(NUM_LANDMARKS):
            start_idx = j * 3
            sequence[i, start_idx:start_idx + 3] -= hip_center

    # Shoulder-width normalization: divide by shoulder distance for scale invariance
    for i in range(SEQUENCE_LENGTH):
        ls_start = LEFT_SHOULDER_IDX * 3
        rs_start = RIGHT_SHOULDER_IDX * 3
        ls = sequence[i, ls_start:ls_start + 3]
        rs = sequence[i, rs_start:rs_start + 3]

        shoulder_width = np.linalg.norm(ls - rs)
        if shoulder_width > 0.01:
            sequence[i] /= shoulder_width

    return sequence


def augment_sequence(sequence, fps=60.0, mirror=False, jitter_frames=0,
                     dropout_rate=0.0, temporal_scale=None, noise_std=0.0):
    """Apply data augmentation to a pose sequence.

    Args:
        sequence: (SEQUENCE_LENGTH, FEATURES_PER_FRAME) array
        mirror: If True, flip X coordinates (for handedness)
        jitter_frames: Random temporal shift (±frames)
        dropout_rate: Fraction of frames to zero out
        temporal_scale: Scale factor for temporal stretch/compress (e.g. 0.8-1.2)
        noise_std: Gaussian noise standard deviation

    Returns:
        Augmented sequence array.
    """
    seq = sequence.copy()

    # Temporal scaling: stretch/compress via interpolation
    if temporal_scale is not None and temporal_scale != 1.0:
        orig_len = SEQUENCE_LENGTH
        scaled_len = int(orig_len * temporal_scale)
        if scaled_len > 2:
            # Resample to scaled length then crop/pad back to original
            indices = np.linspace(0, orig_len - 1, scaled_len).astype(np.float32)
            new_seq = np.zeros_like(seq)
            for feat in range(FEATURES_PER_FRAME):
                new_seq[:, feat] = np.interp(
                    np.arange(orig_len),
                    np.linspace(0, orig_len - 1, scaled_len),
                    seq[np.clip(np.round(np.linspace(0, orig_len - 1, scaled_len)).astype(int), 0, orig_len - 1), feat]
                )
            seq = new_seq

    # Temporal jitter: shift the sequence
    if jitter_frames > 0:
        shift = np.random.randint(-jitter_frames, jitter_frames + 1)
        if shift > 0:
            seq[shift:] = seq[:-shift]
            seq[:shift] = 0.0
        elif shift < 0:
            seq[:shift] = seq[-shift:]
            seq[shift:] = 0.0

    # Frame dropout: zero out random frames
    if dropout_rate > 0:
        n_drop = int(SEQUENCE_LENGTH * dropout_rate)
        drop_indices = np.random.choice(SEQUENCE_LENGTH, n_drop, replace=False)
        seq[drop_indices] = 0.0

    # Mirror: flip X coordinates (every 3rd value starting from 0)
    if mirror:
        for i in range(0, FEATURES_PER_FRAME, 3):
            seq[:, i] *= -1

    # Gaussian noise
    if noise_std > 0:
        mask = np.any(seq != 0, axis=1, keepdims=True)  # only add to non-zero frames
        seq += np.random.normal(0, noise_std, seq.shape).astype(np.float32) * mask

    return seq


def load_model(model_path=None, device=None):
    """Load a trained CNN model.

    Args:
        model_path: Path to .pt file. Default: models/sequence_classifier.pt
        device: torch device

    Returns:
        (model, device) tuple, or (None, None) if torch unavailable
    """
    if not TORCH_AVAILABLE:
        return None, None

    if model_path is None:
        model_path = os.path.join(MODELS_DIR, "sequence_classifier.pt")

    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    if not os.path.exists(model_path):
        return None, device

    state = torch.load(model_path, map_location=device, weights_only=True)

    # Detect num_classes from the saved fc layer — supports legacy 6-class
    # models (e.g. broken_28814eeb with volleys) for evaluation, even though
    # production is 4-class. fc.weight has shape (num_classes, in_features).
    num_classes = len(CLASSES)
    if "fc.weight" in state:
        num_classes = state["fc.weight"].shape[0]

    model = ShotClassifierCNN(num_classes=num_classes)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    # Stash classes used at training time for downstream code
    model.num_classes = num_classes
    return model, device



def predict_single(model, device, frames, center_frame):
    """Predict shot type for a single detection.

    Returns:
        dict with 'class', 'confidence', 'probabilities' (per-class)
    """
    seq = extract_pose_sequence(frames, center_frame)
    if seq is None:
        return None

    tensor = torch.from_numpy(seq).unsqueeze(0).to(device)
    proba = model.predict_proba(tensor)[0].cpu().numpy()

    pred_idx = np.argmax(proba)
    return {
        "class": CLASSES[pred_idx],
        "confidence": float(proba[pred_idx]),
        "probabilities": {c: float(proba[i]) for i, c in enumerate(CLASSES)},
    }
