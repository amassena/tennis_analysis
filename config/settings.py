"""Central configuration for the tennis analysis pipeline."""

import os

# ── Project root ──────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Directory paths ───────────────────────────────────────────
RAW_DIR = os.path.join(PROJECT_ROOT, "raw")
PREPROCESSED_DIR = os.path.join(PROJECT_ROOT, "preprocessed")
POSES_DIR = os.path.join(PROJECT_ROOT, "poses")
TRAINING_DATA_DIR = os.path.join(PROJECT_ROOT, "training_data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
CLIPS_DIR = os.path.join(PROJECT_ROOT, "clips")
HIGHLIGHTS_DIR = os.path.join(PROJECT_ROOT, "highlights")
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")

SHOT_TYPES = ["forehand", "backhand", "serve", "neutral"]

# ── Video processing defaults ─────────────────────────────────
VIDEO = {
    "target_fps": 60,
    "source_fps": 240,
    "codec": "libx264",
    "crf": 18,           # quality (lower = better, 18 is visually lossless)
    "preset": "medium",
    "pixel_format": "yuv420p",
    "audio_codec": "aac",
}

# ── MediaPipe pose settings ───────────────────────────────────
MEDIAPIPE = {
    "model_complexity": 2,          # 0, 1, or 2 (2 = most accurate)
    "min_detection_confidence": 0.5,
    "min_tracking_confidence": 0.5,
    "enable_segmentation": False,
    "smooth_landmarks": True,
}

# ── Model / training hyperparameters ──────────────────────────
MODEL = {
    "sequence_length": 30,    # frames per input window (0.5s at 60fps)
    "hidden_units": 128,
    "num_layers": 2,
    "dropout": 0.3,
    "learning_rate": 1e-3,
    "batch_size": 32,
    "epochs": 50,
    "validation_split": 0.2,
    "early_stopping_patience": 7,
}
