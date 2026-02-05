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

# ── iCloud integration ───────────────────────────────────────
COOKIE_DIR = os.path.join(PROJECT_ROOT, "config", "icloud_session")

ICLOUD = {
    "env_file": os.path.join(PROJECT_ROOT, ".env"),
    "cookie_directory": COOKIE_DIR,
    "download_directory": RAW_DIR,
    "chunk_size": 65536,       # 64KB streaming chunks
    "video_album": "Videos",
    "max_retries": 3,
    "retry_delay": 5,
}

# ── Automated pipeline ──────────────────────────────────────
AUTO_PIPELINE = {
    "album": "Tennis Videos",  # default album (chronological ordering)
    "albums": {                # album name -> shot ordering
        "Tennis Videos": "chronological",
        "Tennis Videos Group By Shot Type": "type",
    },
    "poll_interval": 300,        # seconds between iCloud checks
    "state_file": os.path.join(PROJECT_ROOT, "pipeline_state.json"),
    "slowmo_factor": 4.0,       # setpts multiplier (4.0 = 0.25x speed)
    "slowmo_output_fps": 60,    # output framerate for slow-mo clips
    "youtube_title_format": "Training session ({date}) video {n}",
    "gpu_machines": [
        {"host": "windows", "project": "C:/Users/amass/tennis_analysis"},
        {"host": "tmassena", "project": "C:/Users/amass/tennis_analysis"},
    ],
}

# ── Notifications ────────────────────────────────────────
# Add phone numbers / email addresses here to get notified
# when highlight videos are processed and uploaded.
NOTIFICATIONS = {
    "imessage": [
        "+12068527753",
    ],
    "email": {
        "recipients": [
            # "friend@example.com",
            # "coach@example.com",
        ],
        "smtp_server": "smtp.gmail.com",
        "smtp_port": 587,
        "sender": os.environ.get("GMAIL_SENDER", ""),
        "app_password": os.environ.get("GMAIL_APP_PASSWORD", ""),
    },
}
