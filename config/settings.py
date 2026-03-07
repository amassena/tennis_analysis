"""Central configuration for the tennis analysis pipeline."""

import os

# ── Project root ──────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Directory paths ───────────────────────────────────────────
RAW_DIR = os.path.join(PROJECT_ROOT, "raw")
PREPROCESSED_DIR = os.path.join(PROJECT_ROOT, "preprocessed")
POSES_DIR = os.path.join(PROJECT_ROOT, "poses_full_videos")
TRAINING_DIR = os.path.join(PROJECT_ROOT, "training")
TRAINING_DATA_DIR = os.path.join(PROJECT_ROOT, "training", "poses")
CLIPS_DIR = os.path.join(PROJECT_ROOT, "training", "clips")
LABELS_DIR = os.path.join(PROJECT_ROOT, "labels")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
HIGHLIGHTS_DIR = os.path.join(PROJECT_ROOT, "highlights")
ANALYSIS_DIR = os.path.join(PROJECT_ROOT, "analysis")
BALL_TRACKING_DIR = os.path.join(PROJECT_ROOT, "ball_tracking")
RACKET_DETECTIONS_DIR = os.path.join(PROJECT_ROOT, "racket_detections")
POSES_3D_DIR = os.path.join(PROJECT_ROOT, "poses_3d")
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")

SHOT_TYPES = ["forehand", "backhand", "serve", "neutral"]

# ── View angles (camera positions) ───────────────────────────
VIEW_ANGLES = ["back-court", "left-side", "right-side", "front", "overhead"]
DEFAULT_VIEW_ANGLE = "back-court"

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

# ── Pose estimation settings ─────────────────────────────────
# Two backends available:
# - "yolo": YOLOv8-Pose (17 keypoints, GPU, ~200fps)
# - "mediapipe": MediaPipe (33 keypoints, CPU, ~8fps)
POSE_BACKEND = "yolo"  # Default to YOLO for speed

YOLO_POSE = {
    "model": "yolov8m-pose.pt",  # s=small, m=medium, l=large, x=xlarge
    "num_keypoints": 17,
    "keypoint_names": [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ],
}

MEDIAPIPE = {
    "model_complexity": 2,          # 0, 1, or 2 (2 = most accurate)
    "min_detection_confidence": 0.5,
    "min_tracking_confidence": 0.5,
    "enable_segmentation": False,
    "smooth_landmarks": True,
    "num_keypoints": 33,
}

# ── Model / training hyperparameters ──────────────────────────
MODEL = {
    "sequence_length": 90,    # frames per input window (1.5s at 60fps) - captures full stroke
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
    "poll_interval": 60,         # seconds between iCloud checks
    "state_file": os.path.join(PROJECT_ROOT, "pipeline_state.json"),
    "slowmo_factor": 4.0,       # setpts multiplier (4.0 = 0.25x speed)
    "slowmo_output_fps": 60,    # output framerate for slow-mo clips
    "youtube_title_format": "Training session ({date}) video {n}",
    "gpu_machines": [
        {"host": "windows", "project": "C:/Users/amass/tennis_analysis"},
        {"host": "tmassena", "project": "C:/Users/amass/tennis_analysis"},
        {"host": "desktop3090", "project": "C:/Users/Andrew/tennis_analysis"},
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

# ── Distributed Coordinator ────────────────────────────────
# Portable architecture: works on Raspberry Pi (local) or AWS (cloud)
COORDINATOR = {
    # API settings
    "host": "0.0.0.0",
    "port": 8080,
    "db_path": os.path.join(PROJECT_ROOT, "coordinator.db"),  # SQLite for local

    # AWS settings (for cloud deployment)
    "aws_region": "us-west-2",
    "dynamodb_table": "tennis-pipeline-jobs",

    # Timeouts and intervals
    "stale_claim_seconds": 3600,  # Release claims older than 1 hour
    "icloud_poll_interval": 300,  # 5 minutes
    "wol_check_interval": 60,     # 1 minute
    "worker_poll_interval": 60,   # 1 minute
}

# ── Cloud Infrastructure ───────────────────────────────────
# Hetzner (orchestration) + Cloudflare R2 (storage) + Local GPUs via Tailscale
# NOTE: RunPod removed - using local GPU machines only (no cloud GPU billing)
CLOUD = {
    "enabled": False,  # Set True to use cloud orchestration with local GPUs

    # Hetzner CPX31 - Orchestration server (5.78.96.237)
    # ~$5/mo: Polls iCloud, dispatches to local GPUs via Tailscale SSH
    "hetzner": {
        "host": os.environ.get("HETZNER_HOST", "5.78.96.237"),
        "ssh_key_path": os.path.expanduser("~/.ssh/id_rsa"),
    },

    # Local GPU machines (accessed via Tailscale)
    "gpu_machines": [
        {"name": "windows", "tailscale_ip": "100.x.x.x", "gpu": "RTX 5080"},
        {"name": "tmassena", "tailscale_ip": "100.x.x.x", "gpu": "RTX 4080"},
    ],

    # Cloudflare R2 - Video storage (zero egress fees!)
    # ~$0.015/GB/month, free egress, S3-compatible
    "r2": {
        "account_id": os.environ.get("CF_ACCOUNT_ID", ""),
        "access_key_id": os.environ.get("CF_R2_ACCESS_KEY_ID", ""),
        "secret_access_key": os.environ.get("CF_R2_SECRET_ACCESS_KEY", ""),
        "bucket_name": "tennis-videos",
        "endpoint_url": None,  # Auto-generated from account_id
        "multipart_threshold_mb": 100,  # Use multipart for files > 100MB
        "multipart_chunk_mb": 50,       # 50MB chunks for multipart
        "lifecycle_days": {
            "raw": 7,           # Delete raw videos after 7 days
            "preprocessed": 14, # Delete preprocessed after 14 days
            "highlights": 0,    # Keep highlights forever (0 = no expiry)
        },
    },

    # YouTube API - Final publishing
    "youtube": {
        "client_secrets_file": os.path.join(PROJECT_ROOT, "config", "client_secrets.json"),
        "credentials_file": os.path.join(PROJECT_ROOT, "config", "youtube_credentials.json"),
        "default_privacy": "unlisted",
        "category_id": "17",  # Sports
        "add_chapters": True,  # Auto-generate chapters from shots
    },

}

# Helper to get R2 endpoint URL
def get_r2_endpoint():
    """Generate R2 endpoint URL from account ID."""
    account_id = CLOUD["r2"]["account_id"]
    if account_id:
        return f"https://{account_id}.r2.cloudflarestorage.com"
    return None

# Set endpoint if account_id is configured
if CLOUD["r2"]["account_id"]:
    CLOUD["r2"]["endpoint_url"] = get_r2_endpoint()
