"""Utilities for managing per-video metadata files."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import PROJECT_ROOT, DEFAULT_VIEW_ANGLE


def _metadata_path(video_name: str) -> str:
    """Return the path to a video's metadata JSON file."""
    # Strip extension if present
    base = os.path.splitext(video_name)[0]
    return os.path.join(PROJECT_ROOT, f"{base}_metadata.json")


def load_video_metadata(video_name: str) -> dict:
    """Load metadata for a video, returning empty dict if not found."""
    path = _metadata_path(video_name)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def save_video_metadata(video_name: str, metadata: dict) -> None:
    """Save metadata for a video to its JSON file."""
    path = _metadata_path(video_name)
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)


def get_view_angle(video_name: str) -> str:
    """Get the view angle for a video, defaulting to back-court."""
    metadata = load_video_metadata(video_name)
    return metadata.get("view_angle", DEFAULT_VIEW_ANGLE)


def set_view_angle(video_name: str, view_angle: str) -> None:
    """Set the view angle for a video."""
    metadata = load_video_metadata(video_name)
    metadata["view_angle"] = view_angle
    save_video_metadata(video_name, metadata)
