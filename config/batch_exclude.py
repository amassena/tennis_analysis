"""Helper for reading the permanent batch-exclusion list.

Batch scripts (claude_coach.py --all, backfill_shots_json.py, grade
overlay drivers, any new --all style loops) should call
`is_excluded(vid)` and skip accordingly.
"""
import json
import os
from pathlib import Path

_CONFIG_PATH = Path(__file__).parent / "exclude_from_batch.json"

_CACHE = None


def _load():
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    try:
        with open(_CONFIG_PATH) as f:
            data = json.load(f)
    except FileNotFoundError:
        _CACHE = set()
        return _CACHE
    vids = set()
    for k, v in data.items():
        if isinstance(v, list):
            vids.update(v)
    _CACHE = vids
    return _CACHE


def is_excluded(video_id: str) -> bool:
    """Return True if video_id should be skipped by batch operations."""
    return video_id in _load()


def excluded_set() -> set:
    """Return the full set of excluded video IDs (copy)."""
    return set(_load())
