#!/usr/bin/env python3
"""Patch YOLO pose files to add missing fields for label_clips.py compatibility."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import POSES_DIR

def patch_file(name):
    path = os.path.join(POSES_DIR, f"{name}.json")
    print(f"Patching {path}...")

    with open(path, "r") as f:
        data = json.load(f)

    if "version" not in data:
        data["version"] = 2
        data["source_video"] = name
        if "num_keypoints" not in data.get("video_info", {}):
            data["video_info"]["num_keypoints"] = 17

        with open(path, "w") as f:
            json.dump(data, f)
        print(f"  Patched successfully")
    else:
        print(f"  Already patched")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        for name in sys.argv[1:]:
            patch_file(name)
    else:
        # Patch known YOLO-extracted files
        for name in ["IMG_0870", "IMG_6665", "IMG_6713"]:
            patch_file(name)
