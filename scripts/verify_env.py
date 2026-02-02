#!/usr/bin/env python3
"""Verify that the tennis analysis environment is correctly set up."""

import sys
import os
import subprocess

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
WARN = "\033[93m[WARN]\033[0m"

failures = 0


def check(label, condition, detail=""):
    global failures
    if condition:
        print(f"  {PASS} {label}" + (f"  ({detail})" if detail else ""))
    else:
        print(f"  {FAIL} {label}" + (f"  ({detail})" if detail else ""))
        failures += 1


def warn(label, detail=""):
    print(f"  {WARN} {label}" + (f"  ({detail})" if detail else ""))


# ── Python version ────────────────────────────────────────────
print("\n--- Python ---")
v = sys.version_info
check("Python 3.9+", v.major == 3 and v.minor >= 9, f"{v.major}.{v.minor}.{v.micro}")

# ── Package imports ───────────────────────────────────────────
print("\n--- Package imports ---")

packages = {
    "mediapipe": "mediapipe",
    "tensorflow": "tensorflow",
    "cv2": "opencv-python",
    "numpy": "numpy",
    "matplotlib": "matplotlib",
    "ffmpeg": "ffmpeg-python",
    "sklearn": "scikit-learn",
}

imported = {}
for mod, pip_name in packages.items():
    try:
        imported[mod] = __import__(mod)
        ver = getattr(imported[mod], "__version__", "?")
        check(f"{pip_name}", True, f"v{ver}")
    except ImportError as e:
        check(f"{pip_name}", False, str(e))

# pyicloud uses a different import pattern
try:
    from pyicloud import PyiCloudService
    check("pyicloud", True)
except ImportError as e:
    check("pyicloud", False, str(e))

# ── TensorFlow Metal GPU ─────────────────────────────────────
print("\n--- TensorFlow GPU ---")
if "tensorflow" in imported:
    tf = imported["tensorflow"]
    gpus = tf.config.list_physical_devices("GPU")
    check("TensorFlow detects Metal GPU", len(gpus) > 0, f"{len(gpus)} GPU(s): {gpus}")
else:
    check("TensorFlow detects Metal GPU", False, "tensorflow not imported")

# ── FFmpeg ────────────────────────────────────────────────────
print("\n--- FFmpeg ---")
try:
    result = subprocess.run(
        ["ffmpeg", "-version"], capture_output=True, text=True, timeout=10
    )
    first_line = result.stdout.split("\n")[0] if result.stdout else "unknown"
    check("FFmpeg callable", result.returncode == 0, first_line)
except FileNotFoundError:
    check("FFmpeg callable", False, "ffmpeg not found in PATH")
except subprocess.TimeoutExpired:
    check("FFmpeg callable", False, "timed out")

# ── MediaPipe pose model ─────────────────────────────────────
print("\n--- MediaPipe Pose ---")
if "mediapipe" in imported:
    try:
        mp = imported["mediapipe"]
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=0.5,
        )
        pose.close()
        check("MediaPipe Pose initializes", True)
    except Exception as e:
        check("MediaPipe Pose initializes", False, str(e))
else:
    check("MediaPipe Pose initializes", False, "mediapipe not imported")

# ── Project directories ──────────────────────────────────────
print("\n--- Project directories ---")
from config.settings import (
    RAW_DIR, PREPROCESSED_DIR, POSES_DIR, TRAINING_DATA_DIR,
    MODELS_DIR, CLIPS_DIR, HIGHLIGHTS_DIR, SHOT_TYPES,
)

dirs_to_check = [
    RAW_DIR, PREPROCESSED_DIR, POSES_DIR, MODELS_DIR, HIGHLIGHTS_DIR,
]
for shot in SHOT_TYPES:
    dirs_to_check.append(os.path.join(TRAINING_DATA_DIR, shot))
for shot in ["forehand", "backhand", "serve"]:
    dirs_to_check.append(os.path.join(CLIPS_DIR, shot))

all_dirs_exist = True
for d in dirs_to_check:
    if not os.path.isdir(d):
        check(f"Directory exists: {d}", False)
        all_dirs_exist = False

if all_dirs_exist:
    check(f"All {len(dirs_to_check)} project directories exist", True)

# ── Summary ───────────────────────────────────────────────────
print("\n" + "=" * 50)
if failures == 0:
    print(f"{PASS} All checks passed!")
else:
    print(f"{FAIL} {failures} check(s) failed.")
print("=" * 50 + "\n")

sys.exit(failures)
