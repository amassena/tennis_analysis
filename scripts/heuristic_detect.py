#!/usr/bin/env python3
"""Heuristic-based shot detection using pose biomechanics.

No ML model - uses velocity spikes + geometric rules based on pose keypoints.
Expected accuracy: ~90% serve, ~70% forehand/backhand.

Detection strategy (biomechanics-based):
- SERVE: Wrist raised above head level (trophy position)
- FOREHAND: Wrist velocity spike + wrist crosses body (racket to non-racket side)
- BACKHAND: Wrist velocity spike + wrist crosses body (non-racket to racket side)
- NEUTRAL: Low arm movement, stance is relatively still

Key insight: Raw position isn't enough. We detect shots by:
1. Finding velocity SPIKES (sudden acceleration)
2. Checking direction of movement at spike time
3. Using geometric constraints (body crossing) to classify

Usage:
    python heuristic_detect.py poses/IMG_1234.json -o shots_heuristic.json
    python heuristic_detect.py poses/IMG_1234.json --visualize
"""

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

# Try numpy, fall back to pure Python
try:
    import numpy as np
    def sqrt(x): return np.sqrt(x)
    def mean(x): return np.mean(x) if x else 0.0
except ImportError:
    def sqrt(x): return math.sqrt(x)
    def mean(x): return sum(x) / len(x) if x else 0.0

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ─────────────────────────────────────────────────────────────
# TUNABLE PARAMETERS (documented in plan)
# ─────────────────────────────────────────────────────────────

# Clip extraction offsets (seconds) - TUNABLE
CLIP_OFFSETS = {
    "serve": {"pre": 3.0, "post": 1.5},      # Capture ball toss, trophy, follow-through
    "forehand": {"pre": 1.5, "post": 1.0},   # Capture backswing, contact, follow-through
    "backhand": {"pre": 1.5, "post": 1.0},   # Capture backswing, contact, follow-through
    "volley": {"pre": 1.0, "post": 0.5},     # Shorter motion
}

# Detection thresholds - TUNABLE
THRESHOLDS = {
    "serve_arm_height": 0.25,        # Wrist must be 25cm+ above head (stricter)
    "velocity_min": 12.0,            # Min wrist velocity (m/s) - spike detection threshold
    "velocity_spike_ratio": 3.0,     # Velocity must be 3x rolling avg
    "min_shot_gap_frames": 60,       # 1.0s between shots (prevents duplicate spikes from same swing)
    "body_cross_threshold": 0.08,    # Wrist must clearly cross body (8cm)
    "min_segment_frames": 10,        # Min frames to count as a shot
    "peak_velocity_min": 10.0,       # Min peak velocity during contact (m/s)
}


@dataclass
class Detection:
    """A detected shot segment."""
    shot_type: str
    start_frame: int
    end_frame: int
    confidence: float
    trigger: str  # What triggered the detection


# ─────────────────────────────────────────────────────────────
# Keypoint indices (MediaPipe 33-keypoint model)
# ─────────────────────────────────────────────────────────────

# MediaPipe pose landmarks
NOSE = 0
LEFT_EYE_INNER = 1
LEFT_EYE = 2
LEFT_EYE_OUTER = 3
RIGHT_EYE_INNER = 4
RIGHT_EYE = 5
RIGHT_EYE_OUTER = 6
LEFT_EAR = 7
RIGHT_EAR = 8
MOUTH_LEFT = 9
MOUTH_RIGHT = 10
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_PINKY = 17
RIGHT_PINKY = 18
LEFT_INDEX = 19
RIGHT_INDEX = 20
LEFT_THUMB = 21
RIGHT_THUMB = 22
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
LEFT_HEEL = 29
RIGHT_HEEL = 30
LEFT_FOOT_INDEX = 31
RIGHT_FOOT_INDEX = 32


# ─────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────

def get_pose_confidence(frame_data) -> float:
    """Check if pose is plausible based on landmark visibility.

    Returns average visibility (0.0–1.0) of key body landmarks.
    """
    key_indices = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_WRIST, RIGHT_WRIST]

    # Try landmarks first (normalized image coords with visibility)
    landmarks = frame_data.get("landmarks", [])
    if landmarks and len(landmarks) > max(key_indices):
        visibilities = [landmarks[idx][3] for idx in key_indices
                        if idx < len(landmarks) and len(landmarks[idx]) >= 4]
        if visibilities:
            return sum(visibilities) / len(visibilities)

    # Try keypoints (dict format with visibility key)
    keypoints = frame_data.get("keypoints", [])
    if keypoints:
        visibilities = []
        for idx in key_indices:
            if idx < len(keypoints):
                vis = keypoints[idx].get("visibility", 0) if isinstance(keypoints[idx], dict) else 0
                visibilities.append(vis)
        if visibilities:
            return sum(visibilities) / len(visibilities)

    # Fall back: if world_landmarks exist, assume valid
    world_lm = frame_data.get("world_landmarks", [])
    if world_lm and len(world_lm) >= 17:
        return 0.8

    return 0.0


def get_keypoint(frame_data, idx):
    """Get keypoint coordinates from frame data.

    Returns (x, y, z) or None if not available.
    """
    # Try world_landmarks first (real-world coordinates)
    world_lm = frame_data.get("world_landmarks")
    if world_lm and idx < len(world_lm):
        return world_lm[idx]

    # Fall back to normalized keypoints
    keypoints = frame_data.get("keypoints")
    if keypoints and idx < len(keypoints):
        kp = keypoints[idx]
        return [kp.get("x", 0), kp.get("y", 0), kp.get("z", 0)]

    return None


def body_midline_x(frame_data):
    """Calculate x-coordinate of body midline (average of shoulders)."""
    left_shoulder = get_keypoint(frame_data, LEFT_SHOULDER)
    right_shoulder = get_keypoint(frame_data, RIGHT_SHOULDER)

    if left_shoulder and right_shoulder:
        return (left_shoulder[0] + right_shoulder[0]) / 2
    return None


def head_level_y(frame_data):
    """Calculate y-coordinate of head level (nose or eye average)."""
    nose = get_keypoint(frame_data, NOSE)
    if nose:
        return nose[1]

    left_eye = get_keypoint(frame_data, LEFT_EYE)
    right_eye = get_keypoint(frame_data, RIGHT_EYE)
    if left_eye and right_eye:
        return (left_eye[1] + right_eye[1]) / 2

    return None


def shoulder_level_y(frame_data):
    """Calculate y-coordinate of shoulder level."""
    left_shoulder = get_keypoint(frame_data, LEFT_SHOULDER)
    right_shoulder = get_keypoint(frame_data, RIGHT_SHOULDER)

    if left_shoulder and right_shoulder:
        return (left_shoulder[1] + right_shoulder[1]) / 2
    return None


# ─────────────────────────────────────────────────────────────
# Velocity computation (key for biomechanics-based detection)
# ─────────────────────────────────────────────────────────────

def compute_wrist_velocities(frames: List[dict], wrist_idx: int, fps: float = 60.0) -> List[float]:
    """Compute wrist velocity magnitude for each frame.

    Returns list of velocities (same length as frames, 0.0 for first frame).
    """
    velocities = [0.0]  # First frame has no velocity
    prev_pos = None

    for i, frame in enumerate(frames):
        wrist = get_keypoint(frame, wrist_idx)
        if wrist is None:
            velocities.append(0.0)
            prev_pos = None
            continue

        if prev_pos is None:
            prev_pos = wrist
            if i > 0:
                velocities.append(0.0)
            continue

        # Velocity = distance / time
        dx = wrist[0] - prev_pos[0]
        dy = wrist[1] - prev_pos[1]
        dz = wrist[2] - prev_pos[2] if len(wrist) > 2 and len(prev_pos) > 2 else 0

        dist = sqrt(dx*dx + dy*dy + dz*dz)
        vel = dist * fps  # Convert to units/second
        velocities.append(vel)
        prev_pos = wrist

    return velocities


def rolling_average(values: List[float], window: int = 15) -> List[float]:
    """Compute rolling average with edge padding."""
    result = []
    for i in range(len(values)):
        start = max(0, i - window)
        end = min(len(values), i + window + 1)
        result.append(mean(values[start:end]))
    return result


def find_velocity_spikes(velocities: List[float],
                         min_vel: float = 0.5,
                         spike_ratio: float = 2.5,
                         min_gap: int = 60) -> List[Tuple[int, float]]:
    """Find frames with velocity spikes (potential shot contact points).

    Returns list of (frame_index, velocity) tuples.
    """
    avg_velocities = rolling_average(velocities)
    spikes = []
    last_spike = -min_gap

    for i, vel in enumerate(velocities):
        if i - last_spike < min_gap:
            continue
        if vel < min_vel:
            continue
        if avg_velocities[i] > 0 and vel < avg_velocities[i] * spike_ratio:
            continue

        # This is a spike
        spikes.append((i, vel))
        last_spike = i

    return spikes


# ─────────────────────────────────────────────────────────────
# Jerk detection (d³position/dt³ — immune to rally baseline)
# ─────────────────────────────────────────────────────────────

def compute_wrist_acceleration(velocities, fps, smooth_window=5):
    """Compute wrist acceleration from velocity series.

    Smooths velocity with a triangular kernel (2*smooth_window+1 frames)
    before differentiating to reduce noise.

    Returns array of acceleration values (length = len(velocities) - 1).
    """
    import numpy as np
    v = np.array(velocities, dtype=np.float64)

    # Triangular (Bartlett) smoothing kernel
    kernel_size = 2 * smooth_window + 1
    kernel = np.bartlett(kernel_size)
    kernel /= kernel.sum()

    # Pad and convolve
    v_smooth = np.convolve(v, kernel, mode='same')

    # Acceleration = d(velocity)/dt
    accel = np.diff(v_smooth) * fps
    return accel


def compute_wrist_jerk(accelerations, fps, smooth_window=3):
    """Compute wrist jerk (rate of change of acceleration).

    Uses a tighter triangular kernel (2*smooth_window+1 frames) on
    acceleration before differentiating. Returns absolute jerk values
    (length = len(accelerations) - 1).
    """
    import numpy as np
    a = np.array(accelerations, dtype=np.float64)

    kernel_size = 2 * smooth_window + 1
    kernel = np.bartlett(kernel_size)
    kernel /= kernel.sum()

    a_smooth = np.convolve(a, kernel, mode='same')

    jerk = np.abs(np.diff(a_smooth) * fps)
    return jerk


def find_jerk_spikes(jerk, min_jerk=800.0, min_gap=60, percentile_threshold=92.0):
    """Find frames with jerk spikes using adaptive percentile threshold.

    Uses scipy.signal.find_peaks with a threshold derived from the
    percentile of the jerk series (not ratio-based, so immune to
    inflated rolling averages during active rallies).

    Returns list of (frame_index, jerk_value) tuples.
    Frame indices are offset by +2 from the original velocity array
    (one diff for accel, one for jerk).
    """
    import numpy as np
    from scipy.signal import find_peaks

    jerk = np.array(jerk, dtype=np.float64)
    if len(jerk) == 0:
        return []

    # Adaptive threshold: max of absolute minimum and percentile
    pct_thresh = np.percentile(jerk, percentile_threshold)
    threshold = max(min_jerk, pct_thresh)

    # find_peaks with minimum distance between peaks
    peak_indices, properties = find_peaks(jerk, height=threshold, distance=min_gap)

    spikes = []
    for idx in peak_indices:
        # Offset +2 to map back to original velocity/frame indices
        # (1 diff for accel, 1 diff for jerk)
        original_frame = idx + 2
        spikes.append((original_frame, float(jerk[idx])))

    return spikes


# ─────────────────────────────────────────────────────────────
# Detection rules
# ─────────────────────────────────────────────────────────────

def detect_serve(frame_data) -> tuple:
    """Detect serve: wrist raised above head level.

    In MediaPipe world coordinates, Y increases downward in image space,
    but in world coordinates Y is often positive upward. We need to check
    if wrist.y > head.y (world coords) or wrist.y < head.y (image coords).

    Returns: (is_serve, confidence, trigger_description)
    """
    right_wrist = get_keypoint(frame_data, RIGHT_WRIST)
    left_wrist = get_keypoint(frame_data, LEFT_WRIST)
    head_y = head_level_y(frame_data)
    shoulder_y = shoulder_level_y(frame_data)

    if not head_y or not shoulder_y:
        return False, 0.0, ""

    # Check both wrists
    for wrist, name in [(right_wrist, "right"), (left_wrist, "left")]:
        if not wrist:
            continue

        # In world coordinates, more negative Y = higher
        # Serve = wrist at least 0.3m above head
        height_above_head = head_y - wrist[1]  # positive = wrist higher

        if height_above_head > 0.15:  # 15cm above head
            confidence = min(1.0, height_above_head / 0.4)  # Max at 40cm
            return True, confidence, f"{name} wrist {height_above_head:.2f}m above head"

    return False, 0.0, ""


def detect_forehand(frame_data, dominant_hand="right") -> tuple:
    """Detect forehand: dominant wrist crosses body OR left wrist leads left.

    For right-handed player:
    - Right wrist crosses left of midline, OR
    - Left wrist significantly left (follow-through) with right wrist near midline
    Returns: (is_forehand, confidence, trigger_description)
    """
    midline = body_midline_x(frame_data)
    if midline is None:
        return False, 0.0, ""

    if dominant_hand == "right":
        right_wrist = get_keypoint(frame_data, RIGHT_WRIST)
        left_wrist = get_keypoint(frame_data, LEFT_WRIST)

        if not right_wrist:
            return False, 0.0, ""

        r_wrist_offset = midline - right_wrist[0]  # positive = wrist left of midline
        l_wrist_offset = midline - left_wrist[0] if left_wrist else 0

        # Forehand indicators:
        # 1. Right wrist crosses left with left wrist NOT leading (contact phase)
        # 2. Left wrist leads left but right wrist stays near/right of midline (follow-through)

        # Key: in forehand, right wrist is to the RIGHT of left wrist (or equal)
        # In backhand, right wrist is to the LEFT of left wrist (two-handed)
        wrist_diff = (right_wrist[0] - left_wrist[0]) if left_wrist else 0

        # Forehand: right wrist >= left wrist (right wrist is more right or equal)
        if wrist_diff >= -0.02:  # right wrist not significantly left of left wrist
            if r_wrist_offset > 0.03 or l_wrist_offset > 0.04:  # some crossing
                confidence = min(1.0, max(r_wrist_offset, l_wrist_offset) / 0.15)
                return True, confidence, f"forehand: R={r_wrist_offset:.2f}, L={l_wrist_offset:.2f}"
    else:
        left_wrist = get_keypoint(frame_data, LEFT_WRIST)
        if left_wrist and left_wrist[0] > midline + 0.05:
            offset = left_wrist[0] - midline
            confidence = min(1.0, offset / 0.2)
            return True, confidence, f"forehand: wrist {offset:.2f} right"

    return False, 0.0, ""


def detect_backhand(frame_data, dominant_hand="right") -> tuple:
    """Detect backhand: right wrist crosses LEFT of left wrist (two-handed).

    For right-handed two-handed backhand:
    - Right wrist is to the LEFT of left wrist (hands together, right hand underneath)
    - Both wrists are on the left side of body

    Returns: (is_backhand, confidence, trigger_description)
    """
    midline = body_midline_x(frame_data)
    if midline is None:
        return False, 0.0, ""

    if dominant_hand == "right":
        right_wrist = get_keypoint(frame_data, RIGHT_WRIST)
        left_wrist = get_keypoint(frame_data, LEFT_WRIST)

        if not right_wrist or not left_wrist:
            return False, 0.0, ""

        r_wrist_offset = midline - right_wrist[0]  # positive = left of midline

        # Key indicator: right wrist is LEFT of left wrist (two-handed backhand grip)
        wrist_diff = right_wrist[0] - left_wrist[0]  # negative = right more left

        # Two-handed backhand: right wrist is left of left wrist AND left of midline
        if wrist_diff < -0.01 and r_wrist_offset > 0.02:
            confidence = min(1.0, (abs(wrist_diff) + r_wrist_offset) / 0.15)
            return True, confidence, f"backhand: R-L={wrist_diff:.2f}, R_off={r_wrist_offset:.2f}"

        # One-handed backhand: right wrist clearly left, but right >= left (no two-hand)
        # This is harder to detect reliably, so we use stricter threshold
        if wrist_diff >= 0 and r_wrist_offset > 0.08:
            left_shoulder = get_keypoint(frame_data, LEFT_SHOULDER)
            right_shoulder = get_keypoint(frame_data, RIGHT_SHOULDER)
            if left_shoulder and right_shoulder:
                shoulder_rot = left_shoulder[0] - right_shoulder[0]
                if shoulder_rot < -0.03:  # right shoulder forward
                    confidence = min(1.0, r_wrist_offset / 0.15)
                    return True, confidence, f"1H-backhand: off={r_wrist_offset:.2f}, rot={shoulder_rot:.2f}"
    else:
        left_wrist = get_keypoint(frame_data, LEFT_WRIST)
        right_wrist = get_keypoint(frame_data, RIGHT_WRIST)

        if not left_wrist or not right_wrist:
            return False, 0.0, ""

        wrist_diff = left_wrist[0] - right_wrist[0]
        l_wrist_offset = left_wrist[0] - midline

        if wrist_diff < -0.01 and l_wrist_offset > 0.02:
            confidence = min(1.0, (abs(wrist_diff) + l_wrist_offset) / 0.15)
            return True, confidence, f"backhand: L-R={wrist_diff:.2f}, L_off={l_wrist_offset:.2f}"

    return False, 0.0, ""


def classify_frame(frame_data, dominant_hand="right") -> tuple:
    """Classify a single frame using heuristics.

    Returns: (shot_type, confidence, trigger)
    Priority: serve > forehand > backhand > neutral
    """
    # Check serve first (highest priority)
    is_serve, conf, trigger = detect_serve(frame_data)
    if is_serve and conf > 0.3:
        return "serve", conf, trigger

    # Check forehand
    is_forehand, conf, trigger = detect_forehand(frame_data, dominant_hand)
    if is_forehand and conf > 0.3:
        return "forehand", conf, trigger

    # Check backhand
    is_backhand, conf, trigger = detect_backhand(frame_data, dominant_hand)
    if is_backhand and conf > 0.3:
        return "backhand", conf, trigger

    return "neutral", 0.5, "no shot detected"


# ─────────────────────────────────────────────────────────────
# Segment merging
# ─────────────────────────────────────────────────────────────

def merge_detections(frame_results: list, fps: float = 60.0,
                     min_duration_frames: int = 15,
                     max_gap_frames: int = 10) -> List[Detection]:
    """Merge consecutive same-label frames into segments.

    Args:
        frame_results: List of (shot_type, confidence, trigger) per frame
        fps: Frames per second
        min_duration_frames: Minimum segment length to keep
        max_gap_frames: Maximum gap to bridge within same shot type

    Returns:
        List of Detection objects
    """
    if not frame_results:
        return []

    segments = []
    current_type = frame_results[0][0]
    current_start = 0
    current_confidences = [frame_results[0][1]]
    current_trigger = frame_results[0][2]

    for i, (shot_type, conf, trigger) in enumerate(frame_results[1:], 1):
        if shot_type == current_type:
            current_confidences.append(conf)
        else:
            # End current segment
            if current_type != "neutral" and len(current_confidences) >= min_duration_frames:
                segments.append(Detection(
                    shot_type=current_type,
                    start_frame=current_start,
                    end_frame=i - 1,
                    confidence=sum(current_confidences) / len(current_confidences),
                    trigger=current_trigger,
                ))

            # Start new segment
            current_type = shot_type
            current_start = i
            current_confidences = [conf]
            current_trigger = trigger

    # Handle final segment
    if current_type != "neutral" and len(current_confidences) >= min_duration_frames:
        segments.append(Detection(
            shot_type=current_type,
            start_frame=current_start,
            end_frame=len(frame_results) - 1,
            confidence=sum(current_confidences) / len(current_confidences),
            trigger=current_trigger,
        ))

    # Merge nearby segments of same type
    merged = []
    for seg in segments:
        if merged and merged[-1].shot_type == seg.shot_type:
            gap = seg.start_frame - merged[-1].end_frame
            if gap <= max_gap_frames:
                # Merge
                merged[-1].end_frame = seg.end_frame
                merged[-1].confidence = (merged[-1].confidence + seg.confidence) / 2
                continue
        merged.append(seg)

    return merged


# ─────────────────────────────────────────────────────────────
# Pattern-based shot detection (biomechanics approach)
# ─────────────────────────────────────────────────────────────

# Stroke timing (frames at 60fps)
STROKE_WINDOW = 60       # 1 second to analyze full stroke
MIN_STROKE_GAP = 180     # 3 seconds minimum between shots (realistic rally pace)


def analyze_stroke_pattern(frames: List[dict], center_frame: int,
                           dominant_hand: str = "right", fps: float = 60.0) -> Tuple[str, float, str]:
    """
    Analyze frames around a potential contact point for stroke pattern.

    A real tennis stroke has this sequence over ~1 second:

    FOREHAND (right-handed):
      Phase 1 - Backswing (0.3-0.5s before contact):
        - Wrist moves to RIGHT (racket side)
        - Shoulders rotate (right shoulder back)
      Phase 2 - Forward swing (0.2s before contact):
        - Wrist accelerates LEFT
        - Hips/shoulders rotate forward
      Phase 3 - Contact:
        - Peak velocity
        - Wrist near or crossing midline
      Phase 4 - Follow-through (0.2-0.3s after):
        - Wrist continues LEFT, ends on opposite side
        - Arm wraps around

    BACKHAND is the mirror image.

    Returns (shot_type, confidence, description)
    """
    # Check pose confidence at contact point - reject if too low
    if center_frame < len(frames):
        pose_conf = get_pose_confidence(frames[center_frame])
        if pose_conf < 0.5:  # Need at least 50% visibility on key joints
            return "neutral", 0.0, f"low pose confidence ({pose_conf:.2f})"

    # Define analysis window
    backswing_start = max(0, center_frame - 30)  # 0.5s before
    backswing_end = max(0, center_frame - 10)    # 0.17s before
    contact_start = max(0, center_frame - 5)
    contact_end = min(len(frames) - 1, center_frame + 5)
    followthru_start = min(len(frames) - 1, center_frame + 10)
    followthru_end = min(len(frames) - 1, center_frame + 25)

    wrist_idx = RIGHT_WRIST if dominant_hand == "right" else LEFT_WRIST

    # Collect wrist x-positions relative to body midline for each phase
    def get_wrist_offsets(start, end):
        offsets = []
        for i in range(start, end + 1):
            if i >= len(frames):
                break
            wrist = get_keypoint(frames[i], wrist_idx)
            midline = body_midline_x(frames[i])
            if wrist and midline:
                offsets.append(wrist[0] - midline)
        return offsets

    backswing_offsets = get_wrist_offsets(backswing_start, backswing_end)
    contact_offsets = get_wrist_offsets(contact_start, contact_end)
    followthru_offsets = get_wrist_offsets(followthru_start, followthru_end)

    # Need data from all phases
    if not backswing_offsets or not contact_offsets:
        return "neutral", 0.0, "insufficient data"

    avg_backswing = mean(backswing_offsets)
    avg_contact = mean(contact_offsets)
    avg_followthru = mean(followthru_offsets) if followthru_offsets else avg_contact

    # Calculate the stroke trajectory
    # Positive offset = right of midline, negative = left of midline
    backswing_to_contact = avg_contact - avg_backswing
    contact_to_followthru = avg_followthru - avg_contact
    total_travel = avg_followthru - avg_backswing

    # Compute velocity at contact
    contact_velocities = []
    for i in range(contact_start + 1, contact_end + 1):
        if i >= len(frames):
            break
        w1 = get_keypoint(frames[i-1], wrist_idx)
        w2 = get_keypoint(frames[i], wrist_idx)
        if w1 and w2:
            dx = w2[0] - w1[0]
            dy = w2[1] - w1[1]
            vel = sqrt(dx*dx + dy*dy) * fps
            contact_velocities.append(vel)

    peak_velocity = max(contact_velocities) if contact_velocities else 0

    # ─────────────────────────────────────────────────────────
    # FOREHAND pattern (right-handed):
    # - Backswing: wrist on RIGHT (positive offset)
    # - Travel: moves LEFT (negative total_travel)
    # - Follow-through: wrist on LEFT or center
    # ─────────────────────────────────────────────────────────

    min_peak_vel = THRESHOLDS.get("peak_velocity_min", 10.0)

    if dominant_hand == "right":
        # FOREHAND: started right, ended left/center
        if (avg_backswing > 0.08 and      # Wrist clearly RIGHT during backswing (8cm+)
            total_travel < -0.12 and       # Moved significantly LEFT (12cm+)
            peak_velocity > min_peak_vel): # Had real velocity (10+ m/s)

            # Confidence based on how clear the pattern is
            conf = min(1.0, (
                (avg_backswing / 0.1) * 0.3 +           # Clearer backswing position
                (abs(total_travel) / 0.15) * 0.4 +     # Clearer travel distance
                (peak_velocity / 15.0) * 0.3           # Higher velocity
            ))

            return "forehand", conf, (
                f"backswing={avg_backswing:.2f}R, "
                f"travel={total_travel:.2f}, "
                f"vel={peak_velocity:.1f}"
            )

        # BACKHAND: started left, ended right/center
        if (avg_backswing < -0.08 and     # Wrist clearly LEFT during backswing (8cm+)
            total_travel > 0.12 and        # Moved significantly RIGHT (12cm+)
            peak_velocity > min_peak_vel): # Had real velocity (10+ m/s)

            conf = min(1.0, (
                (abs(avg_backswing) / 0.1) * 0.3 +
                (abs(total_travel) / 0.15) * 0.4 +
                (peak_velocity / 15.0) * 0.3
            ))

            return "backhand", conf, (
                f"backswing={avg_backswing:.2f}L, "
                f"travel={total_travel:.2f}, "
                f"vel={peak_velocity:.1f}"
            )
    else:
        # Left-handed: mirror image
        if (avg_backswing < -0.02 and total_travel > 0.05 and peak_velocity > min_peak_vel):
            conf = min(1.0, (abs(avg_backswing)/0.1)*0.3 + (total_travel/0.15)*0.4 + (peak_velocity/10.0)*0.3)
            return "forehand", conf, f"backswing={avg_backswing:.2f}, travel={total_travel:.2f}"

        if (avg_backswing > 0.02 and total_travel < -0.05 and peak_velocity > min_peak_vel):
            conf = min(1.0, (avg_backswing/0.1)*0.3 + (abs(total_travel)/0.15)*0.4 + (peak_velocity/10.0)*0.3)
            return "backhand", conf, f"backswing={avg_backswing:.2f}, travel={total_travel:.2f}"

    return "neutral", 0.0, f"no pattern: bs={avg_backswing:.2f}, travel={total_travel:.2f}, vel={peak_velocity:.1f}"


def detect_serve_pattern(frames: List[dict], center_frame: int, fps: float = 60.0) -> Tuple[bool, float, str]:
    """
    Detect serve by looking for arm-above-head pattern.

    Serve signature (over ~1.5 seconds):
    1. Ball toss + trophy position: arm rises ABOVE head
    2. Peak: wrist at maximum height (well above head)
    3. Swing down: rapid descent

    We look for frames where wrist was above head BEFORE the velocity spike.
    """
    # Check pose confidence at contact point
    if center_frame < len(frames):
        pose_conf = get_pose_confidence(frames[center_frame])
        if pose_conf < 0.5:
            return False, 0.0, f"low pose confidence ({pose_conf:.2f})"

    # Look 0.5-1.5 seconds before for trophy position
    search_start = max(0, center_frame - 90)  # 1.5s before
    search_end = max(0, center_frame - 20)    # 0.33s before

    max_height_above_head = 0.0
    trophy_frame = -1

    for i in range(search_start, search_end):
        if i >= len(frames):
            break

        # Check both wrists
        for wrist_idx in [RIGHT_WRIST, LEFT_WRIST]:
            wrist = get_keypoint(frames[i], wrist_idx)
            head_y = head_level_y(frames[i])

            if wrist and head_y:
                # In world coords, lower Y = higher position
                height = head_y - wrist[1]
                if height > max_height_above_head:
                    max_height_above_head = height
                    trophy_frame = i

    # Serve requires arm significantly above head (at least 25cm)
    if max_height_above_head < 0.25:
        return False, 0.0, ""

    # Check swing direction at contact — reduce confidence but don't reject
    # The velocity spike can fire on upswing or downswing depending on timing
    look_back = max(0, center_frame - 5)
    look_ahead = min(len(frames) - 1, center_frame + 5)

    wrist_before = get_keypoint(frames[look_back], RIGHT_WRIST)
    wrist_after = get_keypoint(frames[look_ahead], RIGHT_WRIST)

    swing_penalty = 0.0
    if wrist_before and wrist_after:
        dx = abs(wrist_after[0] - wrist_before[0])  # Horizontal movement
        dy = wrist_after[1] - wrist_before[1]       # Vertical movement (positive = downward)

        # Strong horizontal swing with weak trophy = likely groundstroke, not serve
        if dx > abs(dy) * 2.0 and max_height_above_head < 0.30:
            return False, 0.0, f"horizontal swing (dx={dx:.2f} > dy={dy:.2f}) with weak trophy - likely groundstroke"

        # Upswing at spike frame is common for serves — just reduce confidence slightly
        if dy < -0.05:
            swing_penalty = 0.15

    confidence = min(1.0, max_height_above_head / 0.35) - swing_penalty
    confidence = max(0.1, confidence)
    swing_desc = "downward swing" if swing_penalty == 0.0 else "upswing at spike"
    return True, confidence, f"trophy position: {max_height_above_head:.2f}m above head, {swing_desc}"


def detect_shots_pattern_based(frames: List[dict], fps: float = 60.0,
                                dominant_hand: str = "right") -> List[Detection]:
    """
    Detect shots by finding velocity spikes, then verifying stroke PATTERN.

    Two-stage approach:
    1. Find candidate contact points (velocity spikes)
    2. For each candidate, verify it matches a real stroke pattern

    This eliminates false positives from random arm movements.
    """
    wrist_idx = RIGHT_WRIST if dominant_hand == "right" else LEFT_WRIST

    # Stage 1: Find velocity spikes (candidate contact points)
    velocities = compute_wrist_velocities(frames, wrist_idx, fps)

    spikes = find_velocity_spikes(
        velocities,
        min_vel=THRESHOLDS["velocity_min"],
        spike_ratio=THRESHOLDS["velocity_spike_ratio"],
        min_gap=THRESHOLDS["min_shot_gap_frames"]
    )

    detections = []

    # Stage 2: Verify each spike has a real stroke pattern
    for spike_frame, velocity in spikes:
        # Check for serve first
        is_serve, serve_conf, serve_trigger = detect_serve_pattern(frames, spike_frame, fps)

        if is_serve and serve_conf > 0.4:
            detections.append(Detection(
                shot_type="serve",
                start_frame=max(0, spike_frame - 60),
                end_frame=min(len(frames)-1, spike_frame + 30),
                confidence=serve_conf,
                trigger=f"{serve_trigger}, vel={velocity:.1f}"
            ))
            continue

        # Check for groundstroke pattern
        shot_type, conf, trigger = analyze_stroke_pattern(
            frames, spike_frame, dominant_hand, fps
        )

        if shot_type != "neutral" and conf > 0.3:
            detections.append(Detection(
                shot_type=shot_type,
                start_frame=max(0, spike_frame - 30),
                end_frame=min(len(frames)-1, spike_frame + 20),
                confidence=conf,
                trigger=f"{trigger}, vel={velocity:.1f}"
            ))

    return detections


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def process_poses_file(poses_path: str, dominant_hand: str = "right",
                       method: str = "velocity") -> dict:
    """Process a poses JSON file and detect shots.

    Args:
        poses_path: Path to poses JSON
        dominant_hand: "left" or "right"
        method: "velocity" (recommended) or "frame" (legacy)

    Returns dict compatible with detect_shots.py output format.
    """
    with open(poses_path) as f:
        poses = json.load(f)

    fps = poses.get("fps", 60)
    frames = poses.get("frames", [])

    if not frames:
        return {
            "source_video": Path(poses_path).stem,
            "fps": fps,
            "total_frames": 0,
            "shot_counts": {},
            "segments": [],
            "detector": "heuristic",
        }

    # Detect shots
    if method == "velocity":
        detections = detect_shots_pattern_based(frames, fps, dominant_hand)
    else:
        # Legacy frame-by-frame method
        frame_results = []
        for frame in frames:
            if frame.get("keypoints") or frame.get("world_landmarks"):
                shot_type, conf, trigger = classify_frame(frame, dominant_hand)
                frame_results.append((shot_type, conf, trigger))
            else:
                frame_results.append(("neutral", 0.0, "no pose"))
        detections = merge_detections(frame_results, fps)

    # Build output
    segments = []
    shot_counts = {"forehand": 0, "backhand": 0, "serve": 0}

    for det in detections:
        shot_counts[det.shot_type] = shot_counts.get(det.shot_type, 0) + 1

        # Calculate clip extraction times with offsets
        offsets = CLIP_OFFSETS.get(det.shot_type, {"pre": 1.5, "post": 1.0})
        detection_time = det.start_frame / fps
        clip_start = max(0, detection_time - offsets["pre"])
        clip_end = detection_time + offsets["post"]

        segments.append({
            "shot_type": det.shot_type,
            "start_frame": det.start_frame,
            "end_frame": det.end_frame,
            "start_time": round(det.start_frame / fps, 2),
            "end_time": round(det.end_frame / fps, 2),
            "confidence": round(det.confidence, 3),
            "trigger": det.trigger,
            # Clip extraction bounds (with pre/post offsets)
            "clip_start": round(clip_start, 2),
            "clip_end": round(clip_end, 2),
        })

    return {
        "source_video": Path(poses_path).stem.replace("_poses", ""),
        "fps": fps,
        "total_frames": len(frames),
        "shot_counts": shot_counts,
        "segments": segments,
        "detector": "heuristic",
        "method": method,
        "dominant_hand": dominant_hand,
        "thresholds": THRESHOLDS,
        "clip_offsets": CLIP_OFFSETS,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Heuristic-based shot detection (no ML model)")
    parser.add_argument("poses_file", help="Path to poses JSON file")
    parser.add_argument("-o", "--output", help="Output JSON file")
    parser.add_argument("--dominant-hand", choices=["left", "right"],
                        default="right", help="Player's dominant hand")
    parser.add_argument("--method", choices=["velocity", "frame"],
                        default="velocity",
                        help="Detection method: velocity (recommended) or frame (legacy)")
    parser.add_argument("--visualize", action="store_true",
                        help="Print detection summary")
    args = parser.parse_args()

    if not os.path.exists(args.poses_file):
        print(f"Error: {args.poses_file} not found")
        sys.exit(1)

    # Process
    result = process_poses_file(args.poses_file, args.dominant_hand, args.method)

    # Output
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Wrote {args.output}")

    if args.visualize or not args.output:
        print(f"\n{'='*60}")
        print(f"HEURISTIC DETECTION RESULTS")
        print(f"{'='*60}")
        print(f"Source: {result['source_video']}")
        print(f"FPS: {result['fps']}, Total frames: {result['total_frames']}")
        print(f"Dominant hand: {result['dominant_hand']}, Method: {result['method']}")
        print(f"\nShot counts: {result['shot_counts']}")
        print(f"\nDetected {len(result['segments'])} shots:")
        print(f"{'─'*60}")

        for i, seg in enumerate(result["segments"][:30], 1):
            shot = seg['shot_type']
            clip_info = f"clip: {seg.get('clip_start', 0):.1f}s-{seg.get('clip_end', 0):.1f}s"
            print(f"  {i:2d}. {shot:10s} @ {seg['start_time']:6.1f}s  "
                  f"conf={seg['confidence']:.2f}  {clip_info}")
            if args.visualize:
                print(f"      trigger: {seg['trigger']}")

        if len(result["segments"]) > 30:
            print(f"  ... and {len(result['segments']) - 30} more")

        print(f"\nThresholds used: {result.get('thresholds', {})}")


if __name__ == "__main__":
    main()
