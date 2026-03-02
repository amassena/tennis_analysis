#!/usr/bin/env python3
"""Extract engineered features from labeled detections + pose data.

For each labeled shot (forehand/backhand/serve), extracts ~25 biomechanical
features from a 90-frame window centered on the detection frame.

Also extracts `not_shot` negatives from:
  1. Deleted detections (pre_fix_backup diff + IMG_0931 v1→v2 diff)
  2. Random idle frames from pose files (hard negatives)

Cleans contaminated labels: skips FH/BH from serve-only videos.

Usage:
    python scripts/extract_training_features.py
    python scripts/extract_training_features.py --output training/features_v2.json
"""

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import PROJECT_ROOT, POSES_DIR, TRAINING_DIR, MODELS_DIR

# Reuse helpers from heuristic_detect
from scripts.heuristic_detect import (
    get_keypoint, body_midline_x, head_level_y,
    compute_wrist_velocities,
    compute_wrist_acceleration, compute_wrist_jerk,
    RIGHT_WRIST, LEFT_WRIST, RIGHT_ELBOW, LEFT_ELBOW,
    RIGHT_SHOULDER, LEFT_SHOULDER, RIGHT_HIP, LEFT_HIP,
    LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE,
    NOSE,
)

DETECTIONS_DIR = os.path.join(PROJECT_ROOT, "detections")
BACKUP_DIR = os.path.join(DETECTIONS_DIR, "pre_fix_backup")
TRAINABLE_TYPES = {"forehand", "backhand", "serve"}
WINDOW_HALF = 45  # 45 frames each side = 90-frame window (1.5s at 60fps)

# Serve-only videos — FH/BH labels in these are heuristic artifacts
SERVE_ONLY_VIDEOS = {
    "IMG_0864", "IMG_0865", "IMG_0866", "IMG_0867",
    "IMG_0868", "IMG_0869", "IMG_0870",
}


def load_pose_frames(pose_path):
    """Load pose JSON, return (frames_list, fps, total_frames)."""
    with open(pose_path) as f:
        data = json.load(f)
    video_info = data.get("video_info", {})
    fps = video_info.get("fps", 60.0)
    raw_frames = data.get("frames", [])
    total_frames = video_info.get("total_frames", len(raw_frames))

    frames = [{}] * total_frames
    for fr in raw_frames:
        idx = fr.get("frame_idx", 0)
        if idx < total_frames:
            frames[idx] = fr
    return frames, fps, total_frames


def elbow_angle(frame, shoulder_idx, elbow_idx, wrist_idx):
    """Compute elbow angle in degrees (shoulder-elbow-wrist)."""
    s = get_keypoint(frame, shoulder_idx)
    e = get_keypoint(frame, elbow_idx)
    w = get_keypoint(frame, wrist_idx)
    if not s or not e or not w:
        return None
    # Vectors: elbow→shoulder, elbow→wrist
    v1 = [s[i] - e[i] for i in range(3)]
    v2 = [w[i] - e[i] for i in range(3)]
    dot = sum(a * b for a, b in zip(v1, v2))
    mag1 = math.sqrt(sum(a * a for a in v1))
    mag2 = math.sqrt(sum(a * a for a in v2))
    if mag1 < 1e-6 or mag2 < 1e-6:
        return None
    cos_angle = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    return math.degrees(math.acos(cos_angle))


def compute_pose_stability(frames, center_frame, fps, wrist_idx):
    """Compute pose stability: how much wrist moves in ±0.5s window.

    Low variance = idle/walking (not a shot). High variance = swinging.
    Returns standard deviation of wrist positions in the window.
    """
    n = len(frames)
    half_window = int(0.5 * fps)  # ±0.5s
    start = max(0, center_frame - half_window)
    end = min(n, center_frame + half_window + 1)

    positions = []
    for i in range(start, end):
        if not frames[i]:
            continue
        w = get_keypoint(frames[i], wrist_idx)
        if w is not None:
            positions.append(w[:3])

    if len(positions) < 3:
        return 0.0

    # Compute std dev of each axis, return mean
    means = [sum(p[j] for p in positions) / len(positions) for j in range(3)]
    variances = [
        sum((p[j] - means[j]) ** 2 for p in positions) / len(positions)
        for j in range(3)
    ]
    return round(math.sqrt(sum(variances) / 3), 5)


def compute_backswing_duration(frames, center_frame, fps, wrist_idx):
    """Estimate frames from neutral to max backswing before contact.

    Looks backwards from center_frame for when wrist velocity drops below
    a threshold, indicating the start of the backswing preparation.
    Returns duration in frames (0 if not detected).
    """
    n = len(frames)
    search_start = max(0, center_frame - int(1.0 * fps))  # look back up to 1s
    velocities = compute_wrist_velocities(
        frames[search_start:center_frame + 1], wrist_idx, fps
    )
    if len(velocities) < 5:
        return 0

    # Find where velocity first exceeds 5 m/s going backwards from contact
    threshold = 5.0
    backswing_start = 0
    for i in range(len(velocities) - 1, -1, -1):
        if velocities[i] < threshold:
            backswing_start = i
            break

    duration = len(velocities) - 1 - backswing_start
    return duration


def compute_angle_3points(p1, p2, p3):
    """Compute angle at p2 formed by p1-p2-p3, in degrees.

    Args:
        p1, p2, p3: 3D points as [x, y, z] lists
    Returns:
        Angle in degrees, or None if degenerate.
    """
    v1 = [p1[i] - p2[i] for i in range(3)]
    v2 = [p3[i] - p2[i] for i in range(3)]
    dot = sum(a * b for a, b in zip(v1, v2))
    mag1 = math.sqrt(sum(a * a for a in v1))
    mag2 = math.sqrt(sum(a * a for a in v2))
    if mag1 < 1e-6 or mag2 < 1e-6:
        return None
    cos_angle = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    return math.degrees(math.acos(cos_angle))


def _line_angle_xz(p1, p2):
    """Angle of line p1→p2 projected onto XZ plane, in radians."""
    dx = p2[0] - p1[0]
    dz = p2[2] - p1[2] if len(p2) > 2 and len(p1) > 2 else 0.0
    return math.atan2(dz, dx)


def extract_biomechanical_features(frames, center_frame, fps, dominant_hand="right"):
    """Extract ~16 biomechanical features from pose data around center_frame.

    Uses lower body, trunk rotation, arm kinematics, temporal dynamics,
    and phase-specific jerk features.

    Returns dict of feature_name → float.
    """
    import numpy as np
    n = len(frames)
    feats = {}

    # Hand-dependent indices
    if dominant_hand == "right":
        d_wrist, d_elbow, d_shoulder = RIGHT_WRIST, RIGHT_ELBOW, RIGHT_SHOULDER
        d_hip, d_knee, d_ankle = RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE
        nd_wrist, nd_elbow, nd_shoulder = LEFT_WRIST, LEFT_ELBOW, LEFT_SHOULDER
        nd_hip, nd_knee, nd_ankle = LEFT_HIP, LEFT_KNEE, LEFT_ANKLE
    else:
        d_wrist, d_elbow, d_shoulder = LEFT_WRIST, LEFT_ELBOW, LEFT_SHOULDER
        d_hip, d_knee, d_ankle = LEFT_HIP, LEFT_KNEE, LEFT_ANKLE
        nd_wrist, nd_elbow, nd_shoulder = RIGHT_WRIST, RIGHT_ELBOW, RIGHT_SHOULDER
        nd_hip, nd_knee, nd_ankle = RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE

    cf = min(center_frame, n - 1)

    # ── Lower Body (4 features) ──────────────────────────────────
    # knee_bend_at_contact: hip-knee-ankle angle at contact
    knee_bend = None
    if frames[cf]:
        hip_p = get_keypoint(frames[cf], d_hip)
        knee_p = get_keypoint(frames[cf], d_knee)
        ankle_p = get_keypoint(frames[cf], d_ankle)
        if hip_p and knee_p and ankle_p:
            knee_bend = compute_angle_3points(hip_p, knee_p, ankle_p)
    feats["knee_bend_at_contact"] = round(knee_bend, 2) if knee_bend else 160.0

    # min_knee_bend_window: minimum knee angle in ±15 frames
    min_knee = 180.0
    for i in range(max(0, cf - 15), min(n, cf + 16)):
        if not frames[i]:
            continue
        hp = get_keypoint(frames[i], d_hip)
        kp = get_keypoint(frames[i], d_knee)
        ap = get_keypoint(frames[i], d_ankle)
        if hp and kp and ap:
            ang = compute_angle_3points(hp, kp, ap)
            if ang is not None and ang < min_knee:
                min_knee = ang
    feats["min_knee_bend_window"] = round(min_knee, 2)

    # hip_rotation_speed: delta of hip-line angle in XZ plane across ±6 frames
    delta_frames = 6
    hip_rot_speed = 0.0
    f_before = max(0, cf - delta_frames)
    f_after = min(n - 1, cf + delta_frames)
    if frames[f_before] and frames[f_after]:
        lh_b = get_keypoint(frames[f_before], LEFT_HIP)
        rh_b = get_keypoint(frames[f_before], RIGHT_HIP)
        lh_a = get_keypoint(frames[f_after], LEFT_HIP)
        rh_a = get_keypoint(frames[f_after], RIGHT_HIP)
        if lh_b and rh_b and lh_a and rh_a:
            angle_before = _line_angle_xz(lh_b, rh_b)
            angle_after = _line_angle_xz(lh_a, rh_a)
            dt = (f_after - f_before) / fps
            if dt > 0:
                hip_rot_speed = abs(angle_after - angle_before) / dt
    feats["hip_rotation_speed"] = round(hip_rot_speed, 3)

    # stance_width_contact: L_ANKLE to R_ANKLE XZ distance
    stance_width = 0.0
    if frames[cf]:
        la = get_keypoint(frames[cf], LEFT_ANKLE)
        ra = get_keypoint(frames[cf], RIGHT_ANKLE)
        if la and ra:
            dx = la[0] - ra[0]
            dz = (la[2] - ra[2]) if len(la) > 2 and len(ra) > 2 else 0.0
            stance_width = math.sqrt(dx * dx + dz * dz)
    feats["stance_width_contact"] = round(stance_width, 5)

    # ── Trunk Rotation (3 features) ──────────────────────────────
    # shoulder_hip_differential: shoulder vs hip line angle in XZ at contact
    sh_diff = 0.0
    if frames[cf]:
        ls = get_keypoint(frames[cf], LEFT_SHOULDER)
        rs = get_keypoint(frames[cf], RIGHT_SHOULDER)
        lh = get_keypoint(frames[cf], LEFT_HIP)
        rh = get_keypoint(frames[cf], RIGHT_HIP)
        if ls and rs and lh and rh:
            sh_angle = _line_angle_xz(ls, rs)
            hip_angle = _line_angle_xz(lh, rh)
            sh_diff = sh_angle - hip_angle
    feats["shoulder_hip_differential"] = round(sh_diff, 5)

    # trunk_angular_velocity: delta across ±6 frames
    trunk_vel = 0.0
    if frames[f_before] and frames[f_after]:
        ls_b = get_keypoint(frames[f_before], LEFT_SHOULDER)
        rs_b = get_keypoint(frames[f_before], RIGHT_SHOULDER)
        lh_b2 = get_keypoint(frames[f_before], LEFT_HIP)
        rh_b2 = get_keypoint(frames[f_before], RIGHT_HIP)
        ls_a = get_keypoint(frames[f_after], LEFT_SHOULDER)
        rs_a = get_keypoint(frames[f_after], RIGHT_SHOULDER)
        lh_a2 = get_keypoint(frames[f_after], LEFT_HIP)
        rh_a2 = get_keypoint(frames[f_after], RIGHT_HIP)
        if all([ls_b, rs_b, lh_b2, rh_b2, ls_a, rs_a, lh_a2, rh_a2]):
            diff_before = _line_angle_xz(ls_b, rs_b) - _line_angle_xz(lh_b2, rh_b2)
            diff_after = _line_angle_xz(ls_a, rs_a) - _line_angle_xz(lh_a2, rh_a2)
            dt = (f_after - f_before) / fps
            if dt > 0:
                trunk_vel = abs(diff_after - diff_before) / dt
    feats["trunk_angular_velocity"] = round(trunk_vel, 3)

    # peak_trunk_rotation_window: max absolute differential in 90-frame window
    peak_trunk = 0.0
    for i in range(max(0, cf - 45), min(n, cf + 46)):
        if not frames[i]:
            continue
        ls_i = get_keypoint(frames[i], LEFT_SHOULDER)
        rs_i = get_keypoint(frames[i], RIGHT_SHOULDER)
        lh_i = get_keypoint(frames[i], LEFT_HIP)
        rh_i = get_keypoint(frames[i], RIGHT_HIP)
        if ls_i and rs_i and lh_i and rh_i:
            diff_i = abs(_line_angle_xz(ls_i, rs_i) - _line_angle_xz(lh_i, rh_i))
            if diff_i > peak_trunk:
                peak_trunk = diff_i
    feats["peak_trunk_rotation_window"] = round(peak_trunk, 5)

    # ── Arm Kinematics (3 features) ──────────────────────────────
    # elbow_extension_speed: delta of elbow angle across ±6 frames
    elbow_ext_speed = 0.0
    if frames[f_before] and frames[f_after]:
        ea_before = None
        s_b = get_keypoint(frames[f_before], d_shoulder)
        e_b = get_keypoint(frames[f_before], d_elbow)
        w_b = get_keypoint(frames[f_before], d_wrist)
        if s_b and e_b and w_b:
            ea_before = compute_angle_3points(s_b, e_b, w_b)

        ea_after = None
        s_a = get_keypoint(frames[f_after], d_shoulder)
        e_a = get_keypoint(frames[f_after], d_elbow)
        w_a = get_keypoint(frames[f_after], d_wrist)
        if s_a and e_a and w_a:
            ea_after = compute_angle_3points(s_a, e_a, w_a)

        if ea_before is not None and ea_after is not None:
            dt = (f_after - f_before) / fps
            if dt > 0:
                elbow_ext_speed = abs(ea_after - ea_before) / dt
    feats["elbow_extension_speed"] = round(elbow_ext_speed, 2)

    # shoulder_elevation_at_contact: dominant shoulder Y relative to hip Y
    shoulder_elev = 0.0
    if frames[cf]:
        ds = get_keypoint(frames[cf], d_shoulder)
        dh = get_keypoint(frames[cf], d_hip)
        if ds and dh:
            shoulder_elev = dh[1] - ds[1]  # positive = shoulder higher (lower Y)
    feats["shoulder_elevation_at_contact"] = round(shoulder_elev, 5)

    # arm_abduction_angle: angle between shoulder→elbow and spine axis
    arm_abd = 0.0
    if frames[cf]:
        ds2 = get_keypoint(frames[cf], d_shoulder)
        de = get_keypoint(frames[cf], d_elbow)
        dh2 = get_keypoint(frames[cf], d_hip)
        if ds2 and de and dh2:
            # spine axis: hip → shoulder
            spine = [ds2[i] - dh2[i] for i in range(3)]
            # arm: shoulder → elbow
            arm = [de[i] - ds2[i] for i in range(3)]
            dot = sum(a * b for a, b in zip(spine, arm))
            mag_s = math.sqrt(sum(a * a for a in spine))
            mag_a = math.sqrt(sum(a * a for a in arm))
            if mag_s > 1e-6 and mag_a > 1e-6:
                cos_a = max(-1.0, min(1.0, dot / (mag_s * mag_a)))
                arm_abd = math.degrees(math.acos(cos_a))
    feats["arm_abduction_angle"] = round(arm_abd, 2)

    # ── Temporal Dynamics (4 features) ────────────────────────────
    # Need velocities for this section
    win_start = max(0, cf - WINDOW_HALF)
    win_end = min(n, cf + WINDOW_HALF + 1)
    window_frames = frames[win_start:win_end]
    velocities = compute_wrist_velocities(window_frames, d_wrist, fps)
    center_in_window = cf - win_start

    # followthrough_duration: frames from contact until velocity < 5 m/s
    ft_dur = 0
    if velocities:
        for i in range(min(center_in_window, len(velocities) - 1), len(velocities)):
            if velocities[i] < 5.0:
                ft_dur = i - center_in_window
                break
        else:
            ft_dur = len(velocities) - center_in_window
    feats["followthrough_duration"] = ft_dur

    # deceleration_rate: (contact_vel - vel_at_+12) / (12/fps)
    contact_vel = velocities[min(center_in_window, len(velocities) - 1)] if velocities else 0.0
    post_12_idx = min(center_in_window + 12, len(velocities) - 1) if velocities else 0
    vel_at_12 = velocities[post_12_idx] if velocities else 0.0
    dt_12 = 12.0 / fps
    decel = (contact_vel - vel_at_12) / dt_12 if dt_12 > 0 else 0.0
    feats["deceleration_rate"] = round(decel, 2)

    # velocity_asymmetry: (peak_pre - peak_post) / peak_velocity
    peak_vel = max(velocities) if velocities else 0.01
    pre_vels = velocities[:min(center_in_window + 1, len(velocities))] if velocities else [0]
    post_vels = velocities[min(center_in_window, len(velocities) - 1):] if velocities else [0]
    peak_pre = max(pre_vels) if pre_vels else 0.0
    peak_post = max(post_vels) if post_vels else 0.0
    vel_asym = (peak_pre - peak_post) / peak_vel if peak_vel > 0.01 else 0.0
    feats["velocity_asymmetry"] = round(vel_asym, 5)

    # backswing_to_contact_ratio
    backswing_dur = compute_backswing_duration(frames, cf, fps, d_wrist)
    # Total swing: backswing + followthrough
    total_swing = backswing_dur + ft_dur
    bs_ratio = backswing_dur / total_swing if total_swing > 0 else 0.5
    feats["backswing_to_contact_ratio"] = round(bs_ratio, 5)

    # ── Phase-Specific Jerk (2 features) ──────────────────────────
    if len(velocities) > 4:
        accel = compute_wrist_acceleration(velocities, fps, smooth_window=5)
        jerk_arr = compute_wrist_jerk(accel, fps, smooth_window=3)

        # jerk_at_backswing_peak: jerk at max backswing offset frame
        bs_start_idx = max(0, center_in_window - 30)
        bs_end_idx = max(0, center_in_window - 10)
        if bs_end_idx > bs_start_idx and len(jerk_arr) > 0:
            # Find frame with max wrist offset in backswing phase
            max_off_frame = bs_start_idx
            max_off_val = -999
            for i in range(bs_start_idx, min(bs_end_idx + 1, len(window_frames))):
                if not window_frames[i]:
                    continue
                w = get_keypoint(window_frames[i], d_wrist)
                mid = body_midline_x(window_frames[i])
                if w is not None and mid is not None:
                    off = abs(w[0] - mid)
                    if off > max_off_val:
                        max_off_val = off
                        max_off_frame = i
            jerk_bs_idx = max(0, min(max_off_frame - 2, len(jerk_arr) - 1))
            feats["jerk_at_backswing_peak"] = round(float(jerk_arr[jerk_bs_idx]), 1)
        else:
            feats["jerk_at_backswing_peak"] = 0.0

        # jerk_at_followthrough: jerk at peak follow-through frame
        ft_start_idx = min(center_in_window + 10, len(window_frames) - 1)
        ft_end_idx = min(center_in_window + 25, len(window_frames) - 1)
        if ft_end_idx > ft_start_idx and len(jerk_arr) > 0:
            max_ft_frame = ft_start_idx
            max_ft_vel = -1
            for i in range(ft_start_idx, min(ft_end_idx + 1, len(velocities))):
                if velocities[i] > max_ft_vel:
                    max_ft_vel = velocities[i]
                    max_ft_frame = i
            jerk_ft_idx = max(0, min(max_ft_frame - 2, len(jerk_arr) - 1))
            feats["jerk_at_followthrough"] = round(float(jerk_arr[jerk_ft_idx]), 1)
        else:
            feats["jerk_at_followthrough"] = 0.0
    else:
        feats["jerk_at_backswing_peak"] = 0.0
        feats["jerk_at_followthrough"] = 0.0

    return feats


def extract_features(frames, center_frame, fps, dominant_hand="right",
                     detection_meta=None, rally_context=None):
    """Extract ~25 features from a 90-frame window around center_frame.

    Args:
        frames: List of pose frame dicts
        center_frame: Frame index of the detection
        fps: Video FPS
        dominant_hand: "left" or "right"
        detection_meta: Optional dict with source, audio_amplitude, velocity
                        from the detection JSON (for discriminative features)

    Returns dict of feature_name → float, or None if insufficient data.
    """
    n = len(frames)
    if center_frame >= n:
        return None

    wrist_idx = RIGHT_WRIST if dominant_hand == "right" else LEFT_WRIST
    nd_wrist_idx = LEFT_WRIST if dominant_hand == "right" else RIGHT_WRIST
    elbow_idx = RIGHT_ELBOW if dominant_hand == "right" else LEFT_ELBOW
    shoulder_idx = RIGHT_SHOULDER if dominant_hand == "right" else LEFT_SHOULDER
    nd_shoulder_idx = LEFT_SHOULDER if dominant_hand == "right" else RIGHT_SHOULDER

    # Define phases
    win_start = max(0, center_frame - WINDOW_HALF)
    win_end = min(n - 1, center_frame + WINDOW_HALF)

    bs_start = max(0, center_frame - 30)   # backswing: 0.5s before
    bs_end = max(0, center_frame - 10)     # to 0.17s before
    ct_start = max(0, center_frame - 5)    # contact: ±0.08s
    ct_end = min(n - 1, center_frame + 5)
    ft_start = min(n - 1, center_frame + 10)  # follow-through: +0.17s to +0.42s
    ft_end = min(n - 1, center_frame + 25)

    # --- Collect wrist offsets relative to midline per phase ---
    def wrist_offsets(start, end):
        offsets = []
        for i in range(start, end + 1):
            if i >= n or not frames[i]:
                continue
            w = get_keypoint(frames[i], wrist_idx)
            mid = body_midline_x(frames[i])
            if w is not None and mid is not None:
                offsets.append(w[0] - mid)
        return offsets

    bs_off = wrist_offsets(bs_start, bs_end)
    ct_off = wrist_offsets(ct_start, ct_end)
    ft_off = wrist_offsets(ft_start, ft_end)

    # Need at least contact data
    if not ct_off:
        return None

    avg_bs = sum(bs_off) / len(bs_off) if bs_off else 0.0
    avg_ct = sum(ct_off) / len(ct_off)
    avg_ft = sum(ft_off) / len(ft_off) if ft_off else avg_ct

    # --- Velocity features ---
    vel_start = max(0, center_frame - WINDOW_HALF)
    vel_end = min(n, center_frame + WINDOW_HALF + 1)
    window_frames = frames[vel_start:vel_end]
    velocities = compute_wrist_velocities(window_frames, wrist_idx, fps)
    peak_vel = max(velocities) if velocities else 0.0
    center_in_window = center_frame - vel_start
    contact_vel = velocities[min(center_in_window, len(velocities) - 1)] if velocities else 0.0
    avg_vel = sum(velocities) / len(velocities) if velocities else 0.0

    # --- Jerk / acceleration features ---
    import numpy as np
    if len(velocities) > 4:
        accel = compute_wrist_acceleration(velocities, fps, smooth_window=5)
        jerk_arr = compute_wrist_jerk(accel, fps, smooth_window=3)

        # Peak jerk in the window
        peak_jerk = float(np.max(jerk_arr)) if len(jerk_arr) > 0 else 0.0

        # Jerk at contact frame (mapped into the window)
        jerk_contact_idx = max(0, min(center_in_window - 2, len(jerk_arr) - 1))
        contact_jerk = float(jerk_arr[jerk_contact_idx]) if len(jerk_arr) > 0 else 0.0

        # Peak acceleration in the window
        peak_accel = float(np.max(np.abs(accel))) if len(accel) > 0 else 0.0

        # Acceleration at contact
        accel_contact_idx = max(0, min(center_in_window - 1, len(accel) - 1))
        contact_accel = float(np.abs(accel[accel_contact_idx])) if len(accel) > 0 else 0.0

        # Velocity change ratio: how much does velocity change around contact?
        # (captures the deceleration→acceleration "snap")
        pre_idx = max(0, center_in_window - 6)
        post_idx = min(len(velocities) - 1, center_in_window + 6)
        pre_vel = velocities[pre_idx]
        post_vel = velocities[post_idx]
        vel_change = abs(contact_vel - min(pre_vel, post_vel))
    else:
        peak_jerk = 0.0
        contact_jerk = 0.0
        peak_accel = 0.0
        contact_accel = 0.0
        vel_change = 0.0

    # --- Height features ---
    max_above_head = -999.0
    max_above_hip = -999.0
    contact_above_hip = None

    for i in range(win_start, win_end + 1):
        if not frames[i]:
            continue
        w = get_keypoint(frames[i], wrist_idx)
        if w is None:
            continue

        # Above head
        hy = head_level_y(frames[i])
        if hy is not None:
            above_head = hy - w[1]  # positive = wrist higher
            if above_head > max_above_head:
                max_above_head = above_head

        # Above hip
        lh = get_keypoint(frames[i], LEFT_HIP)
        rh = get_keypoint(frames[i], RIGHT_HIP)
        if lh and rh:
            hip_y = (lh[1] + rh[1]) / 2
            above_hip = hip_y - w[1]
            if above_hip > max_above_hip:
                max_above_hip = above_hip
            if ct_start <= i <= ct_end and contact_above_hip is None:
                contact_above_hip = above_hip

    if max_above_head < -900:
        max_above_head = 0.0
    if max_above_hip < -900:
        max_above_hip = 0.0
    if contact_above_hip is None:
        contact_above_hip = max_above_hip

    # --- Contact frame wrist absolute Y ---
    contact_wrist_y = 0.0
    w = get_keypoint(frames[center_frame], wrist_idx) if frames[center_frame] else None
    if w is not None:
        contact_wrist_y = w[1]

    # --- Elbow angle at contact ---
    ea = elbow_angle(frames[center_frame], shoulder_idx, elbow_idx, wrist_idx) if frames[center_frame] else None
    elbow_ang = ea if ea is not None else 90.0  # default to neutral

    # --- Shoulder rotation ---
    ls = get_keypoint(frames[center_frame], LEFT_SHOULDER) if frames[center_frame] else None
    rs = get_keypoint(frames[center_frame], RIGHT_SHOULDER) if frames[center_frame] else None
    shoulder_rot = (ls[0] - rs[0]) if (ls and rs) else 0.0

    # --- Non-dominant wrist position (two-handed BH indicator) ---
    nd_offsets = []
    for i in range(ct_start, ct_end + 1):
        if i >= n or not frames[i]:
            continue
        ndw = get_keypoint(frames[i], nd_wrist_idx)
        mid = body_midline_x(frames[i])
        if ndw is not None and mid is not None:
            nd_offsets.append(ndw[0] - mid)
    nd_wrist_offset = sum(nd_offsets) / len(nd_offsets) if nd_offsets else 0.0

    # --- Wrist separation at contact ---
    dw = get_keypoint(frames[center_frame], wrist_idx) if frames[center_frame] else None
    ndw = get_keypoint(frames[center_frame], nd_wrist_idx) if frames[center_frame] else None
    if dw and ndw:
        wrist_sep = math.sqrt(sum((a - b) ** 2 for a, b in zip(dw[:3], ndw[:3])))
    else:
        wrist_sep = 0.3  # default

    # --- Max wrist X range in window (total lateral movement) ---
    all_wx = []
    for i in range(win_start, win_end + 1):
        if not frames[i]:
            continue
        w = get_keypoint(frames[i], wrist_idx)
        mid = body_midline_x(frames[i])
        if w is not None and mid is not None:
            all_wx.append(w[0] - mid)
    max_x_offset = max(abs(x) for x in all_wx) if all_wx else 0.0
    x_range = (max(all_wx) - min(all_wx)) if all_wx else 0.0

    # --- Wrist vertical velocity at contact (upswing vs downswing) ---
    vert_vel = 0.0
    if center_frame > 0 and center_frame < n:
        w_before = get_keypoint(frames[max(0, center_frame - 3)], wrist_idx) if frames[max(0, center_frame - 3)] else None
        w_after = get_keypoint(frames[min(n - 1, center_frame + 3)], wrist_idx) if frames[min(n - 1, center_frame + 3)] else None
        if w_before and w_after:
            vert_vel = (w_before[1] - w_after[1]) * fps / 6  # positive = moving up

    # --- NEW: Discriminative features from detection metadata ---
    meta = detection_meta or {}

    # Was there a velocity spike at this detection? (audio_only FPs lack this)
    source = meta.get("source", "")
    has_velocity_spike = 1.0 if source in ("audio+heuristic", "heuristic_only") else 0.0

    # Audio amplitude (louder = more likely real hit)
    audio_amplitude = meta.get("audio_amplitude", 0.0) or 0.0

    # Velocity ratio: contact_velocity / peak_velocity
    # Real shots peak at contact, FPs don't
    velocity_ratio = round(contact_vel / peak_vel, 5) if peak_vel > 0.01 else 0.0

    # Pose stability: how much does pose change in ±0.5s?
    pose_stability = compute_pose_stability(frames, center_frame, fps, wrist_idx)

    # Backswing duration: frames from neutral to max backswing
    backswing_dur = compute_backswing_duration(frames, center_frame, fps, wrist_idx)

    features = {
        # Spatial (relative to midline)
        "avg_backswing_offset": round(avg_bs, 5),
        "avg_contact_offset": round(avg_ct, 5),
        "avg_followthru_offset": round(avg_ft, 5),
        "total_x_travel": round(avg_ft - avg_bs, 5),
        "backswing_to_contact": round(avg_ct - avg_bs, 5),
        "contact_to_followthru": round(avg_ft - avg_ct, 5),
        "max_x_offset": round(max_x_offset, 5),
        "x_range": round(x_range, 5),
        # Height
        "max_wrist_above_head": round(max_above_head, 5),
        "wrist_above_hip_contact": round(contact_above_hip, 5),
        "max_wrist_above_hip": round(max_above_hip, 5),
        "contact_wrist_y": round(contact_wrist_y, 5),
        # Velocity
        "peak_velocity": round(peak_vel, 3),
        "contact_velocity": round(contact_vel, 3),
        "avg_velocity": round(avg_vel, 3),
        "vertical_velocity_contact": round(vert_vel, 3),
        # Arm geometry
        "elbow_angle_contact": round(elbow_ang, 2),
        "shoulder_rotation": round(shoulder_rot, 5),
        # Non-dominant hand
        "nd_wrist_offset": round(nd_wrist_offset, 5),
        "wrist_separation": round(wrist_sep, 5),
        # Jerk / acceleration
        "peak_jerk": round(peak_jerk, 1),
        "contact_jerk": round(contact_jerk, 1),
        "peak_acceleration": round(peak_accel, 1),
        "contact_acceleration": round(contact_accel, 1),
        "vel_change_around_contact": round(vel_change, 3),
        # Discriminative features
        "source_has_velocity_spike": has_velocity_spike,
        "audio_amplitude": round(audio_amplitude, 6),
        "velocity_ratio": round(velocity_ratio, 5),
        "pose_stability": pose_stability,
        "backswing_duration": backswing_dur,
    }

    # ── Biomechanical features (Phase 1) ──────────────────────────
    biomech = extract_biomechanical_features(frames, center_frame, fps, dominant_hand)
    features.update(biomech)

    # ── Rally context features (Phase 2) ──────────────────────────
    rc = rally_context or {}
    features["shot_number_in_rally"] = rc.get("shot_number_in_rally", 0.0)
    features["rally_length"] = rc.get("rally_length", 0.0)
    features["time_since_last_shot"] = rc.get("time_since_last_shot", 0.0)
    features["rolling_inter_shot_interval"] = rc.get("rolling_inter_shot_interval", 0.0)
    features["prev_shot_type_serve"] = rc.get("prev_shot_type_serve", 0.0)
    features["prev_shot_type_same"] = rc.get("prev_shot_type_same", 0.0)
    features["rally_stage"] = rc.get("rally_stage", 0.0)

    return features


def compute_rally_context(detections, current_idx):
    """Compute rally context features for a detection at current_idx.

    Rally = consecutive shots within 6s of each other.

    Args:
        detections: Full list of detection dicts (sorted by timestamp)
        current_idx: Index of the current detection in the list

    Returns dict with rally context features.
    """
    MAX_RALLY_GAP = 6.0  # seconds between consecutive shots to be in same rally

    if not detections or current_idx >= len(detections):
        return {}

    timestamps = [d.get("timestamp", 0) for d in detections]
    shot_types = [d.get("shot_type", "unknown") for d in detections]
    current_t = timestamps[current_idx]

    # Find rally boundaries: walk backward and forward from current_idx
    rally_start = current_idx
    while rally_start > 0 and (timestamps[rally_start] - timestamps[rally_start - 1]) < MAX_RALLY_GAP:
        rally_start -= 1

    rally_end = current_idx
    while rally_end < len(detections) - 1 and (timestamps[rally_end + 1] - timestamps[rally_end]) < MAX_RALLY_GAP:
        rally_end += 1

    rally_length = rally_end - rally_start + 1
    shot_number = current_idx - rally_start + 1  # 1-indexed

    # Time since last shot
    if current_idx > 0:
        time_since_last = min(current_t - timestamps[current_idx - 1], 30.0)
    else:
        time_since_last = 0.0

    # Rolling inter-shot interval (median of last 3)
    intervals = []
    for i in range(max(1, current_idx - 2), current_idx + 1):
        if i > 0:
            intervals.append(timestamps[i] - timestamps[i - 1])
    rolling_isi = sorted(intervals)[len(intervals) // 2] if intervals else 0.0

    # Previous shot type features
    prev_is_serve = 0.0
    prev_same_type = 0.0
    if current_idx > 0:
        prev_type = shot_types[current_idx - 1]
        if prev_type == "serve":
            prev_is_serve = 1.0
        if prev_type == shot_types[current_idx]:
            prev_same_type = 1.0

    # Rally stage: position / length (0-1)
    rally_stage = shot_number / rally_length if rally_length > 0 else 0.0

    return {
        "shot_number_in_rally": float(shot_number),
        "rally_length": float(rally_length),
        "time_since_last_shot": round(time_since_last, 3),
        "rolling_inter_shot_interval": round(rolling_isi, 3),
        "prev_shot_type_serve": prev_is_serve,
        "prev_shot_type_same": prev_same_type,
        "rally_stage": round(rally_stage, 3),
    }


def find_deleted_detections(current_dir, backup_dir):
    """Find detections that exist in backup but not in current (user-deleted FPs).

    Compares by timestamp with 0.5s tolerance.
    Returns list of (video_name, detection_dict) tuples.
    """
    deleted = []

    if not os.path.isdir(backup_dir):
        print(f"  Backup dir not found: {backup_dir}")
        return deleted

    for backup_file in sorted(os.listdir(backup_dir)):
        if not backup_file.endswith("_fused.json"):
            continue

        current_file = os.path.join(current_dir, backup_file)
        if not os.path.exists(current_file):
            continue

        with open(os.path.join(backup_dir, backup_file)) as f:
            backup_data = json.load(f)
        with open(current_file) as f:
            current_data = json.load(f)

        video_name = backup_data.get("source_video", backup_file.replace("_fused.json", ""))
        current_times = {d["timestamp"] for d in current_data.get("detections", [])}

        for det in backup_data.get("detections", []):
            t = det["timestamp"]
            # Check if this detection was removed (not within 0.5s of any current)
            if not any(abs(t - ct) < 0.5 for ct in current_times):
                deleted.append((video_name, det))

    return deleted


def find_v1_v2_deleted(det_dir):
    """Find detections deleted between IMG_0931 v1 and v2.

    v2 is the user-reviewed version — detections in v1 but not v2 are FPs.
    """
    deleted = []
    v1_path = os.path.join(det_dir, "IMG_0931_fused.json")
    v2_path = os.path.join(det_dir, "IMG_0931_fused_v2.json")

    if not os.path.exists(v1_path) or not os.path.exists(v2_path):
        return deleted

    with open(v1_path) as f:
        v1_data = json.load(f)
    with open(v2_path) as f:
        v2_data = json.load(f)

    v2_times = {d["timestamp"] for d in v2_data.get("detections", [])}

    for det in v1_data.get("detections", []):
        t = det["timestamp"]
        if not any(abs(t - vt) < 0.5 for vt in v2_times):
            deleted.append(("IMG_0931", det))

    return deleted


def sample_random_negatives(poses_dir, det_dir, n_samples=100, seed=42):
    """Sample random frames from pose files where no detection exists.

    These are hard negatives: idle, walking, ball toss without contact, etc.
    Returns list of (video_name, frame_idx, dominant_hand) tuples.
    """
    rng = random.Random(seed)
    negatives = []

    # Collect all detection timestamps per video for exclusion
    det_times_by_video = {}
    for det_file in os.listdir(det_dir):
        if not det_file.endswith("_fused.json"):
            continue
        # Skip versioned/backup files
        if "_v2" in det_file or "_ml" in det_file:
            continue
        det_path = os.path.join(det_dir, det_file)
        with open(det_path) as f:
            det_data = json.load(f)
        video_name = det_data.get("source_video", det_file.replace("_fused.json", ""))
        det_times_by_video[video_name] = [
            d["timestamp"] for d in det_data.get("detections", [])
        ]

    # Sample from pose files that have corresponding detections
    pose_files = [f for f in os.listdir(poses_dir) if f.endswith(".json")]
    candidates = []

    for pose_file in pose_files:
        video_name = pose_file.replace(".json", "")
        if video_name not in det_times_by_video:
            continue

        pose_path = os.path.join(poses_dir, pose_file)
        with open(pose_path) as f:
            data = json.load(f)
        video_info = data.get("video_info", {})
        fps = video_info.get("fps", 60.0)
        total_frames = video_info.get("total_frames", 0)
        det_times = det_times_by_video[video_name]

        if total_frames < 200:
            continue

        # Sample candidate frames at 2s intervals, excluding detection zones
        for frame_idx in range(90, total_frames - 90, int(fps * 2)):
            t = frame_idx / fps
            # Must be at least 3s from any detection
            if any(abs(t - dt) < 3.0 for dt in det_times):
                continue
            candidates.append((video_name, frame_idx))

    rng.shuffle(candidates)
    return candidates[:n_samples]


def main():
    parser = argparse.ArgumentParser(
        description="Extract training features from labeled detections")
    parser.add_argument("--detections-dir", default=DETECTIONS_DIR,
                        help="Directory with *_fused.json files")
    parser.add_argument("--poses-dir", default=POSES_DIR,
                        help="Directory with pose JSON files")
    parser.add_argument("--output", default=os.path.join(TRAINING_DIR, "features_v4.json"),
                        help="Output features JSON path")
    parser.add_argument("--n-negatives", type=int, default=100,
                        help="Number of random hard negatives to sample (default: 100)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for negative sampling (default: 42)")
    args = parser.parse_args()

    det_dir = args.detections_dir
    poses_dir = args.poses_dir

    # Find all detection files (skip versioned/variant files)
    det_files = sorted(
        f for f in os.listdir(det_dir)
        if f.endswith("_fused.json")
        and "_v2" not in f
        and "_ml" not in f
    )
    if not det_files:
        print(f"No *_fused.json files found in {det_dir}")
        sys.exit(1)

    print(f"Found {len(det_files)} detection files")

    # ── Phase 1: Extract positive samples (serve/FH/BH) ──────────
    print(f"\n{'─'*50}")
    print(f"PHASE 1: Positive samples")
    print(f"{'─'*50}")

    all_samples = []
    skipped_types = {}
    skipped_contaminated = 0
    video_stats = {}
    pose_cache = {}  # video_name → (frames, fps, total_frames)

    for det_file in det_files:
        det_path = os.path.join(det_dir, det_file)
        with open(det_path) as f:
            det_data = json.load(f)

        video_name = det_data.get("source_video", det_file.replace("_fused.json", ""))
        dominant_hand = det_data.get("dominant_hand", "right")
        video_meta = det_data.get("video_metadata", {})
        camera_angle = video_meta.get("camera_angle", None)
        is_serve_only = video_name in SERVE_ONLY_VIDEOS

        # Count trainable shots
        trainable = [
            d for d in det_data.get("detections", [])
            if d["shot_type"] in TRAINABLE_TYPES
        ]
        if not trainable:
            print(f"  {video_name}: 0 trainable shots, skipping")
            continue

        # Find pose file
        pose_path = os.path.join(poses_dir, video_name + ".json")
        if not os.path.exists(pose_path):
            print(f"  {video_name}: pose file not found, skipping")
            continue

        print(f"  {video_name}: {len(trainable)} trainable shots, loading poses...")
        frames, fps, total_frames = load_pose_frames(pose_path)
        pose_cache[video_name] = (frames, fps, total_frames, dominant_hand)

        extracted = 0
        failed = 0
        contaminated = 0
        all_dets = det_data.get("detections", [])
        for det_idx, det in enumerate(all_dets):
            shot_type = det["shot_type"]
            if shot_type not in TRAINABLE_TYPES:
                skipped_types[shot_type] = skipped_types.get(shot_type, 0) + 1
                continue

            # Clean contaminated labels: skip FH/BH from serve-only videos
            if is_serve_only and shot_type in ("forehand", "backhand"):
                contaminated += 1
                skipped_contaminated += 1
                continue

            frame_idx = det.get("frame", 0)
            rally_ctx = compute_rally_context(all_dets, det_idx)
            features = extract_features(
                frames, frame_idx, fps, dominant_hand,
                detection_meta=det,
                rally_context=rally_ctx,
            )
            if features is None:
                failed += 1
                continue

            sample = {
                "video": video_name,
                "frame": frame_idx,
                "timestamp": det.get("timestamp", 0),
                "label": shot_type,
                "dominant_hand": dominant_hand,
                "camera_angle": camera_angle,
                "features": features,
            }
            all_samples.append(sample)
            extracted += 1

        video_stats[video_name] = {"extracted": extracted, "failed": failed}
        msg = f"    Extracted {extracted} samples ({failed} failed)"
        if contaminated:
            msg += f", {contaminated} FH/BH skipped (serve-only video)"
        print(msg)

    print(f"\n  Positive samples: {len(all_samples)}")
    if skipped_contaminated:
        print(f"  Skipped {skipped_contaminated} contaminated FH/BH from serve-only videos")

    # ── Phase 2: Extract not_shot negatives ───────────────────────
    print(f"\n{'─'*50}")
    print(f"PHASE 2: not_shot negatives")
    print(f"{'─'*50}")

    # Source 1: Deleted detections from pre_fix_backup
    print(f"\n  Source 1: Deleted detections (pre_fix_backup diff)...")
    backup_deleted = find_deleted_detections(det_dir, BACKUP_DIR)
    print(f"    Found {len(backup_deleted)} deleted detections")

    # Source 2: IMG_0931 v1→v2 diff
    print(f"\n  Source 2: IMG_0931 v1→v2 deleted...")
    v1v2_deleted = find_v1_v2_deleted(det_dir)
    print(f"    Found {len(v1v2_deleted)} deleted detections")

    # Extract features for all deleted detections
    all_deleted = backup_deleted + v1v2_deleted
    neg_from_deleted = 0
    for video_name, det in all_deleted:
        if video_name in pose_cache:
            frames, fps, total_frames, dominant_hand = pose_cache[video_name]
        else:
            pose_path = os.path.join(poses_dir, video_name + ".json")
            if not os.path.exists(pose_path):
                continue
            frames, fps, total_frames = load_pose_frames(pose_path)
            # Get dominant hand from detection file
            det_path = os.path.join(det_dir, video_name + "_fused.json")
            dominant_hand = "right"
            if os.path.exists(det_path):
                with open(det_path) as f:
                    dominant_hand = json.load(f).get("dominant_hand", "right")
            pose_cache[video_name] = (frames, fps, total_frames, dominant_hand)

        frame_idx = det.get("frame", 0)
        features = extract_features(
            frames, frame_idx, fps, dominant_hand,
            detection_meta=det,
        )
        if features is None:
            continue

        sample = {
            "video": video_name,
            "frame": frame_idx,
            "timestamp": det.get("timestamp", 0),
            "label": "not_shot",
            "dominant_hand": dominant_hand,
            "camera_angle": None,
            "features": features,
            "neg_source": "deleted_detection",
        }
        all_samples.append(sample)
        neg_from_deleted += 1

    print(f"    Extracted {neg_from_deleted} not_shot samples from deleted detections")

    # Source 3: Random hard negatives (idle frames)
    print(f"\n  Source 3: Random hard negatives ({args.n_negatives} target)...")
    random_negs = sample_random_negatives(
        poses_dir, det_dir, n_samples=args.n_negatives, seed=args.seed
    )
    neg_from_random = 0
    for video_name, frame_idx in random_negs:
        if video_name in pose_cache:
            frames, fps, total_frames, dominant_hand = pose_cache[video_name]
        else:
            pose_path = os.path.join(poses_dir, video_name + ".json")
            if not os.path.exists(pose_path):
                continue
            frames, fps, total_frames = load_pose_frames(pose_path)
            dominant_hand = "right"
            pose_cache[video_name] = (frames, fps, total_frames, dominant_hand)

        # Create a minimal detection_meta for random negatives (no audio, no spike)
        meta = {"source": "random_negative", "audio_amplitude": 0.0, "velocity": 0.0}
        features = extract_features(
            frames, frame_idx, fps, dominant_hand,
            detection_meta=meta,
        )
        if features is None:
            continue

        sample = {
            "video": video_name,
            "frame": frame_idx,
            "timestamp": round(frame_idx / fps, 3),
            "label": "not_shot",
            "dominant_hand": dominant_hand,
            "camera_angle": None,
            "features": features,
            "neg_source": "random_idle",
        }
        all_samples.append(sample)
        neg_from_random += 1

    print(f"    Extracted {neg_from_random} not_shot samples from random frames")

    # ── Summary ───────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"EXTRACTION COMPLETE")
    print(f"{'='*50}")
    print(f"Total samples: {len(all_samples)}")

    label_counts = {}
    for s in all_samples:
        label_counts[s["label"]] = label_counts.get(s["label"], 0) + 1
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")

    if skipped_types:
        print(f"\nSkipped types: {skipped_types}")

    print(f"\nPer-video:")
    for video, stats in sorted(video_stats.items()):
        print(f"  {video}: {stats['extracted']} ok, {stats['failed']} failed")

    # Write output
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    output = {
        "version": 4,
        "feature_names": list(all_samples[0]["features"].keys()) if all_samples else [],
        "num_samples": len(all_samples),
        "label_counts": label_counts,
        "samples": all_samples,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
