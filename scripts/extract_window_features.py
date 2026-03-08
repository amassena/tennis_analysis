#!/usr/bin/env python3
"""Extract temporal window features for sliding-window shot detection.

For each labeled shot, extracts a feature vector from the pose sequence
in a window around the shot (positive samples). For random non-shot
windows, extracts the same features (negative samples).

Features capture the SHAPE of motion over a ~1.5s window:
  - Per-joint velocity/acceleration curves → summary stats
  - Pose trajectory shape (wrist path geometry)
  - Audio RMS envelope shape (if available)
  - Temporal patterns (when peak velocity occurs relative to window center)

Usage:
    .venv/bin/python scripts/extract_window_features.py [--window 1.5] [--neg-ratio 3]
"""

import argparse
import json
import math
import os
import random
import struct
import subprocess
import sys
import tempfile
import wave

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

POSES_DIR = os.path.join(PROJECT_ROOT, "poses_full_videos")
DETECTIONS_DIR = os.path.join(PROJECT_ROOT, "detections")
TRAINING_DIR = os.path.join(PROJECT_ROOT, "training")

# Ground truth mapping (same as validate_pipeline.py)
GT_VIDEOS = {
    "IMG_6665": "detections/IMG_6665_fused_v5.json",
    "IMG_0864": "detections/IMG_0864_fused.json",
    "IMG_0865": "detections/IMG_0865_fused.json",
    "IMG_0866": "detections/IMG_0866_fused.json",
    "IMG_0867": "detections/IMG_0867_fused.json",
    "IMG_0868": "detections/IMG_0868_fused.json",
    "IMG_0869": "detections/IMG_0869_fused.json",
    "IMG_0870": "detections/IMG_0870_fused.json",
    "IMG_6703": "detections/IMG_6703_fused_v5.json",
    "IMG_6711": "detections/IMG_6711_fused_v5.json",
    "IMG_6713": "detections/IMG_6713_fused.json",
    "IMG_0929": "detections/IMG_0929_fused_v5.json",
    "IMG_0991": "detections/IMG_0991_fused.json",
    "IMG_0994": "detections/IMG_0994_fused.json",
    "IMG_0995": "detections/IMG_0995_fused.json",
    "IMG_0996": "detections/IMG_0996_fused.json",
    "IMG_0997": "detections/IMG_0997_fused.json",
    "IMG_0999": "detections/IMG_0999_fused.json",
    "IMG_1001": "detections/IMG_1001_fused.json",
    "IMG_1003": "detections/IMG_1003_fused.json",
    "IMG_1004": "detections/IMG_1004_fused.json",
    "IMG_1005": "detections/IMG_1005_fused.json",
    "IMG_1007": "detections/IMG_1007_fused.json",
}

IGNORE_TYPES = {"offscreen", "practice", "unknown_shot"}

# MediaPipe landmark indices
NOSE = 0
L_SHOULDER = 11
R_SHOULDER = 12
L_ELBOW = 13
R_ELBOW = 14
L_WRIST = 15
R_WRIST = 16
L_HIP = 23
R_HIP = 24
L_KNEE = 25
R_KNEE = 26
L_ANKLE = 27
R_ANKLE = 28

# Key joints for feature extraction
KEY_JOINTS = [R_WRIST, L_WRIST, R_ELBOW, L_ELBOW, R_SHOULDER, L_SHOULDER]
JOINT_NAMES = ["r_wrist", "l_wrist", "r_elbow", "l_elbow", "r_shoulder", "l_shoulder"]


def load_pose_data(video_name):
    """Load pose JSON, return (frames_list, fps, total_frames, duration)."""
    pose_path = os.path.join(POSES_DIR, f"{video_name}.json")
    if not os.path.exists(pose_path):
        return None, 60.0, 0, 0.0
    with open(pose_path) as f:
        data = json.load(f)
    video_info = data.get("video_info", {})
    fps = video_info.get("fps", 60.0)
    total_frames = video_info.get("total_frames", 0)
    duration = video_info.get("duration_seconds", total_frames / fps if fps else 0)
    frames = data.get("frames", [])
    return frames, fps, total_frames, duration


def load_gt_shots(video_name):
    """Load ground truth shots, filtering out ignored types."""
    gt_file = GT_VIDEOS.get(video_name)
    if not gt_file:
        return []
    path = os.path.join(PROJECT_ROOT, gt_file)
    if not os.path.exists(path):
        return []
    with open(path) as f:
        data = json.load(f)
    shots = []
    for det in data.get("detections", []):
        if det.get("shot_type") not in IGNORE_TYPES:
            shots.append({
                "timestamp": det["timestamp"],
                "frame": det.get("frame", int(det["timestamp"] * 60)),
                "shot_type": det["shot_type"],
            })
    return shots


def get_world_landmarks(frame_data, joint_idx):
    """Get [x, y, z] for a joint from world_landmarks, or None."""
    if not frame_data or not frame_data.get("detected"):
        return None
    wl = frame_data.get("world_landmarks")
    if wl and joint_idx < len(wl) and wl[joint_idx] is not None:
        pt = wl[joint_idx]
        if pt[3] < 0.01:  # very low visibility
            return None
        return np.array(pt[:3], dtype=np.float64)
    return None


def extract_joint_trajectory(frames, center_frame, half_window, joint_idx):
    """Extract 3D position trajectory for a joint in window.

    Returns: positions array (N, 3), valid_mask (N,), frame_indices (N,)
    """
    n = 2 * half_window + 1
    start = center_frame - half_window
    positions = np.zeros((n, 3))
    valid = np.zeros(n, dtype=bool)

    for i in range(n):
        fi = start + i
        if 0 <= fi < len(frames):
            pt = get_world_landmarks(frames[fi], joint_idx)
            if pt is not None:
                positions[i] = pt
                valid[i] = True

    return positions, valid


def compute_velocity(positions, valid, fps):
    """Compute velocity magnitude at each timestep.

    Uses central differences where possible.
    Returns: velocity array (N,), valid_velocity mask (N,)
    """
    n = len(positions)
    vel = np.zeros(n)
    vel_valid = np.zeros(n, dtype=bool)
    dt = 1.0 / fps

    for i in range(1, n - 1):
        if valid[i - 1] and valid[i + 1]:
            diff = positions[i + 1] - positions[i - 1]
            vel[i] = np.linalg.norm(diff) / (2 * dt)
            vel_valid[i] = True
        elif valid[i] and valid[i - 1]:
            diff = positions[i] - positions[i - 1]
            vel[i] = np.linalg.norm(diff) / dt
            vel_valid[i] = True
        elif valid[i] and valid[i + 1]:
            diff = positions[i + 1] - positions[i]
            vel[i] = np.linalg.norm(diff) / dt
            vel_valid[i] = True

    return vel, vel_valid


def compute_acceleration(velocity, vel_valid, fps):
    """Compute acceleration magnitude from velocity."""
    n = len(velocity)
    accel = np.zeros(n)
    accel_valid = np.zeros(n, dtype=bool)
    dt = 1.0 / fps

    for i in range(1, n - 1):
        if vel_valid[i - 1] and vel_valid[i + 1]:
            accel[i] = abs(velocity[i + 1] - velocity[i - 1]) / (2 * dt)
            accel_valid[i] = True

    return accel, accel_valid


def velocity_curve_features(vel, vel_valid, half_window):
    """Extract summary features from a velocity curve.

    Returns dict of features capturing the shape of the velocity curve.
    """
    valid_vel = vel[vel_valid]
    if len(valid_vel) < 5:
        return {
            "max_vel": 0, "mean_vel": 0, "std_vel": 0,
            "vel_range": 0, "peak_time_ratio": 0.5,
            "vel_skew": 0, "rise_rate": 0, "fall_rate": 0,
            "pre_peak_mean": 0, "post_peak_mean": 0,
        }

    n = len(vel)
    max_vel = float(np.max(valid_vel))
    mean_vel = float(np.mean(valid_vel))
    std_vel = float(np.std(valid_vel))
    vel_range = max_vel - float(np.min(valid_vel))

    # Where does peak velocity occur relative to window center?
    peak_idx = np.argmax(vel)
    peak_time_ratio = peak_idx / n  # 0=start, 0.5=center, 1=end

    # Velocity skewness (shots have asymmetric velocity profiles)
    if std_vel > 1e-6:
        vel_skew = float(np.mean(((valid_vel - mean_vel) / std_vel) ** 3))
    else:
        vel_skew = 0.0

    # Rise rate (velocity increase before peak) and fall rate (after peak)
    center = half_window
    pre_vel = vel[:center][vel_valid[:center]]
    post_vel = vel[center:][vel_valid[center:]]
    pre_peak_mean = float(np.mean(pre_vel)) if len(pre_vel) > 0 else 0
    post_peak_mean = float(np.mean(post_vel)) if len(post_vel) > 0 else 0

    # Rise/fall rate: how quickly velocity changes around peak
    if peak_idx > 2 and vel_valid[peak_idx]:
        lookback = max(0, peak_idx - 10)
        if vel_valid[lookback]:
            rise_rate = (vel[peak_idx] - vel[lookback]) / max(1, peak_idx - lookback)
        else:
            rise_rate = 0
    else:
        rise_rate = 0

    if peak_idx < n - 3 and vel_valid[peak_idx]:
        lookfwd = min(n - 1, peak_idx + 10)
        if vel_valid[lookfwd]:
            fall_rate = (vel[peak_idx] - vel[lookfwd]) / max(1, lookfwd - peak_idx)
        else:
            fall_rate = 0
    else:
        fall_rate = 0

    return {
        "max_vel": max_vel,
        "mean_vel": mean_vel,
        "std_vel": std_vel,
        "vel_range": vel_range,
        "peak_time_ratio": peak_time_ratio,
        "vel_skew": vel_skew,
        "rise_rate": float(rise_rate),
        "fall_rate": float(fall_rate),
        "pre_peak_mean": pre_peak_mean,
        "post_peak_mean": post_peak_mean,
    }


def trajectory_shape_features(positions, valid, half_window):
    """Extract features describing the shape of the joint trajectory.

    Returns dict of features capturing wrist path geometry.
    """
    valid_pos = positions[valid]
    if len(valid_pos) < 5:
        return {
            "path_length": 0, "displacement": 0, "straightness": 0,
            "y_range": 0, "x_range": 0, "z_range": 0,
            "center_y_offset": 0,
        }

    # Total path length vs displacement
    diffs = np.diff(valid_pos, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    path_length = float(np.sum(segment_lengths))
    displacement = float(np.linalg.norm(valid_pos[-1] - valid_pos[0]))
    straightness = displacement / path_length if path_length > 1e-6 else 1.0

    # Range of motion in each axis
    y_range = float(np.ptp(valid_pos[:, 1]))  # vertical range
    x_range = float(np.ptp(valid_pos[:, 0]))  # lateral range
    z_range = float(np.ptp(valid_pos[:, 2]))  # depth range

    # Y position at center vs mean (is wrist raised at contact?)
    center_idx = half_window
    if valid[center_idx]:
        center_y = positions[center_idx, 1]
        mean_y = float(np.mean(valid_pos[:, 1]))
        center_y_offset = center_y - mean_y  # negative = higher at center
    else:
        center_y_offset = 0.0

    return {
        "path_length": path_length,
        "displacement": displacement,
        "straightness": straightness,
        "y_range": y_range,
        "x_range": x_range,
        "z_range": z_range,
        "center_y_offset": center_y_offset,
    }


def pose_relationship_features(frames, center_frame, half_window):
    """Extract features about body pose relationships at key moments.

    Captures: wrist-hip height, elbow angle, trunk rotation, knee bend.
    """
    center = center_frame
    result = {}

    # Sample 3 timepoints: start, center, end of window
    for label, offset in [("pre", -half_window // 2), ("at", 0), ("post", half_window // 2)]:
        fi = center + offset
        if 0 <= fi < len(frames) and frames[fi].get("detected"):
            fr = frames[fi]
            r_wrist = get_world_landmarks(fr, R_WRIST)
            l_wrist = get_world_landmarks(fr, L_WRIST)
            r_hip = get_world_landmarks(fr, R_HIP)
            l_hip = get_world_landmarks(fr, L_HIP)
            r_shoulder = get_world_landmarks(fr, R_SHOULDER)
            l_shoulder = get_world_landmarks(fr, L_SHOULDER)
            r_elbow = get_world_landmarks(fr, R_ELBOW)
            l_elbow = get_world_landmarks(fr, L_ELBOW)
            r_knee = get_world_landmarks(fr, R_KNEE)
            l_knee = get_world_landmarks(fr, L_KNEE)

            # Wrist above hip (negative Y = higher)
            if r_wrist is not None and r_hip is not None:
                result[f"{label}_r_wrist_hip_y"] = float(r_hip[1] - r_wrist[1])
            else:
                result[f"{label}_r_wrist_hip_y"] = 0.0

            if l_wrist is not None and l_hip is not None:
                result[f"{label}_l_wrist_hip_y"] = float(l_hip[1] - l_wrist[1])
            else:
                result[f"{label}_l_wrist_hip_y"] = 0.0

            # Shoulder-hip rotation (trunk twist)
            if r_shoulder is not None and l_shoulder is not None and r_hip is not None and l_hip is not None:
                shoulder_angle = math.atan2(r_shoulder[2] - l_shoulder[2], r_shoulder[0] - l_shoulder[0])
                hip_angle = math.atan2(r_hip[2] - l_hip[2], r_hip[0] - l_hip[0])
                result[f"{label}_trunk_twist"] = float(shoulder_angle - hip_angle)
            else:
                result[f"{label}_trunk_twist"] = 0.0

            # Elbow angle (forearm extension) - right side
            if r_shoulder is not None and r_elbow is not None and r_wrist is not None:
                v1 = r_shoulder - r_elbow
                v2 = r_wrist - r_elbow
                cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                result[f"{label}_r_elbow_angle"] = float(np.arccos(np.clip(cos_a, -1, 1)))
            else:
                result[f"{label}_r_elbow_angle"] = 0.0

            # Knee bend - right side
            if r_hip is not None and r_knee is not None:
                r_ankle = get_world_landmarks(fr, R_ANKLE)
                if r_ankle is not None:
                    v1 = r_hip - r_knee
                    v2 = r_ankle - r_knee
                    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                    result[f"{label}_r_knee_angle"] = float(np.arccos(np.clip(cos_a, -1, 1)))
                else:
                    result[f"{label}_r_knee_angle"] = 0.0
            else:
                result[f"{label}_r_knee_angle"] = 0.0
        else:
            result[f"{label}_r_wrist_hip_y"] = 0.0
            result[f"{label}_l_wrist_hip_y"] = 0.0
            result[f"{label}_trunk_twist"] = 0.0
            result[f"{label}_r_elbow_angle"] = 0.0
            result[f"{label}_r_knee_angle"] = 0.0

    # Delta features: how pose changes pre→at and at→post
    for key in ["r_wrist_hip_y", "trunk_twist", "r_elbow_angle"]:
        result[f"delta_pre_at_{key}"] = result[f"at_{key}"] - result[f"pre_{key}"]
        result[f"delta_at_post_{key}"] = result[f"post_{key}"] - result[f"at_{key}"]

    return result


def extract_audio_rms(video_path, sample_rate=16000):
    """Extract audio RMS envelope from video using ffmpeg + wave.

    Returns: rms_envelope (array), time_per_sample (float), or (None, None)
    """
    if not os.path.exists(video_path):
        return None, None

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-vn", "-ac", "1", "-ar", str(sample_rate),
            "-f", "wav", tmp_path
        ]
        subprocess.run(cmd, capture_output=True, timeout=120)

        with wave.open(tmp_path, "rb") as wf:
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

        # Compute RMS envelope with 10ms windows, 5ms hop
        window_samples = int(sample_rate * 0.01)
        hop_samples = window_samples // 2
        envelope = []
        for i in range(0, len(audio) - window_samples, hop_samples):
            chunk = audio[i:i + window_samples]
            rms = float(np.sqrt(np.mean(chunk ** 2)))
            envelope.append(rms)

        time_per_sample = hop_samples / sample_rate
        return np.array(envelope), time_per_sample
    except Exception as e:
        print(f"  Audio extraction failed: {e}")
        return None, None
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def audio_window_features(rms_envelope, time_per_sample, center_time, window_sec):
    """Extract audio features in a window around center_time.

    Captures the SHAPE of the audio envelope, not just peak amplitude.
    """
    if rms_envelope is None:
        return {
            "audio_max_rms": 0, "audio_mean_rms": 0, "audio_std_rms": 0,
            "audio_peak_ratio": 0, "audio_peak_time_ratio": 0.5,
            "audio_rise_rate": 0, "audio_energy": 0,
        }

    half_window = window_sec / 2.0
    start_idx = max(0, int((center_time - half_window) / time_per_sample))
    end_idx = min(len(rms_envelope), int((center_time + half_window) / time_per_sample))

    if end_idx <= start_idx:
        return {
            "audio_max_rms": 0, "audio_mean_rms": 0, "audio_std_rms": 0,
            "audio_peak_ratio": 0, "audio_peak_time_ratio": 0.5,
            "audio_rise_rate": 0, "audio_energy": 0,
        }

    window = rms_envelope[start_idx:end_idx]
    max_rms = float(np.max(window))
    mean_rms = float(np.mean(window))
    std_rms = float(np.std(window))

    # Peak-to-mean ratio (sharp spikes vs background)
    peak_ratio = max_rms / mean_rms if mean_rms > 1e-6 else 0

    # Where peak occurs in window
    peak_pos = np.argmax(window)
    peak_time_ratio = peak_pos / len(window) if len(window) > 0 else 0.5

    # Rise rate to peak
    if peak_pos > 2:
        lookback = max(0, peak_pos - 20)
        rise_rate = (window[peak_pos] - window[lookback]) / max(1, peak_pos - lookback)
    else:
        rise_rate = 0

    # Total energy
    energy = float(np.sum(window ** 2))

    return {
        "audio_max_rms": max_rms,
        "audio_mean_rms": mean_rms,
        "audio_std_rms": std_rms,
        "audio_peak_ratio": peak_ratio,
        "audio_peak_time_ratio": peak_time_ratio,
        "audio_rise_rate": float(rise_rate),
        "audio_energy": energy,
    }


def extract_window_features(frames, fps, center_frame, half_window,
                            rms_envelope=None, time_per_sample=None):
    """Extract all features for a single window.

    Returns: dict of feature_name → value
    """
    features = {}
    center_time = center_frame / fps

    # Per-joint velocity and trajectory features
    for joint_idx, joint_name in zip(KEY_JOINTS, JOINT_NAMES):
        pos, valid = extract_joint_trajectory(frames, center_frame, half_window, joint_idx)
        vel, vel_valid = compute_velocity(pos, valid, fps)
        accel, accel_valid = compute_acceleration(vel, vel_valid, fps)

        # Velocity curve features
        vel_feats = velocity_curve_features(vel, vel_valid, half_window)
        for k, v in vel_feats.items():
            features[f"{joint_name}_{k}"] = v

        # Acceleration summary
        valid_accel = accel[accel_valid]
        if len(valid_accel) > 0:
            features[f"{joint_name}_max_accel"] = float(np.max(valid_accel))
            features[f"{joint_name}_mean_accel"] = float(np.mean(valid_accel))
        else:
            features[f"{joint_name}_max_accel"] = 0.0
            features[f"{joint_name}_mean_accel"] = 0.0

        # Trajectory shape (only for wrists — most discriminative)
        if joint_name in ("r_wrist", "l_wrist"):
            traj_feats = trajectory_shape_features(pos, valid, half_window)
            for k, v in traj_feats.items():
                features[f"{joint_name}_{k}"] = v

    # Pose relationship features at key moments
    pose_feats = pose_relationship_features(frames, center_frame, half_window)
    features.update(pose_feats)

    # Pose detection density in window (proxy for visibility/confidence)
    start = max(0, center_frame - half_window)
    end = min(len(frames), center_frame + half_window + 1)
    detected_count = sum(1 for i in range(start, end)
                         if i < len(frames) and frames[i].get("detected"))
    total_window = end - start
    features["pose_density"] = detected_count / total_window if total_window > 0 else 0

    # Audio features
    audio_feats = audio_window_features(rms_envelope, time_per_sample,
                                        center_time, half_window * 2 / fps)
    features.update(audio_feats)

    return features


def generate_negative_timestamps(shot_timestamps, duration, fps, min_gap=3.0,
                                 count=None, half_window_sec=0.75):
    """Generate random negative sample timestamps, at least min_gap from any shot."""
    if count is None:
        count = len(shot_timestamps) * 3

    # Build exclusion zones
    exclusion = set()
    for t in shot_timestamps:
        for dt in np.arange(-min_gap, min_gap + 0.1, 0.1):
            exclusion.add(round(t + dt, 1))

    # Generate candidates
    candidates = []
    margin = half_window_sec + 0.5  # stay away from video edges
    t = margin
    while t < duration - margin:
        if round(t, 1) not in exclusion:
            candidates.append(t)
        t += 0.25  # 0.25s step for candidate grid

    if len(candidates) < count:
        count = len(candidates)

    random.seed(42)
    selected = sorted(random.sample(candidates, count))
    return selected


def process_video(video_name, half_window, neg_ratio, include_audio=True, verbose=True):
    """Extract features for all shots and negative samples from one video.

    Returns: list of (features_dict, label, video_name, timestamp, shot_type)
    """
    if verbose:
        print(f"\nProcessing {video_name}...")

    # Load pose data
    frames, fps, total_frames, duration = load_pose_data(video_name)
    if frames is None or len(frames) == 0:
        print(f"  No pose data for {video_name}")
        return []

    # Load GT shots
    shots = load_gt_shots(video_name)
    if not shots:
        print(f"  No GT shots for {video_name}")
        return []

    if verbose:
        print(f"  {len(shots)} shots, {duration:.0f}s, {len(frames)} frames")

    # Load audio
    rms_envelope, time_per_sample = None, None
    if include_audio:
        video_path = os.path.join(PROJECT_ROOT, "preprocessed", f"{video_name}.mp4")
        if os.path.exists(video_path):
            if verbose:
                print(f"  Extracting audio...")
            rms_envelope, time_per_sample = extract_audio_rms(video_path)
            if rms_envelope is not None and verbose:
                print(f"  Audio: {len(rms_envelope)} RMS samples")

    samples = []

    # Positive samples: windows centered on each shot
    for shot in shots:
        center_frame = shot["frame"]
        if center_frame < half_window or center_frame >= len(frames) - half_window:
            continue
        feats = extract_window_features(frames, fps, center_frame, half_window,
                                        rms_envelope, time_per_sample)
        samples.append((feats, "shot", video_name, shot["timestamp"], shot["shot_type"]))

    # Negative samples
    shot_times = [s["timestamp"] for s in shots]
    neg_count = len(shots) * neg_ratio
    neg_timestamps = generate_negative_timestamps(
        shot_times, duration, fps, min_gap=3.0,
        count=neg_count, half_window_sec=half_window / fps
    )

    for t in neg_timestamps:
        center_frame = int(t * fps)
        if center_frame < half_window or center_frame >= len(frames) - half_window:
            continue
        feats = extract_window_features(frames, fps, center_frame, half_window,
                                        rms_envelope, time_per_sample)
        samples.append((feats, "not_shot", video_name, t, "not_shot"))

    if verbose:
        pos = sum(1 for s in samples if s[1] == "shot")
        neg = sum(1 for s in samples if s[1] == "not_shot")
        print(f"  Extracted: {pos} positive, {neg} negative samples")

    return samples


def main():
    parser = argparse.ArgumentParser(description="Extract window features for shot detection")
    parser.add_argument("--window", type=float, default=1.5,
                        help="Window size in seconds (default: 1.5)")
    parser.add_argument("--neg-ratio", type=int, default=3,
                        help="Negative:positive sample ratio (default: 3)")
    parser.add_argument("--no-audio", action="store_true",
                        help="Skip audio feature extraction")
    parser.add_argument("--videos", nargs="+",
                        help="Process specific videos only")
    args = parser.parse_args()

    fps = 60.0
    half_window = int(args.window * fps / 2)  # frames
    print(f"Window: {args.window}s = ±{half_window} frames at {fps}fps")
    print(f"Neg ratio: {args.neg_ratio}x")

    videos = args.videos or sorted(GT_VIDEOS.keys())

    all_samples = []
    for video_name in videos:
        if video_name not in GT_VIDEOS:
            print(f"Skipping {video_name} (no GT)")
            continue
        samples = process_video(video_name, half_window, args.neg_ratio,
                                include_audio=not args.no_audio)
        all_samples.extend(samples)

    if not all_samples:
        print("No samples extracted!")
        return

    # Get feature names from first sample
    feature_names = sorted(all_samples[0][0].keys())
    print(f"\n{'='*60}")
    print(f"Total samples: {len(all_samples)}")
    print(f"  Positive (shot): {sum(1 for s in all_samples if s[1] == 'shot')}")
    print(f"  Negative (not_shot): {sum(1 for s in all_samples if s[1] == 'not_shot')}")
    print(f"Features: {len(feature_names)}")

    # Build feature matrix and labels
    X = np.array([[s[0].get(f, 0) for f in feature_names] for s in all_samples])
    labels = [s[1] for s in all_samples]
    videos_list = [s[2] for s in all_samples]
    timestamps = [s[3] for s in all_samples]
    shot_types = [s[4] for s in all_samples]

    # Save to training directory
    os.makedirs(TRAINING_DIR, exist_ok=True)
    output = {
        "feature_names": feature_names,
        "X": X.tolist(),
        "labels": labels,
        "videos": videos_list,
        "timestamps": timestamps,
        "shot_types": shot_types,
        "window_sec": args.window,
        "half_window_frames": half_window,
        "neg_ratio": args.neg_ratio,
        "include_audio": not args.no_audio,
    }

    out_path = os.path.join(TRAINING_DIR, "window_features.json")
    with open(out_path, "w") as f:
        json.dump(output, f)
    print(f"\nSaved to {out_path}")

    # Also save as npz for faster loading
    npz_path = os.path.join(TRAINING_DIR, "window_features.npz")
    np.savez_compressed(
        npz_path,
        X=X,
        labels=np.array(labels),
        videos=np.array(videos_list),
        timestamps=np.array(timestamps),
        shot_types=np.array(shot_types),
        feature_names=np.array(feature_names),
    )
    print(f"Saved to {npz_path}")
    print(f"Feature matrix shape: {X.shape}")


if __name__ == "__main__":
    main()
