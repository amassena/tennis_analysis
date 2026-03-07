#!/usr/bin/env python3
"""Fused audio + heuristic shot detection.

Combines audio peak detection (detect_audio_hits) with biomechanical pose
analysis (heuristic_detect) to produce higher-confidence shot detections.

Scoring logic:
  - Audio + heuristic agree  -> HIGH confidence
  - Audio only (no pose match) -> MEDIUM confidence (real hit, unknown shot type)
  - Heuristic only (no audio)  -> LOW confidence (likely false positive)

Usage:
    python fused_detect.py /path/to/video.mp4 --poses /path/to/poses.json
    python fused_detect.py /path/to/video.mp4 --poses /path/to/poses.json -o detections.json
"""

import argparse
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import POSES_DIR, PREPROCESSED_DIR, PROJECT_ROOT, MODELS_DIR, RACKET_DETECTIONS_DIR

from scripts.detect_audio_hits import extract_audio, detect_peaks, detect_spectral_onsets, get_video_fps
from scripts.heuristic_detect import (
    analyze_stroke_pattern,
    detect_serve_pattern,
    compute_wrist_velocities,
    compute_wrist_acceleration,
    compute_wrist_jerk,
    find_velocity_spikes,
    find_jerk_spikes,
    get_keypoint,
    RIGHT_WRIST,
    LEFT_WRIST,
    LEFT_HIP,
    RIGHT_HIP,
    THRESHOLDS,
)

# ── ML classifier (lazy-loaded) ─────────────────────────────
_classifier = None
_classifier_meta = None
_classifier_loaded = False

# ── CNN sequence model (lazy-loaded) ────────────────────────
_cnn_model = None
_cnn_device = None
_cnn_loaded = False

# ── Calibrated pipeline thresholds ───────────────────────────
# Default values (backward compatible with models lacking calibrated_thresholds)
_DEFAULT_THRESHOLDS = {
    # not_shot gates (reject if not_shot_prob > threshold)
    'ns_permissive': 0.40, 'ns_moderate': 0.35, 'ns_strict': 0.32,
    'ns_first_pass': 0.30, 'ns_weak_jerk': 0.25, 'ns_weak_heuristic': 0.24,
    # ML confidence floors (reject if ml_conf < threshold)
    'mc_strong': 0.50, 'mc_moderate': 0.40, 'mc_weak': 0.30,
    'mc_floor_audio_heuristic': 0.43, 'mc_floor_heuristic_only': 0.55,
    'mc_floor_jerk': 0.60, 'mc_low_pass': 0.50,
    # Sliding window / rhythm fill ML confidence floors
    'mc_sliding_window': 0.50, 'mc_rhythm_fill': 0.40,
    # Rescue scan thresholds (stricter — last resort)
    'ns_rescue': 0.40, 'ns_rescue_reject': 0.35, 'mc_rescue': 0.62,
    # Jerk spike adaptive thresholds (high jerk >= 1500)
    'ns_jerk_high': 0.40, 'mc_jerk_high': 0.40,
    # Jerk spike adaptive thresholds (low jerk < 1500)
    'ns_jerk_low': 0.30, 'mc_jerk_low': 0.50,
    # Post-dedup weak filter (fused_conf thresholds)
    'fc_weak_jerk': 0.45, 'ns_weak_jerk_floor': 0.10,
    'fc_weak_low': 0.52,
}


class _ThresholdProxy:
    """Lazy threshold lookup — returns calibrated values once classifier is loaded."""
    def __getitem__(self, key):
        if _classifier_meta and 'calibrated_thresholds' in _classifier_meta:
            return _classifier_meta['calibrated_thresholds'].get(
                key, _DEFAULT_THRESHOLDS[key])
        return _DEFAULT_THRESHOLDS[key]


T = _ThresholdProxy()


# ── Racket detection data (optional, for post-dedup filtering) ───
def _load_racket_data(video_name):
    """Load racket detection JSON if available. Returns dict of frame→entry or None."""
    path = os.path.join(RACKET_DETECTIONS_DIR, f"{video_name}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        if data.get("total_frames", 0) == 0:
            return None
        return {f["frame"]: f for f in data.get("frames", [])}
    except Exception:
        return None


def _racket_visibility(racket_frames, frame, window=10):
    """Fraction of frames in [frame-window, frame+window] with racket detected."""
    if racket_frames is None:
        return None
    count = sum(1 for i in range(frame - window, frame + window + 1)
                if racket_frames.get(i, {}).get("detected"))
    return count / (2 * window + 1)


def _load_classifier():
    """Lazy-load the trained shot classifier. Returns (model, meta) or (None, None)."""
    global _classifier, _classifier_meta, _classifier_loaded
    if _classifier_loaded:
        return _classifier, _classifier_meta

    _classifier_loaded = True
    model_path = os.path.join(MODELS_DIR, "shot_classifier.pkl")
    meta_path = os.path.join(MODELS_DIR, "shot_classifier_meta.json")

    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        return None, None

    try:
        with open(model_path, "rb") as f:
            _classifier = pickle.load(f)
        with open(meta_path) as f:
            _classifier_meta = json.load(f)
        print(f"  Loaded ML classifier ({_classifier_meta.get('val_accuracy', 0):.1%} val acc)")
    except Exception as e:
        print(f"  Warning: failed to load classifier: {e}")
        _classifier = None
        _classifier_meta = None

    return _classifier, _classifier_meta


def _load_cnn():
    """Lazy-load the CNN sequence model. Returns (model, device) or (None, None)."""
    global _cnn_model, _cnn_device, _cnn_loaded
    if _cnn_loaded:
        return _cnn_model, _cnn_device

    _cnn_loaded = True
    try:
        from scripts.sequence_model import load_model
        _cnn_model, _cnn_device = load_model()
        if _cnn_model is not None:
            print(f"  Loaded CNN sequence model on {_cnn_device}")
    except Exception as e:
        print(f"  CNN model not available: {e}")
        _cnn_model = None
        _cnn_device = None

    return _cnn_model, _cnn_device


def classify_with_model(frames, center_frame, fps, dominant_hand="right",
                        detection_meta=None, rally_context=None,
                        frames_3d=None):
    """Classify a shot using the trained RF model.

    Returns (shot_type, confidence, class_probs) or (None, 0.0, {}).
    class_probs is a dict of class_name -> probability.
    If frames_3d is provided, uses 3D poses for feature extraction
    (camera-invariant classification) while detection uses 2D frames.
    """
    model, meta = _load_classifier()
    if model is None or meta is None:
        return None, 0.0, {}

    from scripts.extract_training_features import extract_features

    # Use 3D poses for ML features if available, else use 2D
    ml_frames = frames_3d if frames_3d is not None else frames
    features = extract_features(ml_frames, center_frame, fps, dominant_hand,
                                detection_meta=detection_meta,
                                rally_context=rally_context)
    if features is None:
        return None, 0.0, {}

    feature_names = meta["feature_names"]
    X = [[features.get(f, 0.0) for f in feature_names]]

    try:
        rf_proba = model.predict_proba(X)[0]
        rf_classes = list(model.classes_)
    except Exception:
        return None, 0.0, {}

    # Build class probability dict from RF
    rf_probs = {cls: float(p) for cls, p in zip(rf_classes, rf_proba)}

    pred_idx = rf_proba.argmax()
    shot_type = rf_classes[pred_idx]
    confidence = float(rf_proba[pred_idx])
    return shot_type, confidence, rf_probs


# ── Configuration ────────────────────────────────────────────

# Window (in seconds) around a heuristic spike to search for a confirming audio peak
AUDIO_POSE_WINDOW = 1.0  # +/- 1.0s

# Minimum gap between final deduplicated detections (seconds)
# Dedup same-shot duplicates. A single swing can produce multiple peaks
# within ~1s from audio echo + velocity curve. Serve toss + contact
# can produce two audio peaks 1.3-1.7s apart, so 2.0s prevents those
# while still allowing rally shots 3s+ apart.
MIN_DEDUP_GAP = 2.0

# Minimum RMS amplitude for audio peaks.
# Filters out quiet hits from partner/adjacent courts.
# Player's own hits (closest to mic) are typically 0.10-0.30 RMS,
# partner/other court hits are typically 0.03-0.07 RMS.
AUDIO_MIN_AMPLITUDE = 0.09

# Stricter amplitude for MEDIUM tier (audio-only, no heuristic confirmation).
# These need to be louder to be trusted since there's no pose signal.
AUDIO_MIN_AMPLITUDE_MEDIUM = 0.10

# Minimum wrist height above hip (meters) to count as a real stroke.
# During a real swing, wrist is always elevated (backswing/contact/follow-through).
# Ball pickups, dribbling, and walking have wrist near or below hip level.
# Real shots: wrist 0.10-0.50m above hip. FPs: wrist within 0.05m of hip.
MIN_WRIST_ABOVE_HIP = 0.07

# Confidence thresholds for the output tiers
CONFIDENCE_TIERS = {
    "high": 0.8,    # Both signals agree + pattern classified
    "medium": 0.5,  # Audio only, or audio+spike but no pattern
    "low": 0.3,     # Heuristic only
}


def load_pose_frames(pose_path):
    """Load pose JSON and return (frames_list, fps, total_frames).

    Returns the raw frame dicts (with world_landmarks, keypoints, etc.)
    as expected by heuristic_detect functions.
    """
    with open(pose_path) as f:
        data = json.load(f)

    video_info = data.get("video_info", {})
    fps = video_info.get("fps", 60.0)
    raw_frames = data.get("frames", [])
    total_frames = video_info.get("total_frames", len(raw_frames))

    # Build indexed array so frame_idx lines up with list index
    frames = [{}] * total_frames
    for fr in raw_frames:
        idx = fr.get("frame_idx", 0)
        if idx < total_frames:
            frames[idx] = fr

    return frames, fps, total_frames


def deduplicate_detections(detections, min_gap=MIN_DEDUP_GAP):
    """Merge detections within min_gap seconds, keeping the highest confidence.

Merge detections within min_gap seconds, keeping the highest confidence."""
    if not detections:
        return detections

    sorted_dets = sorted(detections, key=lambda d: d["timestamp"])
    merged = [sorted_dets[0]]

    for det in sorted_dets[1:]:
        prev = merged[-1]
        gap = det["timestamp"] - prev["timestamp"]

        if gap < min_gap:
            # Keep whichever has higher confidence, or prefer classified over unknown
            prev_is_classified = prev["shot_type"] not in ("unknown_shot", "neutral")
            det_is_classified = det["shot_type"] not in ("unknown_shot", "neutral")

            if det_is_classified and not prev_is_classified:
                merged[-1] = det
            elif det["confidence"] > prev["confidence"] and not (prev_is_classified and not det_is_classified):
                merged[-1] = det
            # else keep prev
        else:
            merged.append(det)

    return merged


def check_wrist_above_hip(frames, frame_idx, fps, dominant_hand="right"):
    """Check max wrist height above hip in a small window around detection.

    Returns the max height (meters) the wrist was above the hip.
    Positive = wrist above hip. Returns None if no pose data available.

    Checks a ±0.3s window (18 frames at 60fps) to catch the peak of the swing.
    """
    wrist_idx = RIGHT_WRIST if dominant_hand == "right" else LEFT_WRIST
    window = int(0.3 * fps)
    start = max(0, frame_idx - window)
    end = min(len(frames), frame_idx + window + 1)

    max_height = None

    for i in range(start, end):
        if not frames[i]:
            continue
        wrist = get_keypoint(frames[i], wrist_idx)
        l_hip = get_keypoint(frames[i], LEFT_HIP)
        r_hip = get_keypoint(frames[i], RIGHT_HIP)

        if wrist is None or (l_hip is None and r_hip is None):
            continue

        # Average hip Y
        if l_hip and r_hip:
            hip_y = (l_hip[1] + r_hip[1]) / 2
        elif l_hip:
            hip_y = l_hip[1]
        else:
            hip_y = r_hip[1]

        # In world coords, lower Y = higher position
        # So hip_y - wrist_y > 0 means wrist is above hip
        height_above_hip = hip_y - wrist[1]

        if max_height is None or height_above_hip > max_height:
            max_height = height_above_hip

    return max_height


def merge_slomo_clusters(detections, cluster_window=4.0):
    """Merge slo-mo clusters: groups of 3+ same-type detections within cluster_window.

    For clusters of 3+ same-type detections: keep only the highest-confidence one.
    For mixed-type clusters: keep the one with highest velocity (actual contact).
    """
    if len(detections) < 3:
        return detections

    sorted_dets = sorted(detections, key=lambda d: d["timestamp"])

    # Find clusters: groups of detections within cluster_window of each other
    clusters = []
    current_cluster = [sorted_dets[0]]
    for det in sorted_dets[1:]:
        if det["timestamp"] - current_cluster[0]["timestamp"] <= cluster_window:
            current_cluster.append(det)
        else:
            clusters.append(current_cluster)
            current_cluster = [det]
    clusters.append(current_cluster)

    merged = []
    merged_count = 0
    for cluster in clusters:
        if len(cluster) < 3:
            merged.extend(cluster)
            continue

        # Check if cluster is same-type AND dense (slo-mo signature)
        types = set(d["shot_type"] for d in cluster)
        span = cluster[-1]["timestamp"] - cluster[0]["timestamp"]
        avg_gap = span / (len(cluster) - 1) if len(cluster) > 1 else 0

        if len(types) == 1 and avg_gap < 1.5:
            # Dense same-type cluster: slo-mo replay → keep highest confidence
            best = max(cluster, key=lambda d: d["confidence"])
            merged.append(best)
            merged_count += len(cluster) - 1
        else:
            # Mixed-type or sparse cluster: real rally, keep all.
            merged.extend(cluster)

    if merged_count:
        print(f"    Slo-mo cluster merge: removed {merged_count} detections from {len(clusters)} clusters")

    return merged


def _second_pass_low_threshold(velocities, first_pass_times, frames, fps,
                               dominant_hand, verbose, frames_3d=None):
    """Find shots missed by strict heuristic using lower velocity thresholds + ML.

    The first pass uses min_vel=12.0 and spike_ratio=3.0 plus a rigid biomechanical
    pattern filter. This second pass lowers thresholds and relies entirely on the
    ML classifier to distinguish real shots from noise.

    Returns list of detection dicts (same format as first-pass detections).
    """
    model, meta = _load_classifier()
    if model is None:
        if verbose:
            print("    Second pass: no ML model, skipping")
        return []

    # Find spikes with relaxed thresholds
    low_spikes = find_velocity_spikes(
        velocities,
        min_vel=8.0,
        spike_ratio=1.5,
        min_gap=60,  # same 1.0s min gap between spikes
    )

    if verbose:
        print(f"    Second pass: {len(low_spikes)} low-threshold spikes")

    # Filter out spikes already covered by first pass
    new_spikes = []
    for spike_frame, velocity in low_spikes:
        spike_time = spike_frame / fps
        too_close = any(abs(spike_time - t) < MIN_DEDUP_GAP for t in first_pass_times)
        if not too_close:
            new_spikes.append((spike_frame, velocity))

    if verbose:
        print(f"    Second pass: {len(new_spikes)} new spikes after excluding first-pass")

    detections = []
    for spike_frame, velocity in new_spikes:
        spike_time = spike_frame / fps

        spike_meta = {
            "source": "low_threshold_ml",
            "audio_amplitude": 0.0,
            "velocity": velocity,
        }

        ml_type, ml_conf, ml_probs = classify_with_model(
            frames, spike_frame, fps, dominant_hand,
            detection_meta=spike_meta,
            frames_3d=frames_3d,
        )

        not_shot_prob = ml_probs.get("not_shot", 0.0)

        # Stricter ML gate for low-threshold candidates
        if not_shot_prob > T['ns_first_pass']:
            if verbose:
                print(f"      Rejected {spike_time:.1f}s: not_shot={not_shot_prob:.2f} "
                      f"vel={velocity:.1f}")
            continue

        # Filter out not_shot predictions
        if ml_type == "not_shot" or ml_type is None:
            if verbose:
                print(f"      Rejected {spike_time:.1f}s: ml_type={ml_type} vel={velocity:.1f}")
            continue

        # Require decent shot type confidence
        if ml_conf < T['mc_low_pass']:
            if verbose:
                print(f"      Rejected {spike_time:.1f}s: ml_conf={ml_conf:.2f} "
                      f"({ml_type}) vel={velocity:.1f}")
            continue

        fused_conf = min(1.0, ml_conf * 0.5 + 0.2)

        detections.append({
            "timestamp": round(spike_time, 3),
            "frame": spike_frame,
            "shot_type": ml_type,
            "confidence": round(fused_conf, 3),
            "tier": "medium",
            "source": "low_threshold_ml",
            "audio_peak_time": None,
            "audio_amplitude": 0.0,
            "heuristic_frame": spike_frame,
            "pattern_confidence": round(ml_conf, 3),
            "trigger": f"low_thresh ml:{ml_type}({ml_conf:.2f}) vel={velocity:.1f}",
            "velocity": round(velocity, 1),
            "ml_confidence": round(ml_conf, 3),
            "not_shot_prob": round(not_shot_prob, 3),
        })

        if verbose:
            print(f"      Accepted {spike_time:.1f}s: {ml_type}({ml_conf:.2f}) "
                  f"vel={velocity:.1f}")

    return detections


def _sliding_window_scan(existing_times, frames, fps, dominant_hand, verbose,
                         frames_3d=None):
    """Scan entire video at 0.5s intervals, using ML to detect shots from pose alone.

    For shots with no velocity spike and no audio — pure pose-based detection.
    For each candidate, tries multiple offsets (±15 frames) to find the best
    prediction, since the grid won't land exactly on the contact frame.

    Returns list of detection dicts.
    """
    model, meta = _load_classifier()
    if model is None:
        if verbose:
            print("    Sliding window: no ML model, skipping")
        return []

    from scripts.extract_training_features import extract_features
    from scripts.heuristic_detect import get_pose_confidence

    total_frames = len(frames)
    step = int(fps * 0.5)  # 0.5s intervals
    detections = []
    scanned = 0
    skipped_near = 0
    skipped_no_pose = 0

    feature_names = meta["feature_names"]

    # Offsets to try around each grid point (contact frame might be ±0.4s away)
    probe_offsets = [0, -12, 12, -24, 24, -6, 6]

    for frame_idx in range(step, total_frames - step, step):
        frame_time = frame_idx / fps

        # Skip if near any existing detection
        too_close = any(abs(frame_time - t) < 1.5 for t in existing_times)
        if too_close:
            skipped_near += 1
            continue

        # Skip if no pose at this frame
        if not frames[frame_idx] or not frames[frame_idx].get("world_landmarks"):
            skipped_no_pose += 1
            continue

        # Quick pose quality check
        pose_conf = get_pose_confidence(frames[frame_idx])
        if pose_conf < 0.5:
            skipped_no_pose += 1
            continue

        scanned += 1

        # Try multiple offsets around grid point, keep the best prediction
        best_shot_type = None
        best_confidence = 0.0
        best_not_shot = 1.0
        best_probe_frame = frame_idx

        for offset in probe_offsets:
            probe_frame = frame_idx + offset
            if probe_frame < 0 or probe_frame >= total_frames:
                continue
            if not frames[probe_frame]:
                continue

            window_meta = {
                "source": "sliding_window",
                "audio_amplitude": 0.0,
                "velocity": 0.0,
            }
            ml_frames = frames_3d if frames_3d is not None else frames
            features = extract_features(
                ml_frames, probe_frame, fps, dominant_hand,
                detection_meta=window_meta,
            )
            if features is None:
                continue

            X = [[features.get(f, 0.0) for f in feature_names]]

            try:
                proba = model.predict_proba(X)[0]
                pred_idx = proba.argmax()
                shot_type = model.classes_[pred_idx]
                confidence = float(proba[pred_idx])
                not_shot_prob = float(proba[list(model.classes_).index("not_shot")])
            except Exception:
                continue

            if shot_type == "not_shot" or shot_type is None:
                continue

            # Keep the probe with highest confidence and low not_shot
            if confidence > best_confidence and not_shot_prob < T['ns_moderate']:
                best_shot_type = shot_type
                best_confidence = confidence
                best_not_shot = not_shot_prob
                best_probe_frame = probe_frame

        # Apply thresholds on the best probe — relaxed since wrist filter
        # and dedup will catch bad ones downstream
        if best_shot_type is None:
            continue
        if best_not_shot > T['ns_strict']:
            continue
        if best_confidence < T['mc_sliding_window']:
            continue

        best_time = best_probe_frame / fps
        fused_conf = min(1.0, best_confidence * 0.4 + 0.15)

        det = {
            "timestamp": round(best_time, 3),
            "frame": best_probe_frame,
            "shot_type": best_shot_type,
            "confidence": round(fused_conf, 3),
            "tier": "low",
            "source": "sliding_window_ml",
            "audio_peak_time": None,
            "audio_amplitude": 0.0,
            "heuristic_frame": None,
            "pattern_confidence": round(best_confidence, 3),
            "trigger": f"sliding_window ml:{best_shot_type}({best_confidence:.2f})",
            "velocity": 0.0,
            "ml_confidence": round(best_confidence, 3),
            "not_shot_prob": round(best_not_shot, 3),
        }
        detections.append(det)

        # Add to existing_times so subsequent grid points don't double-detect
        existing_times.append(best_time)

        if verbose:
            print(f"      Window hit {best_time:.1f}s: {best_shot_type}({best_confidence:.2f}) "
                  f"not_shot={best_not_shot:.2f}")

    if verbose:
        print(f"    Sliding window: scanned {scanned} frames, "
              f"skipped {skipped_near} near existing, "
              f"{skipped_no_pose} no pose, "
              f"found {len(detections)} candidates")

    return detections


def _jerk_spike_pass(velocities, existing_times, frames, fps,
                     dominant_hand, verbose, frames_3d=None):
    """Find shots via wrist jerk spikes (d³pos/dt³) + ML classification.

    Jerk captures the instantaneous "snap" at ball contact and is immune
    to the inflated rolling-average baseline that kills spike_ratio during
    active rallies.

    Returns list of detection dicts.
    """
    model, meta = _load_classifier()
    if model is None:
        if verbose:
            print("    Jerk pass: no ML model, skipping")
        return []

    accel = compute_wrist_acceleration(velocities, fps, smooth_window=5)
    jerk = compute_wrist_jerk(accel, fps, smooth_window=3)

    jerk_spikes = find_jerk_spikes(jerk, min_jerk=800.0, min_gap=60,
                                   percentile_threshold=92.0)

    if verbose:
        print(f"    Jerk pass: {len(jerk_spikes)} jerk spikes found")

    # Only skip jerk spikes that are exact duplicates of existing detections
    # (within 0.5s). Don't use full MIN_DEDUP_GAP here — that caused FNs when
    # an intermediate detection blocked a jerk spike then got removed by dedup.
    JERK_DEDUP_GAP = 0.5
    new_spikes = []
    for spike_frame, jerk_val in jerk_spikes:
        spike_frame = int(spike_frame)
        if spike_frame >= len(frames):
            continue
        spike_time = spike_frame / fps
        too_close = any(abs(spike_time - t) < JERK_DEDUP_GAP for t in existing_times)
        if not too_close:
            new_spikes.append((spike_frame, jerk_val))

    if verbose:
        print(f"    Jerk pass: evaluating {len(new_spikes)} spikes with ML")

    detections = []
    for spike_frame, jerk_val in new_spikes:
        spike_frame = int(spike_frame)  # ensure native int (not numpy int64)
        spike_time = spike_frame / fps

        spike_meta = {
            "source": "jerk_spike_ml",
            "audio_amplitude": 0.0,
            "velocity": float(velocities[spike_frame]) if spike_frame < len(velocities) else 0.0,
        }

        ml_type, ml_conf, ml_probs = classify_with_model(
            frames, spike_frame, fps, dominant_hand,
            detection_meta=spike_meta,
            frames_3d=frames_3d,
        )

        not_shot_prob = ml_probs.get("not_shot", 0.0)

        # Jerk-adaptive ML gate: higher jerk = more lenient thresholds
        # Strong jerk (>1500) is reliable biomechanical signal
        if jerk_val >= 1500:
            not_shot_limit = T['ns_jerk_high']
            min_conf = T['mc_jerk_high']
        else:
            not_shot_limit = T['ns_jerk_low']
            min_conf = T['mc_jerk_low']

        if not_shot_prob > not_shot_limit:
            if verbose:
                print(f"      Jerk rejected {spike_time:.1f}s: not_shot={not_shot_prob:.2f} "
                      f"jerk={jerk_val:.0f}")
            continue

        if ml_type == "not_shot" or ml_type is None:
            if verbose:
                print(f"      Jerk rejected {spike_time:.1f}s: ml_type={ml_type}")
            continue

        if ml_conf < min_conf:
            if verbose:
                print(f"      Jerk rejected {spike_time:.1f}s: ml_conf={ml_conf:.2f} "
                      f"({ml_type}) jerk={jerk_val:.0f}")
            continue

        fused_conf = min(1.0, ml_conf * 0.5 + 0.2)
        vel = float(velocities[spike_frame]) if spike_frame < len(velocities) else 0.0

        detections.append({
            "timestamp": round(spike_time, 3),
            "frame": spike_frame,
            "shot_type": ml_type,
            "confidence": round(fused_conf, 3),
            "tier": "medium",
            "source": "jerk_spike_ml",
            "audio_peak_time": None,
            "audio_amplitude": 0.0,
            "heuristic_frame": spike_frame,
            "pattern_confidence": round(ml_conf, 3),
            "trigger": f"jerk={jerk_val:.0f} ml:{ml_type}({ml_conf:.2f}) vel={vel:.1f}",
            "velocity": round(vel, 1),
            "ml_confidence": round(ml_conf, 3),
            "not_shot_prob": round(not_shot_prob, 3),
        })

        if verbose:
            print(f"      Jerk accepted {spike_time:.1f}s: {ml_type}({ml_conf:.2f}) "
                  f"jerk={jerk_val:.0f} vel={vel:.1f}")

    return detections


def _rally_rhythm_fill(existing_times, frames, fps, dominant_hand, verbose,
                       frames_3d=None):
    """Fill gaps in rally rhythm using temporal pattern + ML classification.

    Tennis rallies have predictable rhythm (1.5-4s between shots). A gap of
    2x the median interval likely means a missed shot. Uses ML with relaxed
    thresholds (strong rally-context prior).

    Returns list of detection dicts.
    """
    model, meta = _load_classifier()
    if model is None:
        if verbose:
            print("    Rhythm fill: no ML model, skipping")
        return []

    from scripts.extract_training_features import extract_features
    from scripts.heuristic_detect import get_pose_confidence

    sorted_times = sorted(existing_times)
    if len(sorted_times) < 3:
        if verbose:
            print("    Rhythm fill: fewer than 3 detections, skipping")
        return []

    feature_names = meta["feature_names"]

    # Cluster detections into rallies (consecutive shots with gap < 6s)
    rallies = []
    current_rally = [sorted_times[0]]
    for t in sorted_times[1:]:
        if t - current_rally[-1] < 6.0:
            current_rally.append(t)
        else:
            rallies.append(current_rally)
            current_rally = [t]
    rallies.append(current_rally)

    if verbose:
        print(f"    Rhythm fill: {len(rallies)} rallies found "
              f"(sizes: {[len(r) for r in rallies]})")

    # Probe offsets around predicted time
    probe_offsets = [0, -12, 12, -24, 24, -6, 6]
    total_frames = len(frames)

    detections = []
    for rally in rallies:
        if len(rally) < 3:
            continue

        # Compute median inter-shot interval
        intervals = [rally[i+1] - rally[i] for i in range(len(rally) - 1)]
        median_interval = sorted(intervals)[len(intervals) // 2]

        # Find gaps > max(2x median, 4.0s)
        gap_threshold = max(2.0 * median_interval, 4.0)

        for i in range(len(rally) - 1):
            gap = rally[i+1] - rally[i]
            if gap <= gap_threshold:
                continue

            # Predict shot times evenly spaced in the gap
            n_expected = max(1, round(gap / median_interval) - 1)
            for k in range(1, n_expected + 1):
                predicted_time = rally[i] + k * (gap / (n_expected + 1))

                # Skip if too close to any existing detection
                if any(abs(predicted_time - t) < 1.5 for t in sorted_times):
                    continue

                predicted_frame = int(predicted_time * fps)
                if predicted_frame < 0 or predicted_frame >= total_frames:
                    continue

                # Try multiple probe offsets, keep the best
                best_shot_type = None
                best_confidence = 0.0
                best_not_shot = 1.0
                best_probe_frame = predicted_frame

                for offset in probe_offsets:
                    probe_frame = predicted_frame + offset
                    if probe_frame < 0 or probe_frame >= total_frames:
                        continue
                    if not frames[probe_frame]:
                        continue

                    window_meta = {
                        "source": "rhythm_fill_ml",
                        "audio_amplitude": 0.0,
                        "velocity": 0.0,
                    }
                    ml_frames = frames_3d if frames_3d is not None else frames
                    features = extract_features(
                        ml_frames, probe_frame, fps, dominant_hand,
                        detection_meta=window_meta,
                    )
                    if features is None:
                        continue

                    X = [[features.get(f, 0.0) for f in feature_names]]

                    try:
                        proba = model.predict_proba(X)[0]
                        pred_idx = proba.argmax()
                        shot_type = model.classes_[pred_idx]
                        confidence = float(proba[pred_idx])
                        not_shot_prob = float(proba[list(model.classes_).index("not_shot")])
                    except Exception:
                        continue

                    if shot_type == "not_shot" or shot_type is None:
                        continue

                    if confidence > best_confidence and not_shot_prob < T['ns_permissive']:
                        best_shot_type = shot_type
                        best_confidence = confidence
                        best_not_shot = not_shot_prob
                        best_probe_frame = probe_frame

                # Relaxed thresholds for rhythm fill (strong rally-context prior)
                if best_shot_type is None:
                    continue
                if best_not_shot > T['ns_moderate']:
                    continue
                if best_confidence < T['mc_rhythm_fill']:
                    continue

                best_time = best_probe_frame / fps
                fused_conf = min(1.0, best_confidence * 0.4 + 0.15)

                det = {
                    "timestamp": round(best_time, 3),
                    "frame": best_probe_frame,
                    "shot_type": best_shot_type,
                    "confidence": round(fused_conf, 3),
                    "tier": "low",
                    "source": "rhythm_fill_ml",
                    "audio_peak_time": None,
                    "audio_amplitude": 0.0,
                    "heuristic_frame": None,
                    "pattern_confidence": round(best_confidence, 3),
                    "trigger": (f"rhythm_fill gap={gap:.1f}s median={median_interval:.1f}s "
                                f"ml:{best_shot_type}({best_confidence:.2f})"),
                    "velocity": 0.0,
                    "ml_confidence": round(best_confidence, 3),
                    "not_shot_prob": round(best_not_shot, 3),
                }
                detections.append(det)

                # Track so we don't double-fill
                sorted_times.append(best_time)
                sorted_times.sort()

                if verbose:
                    print(f"      Rhythm fill {best_time:.1f}s: {best_shot_type}"
                          f"({best_confidence:.2f}) gap={gap:.1f}s")

    if verbose:
        print(f"    Rhythm fill: found {len(detections)} candidates")

    return detections


def _post_dedup_rescue(pre_filter_dets, post_dedup_dets, frames, fps,
                       dominant_hand, verbose, frames_3d=None):
    """Rescue real shots orphaned by dedup/cluster merge.

    When early pipeline stages create an intermediate detection that blocks
    later candidate generators (sliding window, jerk), then dedup/merge
    removes that intermediate detection, nothing survives at the timestamp.

    This function finds orphaned timestamps and re-probes with ML.

    Returns list of detection dicts.
    """
    model, meta = _load_classifier()
    if model is None:
        return []

    from scripts.extract_training_features import extract_features

    feature_names = meta["feature_names"]
    surviving_times = [d["timestamp"] for d in post_dedup_dets]

    # Find orphaned timestamps: were in pre-filter but have no survivor nearby
    orphaned = []
    for d in pre_filter_dets:
        t = d["timestamp"]
        has_survivor = any(abs(t - st) < MIN_DEDUP_GAP for st in surviving_times)
        if not has_survivor:
            orphaned.append(d)

    if not orphaned:
        return []

    # Deduplicate orphaned list (multiple detections at same timestamp)
    seen_times = set()
    unique_orphaned = []
    for d in orphaned:
        t_round = round(d["timestamp"], 1)
        if t_round not in seen_times:
            seen_times.add(t_round)
            unique_orphaned.append(d)
    orphaned = unique_orphaned

    if verbose:
        print(f"    Post-dedup rescue: {len(orphaned)} orphaned timestamps to probe")

    total_frames = len(frames)
    probe_offsets = [0, -6, 6, -12, 12, -18, 18, -24, 24]
    detections = []

    for orphan in orphaned:
        center_frame = orphan.get("frame") or int(orphan["timestamp"] * fps)

        # Skip if now too close to a surviving detection (could happen if
        # we already rescued something nearby in this loop)
        center_time = center_frame / fps
        too_close = any(abs(center_time - st) < MIN_DEDUP_GAP for st in surviving_times)
        if too_close:
            continue

        best_shot_type = None
        best_confidence = 0.0
        best_not_shot = 1.0
        best_probe_frame = center_frame

        for offset in probe_offsets:
            probe_frame = center_frame + offset
            if probe_frame < 0 or probe_frame >= total_frames:
                continue
            if not frames[probe_frame]:
                continue

            rescue_meta = {
                "source": "rescue_scan",
                "audio_amplitude": orphan.get("audio_amplitude", 0.0),
                "velocity": orphan.get("velocity", 0.0),
            }
            ml_frames = frames_3d if frames_3d is not None else frames
            features = extract_features(
                ml_frames, probe_frame, fps, dominant_hand,
                detection_meta=rescue_meta,
            )
            if features is None:
                continue

            X = [[features.get(f, 0.0) for f in feature_names]]

            try:
                proba = model.predict_proba(X)[0]
                pred_idx = proba.argmax()
                shot_type = model.classes_[pred_idx]
                confidence = float(proba[pred_idx])
                not_shot_prob = float(proba[list(model.classes_).index("not_shot")])
            except Exception:
                continue

            if shot_type == "not_shot" or shot_type is None:
                continue

            if confidence > best_confidence and not_shot_prob < T['ns_rescue']:
                best_shot_type = shot_type
                best_confidence = confidence
                best_not_shot = not_shot_prob
                best_probe_frame = probe_frame

        if best_shot_type is None:
            continue
        if best_not_shot > T['ns_rescue_reject']:
            continue
        # Higher confidence bar than other passes — rescue is a last resort,
        # need stronger ML signal to justify re-adding a removed detection
        if best_confidence < T['mc_rescue']:
            continue

        best_time = best_probe_frame / fps
        fused_conf = min(1.0, best_confidence * 0.4 + 0.15)

        det = {
            "timestamp": round(best_time, 3),
            "frame": best_probe_frame,
            "shot_type": best_shot_type,
            "confidence": round(fused_conf, 3),
            "tier": "low",
            "source": "rescue_scan_ml",
            "audio_peak_time": orphan.get("audio_peak_time"),
            "audio_amplitude": orphan.get("audio_amplitude", 0.0),
            "heuristic_frame": None,
            "pattern_confidence": round(best_confidence, 3),
            "trigger": (f"rescue orphan@{orphan['timestamp']:.1f}s "
                        f"ml:{best_shot_type}({best_confidence:.2f})"),
            "velocity": orphan.get("velocity", 0.0),
            "ml_confidence": round(best_confidence, 3),
            "not_shot_prob": round(best_not_shot, 3),
        }
        detections.append(det)
        surviving_times.append(best_time)

        if verbose:
            print(f"      Rescue {best_time:.1f}s: {best_shot_type}"
                  f"({best_confidence:.2f}) not_shot={best_not_shot:.2f} "
                  f"(orphan from {orphan['timestamp']:.1f}s)")

    return detections


def fused_detect(video_path, pose_path, dominant_hand="right",
                 audio_threshold=98.5, audio_min_gap=800,
                 audio_min_amplitude=AUDIO_MIN_AMPLITUDE,
                 window_sec=AUDIO_POSE_WINDOW,
                 use_model=True, verbose=False,
                 poses_3d_path=None):
    """Run fused audio + heuristic detection.

    Strategy: anchor on heuristic velocity spikes (fewer, higher quality),
    then confirm each with audio. Remaining audio-only peaks are secondary.

    Args:
        video_path: Path to preprocessed video (.mp4)
        pose_path: Path to pose JSON
        dominant_hand: "left" or "right"
        audio_threshold: Percentile threshold for audio peak detection
        audio_min_gap: Minimum gap between audio hits in ms
        audio_min_amplitude: Minimum RMS amplitude (filters distant hits)
        window_sec: Seconds around heuristic spike to search for audio confirmation
        use_model: If True, use trained ML classifier when available
        verbose: If True, log rejected detections (not_shot, low-confidence)

    Returns:
        dict with detections list, metadata, and summary
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # ── Load optional racket detection data ──────────────────
    racket_frames = _load_racket_data(video_name)
    if racket_frames is not None:
        print(f"  Loaded racket detection data ({len(racket_frames)} frames)")

    # ── Step 1: Audio detection ──────────────────────────────
    print(f"  Extracting audio...")
    sample_rate = 16000
    audio = extract_audio(video_path, sample_rate)
    audio_duration = len(audio) / sample_rate
    print(f"    Duration: {audio_duration:.1f}s")

    print(f"  Detecting audio peaks (threshold={audio_threshold}%, min_gap={audio_min_gap}ms)...")
    peaks_with_amp = detect_peaks(audio, sample_rate, audio_threshold, audio_min_gap,
                                  return_amplitudes=True)
    print(f"    Found {len(peaks_with_amp)} audio peaks")

    # Filter by amplitude — removes quiet hits from partner/adjacent courts
    if audio_min_amplitude > 0:
        before = len(peaks_with_amp)
        peaks_with_amp = [(t, a) for t, a in peaks_with_amp if a >= audio_min_amplitude]
        filtered = before - len(peaks_with_amp)
        if filtered:
            print(f"    Filtered {filtered} quiet peaks (below {audio_min_amplitude:.3f} RMS)")
            print(f"    Remaining: {len(peaks_with_amp)} loud peaks")

    # ── Step 1b: Spectral flux onset detection ────────────────
    print(f"  Detecting spectral flux onsets...")
    spectral_onsets = detect_spectral_onsets(
        audio, sample_rate, percentile_threshold=92, min_gap_ms=300
    )
    print(f"    Found {len(spectral_onsets)} spectral onsets")

    # Merge spectral onsets with RMS peaks (skip onsets within 0.2s of existing RMS peak)
    rms_times = set(t for t, _ in peaks_with_amp)
    new_spectral = []
    for onset_time, flux_val in spectral_onsets:
        if not any(abs(onset_time - rt) < 0.2 for rt in rms_times):
            new_spectral.append((onset_time, flux_val))

    if new_spectral:
        print(f"    {len(new_spectral)} new spectral onsets (not near RMS peaks)")

    # Build lookup: time → amplitude, and plain time list for compatibility
    audio_peaks = [t for t, a in peaks_with_amp]
    audio_amplitudes = {t: a for t, a in peaks_with_amp}

    video_fps = get_video_fps(video_path)

    # ── Step 2: Load poses ───────────────────────────────────
    print(f"  Loading poses...")
    frames, pose_fps, total_frames = load_pose_frames(pose_path)
    fps = pose_fps or video_fps
    print(f"    {total_frames} frames at {fps} fps")

    # Load 3D poses for ML classification (if available)
    frames_3d = None
    if poses_3d_path and os.path.exists(poses_3d_path):
        print(f"  Loading 3D poses for classification...")
        frames_3d, _, _ = load_pose_frames(poses_3d_path)
        print(f"    3D poses loaded ({sum(1 for f in frames_3d if f.get('_3d_lifted'))} lifted frames)")

    # ── Step 3: Heuristic velocity spikes ────────────────────
    wrist_idx = RIGHT_WRIST if dominant_hand == "right" else LEFT_WRIST
    velocities = compute_wrist_velocities(frames, wrist_idx, fps)
    heuristic_spikes = find_velocity_spikes(
        velocities,
        min_vel=THRESHOLDS["velocity_min"],
        spike_ratio=THRESHOLDS["velocity_spike_ratio"],
        min_gap=THRESHOLDS["min_shot_gap_frames"],
    )
    print(f"    {len(heuristic_spikes)} heuristic velocity spikes")

    # ── Step 4: Fuse — anchor on heuristic spikes ────────────
    # For each heuristic spike, find the closest audio peak (if any).
    # This avoids the problem of many noisy audio peaks matching one spike.
    detections = []
    matched_audio_times = set()

    for spike_frame, velocity in heuristic_spikes:
        spike_time = spike_frame / fps

        # Find closest audio peak within window
        best_audio = None
        best_audio_dist = float("inf")
        for peak_time in audio_peaks:
            dist = abs(spike_time - peak_time)
            if dist <= window_sec and dist < best_audio_dist:
                best_audio_dist = dist
                best_audio = peak_time

        has_audio = best_audio is not None
        if has_audio:
            matched_audio_times.add(best_audio)

        # Build detection_meta for feature extraction (source-aware features)
        spike_meta = {
            "source": "audio+heuristic" if has_audio else "heuristic_only",
            "audio_amplitude": audio_amplitudes.get(best_audio, 0) if best_audio else 0.0,
            "velocity": velocity,
        }

        # Classify the shot — try ML model first, fall back to heuristic
        ml_type, ml_conf, ml_probs = None, 0.0, {}
        if use_model:
            ml_type, ml_conf, ml_probs = classify_with_model(
                frames, spike_frame, fps, dominant_hand,
                detection_meta=spike_meta,
                frames_3d=frames_3d,
            )

        # ML rejection gate: if model predicts not_shot with high probability, skip
        not_shot_prob = ml_probs.get("not_shot", 0.0)
        if use_model and not_shot_prob > T['ns_permissive']:
            if verbose:
                print(f"    ML rejected: {spike_time:.1f}s not_shot={not_shot_prob:.2f} "
                      f"(vel={velocity:.1f})")
            continue

        # ML confidence-based rejection: if best class confidence < mc_weak, reject
        if use_model and ml_type and ml_type != "not_shot" and ml_conf < T['mc_weak']:
            if verbose:
                print(f"    Low-conf rejected: {spike_time:.1f}s "
                      f"ml:{ml_type}({ml_conf:.2f})")
            continue

        # Heuristic classification (always run — needed for trigger/pattern info)
        is_serve, serve_conf, serve_trigger = detect_serve_pattern(
            frames, spike_frame, fps
        )
        if is_serve and serve_conf > 0.4:
            h_shot_type = "serve"
            pattern_conf = serve_conf
            trigger = serve_trigger
        else:
            h_shot_type, pattern_conf, trigger = analyze_stroke_pattern(
                frames, spike_frame, dominant_hand, fps
            )

        has_pattern = h_shot_type != "neutral" and pattern_conf > 0.3
        pattern_failed = ("low pose confidence" in trigger
                          or "insufficient data" in trigger)

        # Filter out not_shot predictions from shot_type selection
        if ml_type == "not_shot":
            ml_type = None
            ml_conf = 0.0

        # Decide final shot_type: ML model wins if confident enough
        if ml_type and ml_conf >= T['mc_strong']:
            shot_type = ml_type
            trigger = f"ml:{ml_type}({ml_conf:.2f}) h:{trigger}"
        elif ml_type and ml_conf >= T['mc_moderate'] and not has_pattern and not pattern_failed:
            # Heuristic found no pattern but pose data was readable — ML
            # with moderate confidence takes over since there's no competing opinion
            shot_type = ml_type
            trigger = f"ml:{ml_type}({ml_conf:.2f}) h:{trigger}"
        elif ml_type and T['mc_moderate'] <= ml_conf < T['mc_strong'] and has_pattern and ml_type != h_shot_type:
            # ML has moderate signal and disagrees with heuristic.
            # If heuristic also has some pattern, trust ML type with reduced
            # confidence rather than punting to unknown_shot.
            if pattern_conf > 0.3:
                shot_type = ml_type
                trigger = f"ml:{ml_type}({ml_conf:.2f}) over h:{trigger}"
            else:
                shot_type = "unknown_shot"
                trigger = f"ml:{ml_type}({ml_conf:.2f}) disagrees h:{trigger}"
        elif ml_type and T['mc_weak'] <= ml_conf < T['mc_moderate'] and has_pattern and ml_type != h_shot_type:
            # Very low ML confidence — still mark unknown
            shot_type = "unknown_shot"
            trigger = f"ml:{ml_type}({ml_conf:.2f}) disagrees h:{trigger}"
        else:
            shot_type = h_shot_type

        if has_audio and (has_pattern or (ml_type and ml_conf >= T['mc_strong'])) and not pattern_failed:
            # BEST CASE: audio confirms + classified (ML or heuristic)
            if ml_type and ml_conf >= T['mc_strong']:
                fused_conf = min(1.0, ml_conf * 0.6 + 0.4)
            else:
                fused_conf = min(1.0, pattern_conf * 0.6 + 0.4)
            tier = "high"
            source = "audio+heuristic"
        elif has_audio and not pattern_failed:
            # Audio confirms velocity spike, but pattern returned neutral.
            if shot_type == "neutral":
                shot_type = "unknown_shot"
            fused_conf = 0.7
            tier = "high"
            source = "audio+heuristic"
        elif has_audio and pattern_failed:
            # Audio hit near a velocity spike, but pose data was bad.
            if ml_type and ml_conf >= T['mc_strong']:
                shot_type = ml_type
                fused_conf = min(1.0, ml_conf * 0.5 + 0.3)
                trigger = f"ml:{ml_type}({ml_conf:.2f}) pose_failed"
            else:
                shot_type = "unknown_shot"
                fused_conf = 0.65
            tier = "high"
            source = "audio+spike_no_pose"
        elif (has_pattern or (ml_type and ml_conf >= T['mc_strong'])) and not pattern_failed:
            # Clear stroke pattern but no audio confirmation
            if ml_type and ml_conf >= T['mc_strong']:
                fused_conf = min(1.0, ml_conf * 0.5 + 0.15)
            else:
                fused_conf = min(1.0, pattern_conf * 0.5 + 0.15)
            tier = "low"
            source = "heuristic_only"
        else:
            # Velocity spike only, no audio, no pattern — skip
            continue

        effective_pattern_conf = ml_conf if (ml_type and ml_conf >= T['mc_strong']) else (pattern_conf if has_pattern else 0.0)

        detections.append({
            "timestamp": round(spike_time, 3),
            "frame": spike_frame,
            "shot_type": shot_type,
            "confidence": round(fused_conf, 3),
            "tier": tier,
            "source": source,
            "audio_peak_time": round(best_audio, 3) if best_audio else None,
            "audio_amplitude": round(audio_amplitudes.get(best_audio, 0), 6) if best_audio else 0.0,
            "heuristic_frame": spike_frame,
            "pattern_confidence": round(effective_pattern_conf, 3),
            "trigger": trigger,
            "velocity": round(velocity, 1),
            "ml_confidence": round(ml_conf, 3) if ml_type else None,
            "not_shot_prob": round(not_shot_prob, 3),
        })

    # ── Step 4b: Unmatched audio peaks — try pose classification ──
    # For audio peaks without a nearby velocity spike, still try to
    # classify from pose data. The player may be swinging but below
    # the strict velocity threshold.
    from scripts.heuristic_detect import get_pose_confidence

    for peak_time in audio_peaks:
        if peak_time in matched_audio_times:
            continue
        # Skip if too close to an existing detection
        too_close = any(
            abs(peak_time - d["timestamp"]) < MIN_DEDUP_GAP
            for d in detections
        )
        if too_close:
            continue

        peak_frame = int(peak_time * fps)

        # Try to classify from pose data at this timestamp
        shot_type = "unknown_shot"
        pattern_conf = 0.0
        trigger = "audio peak only"
        tier = "medium"
        confidence = 0.4
        audio_ml_conf = None
        audio_not_shot_prob = 0.0

        # Build detection_meta for audio-only peaks
        audio_meta = {
            "source": "audio_only",
            "audio_amplitude": audio_amplitudes.get(peak_time, 0),
            "velocity": 0.0,
        }

        # Try ML model first for audio-only peaks
        if use_model and peak_frame < len(frames):
            ml_type, ml_conf, ml_probs = classify_with_model(
                frames, peak_frame, fps, dominant_hand,
                detection_meta=audio_meta,
                frames_3d=frames_3d,
            )
            audio_not_shot_prob = ml_probs.get("not_shot", 0.0)

            # ML rejection gate for audio-only peaks
            if audio_not_shot_prob > T['ns_permissive']:
                if verbose:
                    print(f"    ML rejected audio-only: {peak_time:.1f}s "
                          f"not_shot={audio_not_shot_prob:.2f}")
                continue

            # Filter out not_shot from shot type
            if ml_type == "not_shot":
                ml_type = None
                ml_conf = 0.0

            # Source-aware threshold: audio_only needs higher confidence (0.6)
            if ml_type and ml_conf >= 0.6:
                shot_type = ml_type
                pattern_conf = ml_conf
                audio_ml_conf = ml_conf
                trigger = f"audio + ml:{ml_type}({ml_conf:.2f})"
                confidence = min(1.0, ml_conf * 0.5 + 0.35)
                tier = "high" if confidence >= 0.7 else "medium"
            elif ml_type and ml_conf >= 0.5:
                # Between 0.5-0.6 for audio_only: accept but lower confidence
                shot_type = ml_type
                pattern_conf = ml_conf
                audio_ml_conf = ml_conf
                trigger = f"audio + ml:{ml_type}({ml_conf:.2f}) [audio-only]"
                confidence = min(1.0, ml_conf * 0.4 + 0.25)
                tier = "medium"

        if peak_frame < len(frames) and pattern_conf == 0.0:
            pose_conf = get_pose_confidence(frames[peak_frame])
            if pose_conf >= 0.5:
                # Good pose data — try stroke pattern classification
                is_serve, serve_conf, serve_trigger = detect_serve_pattern(
                    frames, peak_frame, fps
                )
                if is_serve and serve_conf > 0.4:
                    # Source-aware: audio+pose serves need ML conf > 0.5
                    if use_model and audio_ml_conf is not None and audio_ml_conf < 0.5:
                        if verbose:
                            print(f"    Audio+pose serve rejected: {peak_time:.1f}s "
                                  f"ml_conf={audio_ml_conf:.2f} < 0.5")
                        continue
                    shot_type = "serve"
                    pattern_conf = serve_conf
                    trigger = f"audio + pose: {serve_trigger}"
                    confidence = min(1.0, serve_conf * 0.5 + 0.35)
                    tier = "high" if confidence >= 0.7 else "medium"
                else:
                    st, pc, tr = analyze_stroke_pattern(
                        frames, peak_frame, dominant_hand, fps
                    )
                    if st != "neutral" and pc > 0.3:
                        shot_type = st
                        pattern_conf = pc
                        trigger = f"audio + pose: {tr}"
                        confidence = min(1.0, pc * 0.5 + 0.35)
                        tier = "high" if confidence >= 0.7 else "medium"
                    else:
                        trigger = f"audio peak, pose visible but no pattern: {tr}"
                        confidence = 0.45

        detections.append({
            "timestamp": round(peak_time, 3),
            "frame": peak_frame,
            "shot_type": shot_type,
            "confidence": round(confidence, 3),
            "tier": tier,
            "source": "audio+pose" if pattern_conf > 0 else "audio_only",
            "audio_peak_time": round(peak_time, 3),
            "audio_amplitude": round(audio_amplitudes.get(peak_time, 0), 6),
            "heuristic_frame": None,
            "pattern_confidence": round(pattern_conf, 3),
            "trigger": trigger,
            "velocity": 0.0,
            "ml_confidence": round(audio_ml_conf, 3) if audio_ml_conf is not None else None,
            "not_shot_prob": round(audio_not_shot_prob, 3),
        })

    # ── Step 4b2: Jerk spike pass — d³pos/dt³ + ML ──────────
    # Run jerk EARLY (before spectral/second-pass) so those later steps
    # don't block jerk candidates with detections that may get filtered out.
    if use_model:
        first_pass_times_jerk = [d["timestamp"] for d in detections]
        print(f"  Running jerk spike pass...")
        jerk_hits = _jerk_spike_pass(
            velocities, first_pass_times_jerk, frames, fps,
            dominant_hand, verbose, frames_3d=frames_3d,
        )
        if jerk_hits:
            print(f"    Jerk pass found {len(jerk_hits)} additional detections")
            detections.extend(jerk_hits)

    # ── Step 4b3: Spectral flux onsets — ML classification ──
    if use_model and new_spectral:
        from scripts.heuristic_detect import get_pose_confidence as _gpc2
        print(f"  Processing spectral onsets through ML...")
        spectral_added = 0
        for onset_time, flux_val in new_spectral:
            # Skip if too close to any existing detection
            too_close = any(
                abs(onset_time - d["timestamp"]) < MIN_DEDUP_GAP
                for d in detections
            )
            if too_close:
                continue

            onset_frame = int(onset_time * fps)
            if onset_frame >= len(frames) or not frames[onset_frame]:
                continue

            onset_meta = {
                "source": "spectral_onset",
                "audio_amplitude": 0.0,
                "velocity": 0.0,
            }

            ml_type, ml_conf, ml_probs = classify_with_model(
                frames, onset_frame, fps, dominant_hand,
                detection_meta=onset_meta,
                frames_3d=frames_3d,
            )

            not_shot_prob = ml_probs.get("not_shot", 0.0)
            if not_shot_prob > T['ns_moderate']:
                if verbose:
                    print(f"    Spectral rejected {onset_time:.1f}s: "
                          f"not_shot={not_shot_prob:.2f}")
                continue

            if ml_type == "not_shot" or ml_type is None:
                continue
            if ml_conf < T['mc_strong']:
                continue

            fused_conf = min(1.0, ml_conf * 0.4 + 0.25)
            detections.append({
                "timestamp": round(onset_time, 3),
                "frame": onset_frame,
                "shot_type": ml_type,
                "confidence": round(fused_conf, 3),
                "tier": "medium",
                "source": "spectral_onset",
                "audio_peak_time": round(onset_time, 3),
                "audio_amplitude": 0.0,
                "heuristic_frame": None,
                "pattern_confidence": round(ml_conf, 3),
                "trigger": f"spectral_flux={flux_val:.1f} ml:{ml_type}({ml_conf:.2f})",
                "velocity": 0.0,
                "ml_confidence": round(ml_conf, 3),
                "not_shot_prob": round(not_shot_prob, 3),
            })
            spectral_added += 1

            if verbose:
                print(f"    Spectral accepted {onset_time:.1f}s: "
                      f"{ml_type}({ml_conf:.2f})")

        if spectral_added:
            print(f"    Spectral onsets added {spectral_added} detections")

    # ── Step 4c: Second pass — low-threshold velocity + ML ──
    if use_model:
        first_pass_times = [d["timestamp"] for d in detections]
        print(f"  Running second pass (low-threshold + ML)...")
        second_pass = _second_pass_low_threshold(
            velocities, first_pass_times, frames, fps,
            dominant_hand, verbose, frames_3d=frames_3d,
        )
        if second_pass:
            print(f"    Second pass found {len(second_pass)} additional detections")
            detections.extend(second_pass)

    # ── Step 4d: Sliding window scan — pure ML on pose ────
    if use_model:
        existing_times = [d["timestamp"] for d in detections]
        print(f"  Running sliding window scan...")
        window_hits = _sliding_window_scan(
            existing_times, frames, fps, dominant_hand, verbose,
            frames_3d=frames_3d,
        )
        if window_hits:
            print(f"    Sliding window found {len(window_hits)} additional detections")
            detections.extend(window_hits)

    # ── Step 4e: Rally rhythm fill — temporal gap analysis ──
    if use_model:
        existing_times_rhythm = [d["timestamp"] for d in detections]
        print(f"  Running rally rhythm fill...")
        rhythm_hits = _rally_rhythm_fill(
            existing_times_rhythm, frames, fps, dominant_hand, verbose,
            frames_3d=frames_3d,
        )
        if rhythm_hits:
            print(f"    Rhythm fill found {len(rhythm_hits)} additional detections")
            detections.extend(rhythm_hits)

    # ── Step 5: Filter, sort, and deduplicate ────────────────
    # Save pre-filter timestamps for post-dedup rescue scan
    pre_filter_detections = list(detections)

    # Apply stricter amplitude threshold for MEDIUM-only detections
    # Exempt ML-only sources (low_threshold_ml, sliding_window_ml) — they have no audio
    ml_only_sources = {"low_threshold_ml", "sliding_window_ml", "jerk_spike_ml", "rhythm_fill_ml", "spectral_onset", "rescue_scan_ml"}
    before_med_filter = len(detections)
    detections = [
        d for d in detections
        if d["tier"] != "medium"
        or d.get("source") in ml_only_sources
        or d.get("audio_amplitude", 0) >= AUDIO_MIN_AMPLITUDE_MEDIUM
    ]
    med_filtered = before_med_filter - len(detections)
    if med_filtered:
        print(f"    Filtered {med_filtered} quiet MEDIUM detections (below {AUDIO_MIN_AMPLITUDE_MEDIUM} RMS)")

    # Remove unclassified detections where ML couldn't verify — pure noise
    # audio_only + unknown_shot + no ML = audio peak with nothing backing it
    # audio+spike_no_pose + unknown_shot + no ML = velocity spike + audio but no visible player
    before_unclassified = len(detections)
    detections = [
        d for d in detections
        if not (d["shot_type"] == "unknown_shot"
                and d.get("ml_confidence") is None
                and d.get("source") in ("audio_only", "audio+spike_no_pose"))
    ]
    unclassified_removed = before_unclassified - len(detections)
    if unclassified_removed:
        print(f"    Removed {unclassified_removed} unclassified/unverified detections")

    # Apply wrist height filter — reject detections where wrist isn't elevated
    # (ball pickups, dribbling, walking all have wrist near hip level)
    before_wrist = len(detections)
    wrist_filtered = []
    for d in detections:
        frame_idx = d.get("frame") or d.get("heuristic_frame")
        if frame_idx is None or frame_idx >= len(frames):
            # No frame info — keep the detection (can't check)
            wrist_filtered.append(d)
            continue

        height = check_wrist_above_hip(frames, frame_idx, fps, dominant_hand)
        if height is None:
            # No pose data at this frame — keep the detection
            wrist_filtered.append(d)
            continue

        # Store the measurement for debugging
        d["wrist_above_hip"] = round(height, 4)

        if height >= MIN_WRIST_ABOVE_HIP:
            wrist_filtered.append(d)
        else:
            print(f"    Wrist filter: rejected {d['timestamp']:.1f}s "
                  f"({d['shot_type']}, {d['tier']}) — "
                  f"wrist only {height:.3f}m above hip")
    detections = wrist_filtered
    wrist_rejected = before_wrist - len(detections)
    if wrist_rejected:
        print(f"    Wrist height filter: removed {wrist_rejected} detections "
              f"(below {MIN_WRIST_ABOVE_HIP}m above hip)")

    detections.sort(key=lambda d: d["timestamp"])

    # Slo-mo cluster merging: groups of 3+ same-type within 4s → keep best
    pre_cluster = len(detections)
    detections = merge_slomo_clusters(detections, cluster_window=4.0)

    pre_dedup = len(detections)
    detections = deduplicate_detections(detections, min_gap=MIN_DEDUP_GAP)
    if len(detections) < pre_dedup:
        print(f"    Deduplicated: {pre_dedup} -> {len(detections)}")

    # ── Step 5b: Post-dedup rescue scan ───────────────────────
    # Detections created by early pipeline stages can block later candidate
    # generators (sliding window, jerk) via proximity filters, then get
    # removed by dedup/cluster merge — leaving nothing at that timestamp.
    # Re-probe at orphaned timestamps with ML to rescue real shots.
    if use_model:
        rescue_hits = _post_dedup_rescue(
            pre_filter_detections, detections, frames, fps,
            dominant_hand, verbose, frames_3d=frames_3d,
        )
        if rescue_hits:
            print(f"    Post-dedup rescue: recovered {len(rescue_hits)} detections")
            detections.extend(rescue_hits)
            detections.sort(key=lambda d: d["timestamp"])
            detections = deduplicate_detections(detections, min_gap=MIN_DEDUP_GAP)

    # ── Step 5c: Final weak-detection filter ─────────────────
    # Applied after dedup to avoid cascade effects through proximity filtering.
    # Removes detections where multiple weak signals combine to suggest FP:
    #   - heuristic_only + elevated not_shot + low ML confidence
    #   - jerk_spike_ml + elevated not_shot + low ML confidence
    #   - jerk_spike_ml + low fused confidence + any not_shot signal
    #   - low_threshold_ml with low fused confidence
    #   - audio+heuristic with very low ML confidence (borderline model rejection)
    before_weak = len(detections)
    detections = [
        d for d in detections
        if not (
            # Heuristic-only with weak ML support
            (d.get("source") == "heuristic_only"
             and d.get("not_shot_prob", 0) > T['ns_weak_heuristic']
             and (d.get("ml_confidence") or 0) < T['mc_floor_heuristic_only'])
            or
            # Heuristic-only or audio+heuristic with moderate not_shot
            # (no TP from these sources has ns >= 0.25; validated on 12 GT videos)
            (d.get("source") in ("heuristic_only", "audio+heuristic")
             and d.get("not_shot_prob", 0) >= 0.25)
            or
            # Jerk spike with weak ML support (high not_shot + low ml_confidence)
            (d.get("source") == "jerk_spike_ml"
             and d.get("not_shot_prob", 0) > T['ns_weak_jerk']
             and (d.get("ml_confidence") or 0) < T['mc_floor_jerk'])
            or
            # Jerk spike with low fused confidence + any not_shot signal
            (d.get("source") == "jerk_spike_ml"
             and d.get("confidence", 1.0) < T['fc_weak_jerk']
             and d.get("not_shot_prob", 0) > T['ns_weak_jerk_floor'])
            or
            # Low-threshold pass with low fused confidence
            (d.get("source") == "low_threshold_ml"
             and d.get("confidence", 1.0) < T['fc_weak_low'])
            or
            # Audio+heuristic with very low ML confidence floor
            (d.get("source") == "audio+heuristic"
             and (d.get("ml_confidence") or 0) < T['mc_floor_audio_heuristic'])
        )
    ]
    weak_removed = before_weak - len(detections)
    if weak_removed:
        print(f"    Weak detection filter: removed {weak_removed} "
              f"(low-confidence heuristic/jerk detections)")

    # ── Step 5d: Racket visibility filter ──────────────────────
    # When racket detection data is available, remove heuristic-only
    # detections where the racket is barely visible (player likely off-camera).
    if racket_frames is not None:
        before_racket = len(detections)
        filtered_dets = []
        for d in detections:
            frame = int(round(d["timestamp"] * fps))
            rk_vis = _racket_visibility(racket_frames, frame)
            if (d.get("source") == "heuristic_only"
                    and rk_vis is not None and rk_vis < 0.30):
                if verbose:
                    print(f"    Racket filter: removing {d['timestamp']:.1f}s "
                          f"(heuristic_only, racket_vis={rk_vis:.2f})")
                continue
            filtered_dets.append(d)
        racket_removed = before_racket - len(filtered_dets)
        if racket_removed:
            print(f"    Racket visibility filter: removed {racket_removed} "
                  f"(heuristic_only with racket_vis < 0.30)")
        detections = filtered_dets

    # ── Step 6: Build summary ────────────────────────────────
    tier_counts = {"high": 0, "medium": 0, "low": 0}
    type_counts = {}
    source_counts = {}
    for det in detections:
        tier_counts[det["tier"]] += 1
        st = det["shot_type"]
        type_counts[st] = type_counts.get(st, 0) + 1
        src = det.get("source", "unknown")
        source_counts[src] = source_counts.get(src, 0) + 1

    result = {
        "version": 5,
        "detector": "fused_audio_heuristic_ml",
        "source_video": video_name,
        "video_path": video_path,
        "pose_path": pose_path,
        "fps": fps,
        "total_frames": total_frames,
        "duration": round(audio_duration, 2),
        "dominant_hand": dominant_hand,
        "parameters": {
            "audio_threshold_percentile": audio_threshold,
            "audio_min_gap_ms": audio_min_gap,
            "audio_min_amplitude": audio_min_amplitude,
            "pose_window_sec": window_sec,
            "dedup_gap_sec": MIN_DEDUP_GAP,
        },
        "summary": {
            "total_detections": len(detections),
            "by_tier": tier_counts,
            "by_type": type_counts,
            "by_source": source_counts,
            "audio_peaks": len(audio_peaks),
            "heuristic_spikes": len(heuristic_spikes),
        },
        "detections": detections,
    }

    return result


def print_results(result):
    """Print fused detection results."""
    print()
    print(f"{'='*65}")
    print(f"FUSED DETECTION RESULTS")
    print(f"{'='*65}")
    print(f"Source: {result['source_video']}")
    print(f"Duration: {result['duration']:.1f}s, FPS: {result['fps']}")
    print(f"Audio peaks: {result['summary']['audio_peaks']}, "
          f"Heuristic spikes: {result['summary']['heuristic_spikes']}")
    print()

    summary = result["summary"]
    print(f"Total detections: {summary['total_detections']}")
    print(f"  HIGH   (audio+heuristic): {summary['by_tier']['high']}")
    print(f"  MEDIUM (audio only):      {summary['by_tier']['medium']}")
    print(f"  LOW    (heuristic only):   {summary['by_tier']['low']}")
    print()

    print(f"By shot type:")
    for st, count in sorted(summary["by_type"].items()):
        print(f"  {st}: {count}")
    print()

    if "by_source" in summary:
        print(f"By source:")
        for src, count in sorted(summary["by_source"].items()):
            print(f"  {src}: {count}")
        print()

    print(f"{'Tier':<8} {'Type':<14} {'Time':>7} {'Frame':>7} {'Conf':>6}  Trigger")
    print(f"{'─'*65}")

    for det in result["detections"][:50]:
        tier_str = det["tier"].upper()
        print(f"  {tier_str:<6} {det['shot_type']:<14} {det['timestamp']:6.1f}s "
              f"{det['frame']:>6}  {det['confidence']:.2f}   {det['trigger'][:40]}")

    if len(result["detections"]) > 50:
        print(f"  ... and {len(result['detections']) - 50} more")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Fused audio + heuristic shot detection"
    )
    parser.add_argument("video", help="Path to preprocessed video (.mp4)")
    parser.add_argument("--poses", help="Path to pose JSON (auto-discovers if omitted)")
    parser.add_argument("-o", "--output", help="Output JSON path")
    parser.add_argument("--dominant-hand", choices=["left", "right"],
                        default="right", help="Player's dominant hand (default: right)")
    parser.add_argument("--audio-threshold", type=float, default=98.5,
                        help="Audio peak percentile threshold (default: 98.5)")
    parser.add_argument("--audio-min-gap", type=float, default=800,
                        help="Min gap between audio hits in ms (default: 800)")
    parser.add_argument("--audio-min-amplitude", type=float, default=AUDIO_MIN_AMPLITUDE,
                        help=f"Min RMS amplitude for audio peaks (default: {AUDIO_MIN_AMPLITUDE})")
    parser.add_argument("--window", type=float, default=AUDIO_POSE_WINDOW,
                        help=f"Audio-pose matching window in seconds (default: {AUDIO_POSE_WINDOW})")
    parser.add_argument("--min-confidence", type=float, default=0.0,
                        help="Minimum confidence to include (default: 0.0)")
    parser.add_argument("--no-model", action="store_true",
                        help="Disable ML classifier, use heuristic only")
    parser.add_argument("--poses-dir", default=None,
                        help="Directory with pose JSONs (overrides default POSES_DIR)")
    parser.add_argument("--poses-3d-dir", default=None,
                        help="Directory with 3D-lifted pose JSONs for ML classification")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Log rejected detections (not_shot, low-confidence)")
    args = parser.parse_args()

    video_path = args.video
    if not os.path.isabs(video_path):
        video_path = os.path.join(os.getcwd(), video_path)

    if not os.path.exists(video_path):
        print(f"[ERROR] Video not found: {video_path}")
        sys.exit(1)

    # Auto-discover pose file
    poses_dir = args.poses_dir or POSES_DIR
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    if args.poses:
        pose_path = args.poses
    else:
        pose_path = os.path.join(poses_dir, video_name + ".json")
        if not os.path.exists(pose_path):
            # Try without _poses suffix
            alt = os.path.join(poses_dir, video_name + "_poses.json")
            if os.path.exists(alt):
                pose_path = alt

    if not os.path.exists(pose_path):
        print(f"[ERROR] Pose file not found: {pose_path}")
        print(f"  Run extract_poses.py on the video first.")
        sys.exit(1)

    det_dir = os.path.join(PROJECT_ROOT, "detections")
    os.makedirs(det_dir, exist_ok=True)
    output_path = args.output or os.path.join(
        det_dir, f"{video_name}_fused_detections.json"
    )

    # Auto-discover 3D pose file for ML classification
    poses_3d_path = None
    if args.poses_3d_dir:
        p3d = os.path.join(args.poses_3d_dir, video_name + ".json")
        if os.path.exists(p3d):
            poses_3d_path = p3d
        else:
            print(f"  [WARN] 3D pose file not found: {p3d}, using 2D only")

    print(f"Fused Detection: {video_name}")
    print(f"  Video: {os.path.basename(video_path)}")
    print(f"  Poses: {os.path.basename(pose_path)}")
    if poses_3d_path:
        print(f"  3D Poses: {os.path.basename(poses_3d_path)} (for ML classification)")
    print()

    result = fused_detect(
        video_path, pose_path,
        dominant_hand=args.dominant_hand,
        audio_threshold=args.audio_threshold,
        audio_min_gap=args.audio_min_gap,
        audio_min_amplitude=args.audio_min_amplitude,
        window_sec=args.window,
        use_model=not args.no_model,
        verbose=args.verbose,
        poses_3d_path=poses_3d_path,
    )

    # Filter by confidence
    if args.min_confidence > 0:
        before = len(result["detections"])
        result["detections"] = [
            d for d in result["detections"]
            if d["confidence"] >= args.min_confidence
        ]
        filtered = before - len(result["detections"])
        if filtered:
            print(f"  Filtered {filtered} detections below {args.min_confidence} confidence")

    print_results(result)

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
