#!/usr/bin/env python3
"""Biomechanical analysis pipeline for tennis shots.

Computes per-shot biomechanical reports, session summaries with fatigue tracking,
kinetic chain analysis, and cross-session comparison for coaching insights.

Usage:
    # Single session analysis
    python scripts/biomechanical_analysis.py \
        --detections detections/IMG_6703_fused_v5.json \
        --poses poses_full_videos/IMG_6703.json \
        --output analysis/IMG_6703_biomech.json

    # Cross-session comparison
    python scripts/biomechanical_analysis.py \
        --compare analysis/session1.json analysis/session2.json

    # All reviewed videos
    python scripts/biomechanical_analysis.py --all
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import PROJECT_ROOT, POSES_DIR
from scripts.extract_training_features import (
    load_pose_frames, compute_angle_3points, _line_angle_xz,
    compute_backswing_duration, WINDOW_HALF,
)
from scripts.heuristic_detect import (
    get_keypoint, body_midline_x,
    compute_wrist_velocities, compute_wrist_acceleration, compute_wrist_jerk,
    RIGHT_WRIST, LEFT_WRIST, RIGHT_ELBOW, LEFT_ELBOW,
    RIGHT_SHOULDER, LEFT_SHOULDER, RIGHT_HIP, LEFT_HIP,
    LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE,
)

ANALYSIS_DIR = os.path.join(PROJECT_ROOT, "analysis")
DETECTIONS_DIR = os.path.join(PROJECT_ROOT, "detections")

# Shot types worth analyzing biomechanically
ANALYZABLE_TYPES = {"forehand", "backhand", "serve"}


def _safe(val, decimals=3):
    """Round a value safely, returning 0.0 for None."""
    if val is None:
        return 0.0
    return round(float(val), decimals)


def _smooth_window(seq, window=5):
    """Centered moving-average smoother for a 1D numeric sequence.

    Used on per-frame joint velocities before peak-picking, so noise spikes
    from MediaPipe's per-frame jitter don't masquerade as velocity peaks.
    """
    n = len(seq)
    if window <= 1 or n <= 1:
        return list(seq)
    half = window // 2
    out = [0.0] * n
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out[i] = sum(seq[lo:hi]) / (hi - lo)
    return out


def _peak_with_min_prominence(seq, min_prominence=0.5):
    """Largest local maximum with prominence >= threshold; argmax fallback.

    A "local max" here is i where seq[i] > seq[i-1] and seq[i] >= seq[i+1].
    Prominence is seq[i] minus the higher of the two adjacent valley floors
    (lowest values reached before encountering a higher peak on each side).
    Falls back to plain argmax if no qualifying peak — the relative ordering
    across joints is more important than perfect peak isolation per joint.
    """
    n = len(seq)
    if n == 0:
        return None
    if n <= 2:
        return int(max(range(n), key=lambda k: seq[k]))
    best_idx, best_val = None, -float("inf")
    for i in range(1, n - 1):
        if seq[i] > seq[i - 1] and seq[i] >= seq[i + 1]:
            left_floor = seq[i]
            for j in range(i - 1, -1, -1):
                left_floor = min(left_floor, seq[j])
                if seq[j] > seq[i]:
                    break
            right_floor = seq[i]
            for j in range(i + 1, n):
                right_floor = min(right_floor, seq[j])
                if seq[j] > seq[i]:
                    break
            prom = seq[i] - max(left_floor, right_floor)
            if prom >= min_prominence and seq[i] > best_val:
                best_val = seq[i]
                best_idx = i
    if best_idx is None:
        return int(max(range(n), key=lambda k: seq[k]))
    return best_idx


def _accel_zero_crossing_peak(velocities):
    """Pick a velocity peak via acceleration-sign-change (+ → −).

    Amplitude-insensitive — robust on noisy biomech signals where a single
    spurious spike could otherwise dominate a max-vel detector.
    Returns the index whose velocity is highest among + → − crossings, or
    None if no crossing occurred (caller should fall back).
    """
    n = len(velocities)
    if n < 3:
        return None
    accel = [velocities[i + 1] - velocities[i] for i in range(n - 1)]
    candidates = []
    for i in range(1, len(accel)):
        if accel[i - 1] > 0 and accel[i] <= 0:
            candidates.append(i)
    if not candidates:
        return None
    return max(candidates, key=lambda k: velocities[k])


def analyze_shot(frames, center_frame, fps, dominant_hand="right",
                 kinetic_peak_method="smoothed"):
    """Compute full biomechanical analysis for a single shot.

    Returns dict with swing profile, joint angles, kinetic chain, phase durations.

    kinetic_peak_method:
      - "smoothed"            (default) — moving-average smooth + min-prominence
                              peak. Most robust on noisy MediaPipe outputs.
      - "accel_zero_crossing" — smooth, then pick the largest + → − accel
                              transition. Amplitude-insensitive; complements the
                              "smoothed" path under design-partner's A/B plan.
      - "legacy"              — original argmax-of-raw-velocity. Available for
                              regression comparison via --legacy-kinematic-peak.
    """
    n = len(frames)
    cf = min(center_frame, n - 1)

    # Hand-dependent indices
    if dominant_hand == "right":
        d_wrist, d_elbow, d_shoulder = RIGHT_WRIST, RIGHT_ELBOW, RIGHT_SHOULDER
        d_hip, d_knee, d_ankle = RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE
    else:
        d_wrist, d_elbow, d_shoulder = LEFT_WRIST, LEFT_ELBOW, LEFT_SHOULDER
        d_hip, d_knee, d_ankle = LEFT_HIP, LEFT_KNEE, LEFT_ANKLE

    # ── Swing speed profile (20 timepoints) ──────────────────────
    win_start = max(0, cf - WINDOW_HALF)
    win_end = min(n, cf + WINDOW_HALF + 1)
    window_frames = frames[win_start:win_end]
    velocities = compute_wrist_velocities(window_frames, d_wrist, fps)

    # Downsample to 20 evenly-spaced points
    swing_profile = []
    if velocities:
        step = max(1, len(velocities) // 20)
        for i in range(0, len(velocities), step):
            swing_profile.append(_safe(velocities[i]))
        swing_profile = swing_profile[:20]
    while len(swing_profile) < 20:
        swing_profile.append(0.0)

    # ── Trunk rotation at contact ────────────────────────────────
    trunk_rot = 0.0
    if frames[cf]:
        ls = get_keypoint(frames[cf], LEFT_SHOULDER)
        rs = get_keypoint(frames[cf], RIGHT_SHOULDER)
        lh = get_keypoint(frames[cf], LEFT_HIP)
        rh = get_keypoint(frames[cf], RIGHT_HIP)
        if ls and rs and lh and rh:
            sh_angle = _line_angle_xz(ls, rs)
            hip_angle = _line_angle_xz(lh, rh)
            trunk_rot = math.degrees(sh_angle - hip_angle)
            # Clamp to [-180, 180]: angles near ±π wrap (e.g. 175° vs -175°
            # → raw diff 350° though the physical rotation is -10°). Without
            # this, audit `trunk_extreme:|x|>90` flags fire on tiny rotations.
            trunk_rot = ((trunk_rot + 180.0) % 360.0) - 180.0

    # ── Knee bend depth ──────────────────────────────────────────
    min_knee_angle = 180.0
    knee_at_contact = 180.0
    for i in range(max(0, cf - 15), min(n, cf + 16)):
        if not frames[i]:
            continue
        hp = get_keypoint(frames[i], d_hip)
        kp = get_keypoint(frames[i], d_knee)
        ap = get_keypoint(frames[i], d_ankle)
        if hp and kp and ap:
            ang = compute_angle_3points(hp, kp, ap)
            if ang is not None:
                if ang < min_knee_angle:
                    min_knee_angle = ang
                if i == cf:
                    knee_at_contact = ang

    # ── Weight transfer ──────────────────────────────────────────
    # Approximate via hip center X displacement during swing
    weight_transfer = 0.0
    pre_frame = max(0, cf - 15)
    post_frame = min(n - 1, cf + 15)
    if frames[pre_frame] and frames[post_frame]:
        lh_pre = get_keypoint(frames[pre_frame], LEFT_HIP)
        rh_pre = get_keypoint(frames[pre_frame], RIGHT_HIP)
        lh_post = get_keypoint(frames[post_frame], LEFT_HIP)
        rh_post = get_keypoint(frames[post_frame], RIGHT_HIP)
        if lh_pre and rh_pre and lh_post and rh_post:
            hip_x_pre = (lh_pre[0] + rh_pre[0]) / 2
            hip_x_post = (lh_post[0] + rh_post[0]) / 2
            weight_transfer = hip_x_post - hip_x_pre

    # ── Arm extension at contact ─────────────────────────────────
    arm_ext = 0.0
    if frames[cf]:
        s = get_keypoint(frames[cf], d_shoulder)
        e = get_keypoint(frames[cf], d_elbow)
        w = get_keypoint(frames[cf], d_wrist)
        if s and e and w:
            ang = compute_angle_3points(s, e, w)
            if ang is not None:
                arm_ext = ang

    # ── Follow-through angle ─────────────────────────────────────
    # Wrist position relative to shoulder at peak follow-through
    ft_angle = 0.0
    ft_frame = min(n - 1, cf + 20)
    if frames[ft_frame]:
        s = get_keypoint(frames[ft_frame], d_shoulder)
        w = get_keypoint(frames[ft_frame], d_wrist)
        if s and w:
            dx = w[0] - s[0]
            dy = s[1] - w[1]  # invert Y: lower = higher
            ft_angle = math.degrees(math.atan2(dy, dx))

    # ── Recovery time ────────────────────────────────────────────
    # Frames until wrist velocity drops below 3 m/s after contact
    center_in_window = cf - win_start
    recovery_frames = 0
    if velocities:
        for i in range(min(center_in_window, len(velocities) - 1), len(velocities)):
            if velocities[i] < 3.0:
                recovery_frames = i - center_in_window
                break
        else:
            recovery_frames = len(velocities) - center_in_window
    recovery_time_ms = round(recovery_frames / fps * 1000)

    # ── Kinetic chain timing ─────────────────────────────────────
    # Track peak velocity time of: hip, shoulder, elbow, wrist.
    # Correct order: proximal → distal (hip → shoulder → elbow → wrist).
    # Per-frame velocity from world_landmarks is noise-dominated at high fps,
    # so naïve argmax used to pick a noise spike. We now (a) smooth the
    # velocity series and (b) require a local maximum with min prominence,
    # which fixes the "chain reversed" flag rate (see eval/3d-lifting/REPORT.md).
    chain_times = {}
    search_start = max(0, cf - 30)
    search_end = min(n, cf + 10)
    win_len = max(0, search_end - search_start)

    for joint_name, joint_idx in [("hip", d_hip), ("shoulder", d_shoulder),
                                   ("elbow", d_elbow), ("wrist", d_wrist)]:
        # Build a dense per-frame velocity series over the window. Frames
        # without a detection contribute 0 (a gap-fill that beats skipping —
        # a missing frame between two real ones doesn't fake a peak).
        vels = [0.0] * win_len
        for k in range(1, win_len):
            i = search_start + k
            if not frames[i] or not frames[i - 1]:
                continue
            p_curr = get_keypoint(frames[i], joint_idx)
            p_prev = get_keypoint(frames[i - 1], joint_idx)
            if p_curr and p_prev:
                vels[k] = math.sqrt(
                    sum((a - b) ** 2 for a, b in zip(p_curr[:3], p_prev[:3]))
                ) * fps

        if kinetic_peak_method == "legacy":
            peak_local = int(max(range(win_len), key=lambda k: vels[k])) if win_len else 0
        else:
            smoothed = _smooth_window(vels, window=5)
            if kinetic_peak_method == "accel_zero_crossing":
                peak_local = _accel_zero_crossing_peak(smoothed)
                if peak_local is None:
                    # No accel crossing in window — fall back to smoothed peak.
                    peak_local = _peak_with_min_prominence(smoothed, min_prominence=0.5)
            else:  # "smoothed" — default
                peak_local = _peak_with_min_prominence(smoothed, min_prominence=0.5)
            if peak_local is None:
                peak_local = 0

        chain_times[joint_name] = search_start + peak_local

    # Check if ordering is correct (proximal → distal)
    chain_order = ["hip", "shoulder", "elbow", "wrist"]
    chain_frames = [chain_times.get(j, 0) for j in chain_order]
    kinetic_chain_correct = all(
        chain_frames[i] <= chain_frames[i + 1]
        for i in range(len(chain_frames) - 1)
    )

    chain_timing_ms = {
        joint: round((chain_times[joint] - chain_times.get("hip", cf)) / fps * 1000)
        for joint in chain_order
    }

    # ── Phase durations ──────────────────────────────────────────
    backswing_dur_frames = compute_backswing_duration(frames, cf, fps, d_wrist)
    backswing_ms = round(backswing_dur_frames / fps * 1000)

    # Forward swing: from end of backswing to contact
    forward_swing_frames = max(0, cf - (cf - backswing_dur_frames)) if backswing_dur_frames > 0 else 0
    # More precisely: frames from the velocity-rising point to contact
    forward_swing_ms = round(backswing_dur_frames / fps * 1000)  # approximate

    followthrough_ms = recovery_time_ms  # approximation

    # Peak velocity and contact velocity
    peak_vel = max(velocities) if velocities else 0.0
    contact_vel = velocities[min(center_in_window, len(velocities) - 1)] if velocities else 0.0

    return {
        "swing_speed_profile": swing_profile,
        "peak_swing_speed": _safe(peak_vel),
        "contact_swing_speed": _safe(contact_vel),
        "trunk_rotation_at_contact": _safe(trunk_rot, 1),
        "knee_bend_depth": _safe(min_knee_angle, 1),
        "knee_angle_at_contact": _safe(knee_at_contact, 1),
        "weight_transfer": _safe(weight_transfer, 4),
        "arm_extension_at_contact": _safe(arm_ext, 1),
        "followthrough_angle": _safe(ft_angle, 1),
        "recovery_time_ms": recovery_time_ms,
        "kinetic_chain_timing_ms": chain_timing_ms,
        "kinetic_chain_correct": kinetic_chain_correct,
        "phase_durations": {
            "backswing_ms": backswing_ms,
            "forward_swing_ms": forward_swing_ms,
            "followthrough_ms": followthrough_ms,
        },
    }


def analyze_session(detections_path, poses_path, output_path=None,
                    kinetic_peak_method="smoothed"):
    """Run biomechanical analysis on all shots in a session.

    Returns full session analysis dict.

    kinetic_peak_method: see analyze_shot. Defaults to "smoothed" (the
    fix shipped 2026-05-09); pass "legacy" to reproduce pre-fix behavior
    for regression comparison.
    """
    with open(detections_path) as f:
        det_data = json.load(f)

    dominant_hand = det_data.get("dominant_hand", "right")
    video_name = det_data.get("source_video", "unknown")
    fps_info = det_data.get("fps", 60.0)

    print(f"  Loading poses for {video_name}...")
    frames, fps, total_frames = load_pose_frames(poses_path)

    detections = det_data.get("detections", [])
    analyzable = [d for d in detections if d.get("shot_type") in ANALYZABLE_TYPES]

    print(f"  Analyzing {len(analyzable)} shots ({len(detections)} total)...")

    per_shot = []
    by_type = defaultdict(list)

    for det in analyzable:
        frame_idx = det.get("frame", 0)
        shot_type = det["shot_type"]
        timestamp = det.get("timestamp", 0)

        analysis = analyze_shot(frames, frame_idx, fps, dominant_hand,
                                kinetic_peak_method=kinetic_peak_method)
        analysis["timestamp"] = timestamp
        analysis["frame"] = frame_idx
        analysis["shot_type"] = shot_type

        per_shot.append(analysis)
        by_type[shot_type].append(analysis)

    # ── Per-type averages ────────────────────────────────────────
    type_summaries = {}
    for shot_type, shots in by_type.items():
        n = len(shots)
        if n == 0:
            continue

        avg_swing_speed = sum(s["peak_swing_speed"] for s in shots) / n
        avg_knee_bend = sum(s["knee_bend_depth"] for s in shots) / n
        avg_trunk_rot = sum(abs(s["trunk_rotation_at_contact"]) for s in shots) / n
        avg_arm_ext = sum(s["arm_extension_at_contact"] for s in shots) / n
        avg_recovery = sum(s["recovery_time_ms"] for s in shots) / n
        chain_correct_pct = sum(1 for s in shots if s["kinetic_chain_correct"]) / n

        type_summaries[shot_type] = {
            "count": n,
            "avg_peak_swing_speed": _safe(avg_swing_speed),
            "avg_knee_bend_depth": _safe(avg_knee_bend, 1),
            "avg_trunk_rotation": _safe(avg_trunk_rot, 1),
            "avg_arm_extension": _safe(avg_arm_ext, 1),
            "avg_recovery_time_ms": round(avg_recovery),
            "kinetic_chain_correct_pct": _safe(chain_correct_pct * 100, 1),
        }

    # ── Fatigue indicator ────────────────────────────────────────
    # Rolling average swing speed over time (window of 5 shots)
    fatigue_curve = []
    window = 5
    speeds = [(s["timestamp"], s["peak_swing_speed"]) for s in per_shot]
    for i in range(len(speeds)):
        start = max(0, i - window + 1)
        window_speeds = [s[1] for s in speeds[start:i + 1]]
        avg = sum(window_speeds) / len(window_speeds)
        fatigue_curve.append({
            "timestamp": speeds[i][0],
            "rolling_avg_speed": _safe(avg),
            "instantaneous_speed": _safe(speeds[i][1]),
        })

    # Overall fatigue: compare first quarter vs last quarter
    if len(speeds) >= 8:
        q_size = len(speeds) // 4
        first_q = [s[1] for s in speeds[:q_size]]
        last_q = [s[1] for s in speeds[-q_size:]]
        first_avg = sum(first_q) / len(first_q) if first_q else 0
        last_avg = sum(last_q) / len(last_q) if last_q else 0
        fatigue_pct = ((first_avg - last_avg) / first_avg * 100) if first_avg > 0 else 0.0
    else:
        fatigue_pct = 0.0

    session = {
        "video": video_name,
        "dominant_hand": dominant_hand,
        "fps": fps,
        "total_shots_analyzed": len(per_shot),
        "type_summaries": type_summaries,
        "fatigue_indicator": {
            "speed_decline_pct": _safe(fatigue_pct, 1),
            "curve": fatigue_curve,
        },
        "per_shot": per_shot,
    }

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(session, f, indent=2)
        print(f"  Saved to {output_path}")

    return session


def compare_sessions(paths):
    """Compare biomechanical metrics across sessions.

    Args:
        paths: List of session analysis JSON paths

    Prints trend analysis for coaching.
    """
    sessions = []
    for p in paths:
        with open(p) as f:
            sessions.append(json.load(f))

    print(f"\n{'='*60}")
    print(f"  CROSS-SESSION COMPARISON ({len(sessions)} sessions)")
    print(f"{'='*60}")

    # Collect per-type metrics across sessions
    all_types = set()
    for s in sessions:
        all_types.update(s.get("type_summaries", {}).keys())

    for shot_type in sorted(all_types):
        print(f"\n  {shot_type.upper()}:")
        metrics = ["avg_peak_swing_speed", "avg_knee_bend_depth",
                    "avg_trunk_rotation", "avg_arm_extension",
                    "avg_recovery_time_ms", "kinetic_chain_correct_pct"]
        metric_labels = ["Peak speed", "Knee bend", "Trunk rot",
                         "Arm ext", "Recovery ms", "Chain %"]

        for metric, label in zip(metrics, metric_labels):
            values = []
            for s in sessions:
                ts = s.get("type_summaries", {}).get(shot_type, {})
                values.append(ts.get(metric, None))

            valid = [v for v in values if v is not None]
            if len(valid) < 2:
                continue

            trend = valid[-1] - valid[0]
            arrow = "↑" if trend > 0 else "↓" if trend < 0 else "→"
            vals_str = " → ".join(f"{v:.1f}" if v is not None else "—" for v in values)
            print(f"    {label:<14s}: {vals_str}  {arrow} ({trend:+.1f})")

    # Fatigue comparison
    print(f"\n  FATIGUE:")
    for i, s in enumerate(sessions):
        fi = s.get("fatigue_indicator", {})
        decline = fi.get("speed_decline_pct", 0)
        print(f"    Session {i+1} ({s.get('video', '?')}): "
              f"speed decline = {decline:.1f}%")


def _print_session_summary(session):
    """Print a formatted session summary."""
    v = session["video"]
    n = session["total_shots_analyzed"]

    print(f"\n{'='*60}")
    print(f"  BIOMECHANICAL ANALYSIS: {v}")
    print(f"  {n} shots analyzed")
    print(f"{'='*60}")

    for shot_type, summary in sorted(session["type_summaries"].items()):
        print(f"\n  {shot_type.upper()} ({summary['count']} shots):")
        print(f"    Avg peak swing speed:     {summary['avg_peak_swing_speed']:.1f} m/s")
        print(f"    Avg knee bend depth:      {summary['avg_knee_bend_depth']:.1f}°")
        print(f"    Avg trunk rotation:       {summary['avg_trunk_rotation']:.1f}°")
        print(f"    Avg arm extension:        {summary['avg_arm_extension']:.1f}°")
        print(f"    Avg recovery time:        {summary['avg_recovery_time_ms']} ms")
        print(f"    Kinetic chain correct:    {summary['kinetic_chain_correct_pct']:.0f}%")

    fi = session.get("fatigue_indicator", {})
    decline = fi.get("speed_decline_pct", 0)
    if abs(decline) > 5:
        label = "NOTABLE FATIGUE" if decline > 0 else "SPEED INCREASE"
    else:
        label = "STABLE"
    print(f"\n  Fatigue: {decline:.1f}% speed decline ({label})")


def main():
    parser = argparse.ArgumentParser(
        description="Biomechanical analysis of tennis shots")
    parser.add_argument("--detections", help="Detection JSON file")
    parser.add_argument("--poses", help="Pose JSON file")
    parser.add_argument("--output", help="Output analysis JSON path")
    parser.add_argument("--all", action="store_true",
                        help="Analyze all reviewed videos")
    parser.add_argument("--compare", nargs="+",
                        help="Compare session analysis JSONs")
    parser.add_argument("--legacy-kinematic-peak", action="store_true",
                        help="Use the pre-2026-05-09 raw-velocity argmax for "
                             "kinetic chain timing (regression comparison only).")
    parser.add_argument("--accel-zero-crossing", action="store_true",
                        help="Use acceleration zero-crossing instead of "
                             "smoothed-velocity peak detection. Mutually "
                             "exclusive with --legacy-kinematic-peak.")
    args = parser.parse_args()

    if args.legacy_kinematic_peak and args.accel_zero_crossing:
        parser.error("--legacy-kinematic-peak and --accel-zero-crossing are mutually exclusive")
    kinetic_peak_method = (
        "legacy" if args.legacy_kinematic_peak
        else "accel_zero_crossing" if args.accel_zero_crossing
        else "smoothed"
    )

    if args.compare:
        compare_sessions(args.compare)
        return

    if args.all:
        # Analyze all videos that have both detections and poses
        det_dir = DETECTIONS_DIR
        results = []
        for det_file in sorted(os.listdir(det_dir)):
            if not det_file.endswith("_fused.json"):
                continue
            # Skip versioned files
            if any(x in det_file for x in ["_v2", "_v3", "_v4", "_v5", "_ml", "_pre", "_baseline"]):
                continue

            video_name = det_file.replace("_fused.json", "")
            # Prefer v5 if exists
            v5_path = os.path.join(det_dir, f"{video_name}_fused_v5.json")
            det_path = v5_path if os.path.exists(v5_path) else os.path.join(det_dir, det_file)

            pose_path = os.path.join(POSES_DIR, f"{video_name}.json")
            if not os.path.exists(pose_path):
                print(f"  {video_name}: no poses, skipping")
                continue

            out_path = os.path.join(ANALYSIS_DIR, f"{video_name}_biomech.json")
            print(f"\n{'─'*50}")
            print(f"  {video_name}")
            print(f"{'─'*50}")
            session = analyze_session(det_path, pose_path, out_path,
                                       kinetic_peak_method=kinetic_peak_method)
            _print_session_summary(session)
            results.append(session)

        if len(results) > 1:
            # Auto-compare all sessions
            paths = [os.path.join(ANALYSIS_DIR, f"{s['video']}_biomech.json") for s in results]
            compare_sessions(paths)

        return

    if not args.detections or not args.poses:
        parser.error("--detections and --poses required (or use --all / --compare)")

    det_path = os.path.join(PROJECT_ROOT, args.detections) if not os.path.isabs(args.detections) else args.detections
    poses_path = os.path.join(PROJECT_ROOT, args.poses) if not os.path.isabs(args.poses) else args.poses

    if args.output:
        out_path = os.path.join(PROJECT_ROOT, args.output) if not os.path.isabs(args.output) else args.output
    else:
        video_name = Path(det_path).stem.replace("_fused_v5", "").replace("_fused", "")
        out_path = os.path.join(ANALYSIS_DIR, f"{video_name}_biomech.json")

    session = analyze_session(det_path, poses_path, out_path,
                               kinetic_peak_method=kinetic_peak_method)
    _print_session_summary(session)


if __name__ == "__main__":
    main()
