#!/usr/bin/env python3
"""Render diagnostic PNGs for audit-flagged shots.

For each (video_id, frame_idx) pair, save:
  - The original frame
  - 2D MediaPipe keypoints overlaid in image space
  - A side panel with biomech values + flag reasons
  - A simple side-view diagram of world_landmarks (optional, if requested)

Lets the operator visually confirm whether the 2D pose looks correct AND
whether the world_landmarks projection looks plausible.
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


PROJECT_ROOT = "/Users/andrewhome/tennis_analysis"
POSES_DIR = os.path.join(PROJECT_ROOT, "poses_full_videos")
PREP_DIR = os.path.join(PROJECT_ROOT, "preprocessed")

# MediaPipe POSE_CONNECTIONS indices (33-keypoint topology, body subset).
SKELETON_EDGES = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # arms + shoulders
    (11, 23), (12, 24), (23, 24),                       # torso
    (23, 25), (25, 27), (27, 31), (27, 29),             # left leg
    (24, 26), (26, 28), (28, 32), (28, 30),             # right leg
]

KEY_LABELS = {
    11: "L_SH", 12: "R_SH", 13: "L_EL", 14: "R_EL", 15: "L_WR", 16: "R_WR",
    23: "L_HIP", 24: "R_HIP", 25: "L_KN", 26: "R_KN", 27: "L_AN", 28: "R_AN",
}


def render_one(video_id, frame_idx, biomech, flags, out_path):
    """Render PNG of one frame with 2D pose overlay + biomech panel."""
    import cv2
    import numpy as np

    video_path = os.path.join(PREP_DIR, f"{video_id}.mp4")
    pose_path = os.path.join(POSES_DIR, f"{video_id}.json")

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f"  [warn] could not read {video_id} frame {frame_idx}")
        return False

    h, w = frame.shape[:2]

    with open(pose_path) as f:
        pose = json.load(f)
    pose_frame = next((p for p in pose["frames"] if p["frame_idx"] == frame_idx), None)
    if not pose_frame or not pose_frame.get("detected"):
        print(f"  [warn] no pose for {video_id} frame {frame_idx}")
        return False

    landmarks = pose_frame["landmarks"]

    # Draw skeleton edges
    for a, b in SKELETON_EDGES:
        if landmarks[a] and landmarks[b]:
            x1 = int(landmarks[a][0] * w)
            y1 = int(landmarks[a][1] * h)
            x2 = int(landmarks[b][0] * w)
            y2 = int(landmarks[b][1] * h)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)

    # Draw keypoints + labels
    for idx, label in KEY_LABELS.items():
        if landmarks[idx]:
            x = int(landmarks[idx][0] * w)
            y = int(landmarks[idx][1] * h)
            vis = landmarks[idx][3]
            color = (0, 255, 0) if vis > 0.5 else (0, 0, 255)
            cv2.circle(frame, (x, y), 6, color, -1)
            cv2.putText(frame, label, (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Side panel for biomech values
    panel_h = h
    panel_w = 480
    panel = np.full((panel_h, panel_w, 3), 30, dtype=np.uint8)
    y = 30

    def put(label, value, color=(255, 255, 255)):
        nonlocal y
        cv2.putText(panel, f"{label}", (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, color, 1)
        cv2.putText(panel, str(value), (220, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, color, 1)
        y += 28

    put(f"{video_id} frame {frame_idx}", "")
    put(f"shot type", biomech.get("shot_type", "?"))
    y += 10

    cv2.line(panel, (10, y), (panel_w - 10, y), (80, 80, 80), 1)
    y += 20

    cv2.putText(panel, "Static 3D-derived angles:", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 255), 1)
    y += 28

    put("knee_angle_at_contact", f"{biomech.get('knee_angle_at_contact', '?')}deg")
    put("trunk_rotation",        f"{biomech.get('trunk_rotation_at_contact', '?')}deg")
    put("arm_extension",         f"{biomech.get('arm_extension_at_contact', '?')}deg")
    put("knee_bend_depth",       f"{biomech.get('knee_bend_depth', '?')}deg")
    y += 8

    cv2.putText(panel, "Velocity-derived metrics:", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 255), 1)
    y += 28

    put("peak_swing_speed", biomech.get("peak_swing_speed", "?"))
    put("contact_speed",    biomech.get("contact_swing_speed", "?"))
    timing = biomech.get("kinetic_chain_timing_ms", {})
    if isinstance(timing, dict):
        put("hip ms",      timing.get("hip", "?"))
        put("shoulder ms", timing.get("shoulder", "?"))
        put("elbow ms",    timing.get("elbow", "?"))
        put("wrist ms",    timing.get("wrist", "?"))
    y += 10

    if flags:
        cv2.line(panel, (10, y), (panel_w - 10, y), (80, 80, 80), 1)
        y += 20
        cv2.putText(panel, "FLAGS:", (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 255), 1)
        y += 28
        for flag in flags[:5]:
            cv2.putText(panel, f"  {flag}", (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (0, 100, 255), 1)
            y += 22

    composite = np.hstack([frame, panel])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, composite)
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--audit", default="eval/3d-lifting/audit_phase1.json",
                   help="Audit JSON from audit_world_landmarks.py")
    p.add_argument("--out-dir", default="eval/3d-lifting/audits")
    p.add_argument("--n-per-bucket", type=int, default=3,
                   help="Render N shots per flag-bucket")
    args = p.parse_args()

    with open(args.audit) as f:
        audit = json.load(f)

    # Bucket shots by primary flag
    by_bucket = {}
    clean = []
    for v in audit["videos"]:
        if "error" in v:
            continue
        for s in v["shots_audited"]:
            s = dict(s)
            s["video"] = v["video"]
            if s.get("flags"):
                cat = s["flags"][0].split(":")[0]
                by_bucket.setdefault(cat, []).append(s)
            else:
                clean.append(s)

    # Pick N from each bucket + N clean for control
    picks = []
    for cat, shots in sorted(by_bucket.items()):
        picks.extend([(cat, s) for s in shots[:args.n_per_bucket]])
    picks.extend([("clean", s) for s in clean[:args.n_per_bucket]])

    print(f"[render] rendering {len(picks)} frames")
    for cat, s in picks:
        out_path = os.path.join(
            args.out_dir,
            f"{cat}__{s['video']}__f{s['frame']}__{s['shot_type']}.png"
        )
        render_one(s["video"], s["frame"], s, s.get("flags", []), out_path)
        print(f"  {out_path}")


if __name__ == "__main__":
    main()
