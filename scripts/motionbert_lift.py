"""Lift 2D MediaPipe poses to 3D using MotionBERT.

Reads pose JSON files (with 2D `landmarks`), runs MotionBERT inference,
and replaces `world_landmarks` with camera-invariant 3D coordinates.

Usage (on GPU machine):
    python scripts/motionbert_lift.py poses/IMG_0864.json -o poses_3d/IMG_0864.json
    python scripts/motionbert_lift.py poses/ -o poses_3d/   # batch mode
"""

import argparse
import copy
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from functools import partial

# Add MotionBERT to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MOTIONBERT_DIR = os.path.join(PROJECT_ROOT, "MotionBERT")
sys.path.insert(0, MOTIONBERT_DIR)

from lib.model.DSTformer import DSTformer

# ── Joint Mapping Tables ─────────────────────────────────────────

# MediaPipe 33 → H36M 17 mapping
# H36M: 0=Pelvis, 1=R_Hip, 2=R_Knee, 3=R_Ankle, 4=L_Hip, 5=L_Knee,
#        6=L_Ankle, 7=Spine, 8=Thorax, 9=Neck/Nose, 10=Head,
#        11=L_Shoulder, 12=L_Elbow, 13=L_Wrist, 14=R_Shoulder,
#        15=R_Elbow, 16=R_Wrist

# Direct mappings: H36M_idx -> MediaPipe_idx
H36M_FROM_MP = {
    1: 24,   # R_Hip
    2: 26,   # R_Knee
    3: 28,   # R_Ankle
    4: 23,   # L_Hip
    5: 25,   # L_Knee
    6: 27,   # L_Ankle
    9: 0,    # Neck/Nose <- NOSE
    11: 11,  # L_Shoulder
    12: 13,  # L_Elbow
    13: 15,  # L_Wrist
    14: 12,  # R_Shoulder
    15: 14,  # R_Elbow
    16: 16,  # R_Wrist
}

# Computed joints (midpoints)
# 0: Pelvis = midpoint(L_HIP=23, R_HIP=24)
# 7: Spine = midpoint(Pelvis, Thorax)
# 8: Thorax = midpoint(L_SHOULDER=11, R_SHOULDER=12)
# 10: Head = NOSE + small Y offset

# Reverse: H36M idx -> list of MediaPipe indices it maps back to
H36M_TO_MP = {
    1: [24],   # R_Hip
    2: [26],   # R_Knee
    3: [28],   # R_Ankle
    4: [23],   # L_Hip
    5: [25],   # L_Knee
    6: [27],   # L_Ankle
    9: [0],    # Nose
    11: [11],  # L_Shoulder
    12: [13],  # L_Elbow
    13: [15],  # L_Wrist
    14: [12],  # R_Shoulder
    15: [14],  # R_Elbow
    16: [16],  # R_Wrist
}


def mediapipe_to_h36m(landmarks_2d, video_w, video_h):
    """Convert MediaPipe 33 landmarks (normalized 0-1) to H36M 17 joints.

    Input: list of 33 [x, y, z, visibility] in normalized image coords
    Output: np.array shape (17, 3) — (x, y, confidence) centered and scaled
    """
    lm = np.array(landmarks_2d)  # (33, 4)

    # Convert to pixel coordinates
    px = lm[:, 0] * video_w
    py = lm[:, 1] * video_h
    conf = lm[:, 3]  # visibility as confidence

    h36m = np.zeros((17, 3), dtype=np.float32)

    # Direct mappings
    for h_idx, mp_idx in H36M_FROM_MP.items():
        h36m[h_idx, 0] = px[mp_idx]
        h36m[h_idx, 1] = py[mp_idx]
        h36m[h_idx, 2] = conf[mp_idx]

    # Pelvis = midpoint(L_HIP=23, R_HIP=24)
    h36m[0, 0] = (px[23] + px[24]) / 2
    h36m[0, 1] = (py[23] + py[24]) / 2
    h36m[0, 2] = min(conf[23], conf[24])

    # Thorax = midpoint(L_SHOULDER=11, R_SHOULDER=12)
    h36m[8, 0] = (px[11] + px[12]) / 2
    h36m[8, 1] = (py[11] + py[12]) / 2
    h36m[8, 2] = min(conf[11], conf[12])

    # Spine = midpoint(Pelvis, Thorax)
    h36m[7, 0] = (h36m[0, 0] + h36m[8, 0]) / 2
    h36m[7, 1] = (h36m[0, 1] + h36m[8, 1]) / 2
    h36m[7, 2] = min(h36m[0, 2], h36m[8, 2])

    # Head = NOSE + small upward offset (~10% of nose-thorax distance)
    nose_thorax_dist = abs(py[0] - h36m[8, 1])
    h36m[10, 0] = px[0]
    h36m[10, 1] = py[0] - 0.1 * nose_thorax_dist  # slightly above nose
    h36m[10, 2] = conf[0]

    # Center at frame center and normalize
    h36m[:, 0] = (h36m[:, 0] - video_w / 2) / (min(video_w, video_h) / 2)
    h36m[:, 1] = (h36m[:, 1] - video_h / 2) / (min(video_w, video_h) / 2)

    return h36m


def h36m_to_mediapipe(h36m_3d, original_world_landmarks):
    """Map H36M 17-joint 3D back to MediaPipe 33 world_landmarks format.

    For the 13 directly-mapped joints, replace world_landmarks with 3D output.
    For unmatched joints (face 1-10, hands 17-22, feet 29-32), keep originals.

    Input:
        h36m_3d: np.array (17, 3) — MotionBERT 3D output (x, y, z)
        original_world_landmarks: list of 33 [x, y, z, visibility]
    Output:
        list of 33 [x, y, z, visibility] with 3D-lifted joints replaced
    """
    result = [list(wl) for wl in original_world_landmarks]  # deep copy

    for h_idx, mp_indices in H36M_TO_MP.items():
        for mp_idx in mp_indices:
            result[mp_idx][0] = float(h36m_3d[h_idx, 0])
            result[mp_idx][1] = float(h36m_3d[h_idx, 1])
            result[mp_idx][2] = float(h36m_3d[h_idx, 2])
            # Keep original visibility in [3]

    # Also update midpoint-derived MediaPipe joints from H36M computed joints
    # Pelvis (H36M 0) → update both hips' midpoint isn't a MP joint, skip
    # Thorax (H36M 8) → not a direct MP joint, but we can use it for shoulder midpoint
    # Spine (H36M 7) → not a direct MP joint

    return result


def load_model(checkpoint_path, device):
    """Load MotionBERT Lite model for 3D pose lifting."""
    # Lite config: dim_feat=256, dim_rep=512, depth=5, num_heads=8, mlp_ratio=4
    model = DSTformer(
        dim_in=3,
        dim_out=3,
        dim_feat=256,
        dim_rep=512,
        depth=5,
        num_heads=8,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        maxlen=243,
        num_joints=17,
        att_fuse=True,
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_pos", checkpoint.get("model", checkpoint))

    # Handle DataParallel "module." prefix
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("module.", "")
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict, strict=True)
    model = model.to(device)
    model.eval()

    print(f"Loaded MotionBERT Lite from {checkpoint_path}")
    return model


def process_sliding_window(keypoints_2d, model, device, clip_len=243, stride=121):
    """Run MotionBERT inference with sliding window over full sequence.

    Input: keypoints_2d — np.array (T, 17, 3) centered+normalized 2D keypoints
    Output: np.array (T, 17, 3) — 3D predictions
    """
    T = keypoints_2d.shape[0]

    if T == 0:
        return np.zeros((0, 17, 3), dtype=np.float32)

    # Accumulator for averaging overlapping windows
    output_sum = np.zeros((T, 17, 3), dtype=np.float64)
    output_count = np.zeros(T, dtype=np.float64)

    # Generate window start positions
    if T <= clip_len:
        starts = [0]
    else:
        starts = list(range(0, T - clip_len + 1, stride))
        # Ensure last window covers the end
        if starts[-1] + clip_len < T:
            starts.append(T - clip_len)

    with torch.no_grad():
        for start in starts:
            end = min(start + clip_len, T)
            window = keypoints_2d[start:end]  # (W, 17, 3)
            actual_len = window.shape[0]

            # Pad if shorter than clip_len
            if actual_len < clip_len:
                pad = np.zeros((clip_len - actual_len, 17, 3), dtype=np.float32)
                # Repeat last frame for padding
                pad[:] = window[-1:]
                window = np.concatenate([window, pad], axis=0)

            # To tensor: (1, clip_len, 17, 3)
            x = torch.from_numpy(window).float().unsqueeze(0).to(device)
            pred = model(x)  # (1, clip_len, 17, 3)
            pred = pred.cpu().numpy()[0]  # (clip_len, 17, 3)

            # Only use the valid (non-padded) portion
            output_sum[start:start + actual_len] += pred[:actual_len]
            output_count[start:start + actual_len] += 1

    # Average overlapping regions
    mask = output_count > 0
    output_sum[mask] /= output_count[mask, None, None]

    return output_sum.astype(np.float32)


def lift_poses(pose_json_path, output_path, model, device):
    """Lift 2D poses to 3D using MotionBERT.

    1. Load pose JSON
    2. Extract 2D landmarks → H36M 17-joint format
    3. Sliding window inference (243 frames, stride 121)
    4. Map H36M 17 back to MediaPipe 33 world_landmarks
    5. Save as new pose JSON with same format
    """
    with open(pose_json_path) as f:
        data = json.load(f)

    video_info = data.get("video_info", {})
    video_w = video_info.get("width", 1920)
    video_h = video_info.get("height", 1080)
    frames = data.get("frames", [])

    if not frames:
        print(f"  No frames in {pose_json_path}, skipping")
        return

    # Sort frames by frame_idx
    frames_sorted = sorted(frames, key=lambda f: f.get("frame_idx", 0))
    max_idx = frames_sorted[-1].get("frame_idx", 0)

    # Build frame lookup
    frame_lookup = {}
    for fr in frames_sorted:
        frame_lookup[fr.get("frame_idx", 0)] = fr

    # Collect all detected frames with valid landmarks
    detected_indices = []
    for fr in frames_sorted:
        if fr.get("detected", False) and fr.get("landmarks"):
            if len(fr["landmarks"]) >= 33:
                detected_indices.append(fr["frame_idx"])

    if not detected_indices:
        print(f"  No detected frames with landmarks in {pose_json_path}, skipping")
        return

    print(f"  {len(detected_indices)} detected frames out of {len(frames_sorted)} total")

    # Build continuous 2D keypoints array for detected frames
    # We'll process contiguous segments separately to avoid interpolating across gaps
    segments = find_contiguous_segments(detected_indices, max_gap=5)

    # Process each segment through MotionBERT
    h36m_3d_results = {}  # frame_idx -> (17, 3) 3D prediction

    for seg_start, seg_end in segments:
        seg_indices = [i for i in detected_indices if seg_start <= i <= seg_end]
        if len(seg_indices) < 2:
            continue

        # Build dense array for this segment (interpolate small gaps)
        seg_len = seg_end - seg_start + 1
        keypoints_2d = np.zeros((seg_len, 17, 3), dtype=np.float32)

        for local_idx in range(seg_len):
            global_idx = seg_start + local_idx
            if global_idx in frame_lookup:
                fr = frame_lookup[global_idx]
                if fr.get("detected", False) and fr.get("landmarks") and len(fr["landmarks"]) >= 33:
                    keypoints_2d[local_idx] = mediapipe_to_h36m(
                        fr["landmarks"], video_w, video_h
                    )
                else:
                    # Interpolate from nearest detected frames
                    keypoints_2d[local_idx] = interpolate_nearest(
                        local_idx, seg_len, keypoints_2d, seg_start, detected_indices, frame_lookup,
                        video_w, video_h
                    )
            else:
                keypoints_2d[local_idx] = interpolate_nearest(
                    local_idx, seg_len, keypoints_2d, seg_start, detected_indices, frame_lookup,
                    video_w, video_h
                )

        # Run MotionBERT
        pred_3d = process_sliding_window(keypoints_2d, model, device)  # (seg_len, 17, 3)

        # Store results for detected frames only
        for local_idx in range(seg_len):
            global_idx = seg_start + local_idx
            if global_idx in frame_lookup:
                fr = frame_lookup[global_idx]
                if fr.get("detected", False) and fr.get("landmarks") and len(fr["landmarks"]) >= 33:
                    h36m_3d_results[global_idx] = pred_3d[local_idx]

    # Build output JSON
    output_frames = []
    lifted_count = 0
    for fr in frames_sorted:
        out_fr = copy.deepcopy(fr)
        idx = fr.get("frame_idx", 0)

        if idx in h36m_3d_results and fr.get("world_landmarks"):
            # Replace world_landmarks with 3D-lifted values
            out_fr["world_landmarks"] = h36m_to_mediapipe(
                h36m_3d_results[idx], fr["world_landmarks"]
            )
            out_fr["_3d_lifted"] = True
            lifted_count += 1
        output_frames.append(out_fr)

    output_data = {
        "video_info": video_info,
        "frames": output_frames,
        "_motionbert": {
            "model": "MotionBERT_Lite",
            "checkpoint": "FT_MB_lite_MB_ft_h36m_global_lite",
            "lifted_frames": lifted_count,
            "total_frames": len(output_frames),
        },
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f)

    print(f"  Lifted {lifted_count}/{len(output_frames)} frames -> {output_path}")


def find_contiguous_segments(indices, max_gap=5):
    """Find contiguous segments in sorted frame indices, allowing small gaps."""
    if not indices:
        return []
    segments = []
    seg_start = indices[0]
    prev = indices[0]
    for idx in indices[1:]:
        if idx - prev > max_gap:
            segments.append((seg_start, prev))
            seg_start = idx
        prev = idx
    segments.append((seg_start, prev))
    return segments


def interpolate_nearest(local_idx, seg_len, keypoints_2d, seg_start, detected_indices,
                        frame_lookup, video_w, video_h):
    """Fill missing frame by copying nearest detected frame's keypoints."""
    global_idx = seg_start + local_idx
    # Find nearest detected frame
    best_dist = float("inf")
    best_idx = None
    for di in detected_indices:
        if seg_start <= di <= seg_start + seg_len - 1:
            dist = abs(di - global_idx)
            if dist < best_dist and dist > 0:
                best_dist = dist
                best_idx = di
    if best_idx is not None and best_idx in frame_lookup:
        fr = frame_lookup[best_idx]
        if fr.get("landmarks") and len(fr["landmarks"]) >= 33:
            return mediapipe_to_h36m(fr["landmarks"], video_w, video_h)
    return np.zeros((17, 3), dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Lift 2D MediaPipe poses to 3D with MotionBERT")
    parser.add_argument("input", help="Pose JSON file or directory of pose JSONs")
    parser.add_argument("-o", "--output", help="Output file or directory")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to MotionBERT checkpoint (auto-discovers if omitted)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use (default: cuda if available)")
    args = parser.parse_args()

    # Find checkpoint
    if args.checkpoint:
        ckpt_path = args.checkpoint
    else:
        ckpt_path = os.path.join(
            MOTIONBERT_DIR,
            "checkpoint", "pose3d", "FT_MB_lite_MB_ft_h36m_global_lite", "best_epoch.bin"
        )
    if not os.path.exists(ckpt_path):
        print(f"[ERROR] Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    device = torch.device(args.device)
    model = load_model(ckpt_path, device)

    # Single file or directory
    if os.path.isfile(args.input):
        input_files = [args.input]
        if args.output:
            output_files = [args.output]
        else:
            base = os.path.splitext(args.input)[0]
            output_files = [base + "_3d.json"]
    elif os.path.isdir(args.input):
        input_files = sorted([
            os.path.join(args.input, f)
            for f in os.listdir(args.input)
            if f.endswith(".json") and not f.startswith(".")
            and f != "pose_extract_done.txt"
        ])
        if args.output:
            os.makedirs(args.output, exist_ok=True)
            output_files = [
                os.path.join(args.output, os.path.basename(f))
                for f in input_files
            ]
        else:
            output_files = [
                os.path.splitext(f)[0] + "_3d.json" for f in input_files
            ]
    else:
        print(f"[ERROR] Input not found: {args.input}")
        sys.exit(1)

    # Filter out non-pose files
    filtered = [(i, o) for i, o in zip(input_files, output_files)
                if os.path.basename(i).startswith("IMG_")]
    if not filtered:
        # Fall back to all JSON files
        filtered = list(zip(input_files, output_files))

    print(f"Processing {len(filtered)} pose files on {device}...")

    for i, (inp, out) in enumerate(filtered):
        name = os.path.basename(inp)
        print(f"[{i+1}/{len(filtered)}] {name}")
        try:
            lift_poses(inp, out, model, device)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    print("Done!")


if __name__ == "__main__":
    main()
