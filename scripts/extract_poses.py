#!/usr/bin/env python3
"""Extract MediaPipe pose keypoints from preprocessed videos."""

import json
import os
import sys
import time

# Add project root to path so config is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import PREPROCESSED_DIR, POSES_DIR, MEDIAPIPE


# ── Helpers ──────────────────────────────────────────────────


def format_size(num_bytes):
    """Return human-readable file size string."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(num_bytes) < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"


def format_duration(seconds):
    """Format seconds as MM:SS or HH:MM:SS."""
    seconds = int(seconds)
    if seconds >= 3600:
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        return f"{h}:{m:02d}:{s:02d}"
    m = seconds // 60
    s = seconds % 60
    return f"{m}:{s:02d}"


# ── Discovery ────────────────────────────────────────────────


def find_preprocessed_videos(preprocessed_dir):
    """List all .mp4 files in preprocessed/, sorted by name."""
    if not os.path.isdir(preprocessed_dir):
        return []
    videos = [
        f for f in os.listdir(preprocessed_dir)
        if f.lower().endswith(".mp4")
    ]
    videos.sort()
    return videos


def is_already_processed(video_name, poses_dir):
    """Check if .json exists in poses/ with nonzero size."""
    name = os.path.splitext(video_name)[0]
    json_path = os.path.join(poses_dir, name + ".json")
    return os.path.exists(json_path) and os.path.getsize(json_path) > 0


# ── Pose Model ───────────────────────────────────────────────


def init_pose_model(mp_config):
    """Create MediaPipe Pose instance from MEDIAPIPE config dict."""
    import mediapipe as mp

    return mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=mp_config["model_complexity"],
        smooth_landmarks=mp_config["smooth_landmarks"],
        enable_segmentation=mp_config.get("enable_segmentation", False),
        min_detection_confidence=mp_config["min_detection_confidence"],
        min_tracking_confidence=mp_config["min_tracking_confidence"],
    )


# ── Extraction ───────────────────────────────────────────────


def extract_poses(video_path, pose_model, visualize, poses_dir):
    """Read frames, run MediaPipe, collect landmarks, optionally write skeleton video.

    Returns dict with video_info, detection_stats, landmark_names, and frames,
    or None on failure.
    """
    import cv2
    import mediapipe as mp

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [ERROR] Cannot open {os.path.basename(video_path)}")
        return None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    video_info = {
        "width": width,
        "height": height,
        "fps": fps,
        "total_frames": total_frames,
        "duration_seconds": round(duration, 2),
    }

    # Landmark names for reference
    landmark_names = [lm.name for lm in mp.solutions.pose.PoseLandmark]

    # Skeleton video writer (optional)
    skeleton_writer = None
    skeleton_part_path = None
    if visualize:
        name = os.path.splitext(os.path.basename(video_path))[0]
        skeleton_path = os.path.join(poses_dir, name + "_skeleton.mp4")
        skeleton_part_path = os.path.join(poses_dir, name + "_skeleton.part.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        skeleton_writer = cv2.VideoWriter(
            skeleton_part_path, fourcc, fps, (width, height)
        )

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    frames = []
    frames_detected = 0
    start_time = time.time()

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_idx / fps if fps > 0 else 0.0

        # BGR → RGB, set non-writeable for performance
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = pose_model.process(rgb)

        if result.pose_landmarks:
            frames_detected += 1
            landmarks = [
                [
                    round(lm.x, 6),
                    round(lm.y, 6),
                    round(lm.z, 6),
                    round(lm.visibility, 6),
                ]
                for lm in result.pose_landmarks.landmark
            ]
            world_landmarks = [
                [
                    round(lm.x, 6),
                    round(lm.y, 6),
                    round(lm.z, 6),
                    round(lm.visibility, 6),
                ]
                for lm in result.pose_world_landmarks.landmark
            ]
            frames.append({
                "frame_idx": frame_idx,
                "timestamp": round(timestamp, 6),
                "detected": True,
                "landmarks": landmarks,
                "world_landmarks": world_landmarks,
            })

            if skeleton_writer is not None:
                rgb.flags.writeable = True
                annotated = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                mp_drawing.draw_landmarks(
                    annotated,
                    result.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                )
                skeleton_writer.write(annotated)
        else:
            frames.append({
                "frame_idx": frame_idx,
                "timestamp": round(timestamp, 6),
                "detected": False,
                "landmarks": None,
                "world_landmarks": None,
            })

            if skeleton_writer is not None:
                annotated = frame.copy()
                cv2.putText(
                    annotated,
                    "NO POSE DETECTED",
                    (width // 2 - 200, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 0, 255),
                    3,
                )
                skeleton_writer.write(annotated)

        frame_idx += 1

        # Progress every 100 frames
        if frame_idx % 100 == 0:
            elapsed = time.time() - start_time
            fps_actual = frame_idx / elapsed if elapsed > 0 else 0
            det_rate = frames_detected / frame_idx * 100
            remaining = (total_frames - frame_idx) / fps_actual if fps_actual > 0 else 0
            print(
                f"\r  Processing: {frame_idx}/{total_frames} "
                f"({det_rate:.1f}% detected) "
                f"[{fps_actual:.1f} fps, ETA {format_duration(remaining)}]",
                end="",
                flush=True,
            )

    cap.release()
    elapsed = time.time() - start_time

    # Finalize skeleton video
    if skeleton_writer is not None:
        skeleton_writer.release()
        skeleton_final = os.path.join(
            os.path.dirname(skeleton_part_path),
            os.path.basename(skeleton_part_path).replace(".part", ""),
        )
        os.replace(skeleton_part_path, skeleton_final)
        skeleton_part_path = None  # Mark as completed
        print()
        print(f"  Skeleton:  {os.path.basename(skeleton_final)} ({format_size(os.path.getsize(skeleton_final))})")

    detection_rate = frames_detected / frame_idx if frame_idx > 0 else 0
    fps_avg = frame_idx / elapsed if elapsed > 0 else 0

    print(
        f"\r  Detected: {frames_detected}/{frame_idx} "
        f"({detection_rate * 100:.1f}%)  "
        f"Time: {format_duration(elapsed)}  "
        f"Avg: {fps_avg:.1f} fps"
    )

    detection_stats = {
        "frames_processed": frame_idx,
        "frames_detected": frames_detected,
        "detection_rate": round(detection_rate, 4),
        "processing_time_seconds": round(elapsed, 1),
    }

    return {
        "video_info": video_info,
        "detection_stats": detection_stats,
        "landmark_names": landmark_names,
        "frames": frames,
    }


# ── Output ───────────────────────────────────────────────────


def save_pose_json(pose_data, output_path):
    """Write JSON to .part, then atomic rename."""
    part_path = output_path + ".part"
    with open(part_path, "w") as f:
        json.dump(pose_data, f, separators=(",", ":"))
    os.replace(part_path, output_path)


# ── Main ─────────────────────────────────────────────────────


def main():
    # ── Startup checks ───────────────────────────────────────
    try:
        import mediapipe as mp
        test_pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=MEDIAPIPE["model_complexity"],
        )
        test_pose.close()
    except Exception as e:
        print(f"[ERROR] MediaPipe initialization failed: {e}")
        sys.exit(1)

    try:
        import cv2  # noqa: F401
    except ImportError as e:
        print(f"[ERROR] OpenCV not available: {e}")
        sys.exit(1)

    visualize = "--visualize" in sys.argv

    # Find preprocessed videos
    videos = find_preprocessed_videos(PREPROCESSED_DIR)
    if not videos:
        print(f"No .mp4 files found in {PREPROCESSED_DIR}/")
        return

    os.makedirs(POSES_DIR, exist_ok=True)

    print(f"Found {len(videos)} video(s) in {PREPROCESSED_DIR}/")
    if visualize:
        print("  Skeleton visualization: ON")
    print()

    results = {"ok": [], "skip": [], "fail": []}

    # Track .part files for interrupt cleanup
    active_json_part = None
    active_skeleton_part = None

    try:
        for i, filename in enumerate(videos, 1):
            video_path = os.path.join(PREPROCESSED_DIR, filename)
            name = os.path.splitext(filename)[0]
            json_path = os.path.join(POSES_DIR, name + ".json")

            print(f"[{i}/{len(videos)}] {filename}")

            # Skip if already processed
            if is_already_processed(filename, POSES_DIR):
                size = os.path.getsize(json_path)
                print(f"  [SKIP] {name}.json already exists ({format_size(size)})\n")
                results["skip"].append(filename)
                continue

            # Fresh model per video
            pose_model = init_pose_model(MEDIAPIPE)

            active_json_part = json_path + ".part"
            if visualize:
                active_skeleton_part = os.path.join(POSES_DIR, name + "_skeleton.part.mp4")

            pose_result = extract_poses(video_path, pose_model, visualize, POSES_DIR)

            pose_model.close()
            active_skeleton_part = None  # Skeleton handled inside extract_poses

            if pose_result is None:
                results["fail"].append(filename)
                active_json_part = None
                print()
                continue

            # Build output document
            pose_data = {
                "version": 1,
                "source_video": filename,
                "video_info": pose_result["video_info"],
                "mediapipe_config": {
                    "model_complexity": MEDIAPIPE["model_complexity"],
                    "min_detection_confidence": MEDIAPIPE["min_detection_confidence"],
                    "min_tracking_confidence": MEDIAPIPE["min_tracking_confidence"],
                    "smooth_landmarks": MEDIAPIPE["smooth_landmarks"],
                },
                "detection_stats": pose_result["detection_stats"],
                "landmark_names": pose_result["landmark_names"],
                "frames": pose_result["frames"],
            }

            save_pose_json(pose_data, json_path)
            active_json_part = None

            # Verify output
            size = os.path.getsize(json_path)
            print(f"  Output:  {name}.json ({format_size(size)})")

            det_rate = pose_result["detection_stats"]["detection_rate"]
            if det_rate >= 0.90:
                print(f"  [OK] Detection rate {det_rate * 100:.1f}%")
            else:
                print(f"  [WARN] Detection rate {det_rate * 100:.1f}% (below 90% target)")

            results["ok"].append(filename)
            print()

    except KeyboardInterrupt:
        print("\n\n  Interrupted by user.")
        # Clean up active .part files
        for part in (active_json_part, active_skeleton_part):
            if part and os.path.exists(part):
                os.remove(part)
                print(f"  Cleaned up {os.path.basename(part)}")

    # Summary
    print("=" * 50)
    print("Pose Extraction Summary")
    print("=" * 50)
    print(f"  Extracted: {len(results['ok'])}")
    print(f"  Skipped:   {len(results['skip'])}")
    print(f"  Failed:    {len(results['fail'])}")
    if results["fail"]:
        print("  Failed files:")
        for name in results["fail"]:
            print(f"    - {name}")
    print()


if __name__ == "__main__":
    main()
