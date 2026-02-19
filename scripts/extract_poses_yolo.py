#!/usr/bin/env python3
"""Extract poses using YOLOv8-Pose (GPU accelerated).

Uses YOLO's native video streaming for maximum speed:
- RTX 4090/5080: ~200-300 fps
- MediaPipe CPU: ~8 fps

Output format is compatible with the training pipeline.
Uses 17 COCO keypoints instead of MediaPipe's 33.

Usage:
    python scripts/extract_poses_yolo.py video.mp4
    python scripts/extract_poses_yolo.py --all  # process all preprocessed videos
"""

import argparse
import json
import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import PREPROCESSED_DIR, POSES_DIR

# COCO keypoint names (17 keypoints)
COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]


def extract_poses(video_path, output_path=None):
    """Extract poses from video using YOLOv8-Pose with native video streaming."""
    from ultralytics import YOLO
    import cv2

    # Get video info first
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    print(f"Video: {fps:.0f} fps, {total_frames} frames, {width}x{height}")

    # Load YOLOv8-Pose model
    model = YOLO("yolov8m-pose.pt")

    # Use YOLO's native video streaming - much faster than frame-by-frame
    # stream=True returns a generator, vid_stride=1 processes every frame
    print("Running YOLO pose extraction (streaming mode)...")
    start_time = time.time()

    frames_data = []
    frame_idx = 0

    # Native video processing with batching
    results = model.predict(
        source=video_path,
        stream=True,        # Memory-efficient streaming
        vid_stride=1,       # Process every frame
        verbose=False,
        device=0,           # Use GPU 0
    )

    for result in results:
        frame_data = {"frame_idx": frame_idx, "detected": False}

        if result.keypoints is not None and len(result.keypoints.data) > 0:
            kp = result.keypoints.data[0]  # First person, shape (17, 3)

            if kp.shape[0] == 17:
                frame_data["detected"] = True
                landmarks = []
                for i in range(17):
                    x = float(kp[i, 0]) / width
                    y = float(kp[i, 1]) / height
                    conf = float(kp[i, 2])
                    landmarks.append([x, y, 0.0, conf])

                frame_data["world_landmarks"] = landmarks
                frame_data["landmarks"] = landmarks

        frames_data.append(frame_data)
        frame_idx += 1

        if frame_idx % 1000 == 0:
            elapsed = time.time() - start_time
            fps_rate = frame_idx / elapsed
            eta = (total_frames - frame_idx) / fps_rate if fps_rate > 0 else 0
            print(f"  {frame_idx}/{total_frames} frames ({fps_rate:.0f} fps, ETA: {eta:.0f}s)")

    elapsed = time.time() - start_time
    detected = sum(1 for f in frames_data if f.get("detected"))
    actual_fps = frame_idx / elapsed if elapsed > 0 else 0
    print(f"Done: {detected}/{frame_idx} frames with pose in {elapsed:.1f}s ({actual_fps:.0f} fps)")

    # Build output (compatible with label_clips.py)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output = {
        "version": 2,
        "source_video": video_name,
        "video_info": {
            "fps": fps,
            "total_frames": total_frames,
            "width": width,
            "height": height,
            "pose_model": "yolov8m-pose",
            "keypoint_format": "coco17",
            "num_keypoints": 17,
            "keypoint_names": COCO_KEYPOINTS,
        },
        "frames": frames_data
    }

    # Determine output path
    if output_path is None:
        output_path = os.path.join(POSES_DIR, f"{video_name}.json")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(output, f)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Saved to {output_path} ({size_mb:.1f} MB)")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Extract poses using YOLOv8-Pose (GPU)")
    parser.add_argument("video", nargs="?", help="Video file to process")
    parser.add_argument("--all", action="store_true", help="Process all preprocessed videos")
    parser.add_argument("-o", "--output", help="Output JSON path")
    args = parser.parse_args()

    if args.all:
        # Process all preprocessed videos
        videos = []
        for f in os.listdir(PREPROCESSED_DIR):
            if f.endswith(('.mp4', '.mov', '.MP4', '.MOV')):
                videos.append(os.path.join(PREPROCESSED_DIR, f))

        if not videos:
            print(f"No videos found in {PREPROCESSED_DIR}")
            return

        print(f"Processing {len(videos)} videos...")
        for video in sorted(videos):
            print(f"\n{'='*60}")
            print(f"Processing: {os.path.basename(video)}")
            print('='*60)
            extract_poses(video)

    elif args.video:
        extract_poses(args.video, args.output)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
