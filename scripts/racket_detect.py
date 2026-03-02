#!/usr/bin/env python3
"""Racket detection using YOLOv8 for tennis shot analysis.

Uses pre-trained YOLOv8m with COCO 'tennis racket' class (class 38).
No custom training needed.

IMPORTANT: Run this on Windows GPU machine, not Mac.

Setup:
    pip install ultralytics opencv-python

Usage:
    # Detect rackets in a video
    python scripts/racket_detect.py preprocessed/IMG_6703.mp4

    # Process all videos
    python scripts/racket_detect.py --all
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import PROJECT_ROOT, RACKET_DETECTIONS_DIR, PREPROCESSED_DIR

DETECTIONS_DIR = os.path.join(PROJECT_ROOT, "detections")

# COCO class ID for tennis racket
TENNIS_RACKET_CLASS = 38


def detect_rackets(video_path, output_path=None, conf_threshold=0.3):
    """Run YOLOv8 racket detection on a video.

    Args:
        video_path: Path to preprocessed video
        output_path: Output JSON path
        conf_threshold: Minimum confidence for detection

    Returns:
        Dict with per-frame racket detections
    """
    try:
        import cv2
        from ultralytics import YOLO
    except ImportError:
        print("Required: pip install ultralytics opencv-python")
        sys.exit(1)

    video_name = Path(video_path).stem
    if output_path is None:
        os.makedirs(RACKET_DETECTIONS_DIR, exist_ok=True)
        output_path = os.path.join(RACKET_DETECTIONS_DIR, f"{video_name}.json")

    print(f"  Loading YOLOv8m...")
    model = YOLO("yolov8m.pt")

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"  Processing {video_name}: {total_frames} frames at {fps:.1f} fps...")

    frames_data = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO inference
        results = model(frame, classes=[TENNIS_RACKET_CLASS], conf=conf_threshold,
                        verbose=False)

        frame_entry = {
            "frame": frame_idx,
            "detected": False,
        }

        for r in results:
            boxes = r.boxes
            if len(boxes) > 0:
                # Take highest confidence detection
                best_idx = boxes.conf.argmax()
                box = boxes.xyxy[best_idx].cpu().numpy()
                conf = float(boxes.conf[best_idx].cpu())

                cx = (box[0] + box[2]) / 2
                cy = (box[1] + box[3]) / 2

                frame_entry.update({
                    "detected": True,
                    "bbox": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                    "center": [float(cx), float(cy)],
                    "confidence": round(conf, 3),
                })

        frames_data.append(frame_entry)
        frame_idx += 1

        if frame_idx % 1000 == 0:
            detected = sum(1 for f in frames_data if f["detected"])
            print(f"    Frame {frame_idx}/{total_frames} "
                  f"({detected} racket detections so far)")

    cap.release()

    detected_count = sum(1 for f in frames_data if f["detected"])
    print(f"  Done: {detected_count}/{total_frames} frames with racket "
          f"({detected_count/total_frames*100:.1f}%)")

    result = {
        "video": video_name,
        "video_path": str(video_path),
        "fps": fps,
        "total_frames": total_frames,
        "frames_with_racket": detected_count,
        "detection_rate": round(detected_count / total_frames, 4) if total_frames > 0 else 0,
        "frames": frames_data,
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved to {output_path}")

    return result


def enrich_detections(det_path, racket_path):
    """Add racket-derived features to detection JSON.

    New features per detection:
        - racket_detected_at_contact: 1.0 if racket detected at contact frame
        - racket_center_height: Normalized Y of racket center at contact
        - racket_wrist_distance: Pixel distance from racket center to wrist
    """
    with open(det_path) as f:
        det_data = json.load(f)
    with open(racket_path) as f:
        racket_data = json.load(f)

    racket_frames = {f["frame"]: f for f in racket_data.get("frames", [])
                     if f.get("detected")}

    for det in det_data.get("detections", []):
        frame = det.get("frame", 0)

        # Check ±2 frames for racket detection
        racket = None
        for offset in [0, -1, 1, -2, 2]:
            if frame + offset in racket_frames:
                racket = racket_frames[frame + offset]
                break

        if racket:
            det["racket_detected_at_contact"] = 1.0
            # Normalize height to frame height (approximate)
            det["racket_center_height"] = round(racket["center"][1], 1)
            det["racket_wrist_distance"] = 0.0  # Would need pixel-space wrist coords
        else:
            det["racket_detected_at_contact"] = 0.0
            det["racket_center_height"] = 0.0
            det["racket_wrist_distance"] = 0.0

    with open(det_path, "w") as f:
        json.dump(det_data, f, indent=2)
    print(f"  Enriched {len(det_data.get('detections', []))} detections with racket features")

    return det_data


def main():
    parser = argparse.ArgumentParser(description="Racket detection with YOLOv8")
    parser.add_argument("video", nargs="?", help="Video file to process")
    parser.add_argument("--detections", help="Detection JSON to enrich")
    parser.add_argument("--all", action="store_true", help="Process all videos")
    parser.add_argument("--output", help="Output path override")
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold")
    args = parser.parse_args()

    if args.all:
        for vf in sorted(os.listdir(PREPROCESSED_DIR)):
            if not vf.endswith(".mp4"):
                continue
            video_path = os.path.join(PREPROCESSED_DIR, vf)
            result = detect_rackets(video_path, conf_threshold=args.conf)

            # Auto-enrich if detection JSON exists
            video_name = Path(vf).stem
            for det_file in [f"{video_name}_fused_v5.json", f"{video_name}_fused.json"]:
                det_path = os.path.join(DETECTIONS_DIR, det_file)
                if os.path.exists(det_path):
                    racket_path = os.path.join(RACKET_DETECTIONS_DIR, f"{video_name}.json")
                    enrich_detections(det_path, racket_path)
                    break
        return

    if not args.video:
        parser.error("Provide a video path or use --all")

    result = detect_rackets(args.video, output_path=args.output, conf_threshold=args.conf)

    if args.detections:
        video_name = Path(args.video).stem
        racket_path = os.path.join(RACKET_DETECTIONS_DIR, f"{video_name}.json")
        if os.path.exists(racket_path):
            enrich_detections(args.detections, racket_path)


if __name__ == "__main__":
    main()
