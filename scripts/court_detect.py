#!/usr/bin/env python3
"""Detect tennis court lines and compute image→court homography.

Uses edge detection + Hough lines to find court boundaries, then computes
a perspective transform mapping pixel coordinates to real-world court
coordinates (meters). This enables:
  - Ball speed in mph/kph (pixel displacement → meters → speed)
  - Shot placement heatmaps (ball position → court location)
  - Line calling (bounce point → in/out determination)

Standard tennis court dimensions (meters):
  Full court:  23.77m long × 10.97m wide (doubles) / 8.23m (singles)
  Service box: 6.40m deep × 4.115m wide (each side)
  Net height:  0.914m (center) / 1.07m (posts)

Usage:
    python scripts/court_detect.py preprocessed/IMG_1141.mp4
    python scripts/court_detect.py preprocessed/IMG_1141.mp4 --visualize
    python scripts/court_detect.py --all
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT_ROOT = Path(__file__).parent.parent
PREPROCESSED_DIR = PROJECT_ROOT / "preprocessed"
COURT_DIR = PROJECT_ROOT / "court_calibrations"

# Standard tennis court dimensions in meters
COURT = {
    "length": 23.77,           # baseline to baseline
    "singles_width": 8.23,     # singles sideline to sideline
    "doubles_width": 10.97,    # doubles sideline to sideline
    "service_depth": 6.40,     # net to service line
    "net_to_baseline": 11.885, # half court length
    "center_mark": 0.10,       # center mark length
}

# Key court points in real-world coordinates (meters)
# Origin = center of court at net level
# X = lateral (positive = right from camera), Y = longitudinal (positive = far baseline)
COURT_POINTS_WORLD = {
    "near_baseline_left":    (-COURT["singles_width"]/2, -COURT["net_to_baseline"]),
    "near_baseline_right":   ( COURT["singles_width"]/2, -COURT["net_to_baseline"]),
    "near_service_left":     (-COURT["singles_width"]/2, -COURT["service_depth"]),
    "near_service_right":    ( COURT["singles_width"]/2, -COURT["service_depth"]),
    "net_left":              (-COURT["singles_width"]/2,  0),
    "net_right":             ( COURT["singles_width"]/2,  0),
    "far_service_left":      (-COURT["singles_width"]/2,  COURT["service_depth"]),
    "far_service_right":     ( COURT["singles_width"]/2,  COURT["service_depth"]),
    "far_baseline_left":     (-COURT["singles_width"]/2,  COURT["net_to_baseline"]),
    "far_baseline_right":    ( COURT["singles_width"]/2,  COURT["net_to_baseline"]),
    "center_service_near":   (0, -COURT["service_depth"]),
    "center_service_far":    (0,  COURT["service_depth"]),
    "center_net":            (0, 0),
}


def detect_court_surface(frame):
    """Detect court surface color to create a mask for line search.
    Works for green (grass/hard), blue (hard), and clay (orange/red) courts.
    Returns a binary mask of the court area."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, w = frame.shape[:2]

    masks = []
    # Green court (grass, green hard court)
    masks.append(cv2.inRange(hsv, (30, 30, 40), (85, 255, 200)))
    # Blue court (hard court)
    masks.append(cv2.inRange(hsv, (90, 30, 40), (130, 255, 200)))
    # Clay court (orange/red)
    masks.append(cv2.inRange(hsv, (5, 50, 80), (25, 255, 220)))

    combined = masks[0]
    for m in masks[1:]:
        combined = cv2.bitwise_or(combined, m)

    # Dilate to fill gaps, then find the largest contiguous region
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    combined = cv2.dilate(combined, kernel, iterations=2)

    court_area = np.sum(combined > 0) / (h * w)
    if court_area < 0.05:
        return np.ones((h, w), dtype=np.uint8) * 255
    return combined


def detect_court_lines(frame, min_line_length=100, max_line_gap=10):
    """Detect court lines using multi-threshold approach with court surface awareness.

    Returns list of (x1, y1, x2, y2) line segments.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    court_mask = detect_court_surface(frame)

    all_lines = []
    # Multi-threshold: try different white thresholds to handle varying lighting
    for thresh in [150, 170, 190, 210]:
        _, white_mask = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
        # Focus on lines within/near the court surface
        white_mask = cv2.bitwise_and(white_mask, court_mask)
        edges = cv2.Canny(white_mask, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=60,
                                minLineLength=min_line_length,
                                maxLineGap=max_line_gap)
        if lines is not None:
            all_lines.extend([tuple(l[0]) for l in lines])

    # Also try adaptive threshold for challenging lighting
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 31, -10)
    adaptive = cv2.bitwise_and(adaptive, court_mask)
    edges_a = cv2.Canny(adaptive, 50, 150, apertureSize=3)
    lines_a = cv2.HoughLinesP(edges_a, 1, np.pi/180, threshold=60,
                               minLineLength=min_line_length,
                               maxLineGap=max_line_gap)
    if lines_a is not None:
        all_lines.extend([tuple(l[0]) for l in lines_a])

    return all_lines


def classify_lines(lines, img_w, img_h):
    """Classify detected lines as horizontal (baselines/service lines) or
    vertical (sidelines/center line).

    Returns (horizontal_lines, vertical_lines).
    """
    horizontal = []
    vertical = []

    for x1, y1, x2, y2 in lines:
        angle = abs(math.atan2(y2 - y1, x2 - x1) * 180 / math.pi)
        length = math.sqrt((x2-x1)**2 + (y2-y1)**2)
        if length < 50:
            continue
        if angle < 30 or angle > 150:
            horizontal.append((x1, y1, x2, y2))
        elif 60 < angle < 120:
            vertical.append((x1, y1, x2, y2))

    return horizontal, vertical


def merge_nearby_lines(lines, threshold=20, is_horizontal=True):
    """Merge lines that are close together (same court line detected multiple times)."""
    if not lines:
        return []

    # Sort by the perpendicular coordinate
    if is_horizontal:
        lines = sorted(lines, key=lambda l: (l[1] + l[3]) / 2)
    else:
        lines = sorted(lines, key=lambda l: (l[0] + l[2]) / 2)

    merged = [lines[0]]
    for line in lines[1:]:
        last = merged[-1]
        if is_horizontal:
            dist = abs((line[1] + line[3])/2 - (last[1] + last[3])/2)
        else:
            dist = abs((line[0] + line[2])/2 - (last[0] + last[2])/2)

        if dist < threshold:
            # Merge: keep the longer line
            if line_length(line) > line_length(last):
                merged[-1] = line
        else:
            merged.append(line)

    return merged


def line_length(line):
    x1, y1, x2, y2 = line
    return math.sqrt((x2-x1)**2 + (y2-y1)**2)


def validate_court_geometry(corners, img_w, img_h):
    """Validate that detected corners form a plausible court trapezoid.
    In perspective, the bottom edge (near baseline) should be wider than top (far baseline).
    Returns True if geometry is plausible."""
    tl = corners["top_left"]
    tr = corners["top_right"]
    bl = corners["bottom_left"]
    br = corners["bottom_right"]

    top_w = abs(tr[0] - tl[0])
    bot_w = abs(br[0] - bl[0])
    height = abs((bl[1] + br[1]) / 2 - (tl[1] + tr[1]) / 2)

    # Bottom should be wider than top (perspective)
    if bot_w < top_w * 0.5:
        return False
    # Court should have reasonable proportions
    if height < img_h * 0.1 or top_w < img_w * 0.05:
        return False
    # Court shouldn't be impossibly narrow
    aspect = max(top_w, bot_w) / max(height, 1)
    if aspect < 0.1 or aspect > 10:
        return False
    return True


def find_court_corners(horizontal, vertical, img_w, img_h):
    """Find intersections of horizontal and vertical lines to identify court corners.

    Returns dict of identified court point names → (x, y) pixel positions,
    or None if insufficient points found.
    """
    if len(horizontal) < 2 or len(vertical) < 2:
        return None

    # Find all intersections
    intersections = []
    for h in horizontal:
        for v in vertical:
            pt = line_intersection(h, v)
            if pt and 0 <= pt[0] <= img_w and 0 <= pt[1] <= img_h:
                intersections.append(pt)

    if len(intersections) < 4:
        return None

    # Sort intersections into a grid
    intersections.sort(key=lambda p: (p[1], p[0]))

    # Take the 4 extreme corners
    corners = {
        "top_left": min(intersections[:len(intersections)//2], key=lambda p: p[0]),
        "top_right": max(intersections[:len(intersections)//2], key=lambda p: p[0]),
        "bottom_left": min(intersections[len(intersections)//2:], key=lambda p: p[0]),
        "bottom_right": max(intersections[len(intersections)//2:], key=lambda p: p[0]),
    }

    if not validate_court_geometry(corners, img_w, img_h):
        return None

    return corners, intersections


def line_intersection(line1, line2):
    """Find intersection point of two line segments (extended to infinite lines)."""
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if abs(denom) < 1e-6:
        return None

    t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom
    x = x1 + t*(x2-x1)
    y = y1 + t*(y2-y1)
    return (int(x), int(y))


def compute_homography(image_points, world_points):
    """Compute the perspective transform from image coords to world coords.

    image_points: list of (x, y) pixel positions
    world_points: list of (x, y) meter positions

    Returns 3x3 homography matrix, or None if computation fails.
    """
    if len(image_points) < 4 or len(world_points) < 4:
        return None

    src = np.float32(image_points)
    dst = np.float32(world_points)

    H, status = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    return H


def pixel_to_court(H, px, py):
    """Convert pixel coordinates to court coordinates using homography.

    Returns (x_meters, y_meters) on the court.
    """
    pt = np.float32([[[px, py]]])
    result = cv2.perspectiveTransform(pt, H)
    return float(result[0][0][0]), float(result[0][0][1])


def compute_ball_speed(ball_pos_1, ball_pos_2, H, fps, speed_unit="mph"):
    """Compute ball speed between two frames using homography.

    ball_pos_1, ball_pos_2: (px_x, px_y) pixel positions
    H: homography matrix
    fps: video frame rate

    Returns speed in requested units.
    """
    if ball_pos_1 is None or ball_pos_2 is None or H is None:
        return None

    x1, y1 = pixel_to_court(H, ball_pos_1[0], ball_pos_1[1])
    x2, y2 = pixel_to_court(H, ball_pos_2[0], ball_pos_2[1])

    dist_meters = math.sqrt((x2-x1)**2 + (y2-y1)**2)
    time_seconds = 1.0 / fps
    speed_ms = dist_meters / time_seconds

    if speed_unit == "mph":
        return speed_ms * 2.237
    elif speed_unit == "kph":
        return speed_ms * 3.6
    return speed_ms


def is_ball_in(court_x, court_y, match_type="singles"):
    """Determine if a ball landing at (court_x, court_y) is IN or OUT.

    Returns (bool is_in, float margin_meters).
    Positive margin = inside the line. Negative = outside.
    """
    half_w = COURT["singles_width"]/2 if match_type == "singles" else COURT["doubles_width"]/2
    half_l = COURT["net_to_baseline"]

    # Distance inside each boundary (positive = in)
    margin_x = half_w - abs(court_x)
    margin_y = half_l - abs(court_y)

    margin = min(margin_x, margin_y)
    is_in = margin >= 0

    return is_in, margin


def calibrate_video(video_path, output_path=None, visualize=False):
    """Detect court lines in a video and compute homography.

    Samples multiple frames and picks the best court detection.
    Saves calibration to JSON.
    """
    video_name = Path(video_path).stem
    if output_path is None:
        COURT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = str(COURT_DIR / f"{video_name}.json")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Cannot open {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Calibrating {video_name}: {img_w}x{img_h} @ {fps}fps, {total_frames} frames")

    # Sample more frames for better detection — skip first/last 5%
    sample_start = int(total_frames * 0.05)
    sample_end = int(total_frames * 0.95)
    sample_count = 40
    step = max(1, (sample_end - sample_start) // sample_count)

    best_result = None
    best_score = 0

    for fi in range(sample_start, sample_end, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            continue

        lines = detect_court_lines(frame)
        if not lines:
            continue

        horizontal, vertical = classify_lines(lines, img_w, img_h)
        horizontal = merge_nearby_lines(horizontal, is_horizontal=True)
        vertical = merge_nearby_lines(vertical, is_horizontal=False)

        # Score = number of distinct horizontal + vertical lines found
        score = len(horizontal) + len(vertical)
        if score > best_score and len(horizontal) >= 2 and len(vertical) >= 2:
            result = find_court_corners(horizontal, vertical, img_w, img_h)
            if result:
                best_result = (result, horizontal, vertical, fi, frame if visualize else None)
                best_score = score

    cap.release()

    if not best_result:
        print(f"  Could not detect court lines in {video_name}")
        return None

    (corners, intersections), h_lines, v_lines, best_frame, vis_frame = best_result

    # Map detected corners to world coordinates
    # Assumption: camera is behind one baseline looking toward the other
    image_pts = [
        corners["bottom_left"],
        corners["bottom_right"],
        corners["top_left"],
        corners["top_right"],
    ]
    world_pts = [
        (-COURT["singles_width"]/2, -COURT["net_to_baseline"]),
        ( COURT["singles_width"]/2, -COURT["net_to_baseline"]),
        (-COURT["singles_width"]/2,  COURT["net_to_baseline"]),
        ( COURT["singles_width"]/2,  COURT["net_to_baseline"]),
    ]

    H = compute_homography(image_pts, world_pts)
    if H is None:
        print(f"  Homography computation failed for {video_name}")
        return None

    # Save calibration
    calib = {
        "video": video_name,
        "resolution": [img_w, img_h],
        "fps": fps,
        "total_frames": total_frames,
        "best_frame": best_frame,
        "corners": {k: list(v) for k, v in corners.items()},
        "num_intersections": len(intersections),
        "num_horizontal_lines": len(h_lines),
        "num_vertical_lines": len(v_lines),
        "homography": H.tolist(),
    }

    with open(output_path, "w") as f:
        json.dump(calib, f, indent=2)
    print(f"  Saved calibration: {output_path}")
    print(f"  Corners: {len(corners)}, H-lines: {len(h_lines)}, V-lines: {len(v_lines)}")

    # Visualization
    if visualize and vis_frame is not None:
        vis = vis_frame.copy()
        for x1, y1, x2, y2 in h_lines:
            cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        for x1, y1, x2, y2 in v_lines:
            cv2.line(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
        for name, pt in corners.items():
            cv2.circle(vis, pt, 8, (0, 0, 255), -1)
            cv2.putText(vis, name, (pt[0]+10, pt[1]), cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, (0, 0, 255), 1)
        vis_path = output_path.replace(".json", "_vis.jpg")
        cv2.imwrite(vis_path, vis)
        print(f"  Visualization: {vis_path}")

    return calib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", nargs="?", help="Path to preprocessed video")
    parser.add_argument("--all", action="store_true", help="Process all videos")
    parser.add_argument("--visualize", action="store_true", help="Save visualization images")
    args = parser.parse_args()

    if args.all:
        for vid in sorted(PREPROCESSED_DIR.glob("*.mp4")):
            if "_240fps" in vid.name:
                continue
            calibrate_video(str(vid), visualize=args.visualize)
    elif args.video:
        calibrate_video(args.video, visualize=args.visualize)
    else:
        parser.error("Provide video path or --all")


if __name__ == "__main__":
    main()
