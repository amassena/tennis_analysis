#!/usr/bin/env python3
"""OpenCV-based visual labeling tool for tennis shot segments.

Plays *_skeleton.mp4 videos and lets the user mark shot segments with
keypresses.  Outputs CSV compatible with label_clips.py --csv.

Usage:
    python scripts/visual_label.py                              # Auto-discover
    python scripts/visual_label.py poses/IMG_0870_skeleton.mp4  # Specific video
    python scripts/visual_label.py -o my_labels.csv             # Custom output
"""

import argparse
import csv
import json
import os
import sys
import time
from typing import List, NamedTuple, Optional

import cv2
import numpy as np

# Add project root so config is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import POSES_DIR, PROJECT_ROOT, SHOT_TYPES, VIEW_ANGLES, DEFAULT_VIEW_ANGLE
from scripts.video_metadata import load_video_metadata, save_video_metadata

# ── Data structures ──────────────────────────────────────────


class Mark(NamedTuple):
    """A single-frame mark at ball contact point."""
    shot_type: str
    frame: int


# Shot-type colors (BGR)
SHOT_COLORS = {
    "forehand": (0, 200, 0),      # green
    "backhand": (200, 100, 0),    # blue-ish
    "serve": (0, 100, 255),       # orange
    "neutral": (180, 180, 180),   # gray
}

SPEED_LEVELS = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]

HELP_TEXT = [
    "=== Playback ===",
    "",
    "Space       Play / Pause",
    "Up / Down   Speed up / down (while playing)",
    "Left/Right  Step 1 frame (when paused)",
    ", / .       Jump 5 sec back / forward",
    "Click       Scrub timeline",
    "",
    "=== Marking ===",
    "",
    "F / B / S   Create forehand/backhand/serve",
    "Up / Down   Move mark +/- 1 frame (when paused)",
    "[ / ]       Jump to prev / next mark",
    "X           Delete selected mark",
    "U           Undo last mark",
    "",
    "V  View angle   H  Help   Q  Save & quit",
]


class AppState:
    """Mutable application state for the labeler."""

    def __init__(self, total_frames: int, fps: float):
        self.total_frames = total_frames
        self.fps = fps

        # Playback
        self.current_frame = 0
        self.playing = False
        self.speed_idx = 2  # index into SPEED_LEVELS → 1.0x

        # Completed marks (single-frame contact points)
        self.marks: List[Mark] = []

        # Selection
        self.selected_mark_idx: Optional[int] = None

        # Scrubbing state (for smoother timeline interaction)
        self.scrubbing = False

        # Display
        self.show_help = False

        # Toast message (shown briefly after undo/cancel/etc)
        self.message = ""
        self.message_time = 0.0  # time.time() when set

        # Timeline geometry (set by draw_timeline, read by mouse callback)
        self.tl_x = 0
        self.tl_y = 0
        self.tl_w = 0
        self.tl_h = 0

        # View angle (camera position) - per-video metadata
        self.view_angle = DEFAULT_VIEW_ANGLE
        self.video_name: Optional[str] = None  # set by run_labeler for metadata saving

    def set_message(self, text: str):
        self.message = text
        self.message_time = time.time()

    @property
    def speed(self) -> float:
        return SPEED_LEVELS[self.speed_idx]

    def timestamp(self, frame: Optional[int] = None) -> str:
        if frame is None:
            frame = self.current_frame
        secs = frame / self.fps if self.fps > 0 else 0
        m, s = divmod(int(secs), 60)
        frac = int((secs - int(secs)) * 100)
        return f"{m}:{s:02d}.{frac:02d}"

    def cycle_view_angle(self):
        """Cycle to the next view angle."""
        idx = VIEW_ANGLES.index(self.view_angle) if self.view_angle in VIEW_ANGLES else 0
        idx = (idx + 1) % len(VIEW_ANGLES)
        self.view_angle = VIEW_ANGLES[idx]
        self.set_message(f"View angle: {self.view_angle}")


# ── Video discovery ──────────────────────────────────────────


def find_skeleton_videos(poses_dir: str) -> List[str]:
    """Discover *_skeleton.mp4 files in poses_dir."""
    if not os.path.isdir(poses_dir):
        return []
    files = [
        f
        for f in os.listdir(poses_dir)
        if f.lower().endswith("_skeleton.mp4")
    ]
    files.sort()
    return files


# ── Drawing helpers ──────────────────────────────────────────


def draw_text_with_bg(
    frame,
    text: str,
    pos: tuple,
    font_scale: float = 0.55,
    color=(255, 255, 255),
    bg_color=(0, 0, 0),
    thickness: int = 1,
    padding: int = 4,
):
    """Draw text with a dark background rectangle for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    cv2.rectangle(
        frame,
        (x - padding, y - th - padding),
        (x + tw + padding, y + baseline + padding),
        bg_color,
        cv2.FILLED,
    )
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


def draw_overlay(frame, state: AppState):
    """Frame/time info, selection indicator, and per-type mark counts."""
    h, w = frame.shape[:2]

    # Top-left: frame info
    status = "PLAYING" if state.playing else "PAUSED"
    speed_str = f"{state.speed}x" if state.playing else ""
    info = f"Frame {state.current_frame}/{state.total_frames - 1}  {state.timestamp()}  {status} {speed_str}"
    draw_text_with_bg(frame, info, (10, 25))

    # Top-right: view angle
    angle_text = f"View: {state.view_angle}"
    # Get text size to right-align
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(angle_text, font, 0.55, 1)
    draw_text_with_bg(frame, angle_text, (w - tw - 20, 25), color=(200, 200, 100))

    # Selection indicator (single-frame mark)
    if state.selected_mark_idx is not None and state.selected_mark_idx < len(state.marks):
        sel = state.marks[state.selected_mark_idx]
        sel_color = SHOT_COLORS.get(sel.shot_type, (255, 255, 255))
        sel_text = (
            f"SELECTED: {sel.shot_type.upper()} @ frame {sel.frame} "
            f"({state.timestamp(sel.frame)})  [arrows to move, F/B/S to retype]"
        )
        draw_text_with_bg(frame, sel_text, (10, 55), color=sel_color)

    # Toast message (visible for 2 seconds)
    if state.message and (time.time() - state.message_time) < 2.0:
        draw_text_with_bg(
            frame, state.message, (w // 2 - 150, 25),
            color=(100, 255, 255),
        )

    # Stats bar: per-type counts (above timeline)
    counts = {}
    for m in state.marks:
        counts[m.shot_type] = counts.get(m.shot_type, 0) + 1

    stats_parts = []
    for st in SHOT_TYPES:
        c = counts.get(st, 0)
        stats_parts.append(f"{st[0].upper()}:{c}")
    stats_text = "  ".join(stats_parts) + f"  Total:{len(state.marks)}"
    draw_text_with_bg(frame, stats_text, (10, h - 85))


def draw_timeline(frame, state: AppState):
    """Colored timeline bar with marks as vertical ticks and playhead cursor."""
    h, w = frame.shape[:2]
    bar_h = 60  # Taller bar for easier clicking
    bar_y = h - bar_h - 10
    bar_x = 10
    bar_w = w - 20

    # Store geometry so mouse callback can hit-test
    state.tl_x = bar_x
    state.tl_y = bar_y
    state.tl_w = bar_w
    state.tl_h = bar_h

    # Dark background
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (30, 30, 30), cv2.FILLED)

    if state.total_frames <= 1:
        return

    # Draw marks as vertical ticks (single-frame contact points)
    for i, mark in enumerate(state.marks):
        mx = bar_x + int(mark.frame / (state.total_frames - 1) * bar_w)
        color = SHOT_COLORS.get(mark.shot_type, (180, 180, 180))
        thickness = 3 if i == state.selected_mark_idx else 2
        cv2.line(frame, (mx, bar_y + 2), (mx, bar_y + bar_h - 2), color, thickness)
        # Highlight selected mark with white outline and triangle
        if i == state.selected_mark_idx:
            # White border around tick
            cv2.line(frame, (mx, bar_y), (mx, bar_y + bar_h), (255, 255, 255), 5)
            cv2.line(frame, (mx, bar_y + 2), (mx, bar_y + bar_h - 2), color, 3)
            # Small triangle above
            pts = [(mx - 6, bar_y - 8), (mx + 6, bar_y - 8), (mx, bar_y - 2)]
            cv2.fillPoly(frame, [np.array(pts)], (255, 255, 255))

    # Playhead cursor (thin white line)
    px = bar_x + int(state.current_frame / (state.total_frames - 1) * bar_w)
    cv2.line(frame, (px, bar_y - 2), (px, bar_y + bar_h + 2), (255, 255, 255), 2)


def draw_help_overlay(frame):
    """Full key legend overlay."""
    h, w = frame.shape[:2]

    # Semi-transparent background
    overlay = frame.copy()
    margin = 60
    cv2.rectangle(
        overlay,
        (margin, margin),
        (w - margin, h - margin),
        (20, 20, 20),
        cv2.FILLED,
    )
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    # Draw help text
    y = margin + 35
    for line in HELP_TEXT:
        if line:
            draw_text_with_bg(
                frame, line, (margin + 20, y),
                font_scale=0.5,
                bg_color=(20, 20, 20),
            )
        y += 22


# ── Mark management ──────────────────────────────────────────


def set_or_create_mark(state: AppState, shot_type: str):
    """Set selected mark's type if at same frame, otherwise create new mark.

    If a mark is selected AND we're at its frame, change its shot_type.
    Otherwise, create a new mark at the current frame.
    """
    # Check if we're at the selected mark's position
    at_selected = False
    if state.selected_mark_idx is not None and state.selected_mark_idx < len(state.marks):
        selected_frame = state.marks[state.selected_mark_idx].frame
        at_selected = abs(state.current_frame - selected_frame) < 5  # Within 5 frames

    if at_selected:
        # Change selected mark's type
        old_mark = state.marks[state.selected_mark_idx]
        state.marks[state.selected_mark_idx] = Mark(shot_type, old_mark.frame)
        state.set_message(f"Changed to {shot_type}")
    else:
        # Create new mark at current frame
        new_mark = Mark(shot_type, state.current_frame)
        state.marks.append(new_mark)
        # Sort marks by frame and select the new one
        state.marks.sort(key=lambda m: m.frame)
        state.selected_mark_idx = next(
            i for i, m in enumerate(state.marks) if m.frame == state.current_frame and m.shot_type == shot_type
        )
        state.set_message(f"Created {shot_type} @ {state.current_frame}")


def move_selected_mark(state: AppState, delta: int):
    """Move the selected mark by delta frames."""
    if state.selected_mark_idx is None or state.selected_mark_idx >= len(state.marks):
        state.set_message("No mark selected")
        return

    old_mark = state.marks[state.selected_mark_idx]
    new_frame = max(0, min(state.total_frames - 1, old_mark.frame + delta))

    if new_frame == old_mark.frame:
        return  # No change

    state.marks[state.selected_mark_idx] = Mark(old_mark.shot_type, new_frame)
    # Re-sort marks and update selection index
    old_type = old_mark.shot_type
    state.marks.sort(key=lambda m: m.frame)
    # Find the mark we just moved
    state.selected_mark_idx = next(
        i for i, m in enumerate(state.marks) if m.frame == new_frame and m.shot_type == old_type
    )


def jump_to_mark(state: AppState, direction: int, cap):
    """Jump to the next or previous mark.

    direction: -1 for previous, +1 for next
    """
    if not state.marks:
        state.set_message("No marks")
        return

    if direction > 0:
        # Find next mark after current frame
        for i, m in enumerate(state.marks):
            if m.frame > state.current_frame:
                state.selected_mark_idx = i
                state.current_frame = m.frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, m.frame)
                state.playing = False
                return
        # Wrap to first mark
        state.selected_mark_idx = 0
        state.current_frame = state.marks[0].frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, state.marks[0].frame)
        state.playing = False
    else:
        # Find previous mark before current frame
        for i in range(len(state.marks) - 1, -1, -1):
            if state.marks[i].frame < state.current_frame:
                state.selected_mark_idx = i
                state.current_frame = state.marks[i].frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, state.marks[i].frame)
                state.playing = False
                return
        # Wrap to last mark
        state.selected_mark_idx = len(state.marks) - 1
        state.current_frame = state.marks[-1].frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, state.marks[-1].frame)
        state.playing = False


def undo_last_mark(state: AppState):
    """Pop the last completed mark with feedback."""
    if state.marks:
        removed = state.marks.pop()
        # Clear selection if it pointed at the removed mark
        if state.selected_mark_idx is not None:
            if state.selected_mark_idx >= len(state.marks):
                state.selected_mark_idx = None
        state.set_message(f"Undid {removed.shot_type} @ {removed.frame}")
    else:
        state.set_message("Nothing to undo")


def delete_selected_mark(state: AppState):
    """Delete the currently selected mark."""
    if state.selected_mark_idx is not None and state.selected_mark_idx < len(state.marks):
        removed = state.marks.pop(state.selected_mark_idx)
        state.set_message(f"Deleted {removed.shot_type} @ {removed.frame}")
        # Adjust selection
        if state.selected_mark_idx >= len(state.marks):
            state.selected_mark_idx = len(state.marks) - 1 if state.marks else None
    else:
        state.set_message("No mark selected")


# ── CSV output ───────────────────────────────────────────────


def write_csv(marks: List[Mark], output_path: str):
    """Write marks to CSV (single-frame contact points)."""
    with open(output_path, "w") as f:
        f.write("# Labels from visual_label.py (v2: single-frame contact points)\n")
        f.write("# shot_type, frame\n")
        for mark in sorted(marks, key=lambda m: m.frame):
            f.write(f"{mark.shot_type},{mark.frame}\n")


# ── CSV / pose loading ───────────────────────────────────────


def load_marks_from_csv(csv_path: str) -> List[Mark]:
    """Load marks from a CSV file.

    Supports both formats:
    - New (v2): shot_type, frame
    - Old (v1): shot_type, start_frame, end_frame (uses midpoint as contact)
    """
    marks = []
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].strip().startswith("#"):
                continue
            shot_type = row[0].strip().lower()
            if shot_type not in SHOT_TYPES:
                continue
            try:
                if len(row) == 2:
                    # New v2 format: shot_type, frame
                    frame = int(row[1].strip())
                    if frame >= 0:
                        marks.append(Mark(shot_type, frame))
                elif len(row) >= 3:
                    # Old v1 format: shot_type, start_frame, end_frame
                    # Use midpoint as contact frame
                    start = int(row[1].strip())
                    end = int(row[2].strip())
                    if 0 <= start <= end:
                        mid = (start + end) // 2
                        marks.append(Mark(shot_type, mid))
            except ValueError:
                continue
    # Sort by frame
    marks.sort(key=lambda m: m.frame)
    return marks


def _compute_arm_speed(pose_data: dict) -> dict:
    """Per-frame arm speed from pose landmarks (smoothed).

    Returns dict of frame_idx -> smoothed speed.
    """
    speed_map = {}
    prev = {}
    for fr in pose_data["frames"]:
        if not fr["detected"] or not fr["world_landmarks"]:
            prev = {}
            continue
        wl = fr["world_landmarks"]
        curr = {li: (wl[li][0], wl[li][1], wl[li][2]) for li in (13, 14, 15, 16)}
        if prev:
            dist = sum(
                sum((curr[li][j] - prev[li][j]) ** 2 for j in range(3)) ** 0.5
                for li in curr if li in prev
            )
            speed_map[fr["frame_idx"]] = dist
        prev = curr

    if not speed_map:
        return {}

    # Smooth with window of 7
    smooth = {}
    sorted_keys = sorted(speed_map.keys())
    w = 7
    for i in range(w, len(sorted_keys) - w):
        idx = sorted_keys[i]
        vals = [speed_map[sorted_keys[j]] for j in range(i - w, i + w + 1)]
        smooth[idx] = sum(vals) / len(vals)
    return smooth


def _detect_ball_signal(video_path: str, total_frames: int) -> dict:
    """Per-frame moving-yellow-pixel score from video (ball in flight).

    Returns dict of frame_idx -> score.
    """
    import numpy as np

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {}

    lower_yellow = np.array([20, 50, 100])
    upper_yellow = np.array([55, 255, 255])
    kernel = np.ones((5, 5), np.uint8)

    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        return {}

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    ball_signal = {}

    for fi in range(1, total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray, prev_gray)
        _, motion = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        motion = cv2.dilate(motion, kernel, iterations=2)
        yellow = cv2.inRange(
            cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), lower_yellow, upper_yellow
        )
        score = cv2.countNonZero(cv2.bitwise_and(motion, yellow))
        if score > 50:
            ball_signal[fi] = score
        prev_gray = gray

    cap.release()
    return ball_signal


def detect_shots(pose_path: str, video_path: Optional[str] = None, shot_type: str = "forehand") -> List[Mark]:
    """Auto-detect shot segments using arm speed + ball flight detection.

    Uses two signals:
      1. Arm movement speed from pose landmarks (wrists + elbows)
      2. Moving yellow pixels in video frames (tennis ball in flight)

    Frames where both signals are high → shot contact zone.
    Each contact is windowed with buffer before and after for manual review.

    Args:
        pose_path: Path to pose JSON file
        video_path: Optional path to video for ball detection
        shot_type: Shot type to assign to detected segments (default: forehand)
    """
    with open(pose_path, "r") as f:
        data = json.load(f)

    total = data["video_info"]["total_frames"]
    fps = data["video_info"]["fps"]

    # --- Arm speed signal ---
    arm_smooth = _compute_arm_speed(data)
    if not arm_smooth:
        return []
    arm_vals = sorted(arm_smooth.values())
    arm_thresh = arm_vals[3 * len(arm_vals) // 4]  # P75

    # --- Ball signal (if video available) ---
    ball_signal = {}
    if video_path and os.path.isfile(video_path):
        print(f"  Scanning video for ball flight...")
        ball_signal = _detect_ball_signal(video_path, total)

    # --- Find contact points from ball signal ---
    if ball_signal:
        ball_vals = sorted(ball_signal.values())
        ball_thresh = ball_vals[9 * len(ball_vals) // 10]  # P90 — only strong hits

        # High ball-signal frames
        hot_frames = sorted(k for k, v in ball_signal.items() if v > ball_thresh)
        if not hot_frames:
            return []

        # Cluster ball events (gap > 15 = separate hit)
        clusters = []
        start = end = hot_frames[0]
        for f in hot_frames[1:]:
            if f - end <= 15:
                end = f
            else:
                clusters.append((start, end))
                start = end = f
        clusters.append((start, end))

        # Keep only clusters that overlap with high arm speed (filters noise)
        contacts = []
        for s, e in clusters:
            has_arm = any(arm_smooth.get(f, 0) > arm_thresh for f in range(s, e + 1))
            if has_arm:
                peak = max(range(s, e + 1), key=lambda f: ball_signal.get(f, 0))
                contacts.append(peak)
    else:
        # Fallback: arm speed only — find peak per cluster
        active = sorted(f for f in arm_smooth if arm_smooth[f] > arm_thresh)
        if not active:
            return []
        clusters = []
        start = end = active[0]
        for f in active[1:]:
            if f - end <= 45:
                end = f
            else:
                clusters.append((start, end))
                start = end = f
        clusters.append((start, end))
        contacts = [
            max(range(s, e + 1), key=lambda f: arm_smooth.get(f, 0))
            for s, e in clusters
        ]

    # Return contact points as single-frame marks
    marks = []
    for peak in contacts:
        marks.append(Mark(shot_type, peak))

    return marks


def _find_pose_json(video_path: str) -> Optional[str]:
    """Given a skeleton video path, find the matching pose JSON."""
    basename = os.path.basename(video_path)
    name = basename.replace("_skeleton.mp4", "").replace("_skeleton.MP4", "")
    pose_path = os.path.join(POSES_DIR, f"{name}.json")
    if os.path.isfile(pose_path):
        return pose_path
    return None


def _find_preprocessed_video(pose_path: str) -> Optional[str]:
    """Given a pose JSON, find the matching preprocessed video."""
    with open(pose_path, "r") as f:
        data = json.load(f)
    source = data.get("source_video", "")
    if source:
        from config.settings import PREPROCESSED_DIR
        vid_path = os.path.join(PREPROCESSED_DIR, source)
        if os.path.isfile(vid_path):
            return vid_path
    return None


# ── Key handling ─────────────────────────────────────────────

# OpenCV key codes
KEY_SPACE = 32
KEY_ENTER = 13
KEY_ESCAPE = 27
KEY_LEFT = 63234   # macOS
KEY_RIGHT = 63235  # macOS
KEY_UP = 63232     # macOS
KEY_DOWN = 63233   # macOS
KEY_LEFT_LIN = 65361
KEY_RIGHT_LIN = 65363
KEY_UP_LIN = 65362
KEY_DOWN_LIN = 65364


def handle_key(raw_key: int, state: AppState, cap) -> Optional[str]:
    """Route keypress to action. Returns 'quit' to exit, else None."""
    # Check for shift modifier
    shift_held = (raw_key & 0x10000) != 0

    # Mask platform-specific high bits
    key = raw_key & 0xFFFF

    # Letters (case-insensitive)
    ch = chr(key).lower() if 0 < key < 128 else ""

    if ch == "q":
        return "quit"

    if ch == " " or key == KEY_SPACE:
        state.playing = not state.playing
        return None

    # Step back 1 frame
    if ch == "a":
        state.playing = False
        state.current_frame = max(0, state.current_frame - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, state.current_frame)
        return None

    # Step forward 1 frame
    if ch == "d":
        state.playing = False
        state.current_frame = min(state.total_frames - 1, state.current_frame + 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, state.current_frame)
        return None

    # Left/Right arrows — step video frame by frame
    if key in (KEY_LEFT, KEY_LEFT_LIN, 2):
        state.playing = False
        state.current_frame = max(0, state.current_frame - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, state.current_frame)
        return None

    if key in (KEY_RIGHT, KEY_RIGHT_LIN, 3):
        state.playing = False
        state.current_frame = min(state.total_frames - 1, state.current_frame + 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, state.current_frame)
        return None

    # Up/Down arrows — speed when playing, move mark when paused
    if key in (KEY_UP, KEY_UP_LIN, 0):
        if state.playing:
            # Speed up
            if state.speed_idx < len(SPEED_LEVELS) - 1:
                state.speed_idx += 1
                state.set_message(f"Speed: {state.speed}x")
        else:
            # Move selected mark forward
            move_selected_mark(state, 1)
            if state.selected_mark_idx is not None and state.selected_mark_idx < len(state.marks):
                state.current_frame = state.marks[state.selected_mark_idx].frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, state.current_frame)
        return None

    if key in (KEY_DOWN, KEY_DOWN_LIN, 1):
        if state.playing:
            # Slow down
            if state.speed_idx > 0:
                state.speed_idx -= 1
                state.set_message(f"Speed: {state.speed}x")
        else:
            # Move selected mark backward
            move_selected_mark(state, -1)
            if state.selected_mark_idx is not None and state.selected_mark_idx < len(state.marks):
                state.current_frame = state.marks[state.selected_mark_idx].frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, state.current_frame)
        return None

    # Speed up (+, =, or w)
    if ch in ("+", "=", "w"):
        if state.speed_idx < len(SPEED_LEVELS) - 1:
            state.speed_idx += 1
            state.set_message(f"Speed: {state.speed}x")
        return None

    # Slow down (-, _, or e)
    if ch in ("-", "_", "e"):
        if state.speed_idx > 0:
            state.speed_idx -= 1
            state.set_message(f"Speed: {state.speed}x")
        return None

    # Jump 5 seconds back (comma)
    if ch == ",":
        jump = int(5 * state.fps)
        state.current_frame = max(0, state.current_frame - jump)
        cap.set(cv2.CAP_PROP_POS_FRAMES, state.current_frame)
        state.playing = False
        return None

    # Jump 5 seconds forward (period)
    if ch == ".":
        jump = int(5 * state.fps)
        state.current_frame = min(state.total_frames - 1, state.current_frame + jump)
        cap.set(cv2.CAP_PROP_POS_FRAMES, state.current_frame)
        state.playing = False
        return None

    # Jump to previous mark
    if ch == "[":
        jump_to_mark(state, -1, cap)
        return None

    # Jump to next mark
    if ch == "]":
        jump_to_mark(state, +1, cap)
        return None

    # Shot type keys — set selected mark's type OR create new mark
    if ch == "f":
        set_or_create_mark(state, "forehand")
        return None
    if ch == "b":
        set_or_create_mark(state, "backhand")
        return None
    if ch == "s":
        set_or_create_mark(state, "serve")
        return None
    if ch == "n":
        set_or_create_mark(state, "neutral")
        return None

    # Undo (U or Ctrl+Z)
    if ch == "u" or key == 26:
        undo_last_mark(state)
        return None

    # Delete selected mark (X or Delete)
    if ch == "x" or key == 127 or key == 65535:
        delete_selected_mark(state)
        return None

    # Toggle help
    if ch == "h":
        state.show_help = not state.show_help
        return None

    # Cycle view angle
    if ch == "v":
        state.cycle_view_angle()
        return None

    # Escape — deselect mark
    if key == KEY_ESCAPE:
        state.selected_mark_idx = None
        state.set_message("Deselected")
        return None

    return None


# ── Mouse handling ───────────────────────────────────────────


MARK_SELECT_PX = 15  # pixel threshold for selecting a mark by clicking near it


def _frame_at_x(state: AppState, x: int) -> int:
    """Convert an x pixel on the timeline bar to a frame number."""
    ratio = (x - state.tl_x) / state.tl_w
    target = int(ratio * (state.total_frames - 1))
    return max(0, min(state.total_frames - 1, target))


def _x_at_frame(state: AppState, frame: int) -> int:
    """Convert a frame number to an x pixel on the timeline bar."""
    if state.total_frames <= 1:
        return state.tl_x
    return state.tl_x + int(frame / (state.total_frames - 1) * state.tl_w)


def _nearest_mark_at_x(state: AppState, x: int) -> Optional[int]:
    """Return the index of the nearest mark within threshold, or None."""
    if state.tl_w <= 0 or state.total_frames <= 1 or not state.marks:
        return None

    best_idx = None
    best_dist = MARK_SELECT_PX + 1

    for i, m in enumerate(state.marks):
        mx = _x_at_frame(state, m.frame)
        dist = abs(x - mx)
        if dist < best_dist:
            best_dist = dist
            best_idx = i

    return best_idx if best_dist <= MARK_SELECT_PX else None


def make_mouse_callback(state: AppState, cap):
    """Return a mouse callback for timeline scrub and mark selection."""
    drag_mode = [None]  # 'scrub' or None
    needs_seek = [False]  # Flag to seek on release

    def on_mouse(event, x, y, flags, param):
        # Timeline bar area check - generous hit area
        in_bar = (state.tl_w > 0
                  and state.tl_x - 20 <= x <= state.tl_x + state.tl_w + 20
                  and state.tl_y - 30 <= y <= state.tl_y + state.tl_h + 20)

        if event == cv2.EVENT_LBUTTONDOWN and in_bar:
            # Check if clicking near a mark
            hit = _nearest_mark_at_x(state, x)
            if hit is not None:
                # Select the mark and jump to its frame
                state.selected_mark_idx = hit
                m = state.marks[hit]
                state.current_frame = m.frame
                state.playing = False
                cap.set(cv2.CAP_PROP_POS_FRAMES, m.frame)
                drag_mode[0] = None
            else:
                # Start scrubbing - just update position, don't seek yet
                state.selected_mark_idx = None
                target = _frame_at_x(state, x)
                state.current_frame = target
                state.playing = False
                state.scrubbing = True
                drag_mode[0] = "scrub"
                needs_seek[0] = True

        elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):
            if drag_mode[0] == "scrub" and in_bar:
                # Just update the frame number for visual feedback, don't seek
                target = _frame_at_x(state, x)
                state.current_frame = target
                needs_seek[0] = True

        elif event == cv2.EVENT_LBUTTONUP:
            if drag_mode[0] == "scrub" and needs_seek[0]:
                # Now actually seek the video
                state.scrubbing = False
                cap.set(cv2.CAP_PROP_POS_FRAMES, state.current_frame)
                needs_seek[0] = False
            drag_mode[0] = None
            state.scrubbing = False

        elif event == cv2.EVENT_LBUTTONDOWN and not in_bar:
            state.selected_mark_idx = None

    return on_mouse


# ── Main event loop ──────────────────────────────────────────


def run_labeler(video_path: str, output_csv: str, initial_marks: Optional[List[Mark]] = None):
    """Main event loop: open video, handle keys, draw overlays."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: cannot open video {video_path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    state = AppState(total_frames, fps)
    if initial_marks:
        # Filter marks to valid frame range
        state.marks = [
            m for m in initial_marks
            if 0 <= m.frame < total_frames
        ]
        # Sort by frame
        state.marks.sort(key=lambda m: m.frame)

    # Extract video name for metadata (e.g., IMG_6665 from IMG_6665_skeleton.mp4)
    video_basename = os.path.basename(video_path)
    video_name = video_basename.replace("_skeleton.mp4", "").replace("_skeleton.MP4", "")
    if video_name == video_basename:
        video_name = os.path.splitext(video_basename)[0]
    state.video_name = video_name

    # Load existing view angle from metadata
    metadata = load_video_metadata(video_name)
    state.view_angle = metadata.get("view_angle", DEFAULT_VIEW_ANGLE)

    window_name = f"Visual Labeler — {os.path.basename(video_path)}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    # Register mouse callback for timeline scrubbing
    cv2.setMouseCallback(window_name, make_mouse_callback(state, cap))

    print(f"Opened: {os.path.basename(video_path)}")
    print(f"  {total_frames} frames, {fps:.1f} fps, {total_frames / fps:.1f}s")
    print(f"  Output: {output_csv}")
    print("  Click timeline to scrub, press H for help")

    # Read first frame
    ret, display_frame = cap.read()
    if not ret:
        print("Error: cannot read first frame")
        cap.release()
        sys.exit(1)

    last_read_frame = 0

    while True:
        # ── Draw overlays on a copy ──
        frame = display_frame.copy()
        draw_overlay(frame, state)
        draw_timeline(frame, state)
        if state.show_help:
            draw_help_overlay(frame)
        cv2.imshow(window_name, frame)

        # ── Wait for key ──
        if state.playing:
            target_delay = 1.0 / (fps * state.speed)
            wait_ms = max(1, int(target_delay * 1000))
            raw_key = cv2.waitKeyEx(wait_ms)
        else:
            # Paused — short poll so mouse callbacks fire
            raw_key = cv2.waitKeyEx(1)  # Maximum responsiveness for scrubbing

        # Check if window was closed
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

        # ── Handle key ──
        if raw_key != -1:
            action = handle_key(raw_key, state, cap)
            if action == "quit":
                break

        # ── Advance frame if playing ──
        if state.playing:
            # Calculate frames to skip based on speed
            if state.speed > 1.0:
                skip = int(state.speed)
            else:
                skip = 1

            next_frame = min(state.current_frame + skip, total_frames - 1)
            if next_frame != state.current_frame:
                state.current_frame = next_frame
                if skip > 1:
                    # Must seek for non-sequential reads
                    cap.set(cv2.CAP_PROP_POS_FRAMES, state.current_frame)
                    ret, new_frame = cap.read()
                elif last_read_frame == state.current_frame - 1:
                    # Sequential read is fastest
                    ret, new_frame = cap.read()
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, state.current_frame)
                    ret, new_frame = cap.read()

                if ret:
                    display_frame = new_frame
                    last_read_frame = state.current_frame
                else:
                    state.playing = False
                    state.current_frame = total_frames - 1
            else:
                state.playing = False
        else:
            # Paused — re-read frame if position changed (step keys)
            if state.current_frame != last_read_frame:
                cap.set(cv2.CAP_PROP_POS_FRAMES, state.current_frame)
                ret, new_frame = cap.read()
                if ret:
                    display_frame = new_frame
                    last_read_frame = state.current_frame

    # ── Cleanup ──
    cap.release()
    cv2.destroyAllWindows()

    # ── Save view angle to metadata ──
    if state.video_name:
        metadata = load_video_metadata(state.video_name)
        metadata["view_angle"] = state.view_angle
        save_video_metadata(state.video_name, metadata)
        print(f"Saved view angle: {state.view_angle}")

    # ── Save CSV ──
    if state.marks:
        write_csv(state.marks, output_csv)
        print(f"\nSaved {len(state.marks)} marks to {output_csv}")
        for st in SHOT_TYPES:
            count = sum(1 for m in state.marks if m.shot_type == st)
            if count:
                print(f"  {st}: {count}")
    else:
        print("\nNo marks to save.")


# ── CLI ──────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Visual shot labeling tool for skeleton videos.",
        epilog="Output CSV is compatible with: python scripts/label_clips.py --csv <file>",
    )
    parser.add_argument(
        "video",
        nargs="?",
        help="Path to a *_skeleton.mp4 file (default: auto-discover in poses/)",
    )
    parser.add_argument(
        "-o", "--output",
        help="Custom CSV output path (default: {video_name}_labels.csv in project root)",
    )
    parser.add_argument(
        "--load",
        help="Pre-load marks from a CSV file for review/editing",
    )
    parser.add_argument(
        "--detect",
        action="store_true",
        help="Auto-detect shots from pose data and pre-load for review",
    )
    parser.add_argument(
        "--detect-type",
        choices=SHOT_TYPES,
        default="forehand",
        help="Shot type to assign to auto-detected segments (default: forehand)",
    )
    args = parser.parse_args()

    # ── Resolve video path ──
    if args.video:
        video_path = args.video
        if not os.path.isabs(video_path):
            video_path = os.path.join(os.getcwd(), video_path)
    else:
        skeleton_files = find_skeleton_videos(POSES_DIR)
        if not skeleton_files:
            print(f"No *_skeleton.mp4 files found in {POSES_DIR}/")
            print("Run extract_poses.py first.")
            sys.exit(1)

        if len(skeleton_files) == 1:
            video_path = os.path.join(POSES_DIR, skeleton_files[0])
            print(f"Auto-selected: {skeleton_files[0]}")
        else:
            print("Available skeleton videos:")
            for i, f in enumerate(skeleton_files, 1):
                size = os.path.getsize(os.path.join(POSES_DIR, f))
                size_mb = size / (1024 * 1024)
                print(f"  {i}. {f} ({size_mb:.1f} MB)")
            try:
                choice = input(f"Select [1-{len(skeleton_files)}]: ").strip()
                idx = int(choice) - 1
                if idx < 0 or idx >= len(skeleton_files):
                    print("Invalid selection.")
                    sys.exit(1)
            except (ValueError, EOFError, KeyboardInterrupt):
                print("\nAborted.")
                sys.exit(1)
            video_path = os.path.join(POSES_DIR, skeleton_files[idx])

    if not os.path.isfile(video_path):
        print(f"Error: file not found: {video_path}")
        sys.exit(1)

    # ── Resolve output path ──
    if args.output:
        output_csv = args.output
        if not os.path.isabs(output_csv):
            output_csv = os.path.join(os.getcwd(), output_csv)
    else:
        # Derive from video name: IMG_0870_skeleton.mp4 → IMG_0870_labels.csv
        video_name = os.path.basename(video_path)
        base = video_name.replace("_skeleton.mp4", "").replace("_skeleton.MP4", "")
        if base == video_name:
            # Fallback: strip extension
            base = os.path.splitext(video_name)[0]
        output_csv = os.path.join(PROJECT_ROOT, f"{base}_labels.csv")

    # ── Pre-load marks ──
    initial_marks = None

    if args.load:
        load_path = args.load
        if not os.path.isabs(load_path):
            load_path = os.path.join(os.getcwd(), load_path)
        if not os.path.isfile(load_path):
            print(f"Error: CSV not found: {load_path}")
            sys.exit(1)
        initial_marks = load_marks_from_csv(load_path)
        print(f"Loaded {len(initial_marks)} marks from {os.path.basename(load_path)}")

    if args.detect:
        pose_path = _find_pose_json(video_path)
        if pose_path is None:
            print("Error: no matching pose JSON found in poses/")
            print("  Run extract_poses.py first, or use --load with a CSV.")
            sys.exit(1)
        preproc_path = _find_preprocessed_video(pose_path)
        print(f"Detecting shots from {os.path.basename(pose_path)}...")
        if preproc_path:
            print(f"  + ball tracking from {os.path.basename(preproc_path)}")
        detected = detect_shots(pose_path, preproc_path, shot_type=args.detect_type)
        print(f"  Found {len(detected)} segments (marked as {args.detect_type})")
        if initial_marks:
            initial_marks.extend(detected)
        else:
            initial_marks = detected

    print(f"Video: {video_path}")
    print(f"CSV output: {output_csv}")

    run_labeler(video_path, output_csv, initial_marks)


if __name__ == "__main__":
    main()
