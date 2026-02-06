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

# Add project root so config is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import POSES_DIR, PROJECT_ROOT, SHOT_TYPES, VIEW_ANGLES, DEFAULT_VIEW_ANGLE
from scripts.video_metadata import load_video_metadata, save_video_metadata

# ── Data structures ──────────────────────────────────────────


class Mark(NamedTuple):
    shot_type: str
    start_frame: int
    end_frame: int


# Shot-type colors (BGR)
SHOT_COLORS = {
    "forehand": (0, 200, 0),      # green
    "backhand": (200, 100, 0),    # blue-ish
    "serve": (0, 100, 255),       # orange
    "neutral": (180, 180, 180),   # gray
}

SPEED_LEVELS = [0.25, 0.5, 1.0, 2.0, 4.0]

HELP_TEXT = [
    "=== Controls ===",
    "",
    "Space      Play / Pause",
    "A          Step back 1 frame",
    "D          Step forward 1 frame",
    "Left/Right Jump 5 sec back / forward",
    "[  /  ]    Jump 5 sec back / forward",
    "W          Speed up",
    "E          Slow down",
    "Click      Click timeline bar to scrub",
    "",
    "F          Start/end forehand mark",
    "B          Start/end backhand mark",
    "S          Start/end serve mark",
    "N          Start/end neutral mark",
    "Enter      End any active mark",
    "Escape     Cancel active mark",
    "U/Ctrl+Z   Undo last mark (repeatable)",
    "X/Delete   Delete selected mark",
    "Click seg  Select it, drag edges to resize",
    "",
    "V          Cycle view angle (camera position)",
    "H          Toggle this help",
    "Q          Save CSV and quit",
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

        # Marking
        self.marking_type: Optional[str] = None
        self.marking_start: Optional[int] = None

        # Completed marks
        self.marks: List[Mark] = []

        # Selection and edge-dragging
        self.selected_mark_idx: Optional[int] = None
        self.dragging_edge: Optional[str] = None  # 'start', 'end', or None

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
    """Frame/time info, marking indicator, and per-type mark counts."""
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

    # Marking indicator
    if state.marking_type is not None:
        color = SHOT_COLORS.get(state.marking_type, (255, 255, 255))
        mark_text = (
            f"MARKING: {state.marking_type.upper()} from {state.marking_start} "
            f"({state.timestamp(state.marking_start)})"
        )
        draw_text_with_bg(frame, mark_text, (10, 55), color=color)

    # Selection indicator
    if state.selected_mark_idx is not None and state.selected_mark_idx < len(state.marks):
        sel = state.marks[state.selected_mark_idx]
        sel_color = SHOT_COLORS.get(sel.shot_type, (255, 255, 255))
        if state.dragging_edge:
            hint = f"dragging {state.dragging_edge}"
        else:
            hint = "drag edges to resize, X to delete"
        sel_text = (
            f"SELECTED: {sel.shot_type.upper()} "
            f"{sel.start_frame}-{sel.end_frame}  [{hint}]"
        )
        y_pos = 55 if state.marking_type is None else 85
        draw_text_with_bg(frame, sel_text, (10, y_pos), color=sel_color)

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
    draw_text_with_bg(frame, stats_text, (10, h - 50))


def draw_timeline(frame, state: AppState):
    """Colored timeline bar with playhead cursor at the bottom 40px."""
    h, w = frame.shape[:2]
    bar_y = h - 40
    bar_h = 30
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

    # Draw marks as colored segments
    for i, mark in enumerate(state.marks):
        x1 = bar_x + int(mark.start_frame / (state.total_frames - 1) * bar_w)
        x2 = bar_x + int(mark.end_frame / (state.total_frames - 1) * bar_w)
        x2 = max(x2, x1 + 1)
        color = SHOT_COLORS.get(mark.shot_type, (180, 180, 180))
        cv2.rectangle(frame, (x1, bar_y), (x2, bar_y + bar_h), color, cv2.FILLED)
        # White outline + edge handles on selected mark
        if i == state.selected_mark_idx:
            cv2.rectangle(frame, (x1, bar_y), (x2, bar_y + bar_h), (255, 255, 255), 2)
            # Left handle
            cv2.rectangle(frame, (x1 - 3, bar_y - 4), (x1 + 3, bar_y + bar_h + 4), (255, 255, 255), cv2.FILLED)
            # Right handle
            cv2.rectangle(frame, (x2 - 3, bar_y - 4), (x2 + 3, bar_y + bar_h + 4), (255, 255, 255), cv2.FILLED)

    # Draw active mark in-progress (blinking style: semi-transparent)
    if state.marking_type is not None and state.marking_start is not None:
        x1 = bar_x + int(state.marking_start / (state.total_frames - 1) * bar_w)
        x2 = bar_x + int(state.current_frame / (state.total_frames - 1) * bar_w)
        if x2 < x1:
            x1, x2 = x2, x1
        x2 = max(x2, x1 + 1)
        color = SHOT_COLORS.get(state.marking_type, (180, 180, 180))
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, bar_y), (x2, bar_y + bar_h), color, cv2.FILLED)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    # Playhead cursor
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


def start_mark(state: AppState, shot_type: str):
    """Begin or end marking a segment.

    If already marking the same type, finalize it.
    If marking a different type, finalize current and start new.
    If not marking, start a new mark.
    """
    if state.marking_type == shot_type:
        # Same key pressed again — end the mark
        end_mark(state)
    elif state.marking_type is not None:
        # Different type — end current, start new
        end_mark(state)
        state.marking_type = shot_type
        state.marking_start = state.current_frame
    else:
        state.marking_type = shot_type
        state.marking_start = state.current_frame


def end_mark(state: AppState):
    """Finalize the active mark and append to marks list."""
    if state.marking_type is None or state.marking_start is None:
        return

    start = state.marking_start
    end = state.current_frame

    # Ensure start < end
    if start > end:
        start, end = end, start

    if start == end:
        # Zero-length mark — discard
        state.marking_type = None
        state.marking_start = None
        return

    state.marks.append(Mark(state.marking_type, start, end))
    state.marking_type = None
    state.marking_start = None


def cancel_mark(state: AppState):
    """Discard the in-progress mark."""
    state.marking_type = None
    state.marking_start = None


def undo_last_mark(state: AppState):
    """Pop the last completed mark with feedback."""
    if state.marks:
        removed = state.marks.pop()
        # Clear selection if it pointed at the removed mark
        if state.selected_mark_idx is not None:
            if state.selected_mark_idx >= len(state.marks):
                state.selected_mark_idx = None
        state.set_message(
            f"Undid {removed.shot_type} {removed.start_frame}-{removed.end_frame}"
        )
    else:
        state.set_message("Nothing to undo")


def delete_selected_mark(state: AppState):
    """Delete the currently selected mark."""
    if state.selected_mark_idx is not None and state.selected_mark_idx < len(state.marks):
        removed = state.marks.pop(state.selected_mark_idx)
        state.set_message(
            f"Deleted {removed.shot_type} {removed.start_frame}-{removed.end_frame}"
        )
        state.selected_mark_idx = None
    else:
        state.set_message("No mark selected")


# ── CSV output ───────────────────────────────────────────────


def write_csv(marks: List[Mark], output_path: str):
    """Write marks to CSV compatible with label_clips.py --csv."""
    with open(output_path, "w") as f:
        f.write("# Labels from visual_label.py\n")
        f.write("# shot_type, start_frame, end_frame\n")
        for mark in marks:
            f.write(f"{mark.shot_type},{mark.start_frame},{mark.end_frame}\n")


# ── CSV / pose loading ───────────────────────────────────────


def load_marks_from_csv(csv_path: str) -> List[Mark]:
    """Load marks from a CSV file (same format as output)."""
    marks = []
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].strip().startswith("#"):
                continue
            if len(row) < 3:
                continue
            shot_type = row[0].strip().lower()
            if shot_type not in SHOT_TYPES:
                continue
            try:
                start = int(row[1].strip())
                end = int(row[2].strip())
            except ValueError:
                continue
            if 0 <= start < end:
                marks.append(Mark(shot_type, start, end))
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

    # Window each contact: 1 second before, 1 second after
    # This gives ~2 second segments which comfortably contain a full shot motion
    buf = int(1 * fps)
    marks = []
    for peak in contacts:
        ws = max(0, peak - buf)
        we = min(total - 1, peak + buf)
        marks.append(Mark(shot_type, ws, we))

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
KEY_LEFT_LIN = 65361
KEY_RIGHT_LIN = 65363


def handle_key(raw_key: int, state: AppState, cap) -> Optional[str]:
    """Route keypress to action. Returns 'quit' to exit, else None."""
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

    # Arrow left — jump 5 seconds back
    if key in (KEY_LEFT, KEY_LEFT_LIN):
        jump = int(5 * state.fps)
        state.current_frame = max(0, state.current_frame - jump)
        cap.set(cv2.CAP_PROP_POS_FRAMES, state.current_frame)
        return None

    # Arrow right — jump 5 seconds forward
    if key in (KEY_RIGHT, KEY_RIGHT_LIN):
        jump = int(5 * state.fps)
        state.current_frame = min(state.total_frames - 1, state.current_frame + jump)
        cap.set(cv2.CAP_PROP_POS_FRAMES, state.current_frame)
        return None

    # Speed up
    if ch == "w":
        if state.speed_idx < len(SPEED_LEVELS) - 1:
            state.speed_idx += 1
        return None

    # Slow down
    if ch == "e":
        if state.speed_idx > 0:
            state.speed_idx -= 1
        return None

    # Jump 5 seconds back
    if ch == "[":
        jump = int(5 * state.fps)
        state.current_frame = max(0, state.current_frame - jump)
        cap.set(cv2.CAP_PROP_POS_FRAMES, state.current_frame)
        return None

    # Jump 5 seconds forward
    if ch == "]":
        jump = int(5 * state.fps)
        state.current_frame = min(state.total_frames - 1, state.current_frame + jump)
        cap.set(cv2.CAP_PROP_POS_FRAMES, state.current_frame)
        return None

    # Shot marking keys
    if ch == "f":
        start_mark(state, "forehand")
        return None
    if ch == "b":
        start_mark(state, "backhand")
        return None
    if ch == "s":
        start_mark(state, "serve")
        return None
    if ch == "n":
        start_mark(state, "neutral")
        return None

    # Enter — end any active mark
    if key == KEY_ENTER:
        end_mark(state)
        return None

    # Escape — cancel active mark
    if key == KEY_ESCAPE:
        cancel_mark(state)
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

    return None


# ── Mouse handling ───────────────────────────────────────────


EDGE_GRAB_PX = 8  # pixel threshold for grabbing a mark edge


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


def _mark_at_x(state: AppState, x: int) -> Optional[int]:
    """Return the index of the mark under pixel x, or None."""
    if state.tl_w <= 0 or state.total_frames <= 1:
        return None
    frame = _frame_at_x(state, x)
    # Search in reverse so later (top-drawn) marks get priority
    for i in range(len(state.marks) - 1, -1, -1):
        m = state.marks[i]
        if m.start_frame <= frame <= m.end_frame:
            return i
    return None


def _edge_at_x(state: AppState, x: int) -> Optional[str]:
    """If there's a selected mark and x is near one of its edges, return 'start' or 'end'."""
    idx = state.selected_mark_idx
    if idx is None or idx >= len(state.marks):
        return None
    m = state.marks[idx]
    sx = _x_at_frame(state, m.start_frame)
    ex = _x_at_frame(state, m.end_frame)
    # Prefer whichever edge is closer, if within threshold
    d_start = abs(x - sx)
    d_end = abs(x - ex)
    if d_start <= EDGE_GRAB_PX and d_start <= d_end:
        return "start"
    if d_end <= EDGE_GRAB_PX:
        return "end"
    return None


def make_mouse_callback(state: AppState, cap):
    """Return a mouse callback for timeline scrub, mark selection, and edge resize."""
    drag_mode = [None]  # 'scrub', 'edge', or None

    def on_mouse(event, x, y, flags, param):
        in_bar = (state.tl_w > 0
                  and state.tl_x <= x <= state.tl_x + state.tl_w
                  and state.tl_y - 6 <= y <= state.tl_y + state.tl_h + 6)

        if event == cv2.EVENT_LBUTTONDOWN and in_bar:
            # First: check if grabbing an edge of the selected mark
            edge = _edge_at_x(state, x)
            if edge is not None:
                state.dragging_edge = edge
                drag_mode[0] = "edge"
                return

            # Second: check if clicking on a mark body
            hit = _mark_at_x(state, x)
            if hit is not None:
                if state.selected_mark_idx == hit:
                    state.selected_mark_idx = None
                else:
                    state.selected_mark_idx = hit
                state.dragging_edge = None
                drag_mode[0] = None
            else:
                # Scrub to position
                state.selected_mark_idx = None
                state.dragging_edge = None
                target = _frame_at_x(state, x)
                state.current_frame = target
                state.playing = False
                cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                drag_mode[0] = "scrub"

        elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):
            if drag_mode[0] == "edge" and state.dragging_edge and in_bar:
                idx = state.selected_mark_idx
                if idx is not None and idx < len(state.marks):
                    m = state.marks[idx]
                    new_frame = _frame_at_x(state, x)
                    if state.dragging_edge == "start":
                        # Don't let start go past end - 1
                        new_start = min(new_frame, m.end_frame - 1)
                        new_start = max(0, new_start)
                        state.marks[idx] = Mark(m.shot_type, new_start, m.end_frame)
                        # Scrub video to the edge being dragged
                        state.current_frame = new_start
                        state.playing = False
                        cap.set(cv2.CAP_PROP_POS_FRAMES, new_start)
                    else:
                        # Don't let end go before start + 1
                        new_end = max(new_frame, m.start_frame + 1)
                        new_end = min(state.total_frames - 1, new_end)
                        state.marks[idx] = Mark(m.shot_type, m.start_frame, new_end)
                        # Scrub video to the edge being dragged
                        state.current_frame = new_end
                        state.playing = False
                        cap.set(cv2.CAP_PROP_POS_FRAMES, new_end)
            elif drag_mode[0] == "scrub" and in_bar:
                target = _frame_at_x(state, x)
                state.current_frame = target
                state.playing = False
                cap.set(cv2.CAP_PROP_POS_FRAMES, target)

        elif event == cv2.EVENT_LBUTTONUP:
            if drag_mode[0] == "edge" and state.dragging_edge:
                idx = state.selected_mark_idx
                if idx is not None and idx < len(state.marks):
                    m = state.marks[idx]
                    state.set_message(
                        f"Resized {m.shot_type} to {m.start_frame}-{m.end_frame}"
                    )
                state.dragging_edge = None
            drag_mode[0] = None

        elif event == cv2.EVENT_LBUTTONDOWN and not in_bar:
            state.selected_mark_idx = None
            state.dragging_edge = None

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
            if m.start_frame >= 0 and m.end_frame < total_frames
        ]

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
            raw_key = cv2.waitKeyEx(30)

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
            if state.speed >= 2.0:
                # Frame skipping for high speed
                skip = int(state.speed)
                next_frame = min(state.current_frame + skip, total_frames - 1)
                if next_frame != state.current_frame:
                    state.current_frame = next_frame
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
                next_frame = state.current_frame + 1
                if next_frame < total_frames:
                    state.current_frame = next_frame
                    # Sequential read is fastest when frames are consecutive
                    if last_read_frame == state.current_frame - 1:
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
