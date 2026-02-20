#!/usr/bin/env python3
"""Real-time pipeline status monitor for local GPU processing.

Shows:
- Current video being processed
- Pipeline stage and progress
- GPU machine status
- Recent activity log
- Processing times

Usage:
    python scripts/pipeline_status.py           # live dashboard
    python scripts/pipeline_status.py --tail    # tail log with formatting
    python scripts/pipeline_status.py --summary # quick status summary
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import PROJECT_ROOT, AUTO_PIPELINE

# ANSI colors
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
GRAY = "\033[90m"
WHITE = "\033[97m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"
CLEAR_LINE = "\033[K"
CURSOR_UP = "\033[A"

# Pipeline stages in order
STAGES = [
    ("download", "Downloading from iCloud"),
    ("transfer_raw", "Transferring to GPU"),
    ("preprocess", "Preprocessing (NVENC)"),
    ("prescan", "Pre-scanning for dead sections"),
    ("poses", "Extracting poses"),
    ("detect", "Detecting shots"),
    ("clips", "Extracting clips"),
    ("highlights", "Compiling highlights"),
    ("slowmo", "Creating slow-mo"),
    ("combined", "Combining videos"),
    ("transfer_back", "Transferring to Mac"),
    ("upload", "Uploading to YouTube"),
]

STAGE_PATTERNS = {
    "download": [r"\[STAGE:download\]", r"Downloading", r"Download.*from iCloud"],
    "transfer_raw": [r"\[STAGE:transfer_raw\]", r"SCP to", r"Transfer.*to.*windows|tmassena"],
    "preprocess": [r"\[STAGE:preprocess\]", r"preprocess_nvenc", r"Converting:.*%", r"NVENC"],
    "prescan": [r"\[STAGE:prescan\]", r"Pre-scan", r"dead section"],
    "poses": [r"\[STAGE:poses\]", r"extract_poses", r"Processing:.*frames", r"detected.*fps"],
    "detect": [r"\[STAGE:detect\]", r"detect_shots", r"Shot detection", r"segments"],
    "clips": [r"\[STAGE:clips\]", r"extract_clips", r"Extracting.*clips", r"\[\d+/\d+\].*mp4"],
    "highlights": [r"\[STAGE:highlights\]", r"highlight", r"Compiling"],
    "slowmo": [r"\[STAGE:slowmo\]", r"[Ss]low-?mo", r"setpts"],
    "combined": [r"\[STAGE:combined\]", r"[Cc]ombined", r"concatenat"],
    "transfer_back": [r"\[STAGE:transfer_back\]", r"SCP from", r"Transfer.*from"],
    "upload": [r"\[STAGE:upload\]", r"[Uu]pload.*YouTube", r"youtube_url"],
}


def check_machine_status(host, timeout=5):
    """Check if a GPU machine is reachable and get basic info."""
    try:
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=3", "-o", "BatchMode=yes", host,
             "echo OK && nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null || echo 'NO_GPU'"],
            capture_output=True, text=True, timeout=timeout
        )
        if result.returncode == 0 and "OK" in result.stdout:
            lines = result.stdout.strip().split('\n')
            if len(lines) >= 2 and lines[1] != "NO_GPU":
                parts = lines[1].split(',')
                if len(parts) >= 3:
                    gpu_util = parts[0].strip()
                    mem_used = int(parts[1].strip())
                    mem_total = int(parts[2].strip())
                    return {
                        "status": "online",
                        "gpu_util": f"{gpu_util}%",
                        "mem": f"{mem_used}/{mem_total}MB",
                    }
            return {"status": "online", "gpu_util": "N/A", "mem": "N/A"}
        return {"status": "offline"}
    except Exception:
        return {"status": "unreachable"}


def get_pipeline_state():
    """Read current pipeline state from state file."""
    state_file = AUTO_PIPELINE.get("state_file", os.path.join(PROJECT_ROOT, "pipeline_state.json"))
    if os.path.exists(state_file):
        try:
            with open(state_file) as f:
                return json.load(f)
        except:
            pass
    return {}


def parse_log_line(line):
    """Parse a log line and extract structured info."""
    # Extract timestamp
    ts_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
    timestamp = ts_match.group(1) if ts_match else None

    # Extract video name
    video_match = re.search(r'(IMG_\d+|[A-Za-z0-9_-]+)\.(mov|mp4|MOV|MP4)', line)
    video = video_match.group(0) if video_match else None

    # Check for explicit [STAGE:xxx] or [STAGE:xxx:COMPLETE/FAILED] markers
    stage_marker = re.search(r'\[STAGE:(\w+)(?::(\w+))?\]', line)
    if stage_marker:
        stage = stage_marker.group(1)
        status = stage_marker.group(2)  # COMPLETE, FAILED, or None
        is_complete = status == "COMPLETE"
        is_error = status == "FAILED"
    else:
        # Fall back to pattern matching
        stage = None
        for stage_name, patterns in STAGE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    stage = stage_name
                    break
            if stage:
                break

        # Detect errors
        is_error = bool(re.search(r'\b(error|failed|exception)\b', line, re.IGNORECASE))

        # Detect completion
        is_complete = bool(re.search(r'\b(complete|finished|success|done)\b', line, re.IGNORECASE))

    # Extract progress if present
    progress = None
    prog_match = re.search(r'(\d+(?:\.\d+)?)\s*%', line)
    if prog_match:
        progress = float(prog_match.group(1))

    # Extract frame progress
    frame_match = re.search(r'(\d+)/(\d+)', line)
    if frame_match and ("frame" in line.lower() or "Processing" in line):
        current, total = int(frame_match.group(1)), int(frame_match.group(2))
        progress = (current / total * 100) if total > 0 else 0

    return {
        "timestamp": timestamp,
        "video": video,
        "stage": stage,
        "progress": progress,
        "is_error": is_error,
        "is_complete": is_complete,
        "raw": line.strip(),
    }


def tail_log(log_path, n_lines=50):
    """Read last N lines from log file."""
    if not os.path.exists(log_path):
        return []
    try:
        result = subprocess.run(
            ["tail", "-n", str(n_lines), log_path],
            capture_output=True, text=True
        )
        return result.stdout.strip().split('\n')
    except:
        return []


def format_stage_line(stage_name, stage_desc, is_current=False, is_done=False, progress=None):
    """Format a single stage line for display."""
    if is_done:
        icon = f"{GREEN}✓{RESET}"
        color = GREEN
    elif is_current:
        icon = f"{YELLOW}▶{RESET}"
        color = YELLOW
    else:
        icon = f"{GRAY}○{RESET}"
        color = GRAY

    line = f"  {icon} {color}{stage_desc}{RESET}"

    if progress is not None and is_current:
        bar_width = 20
        filled = int(progress / 100 * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)
        line += f" {CYAN}[{bar}]{RESET} {progress:.0f}%"

    return line


def format_log_entry(parsed):
    """Format a parsed log entry for display."""
    ts = parsed.get("timestamp", "")[:8] if parsed.get("timestamp") else ""

    if parsed["is_error"]:
        icon = f"{RED}✗{RESET}"
        color = RED
    elif parsed["is_complete"]:
        icon = f"{GREEN}✓{RESET}"
        color = GREEN
    elif parsed["stage"]:
        icon = f"{YELLOW}▸{RESET}"
        color = WHITE
    else:
        icon = f"{GRAY}·{RESET}"
        color = GRAY

    # Truncate long lines
    msg = parsed["raw"]
    if len(msg) > 80:
        msg = msg[:77] + "..."

    return f"{GRAY}{ts}{RESET} {icon} {color}{msg}{RESET}"


def display_dashboard(machines, log_path):
    """Display live dashboard with auto-refresh."""
    print("\033[2J\033[H", end="")  # Clear screen

    while True:
        # Move cursor to top
        print("\033[H", end="")

        # Header
        print(f"{BOLD}{CYAN}{'═' * 70}{RESET}")
        print(f"{BOLD}{CYAN}  🎾 TENNIS PIPELINE MONITOR{RESET}")
        print(f"{BOLD}{CYAN}{'═' * 70}{RESET}")
        print()

        # Machine status
        print(f"{BOLD}GPU Machines:{RESET}")
        for machine in machines:
            host = machine["host"]
            status = check_machine_status(host)
            if status["status"] == "online":
                gpu_info = f"GPU: {status['gpu_util']}, Mem: {status['mem']}"
                print(f"  {GREEN}●{RESET} {host}: {GREEN}online{RESET} - {gpu_info}")
            elif status["status"] == "offline":
                print(f"  {RED}○{RESET} {host}: {RED}offline{RESET}")
            else:
                print(f"  {YELLOW}○{RESET} {host}: {YELLOW}unreachable{RESET}")
        print()

        # Pipeline state
        state = get_pipeline_state()
        today = datetime.now().strftime("%Y-%m-%d")
        daily_count = state.get("daily_counts", {}).get(today, 0)
        total_processed = len(state.get("processed", {}))

        print(f"{BOLD}Pipeline State:{RESET}")
        print(f"  Today: {daily_count} video(s) processed")
        print(f"  Total: {total_processed} video(s) all time")
        print()

        # Recent log activity
        print(f"{BOLD}Recent Activity:{RESET}")
        print(f"{GRAY}{'─' * 70}{RESET}")

        lines = tail_log(log_path, 20)
        current_video = None
        current_stage = None
        current_progress = None
        completed_stages = set()

        # Parse recent lines to determine current state
        for line in lines:
            parsed = parse_log_line(line)
            if parsed["video"]:
                current_video = parsed["video"]
            if parsed["stage"]:
                if parsed["is_complete"]:
                    completed_stages.add(parsed["stage"])
                else:
                    current_stage = parsed["stage"]
            if parsed["progress"] is not None:
                current_progress = parsed["progress"]

        # Show current video and stages
        if current_video:
            print(f"\n  {BOLD}Current: {CYAN}{current_video}{RESET}")
            print()
            for stage_name, stage_desc in STAGES:
                is_done = stage_name in completed_stages
                is_current = stage_name == current_stage
                progress = current_progress if is_current else None
                print(format_stage_line(stage_name, stage_desc, is_current, is_done, progress))
            print()

        # Show last few log entries
        print(f"\n{BOLD}Log:{RESET}")
        for line in lines[-10:]:
            if line.strip():
                parsed = parse_log_line(line)
                print(format_log_entry(parsed))

        print()
        print(f"{GRAY}Last updated: {datetime.now().strftime('%H:%M:%S')} | Refresh: 3s | Ctrl+C to exit{RESET}")
        print(f"{CLEAR_LINE}", end="")

        try:
            time.sleep(3)
        except KeyboardInterrupt:
            print(f"\n{GRAY}Monitor stopped.{RESET}")
            break


def display_summary(machines, log_path):
    """Display quick summary of pipeline status."""
    print(f"\n{BOLD}Pipeline Status Summary{RESET}")
    print(f"{'─' * 40}")

    # Machine status
    all_online = True
    for machine in machines:
        host = machine["host"]
        status = check_machine_status(host)
        icon = f"{GREEN}●{RESET}" if status["status"] == "online" else f"{RED}○{RESET}"
        print(f"  {icon} {host}: {status['status']}")
        if status["status"] != "online":
            all_online = False

    print()

    # State
    state = get_pipeline_state()
    today = datetime.now().strftime("%Y-%m-%d")
    daily_count = state.get("daily_counts", {}).get(today, 0)
    print(f"  Videos today: {daily_count}")
    print(f"  Total processed: {len(state.get('processed', {}))}")

    # Recent activity
    lines = tail_log(log_path, 5)
    if lines and lines[0]:
        print(f"\n{BOLD}Last Activity:{RESET}")
        for line in lines[-3:]:
            if line.strip():
                parsed = parse_log_line(line)
                ts = parsed.get("timestamp", "")[:19] if parsed.get("timestamp") else ""
                print(f"  {GRAY}{ts}{RESET} {parsed['raw'][:50]}...")

    print()


def tail_formatted(log_path):
    """Tail log file with formatted output."""
    print(f"{BOLD}Tailing {log_path}...{RESET}")
    print(f"{GRAY}{'─' * 70}{RESET}")

    try:
        process = subprocess.Popen(
            ["tail", "-f", log_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        for line in process.stdout:
            if line.strip():
                parsed = parse_log_line(line)
                print(format_log_entry(parsed))

    except KeyboardInterrupt:
        print(f"\n{GRAY}Stopped.{RESET}")


def main():
    parser = argparse.ArgumentParser(description="Pipeline status monitor")
    parser.add_argument("--tail", action="store_true",
                        help="Tail log with formatting")
    parser.add_argument("--summary", action="store_true",
                        help="Quick status summary")
    parser.add_argument("--log", default=os.path.join(PROJECT_ROOT, "pipeline.log"),
                        help="Log file path")
    args = parser.parse_args()

    machines = AUTO_PIPELINE.get("gpu_machines", [])

    if args.tail:
        tail_formatted(args.log)
    elif args.summary:
        display_summary(machines, args.log)
    else:
        display_dashboard(machines, args.log)


if __name__ == "__main__":
    main()
