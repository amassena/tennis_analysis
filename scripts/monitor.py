#!/usr/bin/env python3
"""Pipeline monitor - shows live status from cloud coordinator.

Usage:
    python3 scripts/monitor.py              # Live dashboard (auto-refresh)
    python3 scripts/monitor.py --logs       # Stream logs from Hetzner
    python3 scripts/monitor.py --status     # Quick status check
    python3 scripts/monitor.py --url URL    # Use custom coordinator URL
"""

import argparse
import json
import subprocess
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime

# Default coordinator URL (Hetzner)
DEFAULT_URL = "http://5.78.96.237:8080"
HETZNER_SSH = "root@5.78.96.237"

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
CLEAR_SCREEN = "\033[2J\033[H"
CLEAR_LINE = "\033[K"

# Processing stages in order
STAGES = [
    "queued", "downloading", "preprocessing", "thumbnail", "prescan",
    "poses", "detection", "clips", "slowmo", "combined", "uploading", "done"
]


def api_get(url: str, endpoint: str) -> dict:
    """Make GET request to coordinator API."""
    try:
        with urllib.request.urlopen(f"{url}{endpoint}", timeout=10) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        return {"error": str(e)}


def time_ago(iso_string: str) -> str:
    """Convert ISO timestamp to relative time."""
    if not iso_string:
        return ""
    try:
        dt = datetime.fromisoformat(iso_string.replace("Z", "+00:00"))
        delta = datetime.now(dt.tzinfo) - dt
        seconds = int(delta.total_seconds())
        if seconds < 60:
            return f"{seconds}s ago"
        if seconds < 3600:
            return f"{seconds // 60}m ago"
        if seconds < 86400:
            return f"{seconds // 3600}h ago"
        return f"{seconds // 86400}d ago"
    except:
        return ""


def stage_progress_bar(stage: str, progress: float = None, width: int = 30) -> str:
    """Create a progress bar showing overall pipeline progress."""
    if not stage:
        return f"{GRAY}[{'░' * width}]{RESET} 0%"

    try:
        idx = STAGES.index(stage)
    except ValueError:
        idx = 0

    stage_pct = idx / len(STAGES) * 100
    in_stage = (progress or 0) / 100 * (100 / len(STAGES))
    total_pct = min(stage_pct + in_stage, 100)

    filled = int(total_pct / 100 * width)
    bar = "█" * filled + "░" * (width - filled)

    return f"{CYAN}[{bar}]{RESET} {total_pct:.0f}%"


def format_job(job: dict, verbose: bool = False) -> str:
    """Format a job for display."""
    status = job.get("status", "unknown")
    filename = job.get("filename", "unknown")
    stage = job.get("current_stage")
    progress = job.get("stage_progress")
    message = job.get("stage_message", "")
    worker = job.get("claimed_by", "")
    pod = job.get("pod_id", "")
    youtube = job.get("youtube_url", "")
    error = job.get("error_message", "")

    # Status colors
    status_colors = {
        "pending": GRAY,
        "claimed": YELLOW,
        "processing": GREEN,
        "completed": CYAN,
        "failed": RED,
    }
    color = status_colors.get(status, GRAY)

    # Status icon
    status_icons = {
        "pending": "○",
        "claimed": "◐",
        "processing": "●",
        "completed": "✓",
        "failed": "✗",
    }
    icon = status_icons.get(status, "?")

    lines = [f"{color}{icon}{RESET} {WHITE}{filename}{RESET} [{color}{status}{RESET}]"]

    if status in ("claimed", "processing") and stage:
        lines.append(f"    {stage_progress_bar(stage, progress)}")
        if stage:
            stage_display = stage.upper()
            pct = f" ({progress:.0f}%)" if progress else ""
            lines.append(f"    Stage: {YELLOW}{stage_display}{pct}{RESET}")
        if message:
            lines.append(f"    {GRAY}{message}{RESET}")
        if worker:
            lines.append(f"    Worker: {GREEN}{worker}{RESET}")
        if pod:
            pod_status = job.get("pod_status", "")
            lines.append(f"    Pod: {YELLOW}{pod}{RESET} ({pod_status})")

    if youtube:
        lines.append(f"    YouTube: {CYAN}{youtube}{RESET}")

    if error:
        lines.append(f"    {RED}Error: {error}{RESET}")

    if verbose:
        updated = job.get("stage_updated_at") or job.get("claimed_at") or job.get("created_at")
        if updated:
            lines.append(f"    {DIM}Updated: {time_ago(updated)}{RESET}")

    return "\n".join(lines)


def display_dashboard(url: str):
    """Display live dashboard with auto-refresh."""
    print(CLEAR_SCREEN, end="")

    while True:
        print("\033[H", end="")  # Move to top

        print(f"{BOLD}{CYAN}{'═' * 60}{RESET}")
        print(f"{BOLD}{CYAN}  🎾 TENNIS PIPELINE - CLOUD MONITOR{RESET}")
        print(f"{BOLD}{CYAN}{'═' * 60}{RESET}")
        print(f"{GRAY}Coordinator: {url}{RESET}")
        print()

        # Get stats
        stats = api_get(url, "/stats")
        if "error" in stats:
            print(f"{RED}Cannot connect to coordinator: {stats['error']}{RESET}")
            print(f"\n{GRAY}Retrying in 5s...{RESET}")
            time.sleep(5)
            continue

        pending = stats.get("pending", 0)
        processing = stats.get("processing", 0) + stats.get("claimed", 0)
        completed = stats.get("completed", 0)
        failed = stats.get("failed", 0)

        print(f"{BOLD}Queue Status:{RESET}")
        print(f"  {YELLOW}○{RESET} Pending: {pending}    "
              f"{GREEN}●{RESET} Processing: {processing}    "
              f"{CYAN}✓{RESET} Completed: {completed}    "
              f"{RED}✗{RESET} Failed: {failed}")
        print()

        # Get active jobs
        active = api_get(url, "/jobs/active")
        active_jobs = active.get("jobs", [])

        print(f"{BOLD}Active Jobs:{RESET}")
        if active_jobs:
            for job in active_jobs:
                print(format_job(job, verbose=True))
                print()
        else:
            print(f"  {GRAY}No active jobs{RESET}")
        print()

        # Get recent completed/failed
        recent = api_get(url, "/jobs?limit=5")
        recent_jobs = [j for j in recent.get("jobs", [])
                       if j.get("status") in ("completed", "failed")]

        print(f"{BOLD}Recent:{RESET}")
        if recent_jobs:
            for job in recent_jobs[:3]:
                print(format_job(job))
        else:
            print(f"  {GRAY}No recent jobs{RESET}")
        print()

        print(f"{GRAY}{'─' * 60}{RESET}")
        print(f"{GRAY}Auto-refresh: 5s | Web: {url}/dash | Ctrl+C to exit{RESET}")
        print(CLEAR_LINE, end="")

        try:
            time.sleep(5)
        except KeyboardInterrupt:
            print(f"\n{GRAY}Monitor stopped.{RESET}")
            break


def display_status(url: str):
    """Display quick status summary."""
    print(f"\n{BOLD}Tennis Pipeline Status{RESET}")
    print(f"{'─' * 40}")

    # Health check
    health = api_get(url, "/health")
    if "error" in health:
        print(f"{RED}✗{RESET} Coordinator: offline ({health['error']})")
        return
    print(f"{GREEN}✓{RESET} Coordinator: online")

    # Stats
    stats = api_get(url, "/stats")
    print(f"\n{BOLD}Queue:{RESET}")
    print(f"  Pending: {stats.get('pending', 0)}")
    print(f"  Processing: {stats.get('processing', 0) + stats.get('claimed', 0)}")
    print(f"  Completed: {stats.get('completed', 0)}")
    print(f"  Failed: {stats.get('failed', 0)}")

    # Active jobs
    active = api_get(url, "/jobs/active")
    active_jobs = active.get("jobs", [])
    if active_jobs:
        print(f"\n{BOLD}Active:{RESET}")
        for job in active_jobs:
            stage = job.get("current_stage", "unknown")
            progress = job.get("stage_progress")
            pct = f" ({progress:.0f}%)" if progress else ""
            print(f"  {job['filename']}: {stage}{pct}")
    print()


def stream_logs():
    """Stream logs from Hetzner via SSH."""
    print(f"\n{BOLD}{CYAN}Streaming logs from Hetzner...{RESET}")
    print(f"{GRAY}Press Ctrl+C to stop{RESET}\n")

    cmd = [
        "ssh", HETZNER_SSH,
        "journalctl -u tennis-coordinator -u tennis-worker -u tennis-watcher -f --no-pager"
    ]

    try:
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1
        )

        for line in process.stdout:
            line = line.strip()
            if not line:
                continue

            # Color-code based on content
            if "error" in line.lower() or "failed" in line.lower():
                print(f"{RED}{line}{RESET}")
            elif "complete" in line.lower() or "success" in line.lower():
                print(f"{GREEN}{line}{RESET}")
            elif "processing" in line.lower() or "stage" in line.lower():
                print(f"{YELLOW}{line}{RESET}")
            else:
                print(line)

    except KeyboardInterrupt:
        print(f"\n{GRAY}Log streaming stopped.{RESET}")
    except Exception as e:
        print(f"{RED}Error: {e}{RESET}")


def main():
    parser = argparse.ArgumentParser(description="Tennis Pipeline Cloud Monitor")
    parser.add_argument("--url", default=DEFAULT_URL,
                        help=f"Coordinator URL (default: {DEFAULT_URL})")
    parser.add_argument("--logs", action="store_true",
                        help="Stream logs from Hetzner instead of dashboard")
    parser.add_argument("--status", action="store_true",
                        help="Quick status check and exit")
    args = parser.parse_args()

    if args.logs:
        stream_logs()
    elif args.status:
        display_status(args.url)
    else:
        display_dashboard(args.url)


if __name__ == "__main__":
    main()
