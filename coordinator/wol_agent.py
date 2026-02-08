"""Wake-on-LAN agent for waking GPU machines.

Runs on a local always-on machine (Pi or Mac) that can send WoL packets.
Monitors the job queue and wakes machines when work is available.
"""

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.wake_machines import MACHINES, send_wol_packet


# Configuration
COORDINATOR_URL = os.environ.get("COORDINATOR_URL", "http://localhost:8080")
CHECK_INTERVAL = int(os.environ.get("WOL_CHECK_INTERVAL", "60"))  # seconds
WAKE_COOLDOWN = 300  # Don't re-wake a machine for 5 minutes


def log(msg: str, level: str = "INFO"):
    """Log with timestamp."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [{level}] {msg}", flush=True)


def api_request(method: str, endpoint: str) -> dict:
    """Make HTTP request to coordinator API."""
    url = f"{COORDINATOR_URL}{endpoint}"
    req = urllib.request.Request(url, method=method)

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        log(f"API error: {e}", "ERROR")
        raise


def is_machine_online(hostname: str, timeout: int = 5) -> bool:
    """Check if a machine responds to SSH."""
    try:
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes", hostname, "echo OK"],
            capture_output=True,
            text=True,
            timeout=timeout + 2,
        )
        return result.returncode == 0
    except Exception:
        return False


def wake_machine(name: str) -> bool:
    """Wake a machine using WoL."""
    config = MACHINES.get(name)
    if not config:
        log(f"Unknown machine: {name}", "WARN")
        return False

    mac = config.get("mac")
    if not mac:
        log(f"No MAC address for {name}", "WARN")
        return False

    log(f"Sending WoL to {name} ({mac})")
    try:
        send_wol_packet(mac)
        return True
    except Exception as e:
        log(f"WoL failed: {e}", "ERROR")
        return False


def wol_agent_loop(coordinator_url: str, check_interval: int, machines: list, once: bool = False):
    """Main WoL agent loop."""
    log("Starting WoL agent")
    log(f"Coordinator: {coordinator_url}")
    log(f"Machines: {machines}")
    log(f"Check interval: {check_interval}s")

    # Update global for api_request function
    global COORDINATOR_URL
    COORDINATOR_URL = coordinator_url

    last_wake = {m: 0 for m in machines}

    while True:
        try:
            # Check for pending jobs
            pending = api_request("GET", "/jobs/pending")
            pending_count = pending.get("count", 0)

            if pending_count > 0:
                log(f"{pending_count} pending jobs")

                # Check each machine
                for machine in machines:
                    # Check cooldown
                    if time.time() - last_wake[machine] < WAKE_COOLDOWN:
                        continue

                    # Check if online
                    if is_machine_online(machine):
                        log(f"{machine} is online")
                        continue

                    # Wake it
                    if wake_machine(machine):
                        last_wake[machine] = time.time()
                        log(f"Sent WoL to {machine}")

            else:
                log("No pending jobs", "DEBUG")

        except Exception as e:
            log(f"Error in WoL agent loop: {e}", "ERROR")

        if once:
            break

        time.sleep(check_interval)


def main():
    parser = argparse.ArgumentParser(description="Wake-on-LAN agent")
    parser.add_argument(
        "--coordinator",
        default=None,
        help="Coordinator API URL",
    )
    parser.add_argument(
        "--machines",
        nargs="+",
        default=None,
        help="Machines to wake (default: all)",
    )
    parser.add_argument(
        "--check-interval",
        type=int,
        default=None,
        help="Seconds between checks",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Check once and exit",
    )
    args = parser.parse_args()

    coordinator_url = args.coordinator or COORDINATOR_URL
    check_interval = args.check_interval or CHECK_INTERVAL
    machines = args.machines or list(MACHINES.keys())

    wol_agent_loop(coordinator_url, check_interval, machines, once=args.once)


if __name__ == "__main__":
    main()
