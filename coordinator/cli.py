#!/usr/bin/env python3
"""CLI for running the distributed coordinator system.

Usage:
    # Start the coordinator API server
    python -m coordinator.cli api

    # Start the iCloud watcher (scans albums, adds jobs)
    python -m coordinator.cli watcher

    # Start the WoL agent (wakes machines when jobs pending)
    python -m coordinator.cli wol

    # Start all components together
    python -m coordinator.cli all

    # Check system status
    python -m coordinator.cli status
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import COORDINATOR


def get_coordinator_url():
    """Get coordinator URL from settings."""
    host = COORDINATOR.get("host", "0.0.0.0")
    port = COORDINATOR.get("port", 8080)
    # Use localhost for client access
    return f"http://localhost:{port}"


def run_api():
    """Start the FastAPI coordinator server."""
    import uvicorn
    from coordinator.api import app

    host = COORDINATOR.get("host", "0.0.0.0")
    port = COORDINATOR.get("port", 8080)

    print(f"Starting coordinator API on {host}:{port}")
    uvicorn.run(app, host=host, port=port)


def run_watcher():
    """Start the iCloud watcher."""
    from coordinator.icloud_watcher import watcher_loop

    coordinator_url = get_coordinator_url()
    poll_interval = COORDINATOR.get("icloud_poll_interval", 300)

    watcher_loop(coordinator_url, poll_interval)


def run_wol():
    """Start the Wake-on-LAN agent."""
    from coordinator.wol_agent import wol_agent_loop
    from scripts.wake_machines import MACHINES

    coordinator_url = get_coordinator_url()
    check_interval = COORDINATOR.get("wol_check_interval", 60)

    wol_agent_loop(coordinator_url, check_interval, list(MACHINES.keys()))


def run_all():
    """Start all components in parallel."""
    import multiprocessing

    processes = [
        multiprocessing.Process(target=run_api, name="api"),
        multiprocessing.Process(target=run_watcher, name="watcher"),
        multiprocessing.Process(target=run_wol, name="wol"),
    ]

    print("Starting all coordinator components...")

    for p in processes:
        p.start()
        print(f"  Started {p.name} (PID {p.pid})")

    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("\nShutting down...")
        for p in processes:
            p.terminate()


def check_status():
    """Check coordinator and worker status."""
    url = get_coordinator_url()

    print(f"Coordinator: {url}")
    print()

    # Check API health
    try:
        req = urllib.request.Request(f"{url}/health")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            print(f"API Status: OK ({data.get('timestamp', '')})")
    except Exception as e:
        print(f"API Status: OFFLINE ({e})")
        return

    # Get stats
    try:
        req = urllib.request.Request(f"{url}/stats")
        with urllib.request.urlopen(req, timeout=5) as resp:
            stats = json.loads(resp.read())
            print()
            print("Job Statistics:")
            print(f"  Pending:    {stats.get('pending', 0)}")
            print(f"  Claimed:    {stats.get('claimed', 0)}")
            print(f"  Processing: {stats.get('processing', 0)}")
            print(f"  Completed:  {stats.get('completed', 0)}")
            print(f"  Failed:     {stats.get('failed', 0)}")
            print(f"  Total:      {stats.get('total', 0)}")
    except Exception as e:
        print(f"Error getting stats: {e}")

    # Check GPU machines
    print()
    print("GPU Machines:")

    from config.settings import AUTO_PIPELINE

    for machine in AUTO_PIPELINE.get("gpu_machines", []):
        host = machine["host"]
        try:
            result = subprocess.run(
                ["ssh", "-o", "ConnectTimeout=3", "-o", "BatchMode=yes", host, "echo OK"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            status = "ONLINE" if result.returncode == 0 else "OFFLINE"
        except Exception:
            status = "UNREACHABLE"

        print(f"  {host}: {status}")


def main():
    parser = argparse.ArgumentParser(
        description="Distributed coordinator CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Components:
  api      - HTTP API server (FastAPI)
  watcher  - iCloud album scanner
  wol      - Wake-on-LAN agent
  all      - Run all components
  status   - Check system status
        """,
    )

    parser.add_argument(
        "command",
        choices=["api", "watcher", "wol", "all", "status"],
        help="Component to run",
    )

    args = parser.parse_args()

    commands = {
        "api": run_api,
        "watcher": run_watcher,
        "wol": run_wol,
        "all": run_all,
        "status": check_status,
    }

    commands[args.command]()


if __name__ == "__main__":
    main()
