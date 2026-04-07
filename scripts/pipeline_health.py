#!/usr/bin/env python3
"""Pipeline health check and diagnostics.

Quick checks:
    python scripts/pipeline_health.py              # Full health check
    python scripts/pipeline_health.py --quick      # Service status only

Test modes:
    python scripts/pipeline_health.py --test-job   # Submit a test job and monitor it
    python scripts/pipeline_health.py --test-auth  # Test iCloud auth on all machines
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

COORDINATOR_URL = "http://5.78.96.237"
HETZNER_HOST = "devserver"
GPU_MACHINES = [
    {"name": "tmassena", "ssh": "tmassena", "role": "primary"},
    {"name": "Andrew-PC", "ssh": "windows", "role": "fallback"},
]
PROJECT_PATH_WIN = r"C:\Users\amass\tennis_analysis"

# ANSI colors
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def ok(msg):
    print(f"  {GREEN}✓{RESET} {msg}")


def fail(msg):
    print(f"  {RED}✗{RESET} {msg}")


def warn(msg):
    print(f"  {YELLOW}!{RESET} {msg}")


def header(msg):
    print(f"\n{BOLD}{CYAN}── {msg} ──{RESET}")


def ssh_cmd(host, cmd, timeout=15):
    """Run SSH command, return (success, stdout)."""
    try:
        result = subprocess.run(
            ["ssh", host, cmd],
            capture_output=True, text=True, timeout=timeout,
        )
        return result.returncode == 0, result.stdout.strip()
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as e:
        return False, str(e)


def http_get(url, timeout=10):
    """HTTP GET, return parsed JSON or None."""
    import urllib.request
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except Exception:
        return None


def check_coordinator():
    """Check coordinator API health."""
    header("Coordinator (Hetzner)")

    # Health endpoint
    health = http_get(f"{COORDINATOR_URL}/health")
    if health:
        ok(f"API responding at {COORDINATOR_URL}")
    else:
        fail(f"API not responding at {COORDINATOR_URL}")
        return False

    # Stats
    stats = http_get(f"{COORDINATOR_URL}/stats")
    if stats:
        ok(f"Jobs: {stats['pending']} pending, {stats.get('processing', 0)} processing, "
           f"{stats['completed']} completed, {stats.get('failed', 0)} failed")
    return True


def check_watcher():
    """Check iCloud watcher service."""
    header("iCloud Watcher (Hetzner)")

    success, output = ssh_cmd(HETZNER_HOST, "systemctl is-active tennis-watcher")
    if success and output == "active":
        ok("tennis-watcher.service is active")
    else:
        fail(f"tennis-watcher.service is {output}")
        # Check if disabled
        _, enabled = ssh_cmd(HETZNER_HOST, "systemctl is-enabled tennis-watcher")
        if enabled != "enabled":
            fail(f"Service is {enabled} — won't start on boot")
        return False

    # Check last log entry for errors
    _, log_output = ssh_cmd(HETZNER_HOST,
        "journalctl -u tennis-watcher --no-pager -n 5 --output=cat", timeout=10)
    if log_output:
        lines = log_output.strip().split("\n")
        last = lines[-1] if lines else ""
        if "ERROR" in last or "Failed" in last:
            fail(f"Recent error: {last}")
        elif "authentication successful" in last.lower() or "Sleeping" in last:
            ok(f"Last activity: {last.strip()}")
        else:
            ok(f"Last log: {last.strip()}")

    # Check if it found videos recently
    _, recent = ssh_cmd(HETZNER_HOST,
        "journalctl -u tennis-watcher --since '1 hour ago' --no-pager --output=cat | grep -c 'Slo-mo detected' || echo 0")
    if recent and recent != "0":
        ok(f"Detected {recent} new slo-mo video(s) in last hour")

    return True


def check_gpu_worker(machine):
    """Check GPU worker on a machine."""
    name = machine["name"]
    ssh = machine["ssh"]
    role = machine["role"]

    header(f"GPU Worker: {name} ({role})")

    # Check SSH connectivity
    success, hostname = ssh_cmd(ssh, "hostname", timeout=10)
    if not success:
        fail(f"Cannot SSH to {ssh} (machine may be off)")
        return False
    ok(f"SSH connected to {hostname}")

    # Check if worker process is running
    success, output = ssh_cmd(ssh,
        'powershell -c "Get-CimInstance Win32_Process | Where-Object {$_.CommandLine -like \'*worker.py*\'} | Select-Object ProcessId,CommandLine | Format-List"',
        timeout=15)

    if success and "worker.py" in output:
        # Extract PIDs
        import re
        pids = re.findall(r"ProcessId\s*:\s*(\d+)", output)
        ok(f"Worker running (PID: {', '.join(pids)})")
    else:
        fail("No worker process found")
        # Check scheduled task
        task_ok, task_out = ssh_cmd(ssh, 'schtasks /query /tn "TennisGPUWorker" 2>nul')
        if task_ok:
            warn("Scheduled task exists but worker not running — try: schtasks /run /tn TennisGPUWorker")
        else:
            fail("No scheduled task configured")
        return False

    # Check iCloud session
    success, session = ssh_cmd(ssh,
        f'dir /b "{PROJECT_PATH_WIN}\\config\\icloud_session\\*.cookiejar" 2>nul')
    if success and session:
        ok("iCloud session cookies present")
    else:
        warn("No iCloud session cookies — downloads will fail")

    # Check GPU availability
    success, gpu = ssh_cmd(ssh,
        'nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader,nounits 2>nul',
        timeout=10)
    if success and gpu:
        ok(f"GPU: {gpu}")
    else:
        warn("nvidia-smi not responding")

    return True


def check_pending_jobs():
    """Show pending and processing jobs."""
    header("Job Queue")

    data = http_get(f"{COORDINATOR_URL}/jobs?status=processing")
    if data and data.get("jobs"):
        for job in data["jobs"]:
            elapsed = ""
            if job.get("claimed_at"):
                try:
                    claimed = datetime.fromisoformat(job["claimed_at"])
                    delta = datetime.now(timezone.utc) - claimed
                    mins = int(delta.total_seconds() / 60)
                    elapsed = f" ({mins}m ago)"
                except Exception:
                    pass
            ok(f"Processing: {job['filename']} on {job.get('claimed_by', '?')}{elapsed}")

    data = http_get(f"{COORDINATOR_URL}/jobs?status=pending")
    if data and data.get("jobs"):
        for job in data["jobs"]:
            warn(f"Pending: {job['filename']}")
    elif data:
        ok("No pending jobs")

    data = http_get(f"{COORDINATOR_URL}/jobs?status=failed&limit=5")
    if data and data.get("jobs"):
        for job in data["jobs"]:
            fail(f"Failed: {job['filename']} on {job.get('claimed_by', '?')}")


def check_r2_gallery():
    """Check R2 gallery status — HTML loads, JS is valid, videos render."""
    header("R2 Gallery")
    try:
        import urllib.request
        req = urllib.request.Request("https://media.playfullife.com/",
                                     headers={"User-Agent": "tennis-healthcheck/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        fail(f"Gallery not responding: {e}")
        return

    # Check video data is embedded
    import re
    vid_match = re.search(r'var VIDEOS = (\[.*?\]);', body, re.DOTALL)
    if not vid_match:
        fail("No VIDEOS data found in HTML")
        return

    try:
        video_data = json.loads(vid_match.group(1))
        video_count = len(video_data)
        if video_count == 0:
            fail("VIDEOS array is empty")
        else:
            ok(f"Gallery has {video_count} videos in data")
    except json.JSONDecodeError as e:
        fail(f"VIDEOS JSON is invalid: {e}")
        return

    # Check thumbnails — all videos should have has_thumb=True
    missing_thumbs = [v['id'] for v in video_data if not v.get('has_thumb')]
    if missing_thumbs:
        warn(f"Missing thumbnails: {', '.join(missing_thumbs)}")
    else:
        ok("All videos have thumbnails")

    # Check JS syntax by running through node (if available)
    try:
        scripts = re.findall(r'<script>(.*?)</script>', body, re.DOTALL)
        if scripts:
            # Write to temp file and check with node
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
                # Wrap in function to avoid DOM errors but catch syntax
                f.write('(function(){\n')
                for s in scripts:
                    f.write(s)
                    f.write('\n')
                f.write('});\n')  # Don't execute, just parse
                tmp_path = f.name
            result = subprocess.run(
                ['node', '--check', tmp_path],
                capture_output=True, text=True, timeout=5
            )
            os.unlink(tmp_path)
            if result.returncode == 0:
                ok("JavaScript syntax valid")
            else:
                fail(f"JavaScript syntax error: {result.stderr.strip()[:100]}")
    except FileNotFoundError:
        pass  # node not available, skip
    except Exception as e:
        warn(f"JS check skipped: {e}")


def test_icloud_auth():
    """Test iCloud authentication on all machines."""
    header("iCloud Auth Test")

    # Test on Hetzner
    print(f"\n  Testing Hetzner watcher...")
    success, output = ssh_cmd(HETZNER_HOST,
        "cd /opt/tennis && venv/bin/python cloud_icloud_watcher.py --once 2>&1 | head -10",
        timeout=60)
    if success:
        if "authentication successful" in output.lower():
            ok(f"Hetzner: iCloud auth OK")
        elif "2FA" in output:
            fail("Hetzner: 2FA required — run interactively")
        else:
            warn(f"Hetzner: {output[:100]}")
    else:
        fail(f"Hetzner: {output[:100]}")

    # Test on GPU machines
    for machine in GPU_MACHINES:
        print(f"\n  Testing {machine['name']}...")
        success, output = ssh_cmd(machine["ssh"],
            f'{PROJECT_PATH_WIN}\\venv\\Scripts\\python.exe -c "'
            f"from pyicloud import PyiCloudService; "
            f"api = PyiCloudService('amassena@gmail.com', open(r'{PROJECT_PATH_WIN}\\.env').read().split('ICLOUD_PASSWORD=')[1].split(chr(10))[0].strip(), "
            f"cookie_directory=r'{PROJECT_PATH_WIN}\\config\\icloud_session'); "
            f"print('2FA' if api.requires_2fa else 'OK')"
            '"',
            timeout=30)
        if success:
            if output.strip() == "OK":
                ok(f"{machine['name']}: iCloud auth OK")
            elif "2FA" in output:
                fail(f"{machine['name']}: 2FA required")
            else:
                warn(f"{machine['name']}: {output[:100]}")
        else:
            fail(f"{machine['name']}: {output[:100]}")


def test_job():
    """Submit a test job and monitor its progress through the pipeline."""
    header("End-to-End Test")

    # Check if there's a small test video we can use
    # For now, we just verify the coordinator can accept and serve jobs
    import urllib.request

    print("  Phase 1: Coordinator API test...")

    # Create a test job
    test_data = json.dumps({
        "icloud_asset_id": "__test__healthcheck__",
        "filename": "__healthcheck_test__.MOV",
        "album_name": "test:healthcheck",
    }).encode()

    try:
        req = urllib.request.Request(
            f"{COORDINATOR_URL}/jobs",
            data=test_data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read())
            if result.get("status") in ("created", "exists"):
                ok(f"Job creation: {result['status']} (video_id: {result.get('video_id', '?')})")
                test_video_id = result.get("video_id")
            else:
                fail(f"Unexpected response: {result}")
                return
    except Exception as e:
        fail(f"Job creation failed: {e}")
        return

    print("  Phase 2: Job visible in queue...")
    stats = http_get(f"{COORDINATOR_URL}/stats")
    if stats:
        ok(f"Queue stats: {stats['pending']} pending")
    else:
        fail("Cannot read stats")

    # Clean up test job — must follow claim → processing → complete sequence
    if test_video_id:
        try:
            for step, data in [
                (f"/jobs/{test_video_id}/claim?worker_id=healthcheck", {}),
                (f"/jobs/{test_video_id}/processing?worker_id=healthcheck", None),
                (f"/jobs/{test_video_id}/complete?worker_id=healthcheck",
                 {"success": False, "error_message": "healthcheck test — not a real job"}),
            ]:
                body = json.dumps(data).encode() if data is not None else b""
                req = urllib.request.Request(
                    f"{COORDINATOR_URL}{step}",
                    data=body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                urllib.request.urlopen(req, timeout=10)
            ok("Test job cleaned up")
        except Exception as e:
            warn(f"Cleanup failed (manual cleanup needed): {e}")

    print("\n  Phase 3: Worker poll connectivity...")
    # Check coordinator logs for recent worker polls (try both journal and nginx access log)
    success, output = ssh_cmd(HETZNER_HOST,
        "journalctl -u tennis-coordinator --since '5 minutes ago' --no-pager --output=cat 2>/dev/null | grep 'jobs/pending' | tail -10;"
        "journalctl -u caddy --since '5 minutes ago' --no-pager --output=cat 2>/dev/null | grep 'jobs/pending' | tail -10;"
        "journalctl -u nginx --since '5 minutes ago' --no-pager --output=cat 2>/dev/null | grep 'jobs/pending' | tail -10",
        timeout=15)
    if success and output and "jobs/pending" in output:
        import re
        ips = set(re.findall(r"(\d+\.\d+\.\d+\.\d+)", output))
        lines = [l for l in output.strip().split("\n") if "jobs/pending" in l]
        ok(f"{len(lines)} poll(s) from {len(ips)} worker(s) in last 5 minutes")
        for ip in ips:
            ok(f"  Worker polling from {ip}")
    else:
        # Fallback: just check if coordinator received any requests recently
        success2, output2 = ssh_cmd(HETZNER_HOST,
            "journalctl -u tennis-coordinator --since '5 minutes ago' --no-pager -n 5 --output=cat")
        if success2 and output2 and "GET" in output2:
            ok("Coordinator receiving requests (workers connected)")
        else:
            warn("No worker polls detected in last 5 minutes")

    print(f"\n  {GREEN}{BOLD}End-to-end test complete.{RESET}")
    print(f"  To test actual video processing, add a video to the 'Tennis Videos' album on iCloud")
    print(f"  or record a new slo-mo video — the watcher will pick it up automatically.")


def main():
    parser = argparse.ArgumentParser(description="Tennis pipeline health check")
    parser.add_argument("--quick", action="store_true", help="Service status only")
    parser.add_argument("--test-job", action="store_true", help="Submit test job to coordinator")
    parser.add_argument("--test-auth", action="store_true", help="Test iCloud auth on all machines")
    args = parser.parse_args()

    print(f"{BOLD}Tennis Pipeline Health Check{RESET}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if args.test_auth:
        test_icloud_auth()
        return

    if args.test_job:
        check_coordinator()
        test_job()
        return

    # Full health check
    check_coordinator()
    check_watcher()
    for machine in GPU_MACHINES:
        check_gpu_worker(machine)
    check_pending_jobs()

    if not args.quick:
        check_r2_gallery()

    print()


if __name__ == "__main__":
    main()
