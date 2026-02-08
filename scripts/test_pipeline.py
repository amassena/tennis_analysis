#!/usr/bin/env python3
"""
Test the full pipeline end-to-end with a small test video.

This script:
1. Copies test video to a GPU machine
2. Runs preprocess -> pose extraction -> shot detection -> clip extraction
3. Compiles highlights (group by shot type)
4. Uploads to YouTube (optional, with --upload flag)
5. Sends notifications
6. Reports success/failure of each step

Usage:
    python scripts/test_pipeline.py                    # Test without YouTube upload
    python scripts/test_pipeline.py --upload           # Test with YouTube upload
    python scripts/test_pipeline.py --machine windows  # Test on specific machine
    python scripts/test_pipeline.py --cleanup          # Clean up test files after
"""

import argparse
import os
import subprocess
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import AUTO_PIPELINE, NOTIFICATIONS

TEST_VIDEO = "TEST_6shots.mov"
TEST_VIDEO_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test", TEST_VIDEO)


def log(msg, status="INFO"):
    icons = {"INFO": "ℹ️", "OK": "✅", "FAIL": "❌", "WARN": "⚠️", "RUN": "🔄"}
    print(f"{icons.get(status, 'ℹ️')} {msg}")


def send_notification(msg):
    """Stub for notifications (ntfy removed)."""
    pass


def run_ssh(host, project, cmd, timeout=600):
    """Run a command on a remote machine via SSH."""
    full_cmd = f"cd {project} && {cmd}"
    result = subprocess.run(
        ["ssh", "-o", "ConnectTimeout=30", host, full_cmd],
        capture_output=True, text=True, timeout=timeout
    )
    return result.returncode == 0, result.stdout, result.stderr


def run_scp_to(host, project, local_path, remote_path):
    """Copy file to remote machine."""
    result = subprocess.run(
        ["scp", local_path, f"{host}:{project}/{remote_path}"],
        capture_output=True, text=True, timeout=300
    )
    return result.returncode == 0


def run_scp_from(host, project, remote_path, local_path):
    """Copy file from remote machine."""
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    result = subprocess.run(
        ["scp", f"{host}:{project}/{remote_path}", local_path],
        capture_output=True, text=True, timeout=300
    )
    return result.returncode == 0


def test_pipeline(machine, upload=False, cleanup=False, dry_run=False):
    """Run full pipeline test.

    Args:
        machine: Machine config dict with host and project path
        upload: If True, upload to YouTube for real
        cleanup: If True, clean up test files after
        dry_run: If True, simulate YouTube upload without actually uploading
    """
    host = machine["host"]
    project = machine["project"]
    python = f"{project}/venv/Scripts/python.exe"

    results = {}
    test_name = TEST_VIDEO.replace(".mov", "")

    print(f"\n{'='*60}")
    print(f"  PIPELINE TEST - {host}")
    print(f"{'='*60}\n")

    # Check machine is reachable
    log(f"Checking {host} is reachable...", "RUN")
    ok, out, err = run_ssh(host, project, "echo OK", timeout=30)
    if not ok:
        log(f"Cannot reach {host}", "FAIL")
        return False
    log(f"{host} is online", "OK")
    results["connectivity"] = True

    # Send start notification
    send_notification(f"🧪 Pipeline test starting on {host}")

    # Step 1: Copy test video
    log(f"Copying test video to {host}...", "RUN")
    if not os.path.exists(TEST_VIDEO_PATH):
        log(f"Test video not found: {TEST_VIDEO_PATH}", "FAIL")
        return False

    ok = run_scp_to(host, project, TEST_VIDEO_PATH, f"raw/{TEST_VIDEO}")
    if not ok:
        log("Failed to copy test video", "FAIL")
        return False
    log("Test video copied", "OK")
    results["copy"] = True

    # Step 2: Preprocess
    log("Running preprocessing (NVENC)...", "RUN")
    start = time.time()
    ok, out, err = run_ssh(host, project, f"{python} preprocess_nvenc.py {TEST_VIDEO}", timeout=120)
    if not ok:
        log(f"Preprocessing failed: {err[:200]}", "FAIL")
        return False
    log(f"Preprocessing done ({time.time()-start:.1f}s)", "OK")
    results["preprocess"] = True

    # Step 3: Pose extraction
    log("Running pose extraction...", "RUN")
    start = time.time()
    ok, out, err = run_ssh(host, project, f"{python} scripts/extract_poses.py preprocessed/{test_name}.mp4", timeout=300)
    if not ok:
        log(f"Pose extraction failed: {err[:200]}", "FAIL")
        return False
    log(f"Pose extraction done ({time.time()-start:.1f}s)", "OK")
    results["poses"] = True

    # Step 4: Shot detection
    log("Running shot detection...", "RUN")
    start = time.time()
    ok, out, err = run_ssh(host, project, f"{python} scripts/detect_shots.py poses/{test_name}.json -o shots_detected_{test_name}.json", timeout=120)
    if not ok:
        log(f"Shot detection failed: {err[:200]}", "FAIL")
        return False
    log(f"Shot detection done ({time.time()-start:.1f}s)", "OK")
    results["detection"] = True

    # Step 5: Extract clips (group by shot type)
    log("Extracting clips (group by shot type)...", "RUN")
    start = time.time()
    ok, out, err = run_ssh(host, project,
        f"{python} scripts/extract_clips.py -i shots_detected_{test_name}.json -v preprocessed/{test_name}.mp4 --highlights --group-by-type",
        timeout=180)
    if not ok:
        log(f"Clip extraction failed: {err[:200]}", "FAIL")
        return False
    log(f"Clip extraction done ({time.time()-start:.1f}s)", "OK")
    results["clips"] = True

    # Step 6: Check highlights exist
    log("Verifying highlights generated...", "RUN")
    ok, out, err = run_ssh(host, project, f"dir highlights\\{test_name}*.mp4", timeout=30)
    if not ok or test_name not in out:
        log("Highlights not found", "FAIL")
        return False
    log("Highlights generated", "OK")
    results["highlights"] = True

    # Step 7: Copy highlights back
    log("Copying highlights to Mac...", "RUN")
    local_highlights = f"/Users/andrewhome/tennis_analysis/test/{test_name}_highlights.mp4"
    # Find the actual highlight file
    ok, out, err = run_ssh(host, project, f"dir /b highlights\\{test_name}*.mp4", timeout=30)
    if ok and out.strip():
        highlight_file = out.strip().split('\n')[0].strip()
        ok = run_scp_from(host, project, f"highlights/{highlight_file}", local_highlights)
        if ok:
            log("Highlights copied to Mac", "OK")
            results["copy_back"] = True
        else:
            log("Failed to copy highlights", "WARN")

    # Step 8: YouTube upload (optional, supports dry-run)
    if upload or dry_run:
        mode = "[DRY-RUN] " if dry_run else ""
        log(f"{mode}Uploading to YouTube...", "RUN")
        dry_run_flag = " --dry-run" if dry_run else ""
        ok, out, err = run_ssh(host, project,
            f"{python} scripts/upload.py highlights/{highlight_file} --title \"[TEST] Pipeline Test\" --youtube{dry_run_flag}",
            timeout=300)
        if ok and "youtu" in out.lower():
            log(f"{mode}YouTube upload done", "OK")
            results["youtube"] = True
            # Extract URL
            for line in out.split('\n'):
                if 'youtu' in line.lower():
                    log(f"  URL: {line.strip()}", "INFO")
        else:
            log(f"{mode}YouTube upload failed", "WARN")
            results["youtube"] = False

    # Send completion notification
    send_notification(f"✅ Pipeline test completed on {host}")

    # Cleanup (optional)
    if cleanup:
        log("Cleaning up test files...", "RUN")
        run_ssh(host, project, f"del raw\\{TEST_VIDEO} preprocessed\\{test_name}.mp4 poses\\{test_name}.json shots_detected_{test_name}.json 2>nul", timeout=30)
        run_ssh(host, project, f"rmdir /s /q clips\\{test_name} 2>nul", timeout=30)
        log("Cleanup done", "OK")

    # Summary
    print(f"\n{'='*60}")
    print("  TEST RESULTS")
    print(f"{'='*60}")
    for step, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {step:15} {status}")

    all_passed = all(results.values())
    print(f"\n  {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    print(f"{'='*60}\n")

    return all_passed


def main():
    parser = argparse.ArgumentParser(description="Test the full pipeline end-to-end")
    parser.add_argument("--machine", default="desktop3090",
                        help="Machine to test on (default: desktop3090)")
    parser.add_argument("--upload", action="store_true",
                        help="Upload to YouTube (unlisted)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Test full pipeline with simulated YouTube upload (no actual upload)")
    parser.add_argument("--cleanup", action="store_true",
                        help="Clean up test files after")
    parser.add_argument("--list-machines", action="store_true",
                        help="List available machines")
    args = parser.parse_args()

    machines = {m["host"]: m for m in AUTO_PIPELINE["gpu_machines"]}

    if args.list_machines:
        print("Available machines:")
        for name, config in machines.items():
            print(f"  {name}: {config['project']}")
        return

    if args.machine not in machines:
        print(f"Unknown machine: {args.machine}")
        print(f"Available: {', '.join(machines.keys())}")
        sys.exit(1)

    # Check test video exists
    if not os.path.exists(TEST_VIDEO_PATH):
        print(f"Test video not found: {TEST_VIDEO_PATH}")
        print("Create it first with: ffmpeg -f concat -safe 0 -i clips_list.txt -c copy test/TEST_6shots.mov")
        sys.exit(1)

    success = test_pipeline(machines[args.machine], upload=args.upload, cleanup=args.cleanup, dry_run=args.dry_run)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
