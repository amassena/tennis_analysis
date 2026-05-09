"""Batch regenerate swing composites for all videos and upload to R2."""
import glob
import os
import subprocess
import sys
import time

os.chdir(r"C:\Users\amass\tennis_analysis")
sys.path.insert(0, r"C:\Users\amass\tennis_analysis")

PY = r"C:\Users\amass\tennis_analysis\venv\Scripts\python.exe"
ENV = {**os.environ, "PYTHONIOENCODING": "utf-8"}

vids = sorted(set(
    os.path.splitext(os.path.basename(f))[0].replace("_fused_detections", "")
    for f in glob.glob(r"detections\*_fused_detections.json")
    if os.path.exists(os.path.join("preprocessed",
        os.path.splitext(os.path.basename(f))[0].replace("_fused_detections", "") + ".mp4"))
    and os.path.exists(os.path.join("poses_full_videos",
        os.path.splitext(os.path.basename(f))[0].replace("_fused_detections", "") + ".json"))
))

print(f"Composite batch: {len(vids)} videos", flush=True)
for i, vid in enumerate(vids, 1):
    print(f"  [{i}/{len(vids)}] {vid}...", flush=True)
    t0 = time.time()
    try:
        r = subprocess.run(
            [PY, "scripts/swing_composite.py", "--video", vid, "--upload"],
            capture_output=True, text=True, timeout=600, env=ENV
        )
        elapsed = time.time() - t0
        if r.returncode == 0:
            for line in r.stdout.strip().split("\n"):
                if "Generated" in line or "Uploaded" in line:
                    print(f"    {line.strip()}", flush=True)
            print(f"    done ({elapsed:.0f}s)", flush=True)
        else:
            print(f"    FAILED ({elapsed:.0f}s): {r.stderr[-200:]}", flush=True)
    except subprocess.TimeoutExpired:
        print(f"    TIMEOUT", flush=True)
    except Exception as e:
        print(f"    ERROR: {e}", flush=True)

print("\n=== COMPOSITE BATCH COMPLETE ===", flush=True)
