"""Batch generate Claude coaching for videos missing coaching.json on R2."""
import glob
import os
import subprocess
import sys
import time

os.chdir(r"C:\Users\amass\tennis_analysis")
sys.path.insert(0, r"C:\Users\amass\tennis_analysis")

from dotenv import load_dotenv
load_dotenv()
from storage.r2_client import R2Client

PY = r"C:\Users\amass\tennis_analysis\venv\Scripts\python.exe"
ENV = {**os.environ, "PYTHONIOENCODING": "utf-8"}

c = R2Client()
keys = c.list(prefix="highlights/", max_keys=10000)
have_coaching = set(k.split("/")[1] for k in keys if k.endswith("/coaching.json"))

# Find local videos with detections + poses
det_files = glob.glob(r"detections\*_fused_detections.json")
local_vids = []
for f in det_files:
    vid = os.path.basename(f).replace("_fused_detections.json", "")
    if (os.path.exists(os.path.join("poses_full_videos", vid + ".json"))
        and os.path.exists(os.path.join("preprocessed", vid + ".mp4"))):
        local_vids.append(vid)

needs_coaching = sorted(set(local_vids) - have_coaching)
print(f"{len(local_vids)} local videos ready, {len(needs_coaching)} need coaching", flush=True)

for i, vid in enumerate(needs_coaching, 1):
    print(f"\n[{i}/{len(needs_coaching)}] {vid}...", flush=True)
    t0 = time.time()
    try:
        r = subprocess.run(
            [PY, "scripts/claude_coach.py", vid, "--upload"],
            capture_output=True, text=True, timeout=300, env=ENV
        )
        elapsed = time.time() - t0
        if r.returncode == 0:
            for line in r.stdout.strip().split("\n")[-3:]:
                print(f"  {line.strip()}", flush=True)
            print(f"  done ({elapsed:.0f}s)", flush=True)
        else:
            print(f"  FAILED ({elapsed:.0f}s): {r.stderr[-300:]}", flush=True)
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT", flush=True)
    except Exception as e:
        print(f"  ERROR: {e}", flush=True)

print("\n=== COACHING BATCH COMPLETE ===", flush=True)
