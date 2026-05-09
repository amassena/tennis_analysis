"""Backfill claude_coach for any video on R2 missing coaching.json."""
import os, sys, subprocess, time
os.chdir(r"C:\Users\amass\tennis_analysis")
sys.path.insert(0, r"C:\Users\amass\tennis_analysis")
from dotenv import load_dotenv
load_dotenv()
from storage.r2_client import R2Client

PY = r"C:\Users\amass\tennis_analysis\venv\Scripts\python.exe"
ENV = {**os.environ, "PYTHONIOENCODING": "utf-8"}

c = R2Client()
keys = c.list(prefix="highlights/", max_keys=10000)
have = set(k.split("/")[1] for k in keys if k.endswith("/coaching.json"))
all_vids = set(k.split("/")[1] for k in keys if "/" in k.split("highlights/", 1)[1])
all_vids.discard("thumbs")
need = sorted(all_vids - have)

# Filter to only videos this machine has data for
local = []
for vid in need:
    if (os.path.exists(f"detections/{vid}_fused_detections.json")
        and os.path.exists(f"poses_full_videos/{vid}.json")):
        local.append(vid)

print(f"{len(need)} videos missing coaching on R2; this machine has data for {len(local)}", flush=True)
if not local:
    print("Nothing to do here.", flush=True); sys.exit(0)

ok = fail = 0
for i, vid in enumerate(local, 1):
    print(f"\n[{i}/{len(local)}] {vid}", flush=True)
    t0 = time.time()
    try:
        r = subprocess.run([PY, "scripts/claude_coach.py", vid, "--upload"],
                           capture_output=True, text=True, timeout=180, env=ENV)
        dt = time.time() - t0
        if r.returncode == 0:
            for line in r.stdout.strip().split("\n")[-3:]:
                print(f"  {line.strip()}", flush=True)
            print(f"  done ({dt:.0f}s)", flush=True); ok += 1
        else:
            print(f"  FAILED ({dt:.0f}s): {r.stderr[-200:]}", flush=True); fail += 1
    except subprocess.TimeoutExpired:
        print("  TIMEOUT", flush=True); fail += 1

print(f"\n=== DONE: {ok} ok, {fail} failed ===", flush=True)
