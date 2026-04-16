"""Run TrackNet ball tracking on assigned video list."""
import os, sys, subprocess
from pathlib import Path

os.chdir(r"C:\Users\amass\tennis_analysis")
sys.path.insert(0, r"C:\Users\amass\tennis_analysis")

PY = r"C:\Users\amass\tennis_analysis\venv\Scripts\python.exe"

vids = sys.argv[1].split(",") if len(sys.argv) > 1 else []
if not vids:
    print("No videos specified"); sys.exit(1)

print(f"Ball-tracking batch: {len(vids)} videos")
for i, vid in enumerate(vids, 1):
    print(f"\n=== [{i}/{len(vids)}] {vid} ===", flush=True)
    video_path = Path("preprocessed") / f"{vid}.mp4"
    out_json = Path("ball_tracking") / f"{vid}.json"
    if out_json.exists():
        print(f"  already have {out_json}, skipping", flush=True); continue
    if not video_path.exists():
        print(f"  SKIP: {video_path} missing", flush=True); continue
    r = subprocess.run(
        [PY, r"scripts\ball_tracking.py", str(video_path)],
        capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  FAILED: {r.stderr.strip()[-200:]}", flush=True)
    else:
        print(f"  DONE {vid}", flush=True)

print("\n=== BALL BATCH COMPLETE ===", flush=True)
