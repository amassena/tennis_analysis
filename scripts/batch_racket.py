"""Batch run YOLO racket detection on all videos."""
import os, sys, subprocess
os.chdir(r"C:\Users\amass\tennis_analysis")
sys.path.insert(0, r"C:\Users\amass\tennis_analysis")

PY = r"C:\Users\amass\tennis_analysis\venv\Scripts\python.exe"

vids = sys.argv[1].split(",") if len(sys.argv) > 1 else []
if not vids:
    print("No videos specified"); sys.exit(1)

print(f"Racket detection batch: {len(vids)} videos", flush=True)
for i, vid in enumerate(vids, 1):
    vid = vid.strip()
    if not vid: continue
    out = os.path.join("racket_detections", f"{vid}.json")
    pp = os.path.join("preprocessed", f"{vid}.mp4")
    if os.path.exists(out):
        print(f"  [{i}/{len(vids)}] {vid}: already done, skip", flush=True)
        continue
    if not os.path.exists(pp):
        print(f"  [{i}/{len(vids)}] {vid}: no preprocessed video, skip", flush=True)
        continue
    print(f"  [{i}/{len(vids)}] {vid}: detecting...", flush=True)
    r = subprocess.run([PY, r"scripts\racket_detect.py", pp],
                       capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        print(f"    FAILED: {r.stderr.strip()[-200:]}", flush=True)
    else:
        print(f"    done", flush=True)

print("\n=== RACKET BATCH COMPLETE ===", flush=True)
