"""Batch run YOLO racket detection on all videos."""
import glob
import os, sys, subprocess
os.chdir(r"C:\Users\amass\tennis_analysis")
sys.path.insert(0, r"C:\Users\amass\tennis_analysis")

PY = r"C:\Users\amass\tennis_analysis\venv\Scripts\python.exe"

if len(sys.argv) > 1 and sys.argv[1] == "--all":
    vids = sorted(
        os.path.splitext(os.path.basename(f))[0]
        for f in glob.glob(r"preprocessed\*.mp4")
    )
elif len(sys.argv) > 1:
    vids = sys.argv[1].split(",")
else:
    print("Usage: batch_racket.py --all  OR  batch_racket.py VID1,VID2"); sys.exit(1)
if not vids:
    print("No videos found"); sys.exit(1)

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
    try:
        r = subprocess.run([PY, r"scripts\racket_detect.py", pp],
                           capture_output=True, text=True, timeout=3600)
        if r.returncode != 0:
            print(f"    FAILED: {r.stderr.strip()[-200:]}", flush=True)
        else:
            print(f"    done", flush=True)
    except subprocess.TimeoutExpired:
        print(f"    TIMEOUT (>60min), skipping", flush=True)
    except Exception as e:
        print(f"    ERROR: {e}", flush=True)

print("\n=== RACKET BATCH COMPLETE ===", flush=True)
