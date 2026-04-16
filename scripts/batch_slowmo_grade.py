"""Batch render grade overlay on slow-mo variants.

For each video in VIDS:
  1. Download rally_slowmo.mp4 + shots.json from R2 to exports/{vid}/
  2. Render grade overlay → exports/{vid}_rally_slowmo_graded.mp4
  3. Transcode via NVENC → exports/{vid}_rally_slowmo_final.mp4
  4. Upload back to R2, overwriting the ungraded version
  5. Delete intermediates

Pass video list as comma-separated: python batch_slowmo_grade.py IMG_1108,IMG_1109,...
Or set VIDS below and run with no args.
"""
import os, subprocess, sys
from pathlib import Path

os.chdir(r"C:\Users\amass\tennis_analysis")
sys.path.insert(0, r"C:\Users\amass\tennis_analysis")

from dotenv import load_dotenv
load_dotenv()
from storage.r2_client import R2Client

# Default list — overridden by CLI arg
DEFAULT_VIDS = []

PY = r"C:\Users\amass\tennis_analysis\venv\Scripts\python.exe"
EXPORTS = Path(r"C:\Users\amass\tennis_analysis\exports")

def main():
    if len(sys.argv) > 1:
        vids = [v.strip() for v in sys.argv[1].split(",") if v.strip()]
    else:
        vids = DEFAULT_VIDS

    if not vids:
        print("No videos specified. Pass as comma-separated arg.")
        sys.exit(1)

    r2 = R2Client()
    print(f"Starting slow-mo grade batch: {len(vids)} videos")

    for i, vid in enumerate(vids, 1):
        print(f"\n=== [{i}/{len(vids)}] {vid} ===", flush=True)
        vid_dir = EXPORTS / vid
        vid_dir.mkdir(parents=True, exist_ok=True)

        slowmo = vid_dir / f"{vid}_rally_slowmo.mp4"
        shots_json = vid_dir / "shots.json"
        graded = EXPORTS / f"{vid}_rally_slowmo_graded.mp4"
        final_mp4 = EXPORTS / f"{vid}_rally_slowmo_final.mp4"

        # 1. Download from R2 if missing
        try:
            if not slowmo.exists():
                print(f"  [download] rally_slowmo.mp4", flush=True)
                r2.download(f"highlights/{vid}/{vid}_rally_slowmo.mp4", str(slowmo))
            if not shots_json.exists():
                print(f"  [download] shots.json", flush=True)
                r2.download(f"highlights/{vid}/shots.json", str(shots_json))
        except Exception as e:
            print(f"  [SKIP] download failed: {e}", flush=True)
            continue

        # 2. Render grade overlay
        print(f"  [render] graded overlay", flush=True)
        r = subprocess.run(
            [PY, r"scripts\render_graded.py",
             "--video", vid,
             "--variant", "rally_slowmo",
             "--input", str(slowmo),
             "--shots-json", str(shots_json),
             "--output", str(graded)],
            capture_output=True, text=True)
        if r.returncode != 0 or not graded.exists():
            print(f"  [SKIP] render failed: {r.stderr.strip()[-300:]}", flush=True)
            continue

        # 3. Transcode via NVENC
        print(f"  [transcode] NVENC", flush=True)
        r = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-i", str(graded),
             "-c:v", "h264_nvenc", "-preset", "p5", "-cq", "30",
             "-an", "-movflags", "+faststart",
             str(final_mp4)],
            capture_output=True, text=True)
        if r.returncode != 0 or not final_mp4.exists():
            print(f"  [SKIP] transcode failed: {r.stderr.strip()[-200:]}", flush=True)
            if graded.exists(): graded.unlink()
            continue

        # 4. Upload
        print(f"  [upload]", flush=True)
        try:
            r2.upload(str(final_mp4),
                      f"highlights/{vid}/{vid}_rally_slowmo.mp4",
                      content_type="video/mp4")
        except Exception as e:
            print(f"  [SKIP] upload failed: {e}", flush=True)
            continue

        # 5. Cleanup intermediates (keep the original slowmo in vid_dir for safety)
        for f in (graded, final_mp4):
            try:
                if f.exists(): f.unlink()
            except Exception:
                pass
        # Also remove the downloaded slowmo (we uploaded a new one)
        try: slowmo.unlink()
        except Exception: pass

        size_mb = final_mp4.stat().st_size / 1e6 if final_mp4.exists() else 0
        print(f"  DONE {vid}", flush=True)

    print("\n=== BATCH COMPLETE ===", flush=True)


if __name__ == "__main__":
    main()
