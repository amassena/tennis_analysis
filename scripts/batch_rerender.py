"""Batch re-render all gallery videos with latest grades + features.

Runs on GPU machines. For each video:
1. Re-export timeline/bytype/rally (with slow-mo variants)
2. Render graded overlay on timeline
3. Re-run court detection (v2)
4. Upload everything to R2
5. Regenerate gallery index

Usage:
    python scripts/batch_rerender.py --all
    python scripts/batch_rerender.py --videos IMG_0864,IMG_1027
    python scripts/batch_rerender.py --all --skip-export  # only grade + upload
"""
import glob
import os
import subprocess
import sys
import time

os.chdir(r"C:\Users\amass\tennis_analysis")
sys.path.insert(0, r"C:\Users\amass\tennis_analysis")

PY = r"C:\Users\amass\tennis_analysis\venv\Scripts\python.exe"
ENV = {**os.environ, "PYTHONIOENCODING": "utf-8"}


def get_all_videos():
    return sorted(
        os.path.splitext(os.path.basename(f))[0]
        for f in glob.glob(r"preprocessed\*.mp4")
        if os.path.exists(os.path.join("detections",
            os.path.splitext(os.path.basename(f))[0] + "_fused_detections.json"))
    )


def run(cmd_args, label="", timeout=1800):
    print(f"    [{label}] {' '.join(cmd_args[:4])}...", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd_args, capture_output=True, text=True, timeout=timeout, env=ENV)
    elapsed = time.time() - t0
    if r.returncode != 0:
        err = (r.stderr or "")[-300:]
        print(f"    [{label}] FAILED ({elapsed:.0f}s): {err}", flush=True)
        return False
    print(f"    [{label}] done ({elapsed:.0f}s)", flush=True)
    return True


def process_video(vid, skip_export=False, skip_court=False, skip_grade=False):
    print(f"\n{'='*50}", flush=True)
    print(f"  {vid}", flush=True)
    print(f"{'='*50}", flush=True)

    pp = os.path.join("preprocessed", f"{vid}.mp4")
    if not os.path.exists(pp):
        print(f"  SKIP: no preprocessed video", flush=True)
        return False

    if not skip_export:
        run([PY, "scripts/export_videos.py", pp,
             "--types", "timeline", "rally", "bytype",
             "--slow-motion", "--upload"],
            label="export", timeout=3600)

    if not skip_court:
        court_out = os.path.join("court_calibrations", f"{vid}.json")
        if not os.path.exists(court_out):
            run([PY, "scripts/court_detect.py", pp], label="court")

    if not skip_grade:
        run([PY, "scripts/render_graded.py", "--video", vid],
            label="grade-timeline", timeout=1800)

        graded = os.path.join("exports", f"{vid}_graded.mp4")
        if os.path.exists(graded):
            run([PY, "-c",
                 f"import sys; sys.path.insert(0,'.'); "
                 f"from scripts.export_videos import upload_to_r2; "
                 f"upload_to_r2('{graded}', 'highlights/{vid}/{vid}_timeline.mp4')"],
                label="upload-graded")

    return True


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--videos", help="Comma-separated video IDs")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--skip-court", action="store_true")
    parser.add_argument("--skip-grade", action="store_true")
    args = parser.parse_args()

    if args.all:
        vids = get_all_videos()
    elif args.videos:
        vids = [v.strip() for v in args.videos.split(",")]
    else:
        parser.error("Provide --all or --videos")
        return

    print(f"\nBatch re-render: {len(vids)} videos", flush=True)
    t0 = time.time()
    success = 0

    for i, vid in enumerate(vids, 1):
        print(f"\n[{i}/{len(vids)}]", flush=True)
        if process_video(vid, skip_export=args.skip_export,
                        skip_court=args.skip_court,
                        skip_grade=args.skip_grade):
            success += 1

    # Regenerate gallery index
    print(f"\nRegenerating gallery index...", flush=True)
    run([PY, "scripts/update_r2_index.py"], label="gallery")

    elapsed = time.time() - t0
    print(f"\n{'='*50}", flush=True)
    print(f"BATCH COMPLETE: {success}/{len(vids)} videos in {elapsed/60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
