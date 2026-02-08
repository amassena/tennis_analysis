"""Preprocess raw VFR .MOV files to 60fps CFR .mp4 using NVENC GPU encoding."""
import os
import subprocess
import sys
import json

# Auto-detect project root from script location
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(PROJECT_ROOT, "raw")
PREPROCESSED_DIR = os.path.join(PROJECT_ROOT, "preprocessed")

TARGET_FPS = 60
CQ = 32
PRESET = "p4"


def probe_duration(filepath):
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", filepath]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode == 0:
        data = json.loads(result.stdout)
        return float(data.get("format", {}).get("duration", 0))
    return 0


def _nvenc_works():
    """Test if NVENC actually works (driver may be too old)."""
    import subprocess as _sp
    try:
        r = _sp.run(
            ["ffmpeg", "-hide_banner", "-y",
             "-f", "lavfi", "-i", "nullsrc=s=64x64:d=0.1",
             "-c:v", "h264_nvenc", "-f", "null", "-"],
            capture_output=True, text=True, timeout=10
        )
        return r.returncode == 0
    except Exception:
        return False


_USE_NVENC = None


def convert(src, dst):
    global _USE_NVENC
    if _USE_NVENC is None:
        _USE_NVENC = _nvenc_works()
        if _USE_NVENC:
            print("  Using NVENC hardware encoder")
        else:
            print("  NVENC unavailable, falling back to libx264")

    part = dst + ".part"
    duration = probe_duration(src)

    if _USE_NVENC:
        codec_args = [
            "-c:v", "h264_nvenc",
            "-preset", PRESET,
            "-rc", "constqp", "-cq", str(CQ),
        ]
    else:
        codec_args = [
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "18",
        ]

    cmd = [
        "ffmpeg", "-y",
        "-i", src,
        "-map", "0:v:0", "-map", "0:a:0?",  # ? makes audio optional
        "-r", str(TARGET_FPS),
        *codec_args,
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-f", "mp4",
        "-progress", "pipe:1",
        "-nostats",
        part,
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)

    for line in proc.stdout:
        line = line.strip()
        if line.startswith("out_time_us="):
            try:
                t = int(line.split("=", 1)[1]) / 1_000_000
                if duration > 0 and t > 0:
                    pct = min(t / duration * 100, 100)
                    m, s = divmod(int(t), 60)
                    dm, ds = divmod(int(duration), 60)
                    print(f"\r  Converting: {pct:5.1f}% [{m}:{s:02d}/{dm}:{ds:02d}]", end="", flush=True)
            except (ValueError, IndexError):
                pass

    proc.wait()
    print()

    if proc.returncode == 0:
        os.replace(part, dst)
        size_mb = os.path.getsize(dst) / (1024 * 1024)
        print(f"  [OK] {os.path.basename(dst)} ({size_mb:.1f} MB)")
        return True
    else:
        print(f"  [FAILED] ffmpeg exit code {proc.returncode}")
        if os.path.exists(part):
            os.remove(part)
        return False


def main():
    os.makedirs(PREPROCESSED_DIR, exist_ok=True)

    videos = sorted(f for f in os.listdir(RAW_DIR) if f.lower().endswith(".mov"))
    if len(sys.argv) > 1:
        # Process specific file(s) passed as arguments
        videos = [f for f in sys.argv[1:] if os.path.exists(os.path.join(RAW_DIR, f))]
    else:
        videos = sorted(f for f in os.listdir(RAW_DIR) if f.lower().endswith(".mov"))

    print(f"Found {len(videos)} video(s) to preprocess with NVENC\n")

    for i, name in enumerate(videos, 1):
        src = os.path.join(RAW_DIR, name)
        dst = os.path.join(PREPROCESSED_DIR, os.path.splitext(name)[0] + ".mp4")

        print(f"[{i}/{len(videos)}] {name}")

        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            size_mb = os.path.getsize(dst) / (1024 * 1024)
            print(f"  [SKIP] Already exists ({size_mb:.1f} MB)\n")
            continue

        size_mb = os.path.getsize(src) / (1024 * 1024)
        print(f"  Source: {size_mb:.0f} MB")
        success = convert(src, dst)
        print()


if __name__ == "__main__":
    main()
