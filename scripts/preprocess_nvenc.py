"""Preprocess raw VFR .MOV files to CFR .mp4 using NVENC GPU encoding.

Detects the native max frame rate (e.g. 240fps for iPhone slo-mo) and outputs
CFR at that rate. Frames from lower-fps sections are duplicated to maintain
constant rate throughout.
"""
import os
import subprocess
import sys
import json
import shutil

# Auto-detect project root from script location
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, "raw")
PREPROCESSED_DIR = os.path.join(PROJECT_ROOT, "preprocessed")

DEFAULT_FPS = 60
CQ = 32
PRESET = "p4"

# Find ffmpeg/ffprobe - check common locations if not in PATH
def _find_ffmpeg():
    """Find ffmpeg executable, checking common Windows locations."""
    # Try PATH first
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg, shutil.which("ffprobe")

    # Common Windows locations - use explicit user paths for scheduled tasks
    locations = [
        # WinGet links for known users
        r"C:\Users\amass\AppData\Local\Microsoft\WinGet\Links",
        r"C:\Users\Andrew\AppData\Local\Microsoft\WinGet\Links",
        # Standard locations
        r"C:\ffmpeg\bin",
        r"C:\Program Files\ffmpeg\bin",
        # Environment variable fallback
        os.path.expandvars(r"%USERPROFILE%\AppData\Local\Microsoft\WinGet\Links"),
        os.path.expandvars(r"%USERPROFILE%\ffmpeg\bin"),
    ]

    for loc in locations:
        ffmpeg_path = os.path.join(loc, "ffmpeg.exe")
        ffprobe_path = os.path.join(loc, "ffprobe.exe")
        if os.path.exists(ffmpeg_path):
            return ffmpeg_path, ffprobe_path

    # Fallback to just the command name
    return "ffmpeg", "ffprobe"

FFMPEG, FFPROBE = _find_ffmpeg()


def probe_duration(filepath):
    cmd = [FFPROBE, "-v", "quiet", "-print_format", "json", "-show_format", filepath]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode == 0:
        data = json.loads(result.stdout)
        return float(data.get("format", {}).get("duration", 0))
    return 0


def probe_nb_frames(filepath):
    """Get total frame count from the video stream."""
    cmd = [
        FFPROBE, "-v", "quiet", "-print_format", "json",
        "-show_streams", "-select_streams", "v:0", filepath,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode == 0:
        data = json.loads(result.stdout)
        streams = data.get("streams", [])
        if streams:
            nb = streams[0].get("nb_frames")
            if nb:
                return int(nb)
    return 0


def probe_max_fps(filepath):
    """Detect the maximum frame rate in a video file.

    iPhone slo-mo files have mixed frame rates (e.g. 60fps + 240fps).
    We want the highest rate so we can output CFR at that rate.
    """
    # Try reading all stream frame rates
    cmd = [
        FFPROBE, "-v", "quiet", "-print_format", "json",
        "-show_streams", "-select_streams", "v:0", filepath,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        return DEFAULT_FPS

    data = json.loads(result.stdout)
    streams = data.get("streams", [])
    if not streams:
        return DEFAULT_FPS

    stream = streams[0]

    # Check avg_frame_rate and r_frame_rate
    max_fps = 0
    for key in ("r_frame_rate", "avg_frame_rate"):
        val = stream.get(key, "")
        if "/" in val:
            num, den = val.split("/")
            try:
                fps = int(num) / int(den)
                if fps > max_fps:
                    max_fps = fps
            except (ValueError, ZeroDivisionError):
                pass

    # For iPhone slo-mo, also check nb_frames / duration for true rate
    nb_frames = stream.get("nb_frames")
    duration = stream.get("duration")
    if nb_frames and duration:
        try:
            real_fps = int(nb_frames) / float(duration)
            if real_fps > max_fps:
                max_fps = real_fps
        except (ValueError, ZeroDivisionError):
            pass

    if max_fps <= 0:
        return DEFAULT_FPS

    # Round to nearest standard rate
    standard_rates = [24, 25, 30, 48, 50, 60, 120, 240]
    closest = min(standard_rates, key=lambda r: abs(r - max_fps))
    if abs(closest - max_fps) / max(closest, 1) < 0.05:
        max_fps = closest

    return int(max_fps)


def _nvenc_works():
    """Test if NVENC actually works (driver may be too old)."""
    import subprocess as _sp
    try:
        r = _sp.run(
            [FFMPEG, "-hide_banner", "-y",
             "-f", "lavfi", "-i", "nullsrc=s=256x256:d=0.1",
             "-c:v", "h264_nvenc", "-f", "null", "-"],
            capture_output=True, text=True, timeout=10
        )
        return r.returncode == 0
    except Exception:
        return False


_USE_NVENC = None


def convert(src, dst, target_fps=None):
    global _USE_NVENC
    if _USE_NVENC is None:
        _USE_NVENC = _nvenc_works()
        if _USE_NVENC:
            print("  Using NVENC hardware encoder")
        else:
            print("  NVENC unavailable, falling back to libx264")

    if target_fps is None:
        target_fps = probe_max_fps(src)

    native_fps = probe_max_fps(src)
    # For slo-mo files (high native fps), output at 60fps real-time
    output_fps = DEFAULT_FPS
    is_slomo = native_fps > 60
    if is_slomo:
        print(f"  Slo-mo detected ({native_fps}fps) — outputting {output_fps}fps real-time")

    part = dst + ".part"
    duration = probe_duration(src)

    if _USE_NVENC:
        codec_args = [
            "-c:v", "h264_nvenc",
            "-preset", PRESET,
            "-rc", "constqp", "-cq", str(CQ),
            "-bf", "0",                   # no B-frames: eliminates CTTS table
        ]
    else:
        codec_args = [
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "18",
            "-bf", "0",                   # no B-frames: eliminates CTTS table
        ]

    # For slo-mo files, setpts assigns constant timestamps at native fps,
    # ignoring Apple's speed ramp metadata. Output fps then downsamples.
    vf_filters = []
    af_filters = []
    if is_slomo:
        vf_filters.append(f"setpts=N/{native_fps}/TB")

        # Audio sync: setpts compresses video to nb_frames/native_fps duration,
        # but audio stays at original duration. Speed up audio to match.
        nb_frames = probe_nb_frames(src)
        if nb_frames > 0 and duration > 0:
            video_duration = nb_frames / native_fps
            atempo_ratio = duration / video_duration
            if abs(atempo_ratio - 1.0) > 0.01:
                # atempo supports 0.5-100.0 range; chain if needed for >2.0
                ratio = atempo_ratio
                while ratio > 2.0:
                    af_filters.append("atempo=2.0")
                    ratio /= 2.0
                if ratio > 0.5:
                    af_filters.append(f"atempo={ratio:.4f}")
                print(f"  Audio tempo: {atempo_ratio:.3f}x ({duration:.1f}s -> {video_duration:.1f}s)")

    vf_arg = ["-vf", ",".join(vf_filters)] if vf_filters else []
    af_arg = ["-af", ",".join(af_filters)] if af_filters else []

    cmd = [
        FFMPEG, "-y",
        "-ignore_editlist", "1",      # ignore Apple slo-mo edit list on input
        "-i", src,
        "-map", "0:v:0", "-map", "0:a:0?",  # ? makes audio optional
        *vf_arg,
        *af_arg,
        "-r", str(output_fps),
        "-vsync", "cfr",                 # force constant frame rate
        *codec_args,
        "-pix_fmt", "yuv420p",
        "-color_range", "tv",            # force limited range (Safari compat)
        "-c:a", "aac",
        "-map_metadata", "-1",        # strip all metadata (Apple slo-mo timing)
        "-movflags", "+faststart",    # web-friendly: moov atom at start
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


def find_original_mov(raw_dir, name):
    """Find the original MOV file, handling 'all photo data' directory structure.

    When downloaded with 'include all photo data', iPhone creates:
      IMG_XXXX/
        IMG_XXXX.MOV     <- original camera recording
        IMG_EXXXX.mov    <- edited/flattened version (has slo-mo baked in)
        IMG_XXXX.AAE     <- edit instructions
    Without 'all photo data', you get IMG_EXXXX.mov (edited) directly.
    """
    stem = os.path.splitext(name)[0]
    direct = os.path.join(raw_dir, name)

    # Check for 'all photo data' directory
    subdir = os.path.join(raw_dir, stem)
    if os.path.isdir(subdir):
        # Prefer the original (non-E prefixed) file
        original = os.path.join(subdir, stem + ".MOV")
        if os.path.exists(original):
            return original
        # Fallback to any MOV in the directory
        for f in os.listdir(subdir):
            if f.lower().endswith(".mov") and not f.startswith("IMG_E"):
                return os.path.join(subdir, f)

    # Direct file in raw directory
    if os.path.exists(direct):
        return direct

    return None


def main():
    os.makedirs(PREPROCESSED_DIR, exist_ok=True)

    # Collect video sources: direct .mov files and 'all photo data' directories
    sources = []
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if os.path.exists(arg):
                sources.append(os.path.basename(arg))
            elif os.path.exists(os.path.join(RAW_DIR, arg)):
                sources.append(arg)
    else:
        for entry in sorted(os.listdir(RAW_DIR)):
            path = os.path.join(RAW_DIR, entry)
            if entry.lower().endswith(".mov"):
                sources.append(entry)
            elif os.path.isdir(path):
                # 'all photo data' directory — look for original MOV inside
                for f in os.listdir(path):
                    if f.upper().endswith(".MOV") and not f.startswith("IMG_E"):
                        sources.append(entry + "/" + f)
                        break

    print(f"Found {len(sources)} video(s) to preprocess\n")

    for i, name in enumerate(sources, 1):
        # Determine source path and output name
        if "/" in name:
            src = os.path.join(RAW_DIR, name)
            out_stem = os.path.splitext(name.split("/")[-1])[0]
        else:
            src = find_original_mov(RAW_DIR, name) or os.path.join(RAW_DIR, name)
            out_stem = os.path.splitext(os.path.basename(name))[0]

        dst = os.path.join(PREPROCESSED_DIR, out_stem + ".mp4")

        print(f"[{i}/{len(sources)}] {os.path.basename(src)}")

        if not os.path.exists(src):
            print(f"  [SKIP] Source not found: {src}\n")
            continue

        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            size_mb = os.path.getsize(dst) / (1024 * 1024)
            print(f"  [SKIP] Already exists ({size_mb:.1f} MB)\n")
            continue

        size_mb = os.path.getsize(src) / (1024 * 1024)
        target_fps = probe_max_fps(src)
        print(f"  Source: {size_mb:.0f} MB, native max fps: {target_fps}")
        success = convert(src, dst, target_fps=target_fps)
        print()


if __name__ == "__main__":
    main()
