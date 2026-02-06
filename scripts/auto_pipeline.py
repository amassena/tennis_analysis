#!/usr/bin/env python3
"""Automated tennis pipeline: poll iCloud -> GPU processing -> YouTube upload.

Downloads iPhone videos from the "tennis_training" iCloud album, processes
them on GPU machines (preprocess, pose extraction, shot detection, clip
extraction), compiles combined normal + 0.25x slow-mo highlights, and
uploads to YouTube.  Supports multiple GPU machines with parallel dispatch.

Usage:
    python scripts/auto_pipeline.py          # daemon mode (polls every 5 min)
    python scripts/auto_pipeline.py --once   # single pass then exit
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load .env before importing settings (so env-based config values work)
_env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

from config.settings import (
    RAW_DIR, HIGHLIGHTS_DIR, ICLOUD, AUTO_PIPELINE, NOTIFICATIONS, PROJECT_ROOT,
)
from scripts.email_notify import (
    notify_processing_started, notify_upload_complete, notify_processing_failed,
    generate_youtube_title,
)
from scripts.video_metadata import get_view_angle_auto

# ── Logging ─────────────────────────────────────────────────

log = logging.getLogger("auto_pipeline")

def setup_logging(debug=False):
    log.setLevel(logging.DEBUG if debug else logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    # Console
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    log.addHandler(ch)
    # File
    fh = logging.FileHandler(os.path.join(PROJECT_ROOT, "pipeline.log"))
    fh.setFormatter(fmt)
    log.addHandler(fh)


# ── State Management ────────────────────────────────────────

def load_state():
    path = AUTO_PIPELINE["state_file"]
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"processed": {}, "daily_counts": {}}


def save_state(state):
    path = AUTO_PIPELINE["state_file"]
    with open(path, "w") as f:
        json.dump(state, f, indent=2)


# ── iCloud Authentication ──────────────────────────────────

def _load_dotenv(path):
    if not os.path.exists(path):
        log.error(".env file not found: %s", path)
        sys.exit(1)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ[key.strip()] = value.strip()


def authenticate():
    from pyicloud import PyiCloudService

    _load_dotenv(ICLOUD["env_file"])
    username = os.environ.get("ICLOUD_USERNAME")
    password = os.environ.get("ICLOUD_PASSWORD")
    if not username or not password:
        log.error("ICLOUD_USERNAME and ICLOUD_PASSWORD must be set in .env")
        sys.exit(1)

    cookie_dir = ICLOUD["cookie_directory"]
    os.makedirs(cookie_dir, exist_ok=True)

    log.info("Authenticating to iCloud as %s...", username)
    api = PyiCloudService(username, password, cookie_directory=cookie_dir)

    if api.requires_2fa:
        if not sys.stdin.isatty():
            log.error("2FA required but running non-interactively. "
                      "Run once from a terminal to refresh the session.")
            return None
        log.info("Two-factor authentication required.")
        for attempt in range(1, 4):
            code = input(f"  Enter 2FA code (attempt {attempt}/3): ").strip()
            if api.validate_2fa_code(code):
                log.info("2FA verified.")
                break
            elif attempt == 3:
                log.error("Failed 2FA after 3 attempts.")
                sys.exit(1)
        if not api.is_trusted_session:
            api.trust_session()

    log.info("iCloud authentication successful.")
    return api


# ── Poll iCloud ────────────────────────────────────────────

def poll_icloud(api, album_name, state):
    """Check the specified iCloud album for new unprocessed videos."""
    try:
        album = api.photos.albums[album_name]
    except KeyError:
        try:
            available = list(api.photos.albums)
        except Exception:
            available = "(unable to list)"
        log.error("Album '%s' not found. Available: %s", album_name, available)
        return []

    new_videos = []
    for asset in album:
        if asset.item_type != "movie":
            continue

        asset_id = str(asset.id)
        if asset_id in state.get("processed", {}):
            continue

        log.info("Found new video: %s", asset.filename)
        new_videos.append(asset)

    return new_videos


def poll_all_albums(api, state):
    """Poll all configured albums. Returns list of (asset, ordering) tuples."""
    albums_config = AUTO_PIPELINE.get("albums", {})
    if not albums_config:
        # Fall back to single album setting
        album_name = AUTO_PIPELINE.get("album", "Tennis Videos")
        albums_config = {album_name: "chronological"}

    all_new = []
    seen_ids = set()
    for album_name, ordering in albums_config.items():
        new_videos = poll_icloud(api, album_name, state)
        for asset in new_videos:
            asset_id = str(asset.id)
            if asset_id not in seen_ids:
                seen_ids.add(asset_id)
                all_new.append((asset, ordering))
                log.info("  Album '%s' -> %s ordering", album_name, ordering)

    return all_new


# ── Download ───────────────────────────────────────────────

def format_size(num_bytes):
    for unit in ("B", "KB", "MB", "GB"):
        if abs(num_bytes) < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"


def download_video(asset, raw_dir):
    """Download original-quality video from iCloud to raw_dir."""
    filename = asset.filename
    dest_path = os.path.join(raw_dir, filename)

    expected_size = asset.versions.get("original", {}).get("size") or asset.size
    if os.path.exists(dest_path) and expected_size:
        if os.path.getsize(dest_path) == expected_size:
            log.info("Skip download (exists): %s", filename)
            return dest_path

    os.makedirs(raw_dir, exist_ok=True)
    part_path = dest_path + ".part"

    for attempt in range(1, ICLOUD["max_retries"] + 1):
        try:
            response = asset.download("original")
            if response is None:
                log.error("No download URL for %s", filename)
                return None

            downloaded = 0
            with open(part_path, "wb") as f:
                if isinstance(response, bytes):
                    f.write(response)
                    downloaded = len(response)
                else:
                    for chunk in response.iter_content(chunk_size=ICLOUD["chunk_size"]):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)

            os.replace(part_path, dest_path)
            log.info("Downloaded %s (%s)", filename, format_size(downloaded))
            return dest_path

        except Exception as e:
            log.warning("Download attempt %d/%d failed for %s: %s",
                        attempt, ICLOUD["max_retries"], filename, e)
            if os.path.exists(part_path):
                os.remove(part_path)
            if attempt < ICLOUD["max_retries"]:
                time.sleep(ICLOUD["retry_delay"])

    log.error("Download failed after %d attempts: %s", ICLOUD["max_retries"], filename)
    return None


# ── SSH/SCP Helpers ────────────────────────────────────────

def _run_ssh(host, cmd, timeout=3600):
    """Run a command on a remote machine via SSH. Returns (success, stdout)."""
    full_cmd = ["ssh", host, cmd]
    log.info("SSH [%s]: %s", host, cmd)
    result = subprocess.run(full_cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        log.error("SSH [%s] failed (rc=%d): %s", host, result.returncode, result.stderr.strip())
        return False, result.stderr
    return True, result.stdout


def _scp_to(host, project, local_path, remote_relative):
    """Copy a file to a remote machine via SCP."""
    remote_path = f"{host}:{project}/{remote_relative}"
    log.info("SCP to %s: %s -> %s", host, os.path.basename(local_path), remote_relative)
    result = subprocess.run(
        ["scp", local_path, remote_path],
        capture_output=True, text=True, timeout=600,
    )
    return result.returncode == 0


def _scp_from(host, project, remote_relative, local_path):
    """Copy a file from a remote machine via SCP."""
    remote_path = f"{host}:{project}/{remote_relative}"
    log.info("SCP from %s: %s -> %s", host, remote_relative, os.path.basename(local_path))
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    result = subprocess.run(
        ["scp", remote_path, local_path],
        capture_output=True, text=True, timeout=600,
    )
    return result.returncode == 0


# ── GPU Processing ─────────────────────────────────────────

def process_on_gpu(filename, machine, ordering="chronological"):
    """Run the full pipeline on a GPU machine via SSH: preprocess, poses, detect, clips."""
    base = os.path.splitext(filename)[0]
    host = machine["host"]
    project = machine["project"]
    py = f"{project}/venv/Scripts/python.exe"

    # 1. Transfer raw video
    local_raw = os.path.join(RAW_DIR, filename)
    if not _scp_to(host, project, local_raw, f"raw/{filename}"):
        log.error("Failed to transfer %s to %s", filename, host)
        return False

    # 2. Preprocess (NVENC, 60fps CFR)
    ok, _ = _run_ssh(
        host,
        f"cd {project} && {py} preprocess_nvenc.py {filename}",
        timeout=1800,
    )
    if not ok:
        log.error("Preprocess failed for %s on %s", filename, host)
        return False

    # 3. Extract poses (large videos can take 90+ min at ~34fps)
    ok, _ = _run_ssh(
        host,
        f"cd {project} && {py} scripts/extract_poses.py preprocessed/{base}.mp4",
        timeout=10800,
    )
    if not ok:
        log.error("Pose extraction failed for %s on %s", filename, host)
        return False

    # 4. Detect shots
    shots_file = f"shots_detected_{base}.json"
    ok, _ = _run_ssh(
        host,
        f"cd {project} && {py} scripts/detect_shots.py poses/{base}.json -o {shots_file}",
        timeout=1800,
    )
    if not ok:
        log.error("Shot detection failed for %s on %s", filename, host)
        return False

    # 5. Extract clips + highlights
    extract_cmd = f"cd {project} && {py} scripts/extract_clips.py -i {shots_file} --highlights"
    if ordering == "type":
        extract_cmd += " --group-by-type"
    ok, _ = _run_ssh(
        host,
        extract_cmd,
        timeout=3600,
    )
    if not ok:
        log.error("Clip extraction failed for %s on %s", filename, host)
        return False

    return True


# ── Combined Video (Normal + Slow-Mo) ─────────────────────

def compile_combined_on_gpu(filename, machine, ordering="chronological"):
    """Compile the combined normal + slow-mo highlight on a GPU machine.

    Runs the slow-mo extraction from the raw 240fps source, then
    concatenates with the normal-speed highlights already produced.
    """
    base = os.path.splitext(filename)[0]
    host = machine["host"]
    project = machine["project"]
    slowmo_factor = AUTO_PIPELINE["slowmo_factor"]
    slowmo_fps = AUTO_PIPELINE["slowmo_output_fps"]
    group_by_type = "True" if ordering == "type" else "False"
    shots_file = f"shots_detected_{base}.json"

    # Build and run a Python one-liner that calls the functions
    # in extract_clips.py to produce slow-mo + combined video
    script = (
        "import json, sys; "
        "sys.path.insert(0, '.'); "
        "from scripts.extract_clips import compile_slowmo_highlights, compile_combined_video; "
        "from config.settings import HIGHLIGHTS_DIR, RAW_DIR; "
        "import os; "
        f"data = json.load(open('{shots_file}')); "
        f"raw_path = os.path.join(RAW_DIR, '{filename}'); "
        f"slowmo_path = os.path.join(HIGHLIGHTS_DIR, '{base}_slowmo_highlights.mp4'); "
        f"normal_path = os.path.join(HIGHLIGHTS_DIR, '{base}_all_highlights.mp4'); "
        f"combined_path = os.path.join(HIGHLIGHTS_DIR, '{base}_combined.mp4'); "
        f"ok1 = compile_slowmo_highlights(raw_path, data['segments'], slowmo_path, {slowmo_factor}, {slowmo_fps}, group_by_type={group_by_type}); "
        "print('Slow-mo:', ok1); "
        "ok2 = compile_combined_video(normal_path, slowmo_path, combined_path) if ok1 else False; "
        "print('Combined:', ok2); "
        "sys.exit(0 if ok2 else 1)"
    )

    py = f"{project}/venv/Scripts/python.exe"
    ok, output = _run_ssh(
        host,
        f'cd {project} && {py} -c "{script}"',
        timeout=3600,
    )
    if not ok:
        # Fall back: if slow-mo fails, just use the normal highlights
        log.warning("Combined video failed on %s, will use normal highlights only for %s", host, base)
        return False

    log.info("Combined video compiled for %s on %s", base, host)
    return True


def transfer_to_mac(filename, machine):
    """Transfer the combined (or fallback normal) highlight from a GPU machine to Mac."""
    base = os.path.splitext(filename)[0]
    host = machine["host"]
    project = machine["project"]
    combined_name = f"{base}_combined.mp4"
    normal_name = f"{base}_all_highlights.mp4"

    local_combined = os.path.join(HIGHLIGHTS_DIR, combined_name)
    os.makedirs(HIGHLIGHTS_DIR, exist_ok=True)

    # Try combined first, fall back to normal highlights
    if _scp_from(host, project, f"highlights/{combined_name}", local_combined):
        return local_combined

    log.warning("Combined not found on %s, trying normal highlights for %s", host, base)
    local_normal = os.path.join(HIGHLIGHTS_DIR, normal_name)
    if _scp_from(host, project, f"highlights/{normal_name}", local_normal):
        return local_normal

    log.error("No highlight video found on %s for %s", host, base)
    return None


# ── YouTube Upload ─────────────────────────────────────────

def upload_to_youtube(video_path, creation_date, video_number, view_angle="back-court", ordering="chronological"):
    """Upload the highlight video to YouTube with formatted title."""
    from scripts.upload import upload_to_youtube as _yt_upload

    date_str = creation_date.strftime("%Y-%m-%d") if creation_date else "unknown"

    # Human-readable labels
    view_labels = {
        "back-court": "Back View",
        "left-side": "Left Side",
        "right-side": "Right Side",
        "front": "Front View",
        "overhead": "Overhead"
    }
    view_label = view_labels.get(view_angle, view_angle.title())
    order_label = "By Shot Type" if ordering == "by_shot_type" else "Chronological"

    # Generate title with view and ordering info
    title = f"Tennis Practice ({date_str}) #{video_number} - {view_label}, {order_label}"

    description = (
        f"Tennis training session highlights - {date_str}\n"
        f"View: {view_label}\n"
        f"Ordering: {order_label}\n"
        f"Normal speed + 0.25x slow motion\n\n"
        f"Auto-generated by tennis_analysis pipeline"
    )

    log.info("Uploading to YouTube: %s", title)
    url = _yt_upload(video_path, title=title, description=description)
    if url:
        log.info("YouTube upload complete: %s", url)
    else:
        log.error("YouTube upload failed for %s", video_path)
    return url


# ── Push Notification (ntfy.sh) ───────────────────────────

def notify_push(message, title=None):
    """Send a push notification via ntfy.sh."""
    topic = NOTIFICATIONS.get("ntfy_topic")
    if not topic:
        log.warning("No ntfy_topic configured, skipping push notification")
        return
    try:
        import urllib.request
        url = f"https://ntfy.sh/{topic}"
        data = message.encode("utf-8")
        req = urllib.request.Request(url, data=data, method="POST")
        if title:
            req.add_header("Title", title)
        urllib.request.urlopen(req, timeout=10)
        log.info("Push notification sent to ntfy.sh/%s", topic)
    except Exception as e:
        log.warning("Failed to send push notification: %s", e)


def send_email(recipients, subject, body):
    """Send an email notification via SMTP (e.g. Gmail)."""
    import smtplib
    from email.mime.text import MIMEText

    cfg = NOTIFICATIONS.get("email", {})
    sender = cfg.get("sender", "")
    password = cfg.get("app_password", "")
    if not sender or not password or not recipients:
        return

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)

    try:
        with smtplib.SMTP(cfg.get("smtp_server", "smtp.gmail.com"),
                          cfg.get("smtp_port", 587)) as server:
            server.starttls()
            server.login(sender, password)
            server.sendmail(sender, recipients, msg.as_string())
        log.info("Email sent to %s", recipients)
    except Exception as e:
        log.warning("Failed to send email: %s", e)


def notify_complete(results):
    """Send notifications (iMessage + email) after processing a batch."""
    lines = ["Tennis highlights ready:"]
    for filename, url in results:
        if url:
            lines.append(f"  {filename} -> {url}")
        else:
            lines.append(f"  {filename} -> FAILED")
    summary = "\n".join(lines)

    # Push notification
    notify_push(summary, title="Tennis highlights ready")

    # Email
    email_cfg = NOTIFICATIONS.get("email", {})
    recipients = email_cfg.get("recipients", [])
    if recipients:
        send_email(recipients, "Tennis highlights ready", summary)


# ── Preflight Checks ──────────────────────────────────────

def preflight():
    """Run all pre-flight checks before starting the pipeline.

    Returns True if all critical checks pass, False otherwise.
    Prints a report of all checks with PASS/FAIL/WARN status.
    """
    setup_logging()
    checks = []  # (name, status, detail)  status: "PASS", "FAIL", "WARN"

    machines = AUTO_PIPELINE.get("gpu_machines", [])

    # ── 1. Config validation ──────────────────────────────
    if not machines:
        checks.append(("Config: gpu_machines", "FAIL", "No GPU machines configured"))
    else:
        hosts = [m["host"] for m in machines]
        checks.append(("Config: gpu_machines", "PASS", f"{len(machines)} machine(s): {hosts}"))

    albums = AUTO_PIPELINE.get("albums", {})
    if not albums and not AUTO_PIPELINE.get("album"):
        checks.append(("Config: albums", "FAIL", "No albums configured"))
    else:
        album_names = list(albums.keys()) if albums else [AUTO_PIPELINE["album"]]
        checks.append(("Config: albums", "PASS",
                       f"{len(album_names)} album(s): {album_names}"))

    # ── 2. iCloud credentials ─────────────────────────────
    env_file = ICLOUD.get("env_file", "")
    if not os.path.exists(env_file):
        checks.append(("iCloud: .env file", "FAIL", f"Not found: {env_file}"))
    else:
        _load_dotenv(env_file)
        user = os.environ.get("ICLOUD_USERNAME")
        pwd = os.environ.get("ICLOUD_PASSWORD")
        if user and pwd:
            checks.append(("iCloud: credentials", "PASS", f"User: {user}"))
        else:
            checks.append(("iCloud: credentials", "FAIL",
                           "ICLOUD_USERNAME or ICLOUD_PASSWORD missing in .env"))

    # ── 3. iCloud authentication ──────────────────────────
    try:
        api = authenticate()
        if api is None:
            checks.append(("iCloud: auth", "FAIL", "Authentication returned None (2FA needed?)"))
        else:
            checks.append(("iCloud: auth", "PASS", "Authenticated"))
            # Check all configured albums exist
            album_names = list(albums.keys()) if albums else [AUTO_PIPELINE.get("album", "Tennis Videos")]
            for aname in album_names:
                try:
                    alb = api.photos.albums[aname]
                    count = sum(1 for a in alb if a.item_type == "movie")
                    ordering = albums.get(aname, "chronological")
                    checks.append((f"iCloud: album '{aname}'", "PASS",
                                   f"{count} video(s), ordering={ordering}"))
                except KeyError:
                    available = list(api.photos.albums)[:10]
                    checks.append((f"iCloud: album '{aname}'", "FAIL",
                                   f"Not found. Available: {available}"))
    except Exception as e:
        checks.append(("iCloud: auth", "FAIL", str(e)))

    # ── 4. SSH connectivity to each GPU machine ───────────
    for machine in machines:
        host = machine["host"]
        project = machine["project"]
        try:
            result = subprocess.run(
                ["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes",
                 host, "echo ok"],
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode == 0:
                checks.append((f"SSH: {host}", "PASS", "Connected"))
            else:
                checks.append((f"SSH: {host}", "FAIL",
                               f"rc={result.returncode}: {result.stderr.strip()}"))
                continue  # skip further checks on this machine
        except subprocess.TimeoutExpired:
            checks.append((f"SSH: {host}", "FAIL", "Connection timed out"))
            continue
        except Exception as e:
            checks.append((f"SSH: {host}", "FAIL", str(e)))
            continue

        py = f"{project}/venv/Scripts/python.exe"

        # ── 4a. Python + venv ─────────────────────────────
        result = subprocess.run(
            ["ssh", host, f"{py} --version"],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0:
            checks.append((f"  {host}: python venv", "PASS",
                           result.stdout.strip()))
        else:
            checks.append((f"  {host}: python venv", "FAIL",
                           f"venv python not found at {py}"))

        # ── 4b. FFmpeg ────────────────────────────────────
        result = subprocess.run(
            ["ssh", host, "ffmpeg -version"],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0:
            ver = result.stdout.split("\n")[0] if result.stdout else "unknown"
            checks.append((f"  {host}: ffmpeg", "PASS", ver))
        else:
            checks.append((f"  {host}: ffmpeg", "FAIL", "ffmpeg not found"))

        # ── 4c. Key Python packages ──────────────────────
        result = subprocess.run(
            ["ssh", host,
             f'{py} -c "import mediapipe, tensorflow; '
             f'print(mediapipe.__version__, tensorflow.__version__)"'],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            checks.append((f"  {host}: mediapipe+tf", "PASS",
                           result.stdout.strip()))
        else:
            checks.append((f"  {host}: mediapipe+tf", "FAIL",
                           result.stderr.strip()[:100]))

        # ── 4d. Model file ───────────────────────────────
        result = subprocess.run(
            ["ssh", host,
             f'cd {project} && {py} -c "'
             f'import os; '
             f'h5 = [f for f in os.listdir(\\"models\\") if f.endswith(\\".h5\\")]; '
             f'print(\\", \\".join(h5) if h5 else \\"NONE\\")"'],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0:
            models = result.stdout.strip()
            if models and models != "NONE":
                checks.append((f"  {host}: model files", "PASS", models))
            else:
                checks.append((f"  {host}: model files", "FAIL",
                               "No .h5 model files in models/"))
        else:
            checks.append((f"  {host}: model files", "FAIL",
                           "Could not check models/ dir"))

        # ── 4e. Disk space ───────────────────────────────
        result = subprocess.run(
            ["ssh", host,
             f'{py} -c "'
             f'import shutil; u = shutil.disk_usage(\\"C:/\\"); '
             f'free_gb = u.free / (1024**3); '
             f'print(f\\"{{free_gb:.1f}} GB free\\")"'],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0:
            info = result.stdout.strip()
            try:
                free_gb = float(info.split()[0])
                status = "PASS" if free_gb > 20 else "WARN" if free_gb > 5 else "FAIL"
            except (ValueError, IndexError):
                status, free_gb = "WARN", 0
            checks.append((f"  {host}: disk space", status, info))
        else:
            checks.append((f"  {host}: disk space", "WARN", "Could not check"))

        # ── 4f. NVENC ────────────────────────────────────
        result = subprocess.run(
            ["ssh", host,
             'ffmpeg -hide_banner -y -f lavfi -i nullsrc=s=64x64:d=0.1 '
             '-c:v h264_nvenc -f null -'],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0:
            checks.append((f"  {host}: NVENC", "PASS", "h264_nvenc working"))
        else:
            checks.append((f"  {host}: NVENC", "WARN",
                           "h264_nvenc failed (will use libx264 fallback)"))

    # ── 5. Mac disk space ─────────────────────────────────
    import shutil
    usage = shutil.disk_usage(PROJECT_ROOT)
    free_gb = usage.free / (1024**3)
    status = "PASS" if free_gb > 50 else "WARN" if free_gb > 10 else "FAIL"
    checks.append(("Mac: disk space", status, f"{free_gb:.1f} GB free"))

    # ── 6. YouTube credentials ────────────────────────────
    secrets_path = os.path.join(PROJECT_ROOT, "config", "client_secrets.json")
    creds_path = os.path.join(PROJECT_ROOT, "config", "youtube_credentials.json")

    if not os.path.exists(secrets_path):
        checks.append(("YouTube: client_secrets.json", "FAIL", "Not found"))
    else:
        checks.append(("YouTube: client_secrets.json", "PASS", "Found"))

    if not os.path.exists(creds_path):
        checks.append(("YouTube: credentials", "FAIL",
                       "youtube_credentials.json not found (run upload.py once manually)"))
    else:
        try:
            import google.oauth2.credentials
            with open(creds_path) as f:
                cred_data = json.load(f)
            if cred_data.get("token") and cred_data.get("refresh_token"):
                checks.append(("YouTube: credentials", "PASS",
                               "Token + refresh token present"))
            else:
                checks.append(("YouTube: credentials", "WARN",
                               "Credentials file exists but may be incomplete"))
        except Exception as e:
            checks.append(("YouTube: credentials", "WARN", f"Could not validate: {e}"))

    # ── 7. Notifications ──────────────────────────────────
    imessage_phones = NOTIFICATIONS.get("imessage", [])
    if imessage_phones:
        checks.append(("Notify: iMessage", "PASS",
                       f"{len(imessage_phones)} recipient(s)"))
    else:
        checks.append(("Notify: iMessage", "WARN", "No recipients configured"))

    email_cfg = NOTIFICATIONS.get("email", {})
    email_recipients = email_cfg.get("recipients", [])
    email_sender = email_cfg.get("sender", "")
    email_password = email_cfg.get("app_password", "")
    if email_recipients and email_sender and email_password:
        # Try SMTP login
        try:
            import smtplib
            with smtplib.SMTP(email_cfg.get("smtp_server", "smtp.gmail.com"),
                              email_cfg.get("smtp_port", 587), timeout=10) as server:
                server.starttls()
                server.login(email_sender, email_password)
            checks.append(("Notify: email SMTP", "PASS",
                           f"Login OK as {email_sender} -> {email_recipients}"))
        except Exception as e:
            checks.append(("Notify: email SMTP", "FAIL", f"Login failed: {e}"))
    elif email_sender and email_password:
        checks.append(("Notify: email", "WARN",
                       "SMTP configured but no recipients"))
    else:
        checks.append(("Notify: email", "WARN",
                       "Not configured (sender/app_password empty)"))

    # ── Print report ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PREFLIGHT CHECK REPORT")
    print("=" * 60)

    pass_count = sum(1 for _, s, _ in checks if s == "PASS")
    warn_count = sum(1 for _, s, _ in checks if s == "WARN")
    fail_count = sum(1 for _, s, _ in checks if s == "FAIL")

    for name, status, detail in checks:
        icon = {"PASS": "+", "FAIL": "X", "WARN": "!"}[status]
        print(f"  [{icon}] {name}: {detail}")

    print("=" * 60)
    print(f"  {pass_count} passed, {warn_count} warnings, {fail_count} failed")
    if fail_count == 0:
        print("  Pipeline is ready to run.")
    else:
        print("  Fix FAIL items before running the pipeline.")
    print("=" * 60 + "\n")

    return fail_count == 0


# ── Main Loop ──────────────────────────────────────────────

def process_single_video(asset, state, machine, ordering="chronological"):
    """Process one video through the full pipeline. Returns youtube URL or None."""
    filename = asset.filename
    base = os.path.splitext(filename)[0]
    asset_id = str(asset.id)
    host = machine["host"]

    log.info("=" * 60)
    log.info("Processing: %s on %s (ordering=%s)", filename, host, ordering)
    log.info("=" * 60)

    notify_push(f"New tennis video detected: {filename}\nProcessing on {host} ({ordering} order)")

    # 1. Download from iCloud
    local_path = download_video(asset, RAW_DIR)
    if not local_path:
        return None

    # Get view angle (will be auto-detected after poses are extracted)
    view_angle = "back-court"  # Default, will be updated after pose extraction

    # Send email notification for processing start
    order_label = "by_shot_type" if ordering == "group_by_type" else "chronological"
    notify_processing_started(filename, host, order_label, view_angle)

    # 2. GPU pipeline: preprocess -> poses -> detect -> clips
    if not process_on_gpu(filename, machine, ordering=ordering):
        return None

    # 3. Compile combined normal + slow-mo highlight on GPU
    compile_combined_on_gpu(filename, machine, ordering=ordering)

    # 4. Transfer highlight back to Mac
    highlight_path = transfer_to_mac(filename, machine)
    if not highlight_path:
        return None

    # 5. Upload to YouTube
    creation_date = asset.asset_date if hasattr(asset, "asset_date") else asset.created
    date_str = creation_date.strftime("%Y-%m-%d") if creation_date else "unknown"
    daily_counts = state.get("daily_counts", {})
    count = daily_counts.get(date_str, 0) + 1

    # Get actual view angle after pose extraction
    view_angle = get_view_angle_auto(base)
    youtube_url = upload_to_youtube(highlight_path, creation_date, count, view_angle, order_label)

    if youtube_url:
        notify_push(f"Tennis highlights uploaded: {filename}\n{youtube_url}")
        # Send email with stats
        stats = {
            "shot_counts": {},  # TODO: load from shots_detected.json
            "total_clips": 0,
            "duration_seconds": 0,
        }
        # Try to load stats from detection file
        detect_file = os.path.join(PROJECT_ROOT, f"shots_detected_{base}.json")
        if os.path.exists(detect_file):
            try:
                with open(detect_file) as f:
                    detect_data = json.load(f)
                segments = detect_data.get("segments", [])
                for seg in segments:
                    st = seg.get("shot_type", "unknown")
                    if st != "neutral":
                        stats["shot_counts"][st] = stats["shot_counts"].get(st, 0) + 1
                stats["total_clips"] = sum(stats["shot_counts"].values())
            except:
                pass
        notify_upload_complete(filename, youtube_url, stats, view_angle, order_label)
    else:
        notify_push(f"Tennis highlights upload failed: {filename}")
        notify_processing_failed(filename, host, "YouTube upload failed")

    # 6. Update state
    state.setdefault("processed", {})[asset_id] = {
        "filename": filename,
        "date": date_str,
        "youtube_url": youtube_url,
        "processed_at": datetime.now().isoformat(),
        "machine": host,
    }
    state.setdefault("daily_counts", {})[date_str] = count
    save_state(state)

    log.info("Finished processing %s on %s", filename, host)
    return youtube_url


def main_loop(once=False, debug=False):
    """Main daemon loop: poll iCloud, process new videos, upload."""
    setup_logging(debug=debug)
    albums = AUTO_PIPELINE.get("albums", {})
    album_names = list(albums.keys()) if albums else [AUTO_PIPELINE.get("album", "Tennis Videos")]
    poll_interval = AUTO_PIPELINE["poll_interval"]
    machines = AUTO_PIPELINE["gpu_machines"]

    log.info("Auto pipeline started (albums=%s, poll=%ds, once=%s, machines=%s)",
             album_names, poll_interval, once,
             [m["host"] for m in machines])

    api = authenticate()
    state = load_state()

    while True:
        try:
            new_videos = poll_all_albums(api, state)

            if new_videos:
                log.info("Found %d new video(s) to process across %d machine(s)",
                         len(new_videos), len(machines))

                results = []  # (filename, url_or_None)

                if len(new_videos) > 1 and len(machines) > 1:
                    # Parallel: distribute videos across machines
                    log.info("Dispatching %d videos in parallel", len(new_videos))
                    with ThreadPoolExecutor(max_workers=len(machines)) as pool:
                        futures = {}
                        for i, (asset, ordering) in enumerate(new_videos):
                            machine = machines[i % len(machines)]
                            fut = pool.submit(process_single_video, asset, state,
                                              machine, ordering=ordering)
                            futures[fut] = (asset, machine)
                        for fut in as_completed(futures):
                            asset, machine = futures[fut]
                            try:
                                url = fut.result()
                                results.append((asset.filename, url))
                                if url:
                                    log.info("Completed %s on %s -> %s",
                                             asset.filename, machine["host"], url)
                            except Exception as e:
                                results.append((asset.filename, None))
                                log.error("Failed to process %s on %s: %s",
                                          asset.filename, machine["host"], e,
                                          exc_info=True)
                else:
                    # Sequential: single video or single machine
                    for i, (asset, ordering) in enumerate(new_videos):
                        machine = machines[i % len(machines)]
                        try:
                            url = process_single_video(asset, state, machine,
                                                       ordering=ordering)
                            results.append((asset.filename, url))
                        except Exception as e:
                            results.append((asset.filename, None))
                            log.error("Failed to process %s on %s: %s",
                                      asset.filename, machine["host"], e,
                                      exc_info=True)

                notify_complete(results)
            else:
                log.info("No new videos found.")

        except Exception as e:
            log.error("Poll cycle error: %s", e, exc_info=True)

        if once:
            log.info("Single pass complete, exiting.")
            break

        log.info("Sleeping %d seconds until next poll...", poll_interval)
        time.sleep(poll_interval)


# ── CLI ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Automated tennis pipeline: iCloud -> GPU -> YouTube",
    )
    parser.add_argument("--once", action="store_true",
                        help="Single pass: process any new videos and exit")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging (dumps iCloud field names)")
    parser.add_argument("--preflight", action="store_true",
                        help="Run pre-flight checks and exit (no processing)")
    args = parser.parse_args()

    if args.preflight:
        ok = preflight()
        sys.exit(0 if ok else 1)

    main_loop(once=args.once, debug=args.debug)


if __name__ == "__main__":
    main()
