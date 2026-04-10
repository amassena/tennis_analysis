#!/usr/bin/env python3
"""Email notifications for tennis pipeline events."""

import os
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

# Load .env file if present
from pathlib import Path
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())

# Email configuration - uses existing .env variables
EMAIL_CONFIG = {
    "smtp_server": os.environ.get("SMTP_SERVER", "smtp.gmail.com"),
    "smtp_port": int(os.environ.get("SMTP_PORT", "587")),
    "sender_email": os.environ.get("GMAIL_SENDER", os.environ.get("SENDER_EMAIL", "")),
    "sender_password": os.environ.get("GMAIL_APP_PASSWORD", os.environ.get("SENDER_PASSWORD", "")),
    "recipient_email": os.environ.get("RECIPIENT_EMAIL", os.environ.get("GMAIL_SENDER", "")),  # Default: send to self
}


def send_sms(message: str) -> bool:
    """Send an SMS via email-to-SMS gateway.

    Args:
        message: Short text message (SMS has 160 char limit)

    Returns:
        True if sent successfully, False otherwise
    """
    sms_email = os.environ.get("SMS_EMAIL", "")
    if not sms_email:
        print("[SMS] Not configured - set SMS_EMAIL in .env")
        return False

    config = EMAIL_CONFIG
    if not config["sender_email"] or not config["sender_password"]:
        print("[SMS] Email not configured")
        return False

    try:
        # SMS via email gateway - no subject, plain text only
        msg = MIMEText(message[:160], "plain")  # SMS limit
        msg["From"] = config["sender_email"]
        msg["To"] = sms_email

        context = ssl.create_default_context()
        with smtplib.SMTP(config["smtp_server"], config["smtp_port"]) as server:
            server.starttls(context=context)
            server.login(config["sender_email"], config["sender_password"])
            server.sendmail(config["sender_email"], sms_email, msg.as_string())

        print(f"[SMS] Sent: {message[:50]}...")
        return True

    except Exception as e:
        print(f"[SMS] Failed: {e}")
        return False


def send_email(subject: str, body: str, html_body: str = None) -> bool:
    """Send an email notification.

    Args:
        subject: Email subject line
        body: Plain text body
        html_body: Optional HTML body for rich formatting

    Returns:
        True if sent successfully, False otherwise
    """
    config = EMAIL_CONFIG

    if not config["sender_email"] or not config["sender_password"]:
        print("[EMAIL] Not configured - set SENDER_EMAIL and SENDER_PASSWORD")
        return False

    if not config["recipient_email"]:
        print("[EMAIL] No recipient - set RECIPIENT_EMAIL")
        return False

    try:
        # Create message
        if html_body:
            msg = MIMEMultipart("alternative")
            msg.attach(MIMEText(body, "plain"))
            msg.attach(MIMEText(html_body, "html"))
        else:
            msg = MIMEText(body, "plain")

        msg["Subject"] = subject
        msg["From"] = config["sender_email"]
        msg["To"] = config["recipient_email"]

        # Connect and send
        context = ssl.create_default_context()
        with smtplib.SMTP(config["smtp_server"], config["smtp_port"]) as server:
            server.starttls(context=context)
            server.login(config["sender_email"], config["sender_password"])
            server.sendmail(
                config["sender_email"],
                config["recipient_email"],
                msg.as_string()
            )

        print(f"[EMAIL] Sent: {subject}")
        return True

    except Exception as e:
        print(f"[EMAIL] Failed to send: {e}")
        return False


def sms_processing_started(video_name: str, host: str):
    """Send SMS when processing starts."""
    msg = f"Tennis: Processing {video_name} on {host}"
    return send_sms(msg)


def sms_upload_complete(video_name: str, highlights_url: str):
    """Send SMS when upload completes."""
    short_name = video_name.replace("IMG_", "").replace(".mov", "").replace(".MOV", "")
    msg = f"Tennis ready: {short_name} - {highlights_url}"
    return send_sms(msg)


def sms_videos_detected(videos_info: list):
    """Send SMS when new videos are detected with queue info.

    Args:
        videos_info: List of (video_name, machine_host, ordering) tuples
    """
    if len(videos_info) == 1:
        name, host, _ = videos_info[0]
        short = name.replace("IMG_", "").replace(".MOV", "").replace(".mov", "")
        msg = f"Tennis: {short} detected, processing on {host}"
    else:
        names = [v[0].replace("IMG_", "").replace(".MOV", "").replace(".mov", "") for v in videos_info]
        msg = f"Tennis: {len(videos_info)} videos detected: {', '.join(names)}"
    return send_sms(msg)


def notify_videos_detected(videos_info: list):
    """Send email/SMS when new videos are detected, showing queue status.

    Args:
        videos_info: List of (video_name, machine_host, ordering) tuples
    """
    if not videos_info:
        return

    # Send SMS first
    sms_videos_detected(videos_info)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    if len(videos_info) == 1:
        name, host, ordering = videos_info[0]
        subject = f"🎾 Video detected: {name}"
        body = f"""Tennis Video Detected

Video: {name}
Time: {timestamp}
Machine: {host}
Ordering: {ordering}

Processing will begin shortly.
"""
    else:
        subject = f"🎾 {len(videos_info)} videos detected"
        lines = []
        for i, (name, host, ordering) in enumerate(videos_info, 1):
            lines.append(f"  {i}. {name} -> {host} ({ordering})")

        body = f"""Tennis Videos Detected

Time: {timestamp}
Queue ({len(videos_info)} videos):
{chr(10).join(lines)}

Videos will be processed in parallel across machines.
"""

    return send_email(subject, body)


def notify_processing_started(video_name: str, host: str, ordering: str, view_angle: str):
    """Send email when video processing starts.

    Args:
        video_name: Name of the video file
        host: GPU machine processing the video
        ordering: 'chronological' or 'by_shot_type'
        view_angle: 'back-court', 'left-side', 'right-side', etc.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    subject = f"🎾 Processing started: {video_name}"

    body = f"""Tennis Video Processing Started

Video: {video_name}
Time: {timestamp}
Machine: {host}
View Angle: {view_angle}
Clip Ordering: {ordering}

Pipeline stages:
1. Preprocess (VFR -> 60fps CFR)
2. Extract poses (MediaPipe)
3. Detect shots (CNN model)
4. Export highlight videos
5. Upload to Playful Life

You'll receive another email when processing completes.
"""

    html_body = f"""
<html>
<body style="font-family: Arial, sans-serif; max-width: 600px;">
<h2 style="color: #2e7d32;">Tennis Video Processing Started</h2>

<table style="border-collapse: collapse; width: 100%;">
<tr><td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>Video</strong></td>
    <td style="padding: 8px; border-bottom: 1px solid #ddd;">{video_name}</td></tr>
<tr><td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>Time</strong></td>
    <td style="padding: 8px; border-bottom: 1px solid #ddd;">{timestamp}</td></tr>
<tr><td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>Machine</strong></td>
    <td style="padding: 8px; border-bottom: 1px solid #ddd;">{host}</td></tr>
</table>

<p style="color: #888; margin-top: 16px;">You'll receive another email when processing completes.</p>
</body>
</html>
"""

    # Send both email and SMS
    sms_processing_started(video_name, host)
    return send_email(subject, body, html_body)


def notify_upload_complete(video_name: str, video_links: dict, stats: dict):
    """Send email when video highlights are uploaded.

    Args:
        video_name: Name of the video file
        video_links: Dict of {label: url} for each video type
        stats: Dict with shot counts, duration, etc.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    shot_counts = stats.get("shot_counts", {})
    total_clips = stats.get("total_clips", sum(shot_counts.values()))
    duration = stats.get("duration_seconds", 0)
    duration_str = f"{int(duration // 60)}m {int(duration % 60)}s" if duration else "Unknown"

    shots_summary = ", ".join(f"{k}: {v}" for k, v in shot_counts.items()) or "N/A"

    subject = f"Tennis: {video_name} ready ({total_clips} shots)"

    # Plain text body
    links_text = "\n".join(f"  {label}: {url}" for label, url in video_links.items())
    body = f"""{video_name} - {total_clips} shots ({shots_summary})
Duration: {duration_str}

{links_text}
"""

    # HTML body with styled link buttons
    colors = {
        "Timeline": "#FF8C00",
        "Rally": "#9b59b6",
        "Rally (Slow-Mo)": "#8e44ad",
        "Grouped": "#27ae60",
        "Grouped (Slow-Mo)": "#219a52",
    }

    link_buttons = ""
    for label, url in video_links.items():
        color = colors.get(label, "#555")
        link_buttons += (
            f'<a href="{url}" style="background: {color}; color: white; padding: 10px 20px; '
            f'text-decoration: none; border-radius: 8px; display: inline-block; '
            f'font-weight: 600; margin: 4px 4px 4px 0; font-size: 14px;">{label}</a>\n'
        )

    html_body = f"""
<html>
<body style="font-family: -apple-system, system-ui, sans-serif; max-width: 600px; background: #f5f5f5; padding: 20px;">
<div style="background: white; border-radius: 12px; padding: 24px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
<h2 style="color: #333; margin-top: 0;">{video_name}</h2>
<p style="color: #666; margin: 4px 0;">{total_clips} shots &mdash; {shots_summary}</p>
<p style="color: #999; margin: 4px 0; font-size: 13px;">Duration: {duration_str}</p>

<div style="margin-top: 20px;">
{link_buttons}
</div>
</div>
</body>
</html>
"""

    # SMS with just the gallery link
    gallery_url = "https://tennis.playfullife.com/"
    sms_upload_complete(video_name, gallery_url)
    return send_email(subject, body, html_body)


def notify_processing_failed(video_name: str, host: str, error: str):
    """Send email when processing fails."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    subject = f"❌ Processing failed: {video_name}"

    body = f"""Tennis Video Processing Failed

Video: {video_name}
Time: {timestamp}
Machine: {host}

Error:
{error}

Please check the logs for more details.
"""

    return send_email(subject, body)


def generate_youtube_title(video_name: str, view_angle: str, ordering: str,
                           date: datetime = None) -> str:
    """Generate descriptive YouTube title.

    Args:
        video_name: Original video filename
        view_angle: Camera position
        ordering: Clip ordering style
        date: Recording date (default: today)

    Returns:
        Title like "Tennis Practice (2026-02-06) - Back View, By Shot Type"
    """
    if date is None:
        date = datetime.now()

    date_str = date.strftime("%Y-%m-%d")

    view_labels = {
        "back-court": "Back View",
        "left-side": "Left Side",
        "right-side": "Right Side",
        "front": "Front View",
        "overhead": "Overhead"
    }
    view_label = view_labels.get(view_angle, view_angle.title())

    order_label = "By Shot Type" if ordering == "by_shot_type" else "Chronological"

    return f"Tennis Practice ({date_str}) - {view_label}, {order_label}"


if __name__ == "__main__":
    # Test email configuration
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        print("Testing email configuration...")
        print(f"  SMTP: {EMAIL_CONFIG['smtp_server']}:{EMAIL_CONFIG['smtp_port']}")
        print(f"  From: {EMAIL_CONFIG['sender_email'] or '(not set)'}")
        print(f"  To: {EMAIL_CONFIG['recipient_email'] or '(not set)'}")

        if send_email(
            "🎾 Tennis Pipeline Test",
            "This is a test email from the tennis analysis pipeline.",
            "<h1>Test Email</h1><p>If you see this, email notifications are working!</p>"
        ):
            print("\n✅ Test email sent successfully!")
        else:
            print("\n❌ Failed to send test email")
    else:
        print("Usage: python email_notify.py test")
        print("\nRequired environment variables:")
        print("  SENDER_EMAIL - Gmail address to send from")
        print("  SENDER_PASSWORD - Gmail app password (not your regular password)")
        print("  RECIPIENT_EMAIL - Email address to receive notifications")
        print("\nFor Gmail, create an App Password at:")
        print("  https://myaccount.google.com/apppasswords")
