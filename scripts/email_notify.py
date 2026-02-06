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
1. Preprocess (VFR → 60fps CFR)
2. Extract poses (MediaPipe)
3. Detect shots (GRU model)
4. Extract clips
5. Compile highlights
6. Upload to YouTube

You'll receive another email when processing completes.
"""

    html_body = f"""
<html>
<body style="font-family: Arial, sans-serif; max-width: 600px;">
<h2 style="color: #2e7d32;">🎾 Tennis Video Processing Started</h2>

<table style="border-collapse: collapse; width: 100%;">
<tr><td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>Video</strong></td>
    <td style="padding: 8px; border-bottom: 1px solid #ddd;">{video_name}</td></tr>
<tr><td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>Time</strong></td>
    <td style="padding: 8px; border-bottom: 1px solid #ddd;">{timestamp}</td></tr>
<tr><td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>Machine</strong></td>
    <td style="padding: 8px; border-bottom: 1px solid #ddd;">{host}</td></tr>
<tr><td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>View Angle</strong></td>
    <td style="padding: 8px; border-bottom: 1px solid #ddd;">{view_angle}</td></tr>
<tr><td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>Clip Ordering</strong></td>
    <td style="padding: 8px; border-bottom: 1px solid #ddd;">{ordering}</td></tr>
</table>

<h3 style="color: #666;">Pipeline Stages:</h3>
<ol>
<li>Preprocess (VFR → 60fps CFR)</li>
<li>Extract poses (MediaPipe)</li>
<li>Detect shots (GRU model)</li>
<li>Extract clips</li>
<li>Compile highlights</li>
<li>Upload to YouTube</li>
</ol>

<p style="color: #888;">You'll receive another email when processing completes.</p>
</body>
</html>
"""

    return send_email(subject, body, html_body)


def notify_upload_complete(video_name: str, youtube_url: str, stats: dict,
                           view_angle: str, ordering: str):
    """Send email when video is uploaded to YouTube.

    Args:
        video_name: Name of the video file
        youtube_url: YouTube URL of uploaded video
        stats: Dict with shot counts, duration, etc.
        view_angle: Camera view angle used
        ordering: Clip ordering used
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Build stats summary
    shot_counts = stats.get("shot_counts", {})
    total_clips = stats.get("total_clips", sum(shot_counts.values()))
    duration = stats.get("duration_seconds", 0)
    duration_str = f"{int(duration // 60)}m {int(duration % 60)}s" if duration else "Unknown"

    shots_summary = ", ".join(f"{k}: {v}" for k, v in shot_counts.items()) or "N/A"

    # Format video title for YouTube
    view_label = {
        "back-court": "Back View",
        "left-side": "Left Side",
        "right-side": "Right Side",
        "front": "Front View",
        "overhead": "Overhead"
    }.get(view_angle, view_angle)

    order_label = "Grouped by Shot" if ordering == "by_shot_type" else "Chronological"

    subject = f"✅ Uploaded: {video_name} ({view_label}, {order_label})"

    body = f"""Tennis Highlights Uploaded to YouTube

Video: {video_name}
YouTube: {youtube_url}
Time: {timestamp}

Details:
- View Angle: {view_label}
- Ordering: {order_label}
- Total Clips: {total_clips}
- Shots: {shots_summary}
- Duration: {duration_str}

Watch now: {youtube_url}
"""

    html_body = f"""
<html>
<body style="font-family: Arial, sans-serif; max-width: 600px;">
<h2 style="color: #2e7d32;">✅ Tennis Highlights Uploaded!</h2>

<p style="font-size: 18px;">
<a href="{youtube_url}" style="color: #1976d2; text-decoration: none;">
▶️ Watch on YouTube
</a>
</p>

<table style="border-collapse: collapse; width: 100%;">
<tr><td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>Video</strong></td>
    <td style="padding: 8px; border-bottom: 1px solid #ddd;">{video_name}</td></tr>
<tr><td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>View Angle</strong></td>
    <td style="padding: 8px; border-bottom: 1px solid #ddd;">{view_label}</td></tr>
<tr><td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>Ordering</strong></td>
    <td style="padding: 8px; border-bottom: 1px solid #ddd;">{order_label}</td></tr>
<tr><td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>Total Clips</strong></td>
    <td style="padding: 8px; border-bottom: 1px solid #ddd;">{total_clips}</td></tr>
<tr><td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>Shots</strong></td>
    <td style="padding: 8px; border-bottom: 1px solid #ddd;">{shots_summary}</td></tr>
<tr><td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>Duration</strong></td>
    <td style="padding: 8px; border-bottom: 1px solid #ddd;">{duration_str}</td></tr>
</table>

<p style="margin-top: 20px;">
<a href="{youtube_url}" style="background: #c62828; color: white; padding: 12px 24px;
   text-decoration: none; border-radius: 4px; display: inline-block;">
Watch on YouTube
</a>
</p>

</body>
</html>
"""

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
