#!/usr/bin/env python3
"""Generate Claude-authored coaching summary for a tennis video.

Reads detection JSON and biomechanical analysis, builds a metrics table,
sends it to Claude with a strict JSON response schema, saves the output
to R2 at highlights/{video_id}/coaching.json.

Usage:
    python scripts/claude_coach.py IMG_1136
    python scripts/claude_coach.py IMG_1136 --upload
    python scripts/claude_coach.py --all --upload   # all videos with biomech
"""

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT_ROOT = Path(__file__).parent.parent
DETECTIONS_DIR = PROJECT_ROOT / "detections"
ANALYSIS_DIR = PROJECT_ROOT / "analysis"

MODEL = "claude-opus-4-7"  # latest Opus — step-change reasoning (4.7 exists only for Opus)
MAX_TOKENS = 1800

SYSTEM_PROMPT = """You are an experienced tennis coach analyzing biomechanical metrics \
and a per-shot timeline from a recreational player's session. Give specific, \
actionable feedback grounded in the numbers. Reference metric values directly \
("your average knee bend was 128° — pros average 110-120°"). Avoid generic advice.

For EACH strength and work_on point, pick 1-3 specific example shots from the \
per-shot timeline that best illustrate the point, and include their timestamps \
so the viewer can jump to them in the video.

Respond ONLY with valid JSON matching this schema:
{
  "headline": "One-sentence summary (<100 chars) of the session's theme",
  "strengths": [
    {
      "point": "short label (3-6 words)",
      "detail": "1-2 sentence explanation referencing metrics",
      "examples": [
        {"t": 45.2, "type": "forehand", "note": "best knee drive"}
      ]
    }
  ],
  "work_on": [
    {
      "point": "short label",
      "detail": "1-2 sentence actionable suggestion referencing metrics",
      "examples": [
        {"t": 120.5, "type": "backhand", "note": "rushed recovery"}
      ]
    }
  ],
  "drill": "One concrete drill for the next practice (2-3 sentences)"
}

Rules:
- "t" is the shot timestamp in seconds (use the exact value from the timeline).
- "type" is the shot type from the timeline.
- "note" is a very short phrase (<8 words) describing what the shot exemplifies.
- Give 2-3 strengths and 2-3 work_on items.
- Be concise and specific. No platitudes."""


def load_metrics(vid: str) -> dict | None:
    """Gather detection + biomech data for a video."""
    metrics = {"video_id": vid}

    # Detection JSON
    det = None
    for name in [f"{vid}_fused.json", f"{vid}_fused_detections.json"]:
        p = DETECTIONS_DIR / name
        if p.exists():
            with open(p) as f:
                det = json.load(f)
            break
    if not det:
        print(f"No detection JSON for {vid}", file=sys.stderr)
        return None

    dets = det.get("detections", [])
    metrics["duration_seconds"] = round(det.get("duration", 0), 1)
    metrics["total_shots"] = len(dets)

    # Per-shot timeline (up to 50 shots evenly distributed — keeps prompt small)
    timeline = []
    step = max(1, len(dets) // 50)
    for i in range(0, len(dets), step):
        d = dets[i]
        t = d.get("timestamp") or d.get("contact_time") or d.get("peak_time") or d.get("time") or d.get("t")
        if t is None and d.get("frame") is not None:
            t = d["frame"] / 60.0
        if t is None and d.get("contact_frame") is not None:
            t = d["contact_frame"] / 60.0
        if t is None:
            continue
        timeline.append({
            "t": round(float(t), 1),
            "type": d.get("shot_type", "unknown"),
        })
    metrics["per_shot_timeline"] = timeline

    # Shot breakdown
    breakdown = {}
    for d in dets:
        st = d.get("shot_type", "unknown")
        breakdown[st] = breakdown.get(st, 0) + 1
    metrics["shot_breakdown"] = breakdown

    # Biomech
    bp = ANALYSIS_DIR / f"{vid}_biomech.json"
    if bp.exists():
        with open(bp) as f:
            biomech = json.load(f)
        metrics["dominant_hand"] = biomech.get("dominant_hand", "unknown")
        metrics["per_type_biomech"] = biomech.get("type_summaries", {})
        fi = biomech.get("fatigue_indicator", {})
        if fi:
            metrics["speed_decline_pct"] = fi.get("speed_decline_pct")
    else:
        metrics["per_type_biomech"] = None

    return metrics


def format_user_message(metrics: dict) -> str:
    """Render metrics as a compact markdown summary for Claude."""
    lines = [
        f"# Session metrics — {metrics['video_id']}",
        "",
        f"- Duration: {metrics['duration_seconds']}s",
        f"- Total shots: {metrics['total_shots']}",
        f"- Shot breakdown: {metrics.get('shot_breakdown', {})}",
    ]
    if metrics.get("dominant_hand"):
        lines.append(f"- Dominant hand: {metrics['dominant_hand']}")
    if metrics.get("speed_decline_pct") is not None:
        lines.append(f"- Speed decline over session: {metrics['speed_decline_pct']}%")
    lines.append("")

    per_type = metrics.get("per_type_biomech")
    if per_type:
        lines.append("## Per-shot-type biomechanics")
        lines.append("")
        lines.append("| Shot | Count | Peak Swing Speed | Knee Bend | Trunk Rotation | Arm Extension | Recovery ms | Kinetic Chain % |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for st, s in per_type.items():
            lines.append(
                f"| {st} | {s.get('count','?')} | "
                f"{s.get('avg_peak_swing_speed','?')} | "
                f"{s.get('avg_knee_bend_depth','?')}° | "
                f"{s.get('avg_trunk_rotation','?')}° | "
                f"{s.get('avg_arm_extension','?')}° | "
                f"{s.get('avg_recovery_time_ms','?')} | "
                f"{s.get('kinetic_chain_correct_pct','?')}% |"
            )
        lines.append("")
        lines.append("### Reference ranges (recreational → advanced)")
        lines.append("- Knee bend at contact: 110-130° (deeper = more power transfer)")
        lines.append("- Trunk rotation: 30-60° (more = more racket head speed)")
        lines.append("- Arm extension at contact: 155-175° (straighter = better)")
        lines.append("- Kinetic chain correct: >80% is solid")
    else:
        lines.append("(No per-shot biomechanical data — analyze from shot breakdown only)")

    # Per-shot timeline
    timeline = metrics.get("per_shot_timeline") or []
    if timeline:
        lines.append("")
        lines.append("## Per-shot timeline (timestamp in seconds, shot type)")
        lines.append("```")
        for s in timeline:
            lines.append(f"t={s['t']:7.1f}s  {s['type']}")
        lines.append("```")
        lines.append("")
        lines.append("When you reference example shots in your feedback, pick timestamps "
                     "from this list that best illustrate the point.")

    return "\n".join(lines)


def parse_claude_response(text: str) -> dict | None:
    """Extract JSON from Claude's response (direct / fenced / regex fallback)."""
    text = text.strip()
    # Direct
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Fenced
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # Loose object match
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


def generate_coaching(vid: str) -> dict | None:
    """Call Claude to produce a coaching summary for a video."""
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set in environment", file=sys.stderr)
        return None

    metrics = load_metrics(vid)
    if not metrics:
        return None
    if metrics["total_shots"] < 5:
        print(f"Skipping {vid}: only {metrics['total_shots']} shots (need >=5)")
        return None

    user_msg = format_user_message(metrics)
    client = anthropic.Anthropic(api_key=api_key)

    try:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
    except Exception as e:
        print(f"Claude API error for {vid}: {e}", file=sys.stderr)
        return None

    text = "".join(b.text for b in resp.content if hasattr(b, "text"))
    coaching = parse_claude_response(text)
    if not coaching:
        print(f"Failed to parse Claude response for {vid}", file=sys.stderr)
        print(f"Raw text: {text[:500]}", file=sys.stderr)
        return None

    coaching["_meta"] = {
        "video_id": vid,
        "model": MODEL,
        "total_shots": metrics["total_shots"],
        "shot_breakdown": metrics.get("shot_breakdown", {}),
    }
    return coaching


def upload_to_r2(vid: str, coaching: dict):
    """Upload coaching.json to R2."""
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
    from storage.r2_client import R2Client

    c = R2Client()
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w")
    json.dump(coaching, tmp, indent=2)
    tmp.close()
    c.upload(tmp.name, f"highlights/{vid}/coaching.json", content_type="application/json")
    os.unlink(tmp.name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video_id", nargs="?", help="Video ID (e.g. IMG_1136)")
    parser.add_argument("--all", action="store_true", help="Process all videos with biomech data")
    parser.add_argument("--upload", action="store_true", help="Upload coaching.json to R2")
    parser.add_argument("--force", action="store_true", help="Regenerate even if already exists locally")
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")

    targets = []
    if args.all:
        for p in sorted(ANALYSIS_DIR.glob("*_biomech.json")):
            targets.append(p.stem.replace("_biomech", ""))
    elif args.video_id:
        targets = [args.video_id]
    else:
        parser.error("Provide video_id or --all")

    # Respect the permanent batch-exclusion list
    try:
        from config.batch_exclude import excluded_set
        excl = excluded_set()
        if args.all and excl:
            before = len(targets)
            targets = [t for t in targets if t not in excl]
            skipped = before - len(targets)
            if skipped:
                print(f"Skipping {skipped} videos per config/exclude_from_batch.json")
    except ImportError:
        pass

    out_dir = PROJECT_ROOT / "coaching"
    out_dir.mkdir(exist_ok=True)

    for vid in targets:
        out_file = out_dir / f"{vid}_coaching.json"
        if out_file.exists() and not args.force:
            print(f"[SKIP] {vid} (exists). Use --force to regenerate.")
            if args.upload:
                with open(out_file) as f:
                    upload_to_r2(vid, json.load(f))
            continue

        print(f"\n=== {vid} ===")
        coaching = generate_coaching(vid)
        if not coaching:
            continue

        with open(out_file, "w") as f:
            json.dump(coaching, f, indent=2)
        print(f"[OK] Saved {out_file}")
        print(f"  Headline: {coaching.get('headline','')}")

        if args.upload:
            upload_to_r2(vid, coaching)


if __name__ == "__main__":
    main()
