#!/usr/bin/env python3
"""Harvest per-video inspector feedback into a corrections dataset.

The /inspect tool writes per-shot feedback to R2 at
``highlights/<user_hash>/<vid>/feedback.json`` (entries:
{kind, shot_idx, note, value, by, at}). This script scans every such file,
cross-references each entry against that video's ``shots.json`` to attach the
detected shot's type / timestamp / confidence, and emits a single consolidated
dataset the pipeline work can act on:

  - wrong_type / not_a_shot  -> detection labels to correct (model eval/retrain)
  - contact_off              -> contact-frame tuning targets (#3)
  - bad_comparison           -> matcher / pro-library signal (#18)
  - good                     -> positive confirmations
  - note / other             -> free-form, for a human to read

Output: eval/feedback/corrections.json (+ a printed summary). Read-only against
R2 (only GETs); writes one local file. Safe to run repeatedly.

Usage:
    python scripts/harvest_feedback.py
    python scripts/harvest_feedback.py --out eval/feedback/corrections.json
    python scripts/harvest_feedback.py --user u_ae629639   # one user only
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from storage.r2_client import R2Client

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_OUT = PROJECT_ROOT / "eval" / "feedback" / "corrections.json"

# Which kinds map to which downstream concern.
KIND_BUCKETS = {
    "wrong_type": "detection",
    "not_a_shot": "detection",
    "contact_off": "contact",
    "bad_comparison": "comparison",
    "good": "confirmed",
}


def _get_json(client, bucket, key):
    try:
        obj = client.client.get_object(Bucket=bucket, Key=key)
        return json.loads(obj["Body"].read())
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--user", default=None,
                    help="Restrict to one user_hash (e.g. u_ae629639).")
    args = ap.parse_args()

    c = R2Client()
    bucket = c.bucket_name
    prefix = f"highlights/{args.user}/" if args.user else "highlights/"

    # 1) Find every feedback.json.
    fb_keys = []
    paginator = c.client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for o in page.get("Contents", []):
            if o["Key"].endswith("/feedback.json"):
                fb_keys.append(o["Key"])

    if not fb_keys:
        print("No feedback.json found yet — nothing to harvest.")
        print("(Feedback is created from the /inspect tool's per-shot buttons.)")
        return 0

    # 2) For each, load entries + the sibling shots.json for cross-reference.
    corrections = []          # one record per feedback entry, enriched
    by_kind = Counter()
    by_video = Counter()
    shots_cache = {}

    for fb_key in fb_keys:
        # .../highlights/<user>/<vid>/feedback.json
        parts = fb_key.split("/")
        vid = parts[-2]
        user_hash = parts[-3] if len(parts) >= 3 else None
        doc = _get_json(c, bucket, fb_key)
        if not doc or not isinstance(doc.get("entries"), list):
            continue

        shots_key = fb_key.rsplit("/", 1)[0] + "/shots.json"
        if shots_key not in shots_cache:
            sj = _get_json(c, bucket, shots_key) or {}
            shots_cache[shots_key] = {
                s.get("idx"): s for s in (sj.get("shots") or [])
            }
        shot_by_idx = shots_cache[shots_key]

        for e in doc["entries"]:
            kind = e.get("kind", "other")
            idx = e.get("shot_idx")
            shot = shot_by_idx.get(idx) if idx is not None else None
            rec = {
                "video_id": vid,
                "user_hash": user_hash,
                "kind": kind,
                "bucket": KIND_BUCKETS.get(kind, "other"),
                "shot_idx": idx,
                "note": e.get("note") or "",
                "by": e.get("by"),
                "at": e.get("at"),
                # Enriched from shots.json (None if shot/idx unknown):
                "shot_type": (shot or {}).get("type"),
                "shot_t": (shot or {}).get("t"),
                "shot_confidence": (shot or {}).get("confidence"),
            }
            corrections.append(rec)
            by_kind[kind] += 1
            by_video[vid] += 1

    # 3) Group for the downstream consumers.
    grouped = defaultdict(list)
    for r in corrections:
        grouped[r["bucket"]].append(r)

    out = {
        "generated_from": prefix,
        "total_entries": len(corrections),
        "videos_with_feedback": len(by_video),
        "by_kind": dict(by_kind),
        "by_bucket": {k: len(v) for k, v in grouped.items()},
        "corrections": corrections,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    # 4) Human summary.
    print(f"Harvested {len(corrections)} feedback entries "
          f"across {len(by_video)} videos -> {out_path}")
    print("By kind:")
    for k, n in by_kind.most_common():
        print(f"  {k:16} {n}")
    print("By downstream bucket:")
    for bkt in ("detection", "contact", "comparison", "confirmed", "other"):
        items = grouped.get(bkt, [])
        if items:
            print(f"  {bkt:11} {len(items)}")
    # Surface the actionable detection/contact corrections with context.
    actionable = [r for r in corrections if r["bucket"] in ("detection", "contact")]
    if actionable:
        print(f"\nActionable ({len(actionable)}):")
        for r in actionable[:25]:
            t = f"{r['shot_t']:.1f}s" if r.get("shot_t") is not None else "?"
            conf = (f"{r['shot_confidence']:.2f}"
                    if r.get("shot_confidence") is not None else "?")
            note = f" — {r['note']}" if r["note"] else ""
            print(f"  {r['video_id']} shot#{r['shot_idx']} "
                  f"({r.get('shot_type') or '?'} @ {t}, conf {conf}) "
                  f"[{r['kind']}]{note}")
        if len(actionable) > 25:
            print(f"  … +{len(actionable) - 25} more in {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
