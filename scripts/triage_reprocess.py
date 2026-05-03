#!/usr/bin/env python3
"""Triage broken-model jobs: run good vs broken model on cached poses,
recommend reprocess based on per-shot disagreement.

For each video:
  - Locate pre-computed pose JSON. Skip + log if missing.
  - Run good model and broken model via the existing detect_video()
    inference path. No re-extraction of poses.
  - Compare detections at temporal-IoU tolerance 0.5s:
      added_by_broken    — broken detected, good didn't (FP in broken; visible)
      removed_by_broken  — good detected, broken missed (FN; SILENT loss)
      reclassified       — both detected at same time, different shot_type
  - Apply triage rule:
      REPROCESS if removed_by_broken >= 1 OR total_disagreement > 5
      KEEP otherwise
  - Write JSON + Markdown table.

Usage:
  python scripts/triage_reprocess.py \\
    --good models/baseline_bbe8a42b_20260502.pt \\
    --bad  models/baseline_28814eeb_BROKEN_20260502.pt \\
    --videos eval/triage/andrew_pc_jobs_since_20260403.txt \\
    --output eval/triage/triage_results_20260502.json

The videos file is the per-video manifest produced by hand from
coordinator.db. One line per video. Lines may be either:
  - bare IMG_xxxx (matches poses_full_videos/IMG_xxxx.json)
  - <coord_video_id> <filename.MOV> — uses filename stem for pose lookup
  - lines starting with '#' are comments
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.detect_shots_sequence import detect_video
from scripts.sequence_model import load_model

DEFAULT_TOLERANCE_SEC = 0.5

# Detection knobs — match production
DETECT_THRESHOLD = 0.90
DETECT_NMS_GAP = 1.5
DETECT_STEP_SEC = 0.1


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_videos_file(p: Path) -> list[tuple[str, str]]:
    """Parse the manifest. Returns [(label, video_id_for_pose_lookup), ...]."""
    out = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) == 1:
            stem = parts[0]
            out.append((stem, stem))
        else:
            coord_id = parts[0]
            filename = parts[1]
            stem = Path(filename).stem
            out.append((coord_id, stem))
    return out


def find_pose_path(stem: str) -> Path | None:
    candidates = [
        PROJECT_ROOT / "poses_full_videos" / f"{stem}.json",
        PROJECT_ROOT / "poses" / f"{stem}.json",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def find_video_path(stem: str) -> Path | None:
    candidates = [
        PROJECT_ROOT / "preprocessed" / f"{stem}.mp4",
        PROJECT_ROOT / "raw" / f"{stem}.MOV",
        PROJECT_ROOT / "raw" / f"{stem}.mov",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def detect(stem: str, video_path: Path, model, device) -> list[dict] | None:
    """Run detect_video and return its detection list (or None on failure)."""
    try:
        result = detect_video(
            str(video_path), model, device,
            threshold=DETECT_THRESHOLD,
            nms_gap=DETECT_NMS_GAP,
            step_sec=DETECT_STEP_SEC,
        )
    except Exception as e:
        print(f"  [ERROR] detect_video failed for {stem}: {e}", file=sys.stderr)
        return None
    if not result:
        return None
    return result.get("detections", [])


def diff_detections(good: list[dict], bad: list[dict],
                    tol_sec: float) -> tuple[int, int, int, list[dict]]:
    """Return (added_by_broken, removed_by_broken, reclassified, pair_log).

    Pair-log is a list of {kind, t_good, t_bad, type_good, type_bad} per
    disagreement, useful for spot-checks.

    Greedy nearest-neighbor matching: each good detection consumes the
    nearest bad detection within tol_sec. Unmatched good = removed,
    unmatched bad = added. Same time, different type = reclassified.
    """
    matched_bad = set()
    pairs = []
    added = removed = reclassified = 0

    good_sorted = sorted(enumerate(good), key=lambda x: x[1]["timestamp"])
    bad_sorted = sorted(enumerate(bad), key=lambda x: x[1]["timestamp"])

    for gi, g in good_sorted:
        best_bi = None
        best_err = tol_sec + 1
        for bi, b in bad_sorted:
            if bi in matched_bad:
                continue
            err = abs(g["timestamp"] - b["timestamp"])
            if err < best_err:
                best_err = err
                best_bi = bi
        if best_bi is not None and best_err <= tol_sec:
            matched_bad.add(best_bi)
            b = bad[best_bi]
            if g.get("shot_type") != b.get("shot_type"):
                reclassified += 1
                pairs.append({
                    "kind": "reclassified",
                    "t_good": round(g["timestamp"], 2),
                    "t_bad":  round(b["timestamp"], 2),
                    "type_good": g.get("shot_type"),
                    "type_bad":  b.get("shot_type"),
                })
        else:
            removed += 1
            pairs.append({
                "kind": "removed_by_broken",
                "t_good": round(g["timestamp"], 2),
                "type_good": g.get("shot_type"),
            })

    for bi, b in enumerate(bad):
        if bi not in matched_bad:
            added += 1
            pairs.append({
                "kind": "added_by_broken",
                "t_bad": round(b["timestamp"], 2),
                "type_bad": b.get("shot_type"),
            })

    return added, removed, reclassified, pairs


def recommend(removed: int, added: int, reclassified: int) -> tuple[str, str]:
    total = added + removed + reclassified
    if removed >= 1:
        return "REPROCESS", f"removed_by_broken >= 1 ({removed} false negative{'s' if removed != 1 else ''})"
    if total > 5:
        return "REPROCESS", f"total_disagreement > 5 ({total} = +{added} added / {reclassified} reclassified)"
    return "KEEP", f"within expected noise (total_disagreement={total}, no false negatives)"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--good", required=True, help="Canonical / good model .pt")
    p.add_argument("--bad",  required=True, help="Broken model .pt")
    p.add_argument("--videos", required=True, help="Manifest text file")
    p.add_argument("--output", required=True, help="Output triage JSON")
    p.add_argument("--tolerance-sec", type=float, default=DEFAULT_TOLERANCE_SEC)
    args = p.parse_args()

    good_path = Path(args.good).resolve()
    bad_path = Path(args.bad).resolve()
    videos_path = Path(args.videos).resolve()
    out_path = Path(args.output).resolve()

    if not good_path.exists():
        sys.exit(f"FATAL: --good not found: {good_path}")
    if not bad_path.exists():
        sys.exit(f"FATAL: --bad not found: {bad_path}")
    if not videos_path.exists():
        sys.exit(f"FATAL: --videos not found: {videos_path}")

    good_sha = sha256_of(good_path)
    bad_sha = sha256_of(bad_path)
    videos = parse_videos_file(videos_path)

    print(f"[triage] good model:  {good_path.name} ({good_sha[:8]})")
    print(f"[triage] bad model:   {bad_path.name} ({bad_sha[:8]})")
    print(f"[triage] {len(videos)} videos to triage, tolerance={args.tolerance_sec}s")

    good_model, device = load_model(str(good_path))
    bad_model, _ = load_model(str(bad_path), device=device)
    if good_model is None or bad_model is None:
        sys.exit("FATAL: failed to load one or both models")

    results = []
    t0 = time.time()
    for label, stem in videos:
        print(f"\n  -- {label} ({stem}) --", flush=True)
        pose_path = find_pose_path(stem)
        if pose_path is None:
            print(f"     skip: no pose JSON for {stem}")
            results.append({
                "video_id": label,
                "filename_stem": stem,
                "skipped": True,
                "skipped_reason": f"no pose file at poses_full_videos/{stem}.json",
            })
            continue

        video_path = find_video_path(stem)
        if video_path is None:
            print(f"     skip: no preprocessed video for {stem}")
            results.append({
                "video_id": label,
                "filename_stem": stem,
                "skipped": True,
                "skipped_reason": f"no video file (preprocessed/{stem}.mp4)",
            })
            continue

        good_dets = detect(stem, video_path, good_model, device)
        bad_dets = detect(stem, video_path, bad_model, device)
        if good_dets is None or bad_dets is None:
            results.append({
                "video_id": label,
                "filename_stem": stem,
                "skipped": True,
                "skipped_reason": "detect_video failed for at least one model",
            })
            continue

        added, removed, reclassed, pairs = diff_detections(
            good_dets, bad_dets, args.tolerance_sec)
        total_disagreement = added + removed + reclassed
        rec, reason = recommend(removed, added, reclassed)

        print(f"     shots good={len(good_dets)} bad={len(bad_dets)} "
              f"add+={added} rm-={removed} reclass={reclassed} "
              f"total={total_disagreement} → {rec}")

        results.append({
            "video_id": label,
            "filename_stem": stem,
            "shots_good": len(good_dets),
            "shots_bad": len(bad_dets),
            "added_by_broken": added,
            "removed_by_broken": removed,
            "reclassified": reclassed,
            "total_disagreement": total_disagreement,
            "recommendation": rec,
            "reason": reason,
            "disagreement_log": pairs,
        })

    elapsed = time.time() - t0

    evaluated = [r for r in results if not r.get("skipped")]
    skipped = [r for r in results if r.get("skipped")]
    reprocess = [r for r in evaluated if r["recommendation"] == "REPROCESS"]
    keep = [r for r in evaluated if r["recommendation"] == "KEEP"]

    summary = {
        "total_videos": len(videos),
        "evaluated": len(evaluated),
        "skipped": len(skipped),
        "reprocess_count": len(reprocess),
        "keep_count": len(keep),
        "total_added_by_broken": sum(r["added_by_broken"] for r in evaluated),
        "total_removed_by_broken": sum(r["removed_by_broken"] for r in evaluated),
        "total_reclassified": sum(r["reclassified"] for r in evaluated),
    }

    out = {
        "schema_version": 1,
        "good_model": {"path": str(good_path.relative_to(PROJECT_ROOT)) if str(good_path).startswith(str(PROJECT_ROOT)) else str(good_path),
                       "sha256": good_sha},
        "bad_model":  {"path": str(bad_path.relative_to(PROJECT_ROOT)) if str(bad_path).startswith(str(PROJECT_ROOT)) else str(bad_path),
                       "sha256": bad_sha},
        "tolerance_seconds": args.tolerance_sec,
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "evaluated_on": socket.gethostname(),
        "elapsed_seconds": round(elapsed, 1),
        "videos": results,
        "summary": summary,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[triage] wrote {out_path}")

    # Markdown table sorted by total_disagreement desc
    print()
    print("# Triage results")
    print()
    print(f"good={good_sha[:8]} bad={bad_sha[:8]} tolerance={args.tolerance_sec}s")
    print()
    print("| video | shots good | shots bad | add+ | rm- | reclass | total | recommendation |")
    print("|---|---:|---:|---:|---:|---:|---:|---|")
    sorted_evaluated = sorted(evaluated, key=lambda r: -r["total_disagreement"])
    for r in sorted_evaluated:
        marker = "🚨" if r["recommendation"] == "REPROCESS" else "✓"
        print(f"| `{r['video_id']}` ({r['filename_stem']}) "
              f"| {r['shots_good']} | {r['shots_bad']} "
              f"| {r['added_by_broken']} | {r['removed_by_broken']} "
              f"| {r['reclassified']} | {r['total_disagreement']} "
              f"| {marker} {r['recommendation']} |")
    for r in skipped:
        print(f"| `{r['video_id']}` ({r['filename_stem']}) | — | — | — | — | — | — | ⏭ skipped: {r['skipped_reason']} |")

    print()
    print("## Summary")
    print(f"- evaluated: {summary['evaluated']}/{summary['total_videos']}")
    print(f"- skipped: {summary['skipped']}")
    print(f"- REPROCESS: {summary['reprocess_count']}")
    print(f"- KEEP:      {summary['keep_count']}")
    print(f"- totals across evaluated: +{summary['total_added_by_broken']} added / "
          f"-{summary['total_removed_by_broken']} removed / {summary['total_reclassified']} reclassified")
    print(f"- elapsed: {elapsed:.0f}s")

    # Don't auto-trigger reprocessing — that's a separate human-confirm step.
    print()
    print("Triage produces a recommendation. Reprocessing requeue is separate "
          "(see brief: UPDATE jobs SET status='pending'... for REPROCESS list).")


if __name__ == "__main__":
    main()
