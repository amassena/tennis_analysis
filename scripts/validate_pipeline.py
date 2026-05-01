#!/usr/bin/env python3
"""End-to-end validation framework for the tennis shot detection pipeline.

Compares pipeline detections against ground truth (label files or
user-reviewed detection JSONs). Reports per-video, per-shot-type,
per-source metrics, temporal error analysis, and regression baselines.

Usage:
    # Validate all GT videos
    python scripts/validate_pipeline.py

    # Validate specific video against label file
    python scripts/validate_pipeline.py --video IMG_6665 --gt labels/IMG_6665_ground_truth.txt

    # Compare against saved baseline
    python scripts/validate_pipeline.py --check-regression

    # Save current results as new baseline
    python scripts/validate_pipeline.py --save-baseline
"""

import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import PROJECT_ROOT, POSES_DIR, PREPROCESSED_DIR

DETECTIONS_DIR = os.path.join(PROJECT_ROOT, "detections")
LABELS_DIR = os.path.join(PROJECT_ROOT, "labels")
BASELINES_DIR = os.path.join(PROJECT_ROOT, "training", "regression_baselines")

# Match tolerance: detection within this many seconds of GT counts as TP.
# This is the legacy "window-level" tolerance — what the regression baseline was
# trained on. Loose enough to absorb stride/peak-finding jitter (1.5s ≈ NMS gap).
MATCH_TOLERANCE = 1.5

# Event-level tolerance: ±6 frames @ 60fps ≈ 100ms (TenniSet convention).
# This is the metric biomech actually cares about — the contact-time alignment
# downstream depends on. Always reported alongside the legacy window F1.
EVENT_TOLERANCE_FRAMES = 6
EVENT_TOLERANCE_FALLBACK_SEC = 0.1  # used when fps unknown

# Videos with known ground truth (reviewed in shot_review.py)
# Map: video_name → GT source file (detection JSON or label file)
GT_VIDEOS = {
    "IMG_6665": {"gt_file": "detections/IMG_6665_fused_v5.json", "format": "detection_json"},
    "IMG_0864": {"gt_file": "detections/IMG_0864_fused.json", "format": "detection_json"},
    "IMG_0865": {"gt_file": "detections/IMG_0865_fused.json", "format": "detection_json"},
    "IMG_0866": {"gt_file": "detections/IMG_0866_fused.json", "format": "detection_json"},
    "IMG_0867": {"gt_file": "detections/IMG_0867_fused.json", "format": "detection_json"},
    "IMG_0868": {"gt_file": "detections/IMG_0868_fused.json", "format": "detection_json"},
    "IMG_0869": {"gt_file": "detections/IMG_0869_fused.json", "format": "detection_json"},
    "IMG_0870": {"gt_file": "detections/IMG_0870_fused.json", "format": "detection_json"},
    "IMG_6703": {"gt_file": "detections/IMG_6703_fused_v5.json", "format": "detection_json"},
    "IMG_6711": {"gt_file": "detections/IMG_6711_fused_v5.json", "format": "detection_json"},
    "IMG_6713": {"gt_file": "detections/IMG_6713_fused.json", "format": "detection_json"},
    "IMG_0929": {"gt_file": "detections/IMG_0929_fused_v5.json", "format": "detection_json"},
    "IMG_0991": {"gt_file": "detections/IMG_0991_fused.json", "format": "detection_json"},
    "IMG_0994": {"gt_file": "detections/IMG_0994_fused.json", "format": "detection_json"},
    "IMG_0996": {"gt_file": "detections/IMG_0996_fused.json", "format": "detection_json"},
    "IMG_1003": {"gt_file": "detections/IMG_1003_fused.json", "format": "detection_json"},
    "IMG_1004": {"gt_file": "detections/IMG_1004_fused.json", "format": "detection_json"},
    "IMG_1005": {"gt_file": "detections/IMG_1005_fused.json", "format": "detection_json"},
    "IMG_1007": {"gt_file": "detections/IMG_1007_fused.json", "format": "detection_json"},
    "IMG_1008": {"gt_file": "detections/IMG_1008_fused.json", "format": "detection_json"},
    "IMG_1026": {"gt_file": "detections/IMG_1026_fused.json", "format": "detection_json"},
    "IMG_1027": {"gt_file": "detections/IMG_1027_fused.json", "format": "detection_json"},
    "IMG_1030": {"gt_file": "detections/IMG_1030_fused.json", "format": "detection_json"},
    "IMG_1031": {"gt_file": "detections/IMG_1031_fused.json", "format": "detection_json"},
    "IMG_6851": {"gt_file": "detections/IMG_6851_fused.json", "format": "detection_json"},
    "IMG_6852": {"gt_file": "detections/IMG_6852_fused.json", "format": "detection_json"},
}

# Shot types to ignore in metrics (not real player shots)
IGNORE_TYPES = {"offscreen", "practice", "unknown_shot"}


def parse_label_file(path):
    """Parse a ground truth label file.

    Format: 'MM:SS shot_type' or 'MM:SS.f shot_type'
    Returns list of {timestamp, shot_type} dicts.
    """
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Remove inline comments
            line = re.sub(r"#.*$", "", line).strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            time_str = parts[0]
            shot_type = parts[1].lower()

            # Parse time: M:SS or MM:SS or MM:SS.f
            match = re.match(r"(\d+):(\d+)(?:\.(\d+))?", time_str)
            if not match:
                continue
            minutes = int(match.group(1))
            seconds = int(match.group(2))
            frac = float(f"0.{match.group(3)}") if match.group(3) else 0.0
            timestamp = minutes * 60 + seconds + frac

            entries.append({"timestamp": timestamp, "shot_type": shot_type})

    return entries


def parse_detection_json(path):
    """Parse a detection JSON file as ground truth.

    Returns list of {timestamp, shot_type, source} dicts.
    """
    with open(path) as f:
        data = json.load(f)

    entries = []
    for det in data.get("detections", []):
        entries.append({
            "timestamp": det["timestamp"],
            "shot_type": det.get("shot_type", "unknown"),
            "source": det.get("source", "unknown"),
        })
    return entries


def load_ground_truth(video_name, gt_override=None):
    """Load ground truth for a video.

    Args:
        video_name: e.g. "IMG_6703"
        gt_override: Optional path to a specific GT file

    Returns (entries, gt_meta) where entries is a list of
    {timestamp, shot_type} dicts (trackable only), and gt_meta carries
    handedness/camera/session/fps fields from the GT JSON when present.
    """
    path = None
    if gt_override:
        path = os.path.join(PROJECT_ROOT, gt_override) if not os.path.isabs(gt_override) else gt_override
        if path.endswith(".json"):
            entries = parse_detection_json(path)
        else:
            entries = parse_label_file(path)
    elif video_name in GT_VIDEOS:
        info = GT_VIDEOS[video_name]
        path = os.path.join(PROJECT_ROOT, info["gt_file"])
        if not os.path.exists(path):
            print(f"  WARNING: GT file not found: {path}")
            return [], {}
        if info["format"] == "label_file":
            entries = parse_label_file(path)
        else:
            entries = parse_detection_json(path)
    else:
        return [], {}

    gt_meta = {}
    if path and path.endswith(".json"):
        try:
            with open(path) as f:
                raw = json.load(f)
            gt_meta = {
                "fps": raw.get("fps"),
                "dominant_hand": raw.get("dominant_hand"),
                "camera_angle": raw.get("camera_angle"),
                "session_type": raw.get("session_type"),
            }
        except (json.JSONDecodeError, OSError):
            pass

    # Filter out non-trackable types
    entries = [e for e in entries if e["shot_type"] not in IGNORE_TYPES]
    return entries, gt_meta


def load_detections(video_name, det_path=None, gt_path=None):
    """Load pipeline detections for a video.

    Args:
        video_name: e.g. "IMG_6703"
        det_path: Optional specific detection file path
        gt_path: GT file path to exclude (avoids comparing GT against itself)

    Returns list of {timestamp, shot_type, source, confidence} dicts.
    """
    gt_abs = os.path.abspath(gt_path) if gt_path else None

    if det_path:
        path = det_path
    else:
        # Find pipeline output file, excluding the GT file
        # Priority: pipeline default output (_fused_detections), then other variants
        candidates = [
            os.path.join(DETECTIONS_DIR, f"{video_name}_fused_detections.json"),
            os.path.join(PROJECT_ROOT, f"{video_name}_fused_detections.json"),
            os.path.join(DETECTIONS_DIR, f"{video_name}_fused.json"),
            os.path.join(DETECTIONS_DIR, f"{video_name}_fused_v5.json"),
        ]
        path = None
        for c in candidates:
            if os.path.exists(c):
                if gt_abs and os.path.abspath(c) == gt_abs:
                    continue  # Skip GT file
                path = c
                break

    if not path or not os.path.exists(path):
        return [], None, {}

    # Warn if detection and GT are the same file
    if gt_abs and os.path.abspath(path) == gt_abs:
        print(f"  WARNING: Detection file is same as GT file: {path}")
        print(f"  Run fused_detect.py first to generate fresh pipeline output.")
        return [], path, {}

    with open(path) as f:
        data = json.load(f)

    entries = []
    for det in data.get("detections", []):
        shot_type = det.get("shot_type", "unknown")
        if shot_type in IGNORE_TYPES:
            continue
        entries.append({
            "timestamp": det["timestamp"],
            "shot_type": shot_type,
            "source": det.get("source", "unknown"),
            "confidence": det.get("confidence", 0.0),
            "not_shot_prob": det.get("not_shot_prob", 0.0),
        })

    # Stratification metadata: fps for event tolerance; handedness/camera/session
    # are saved by shot_review.py at the top level of the detection JSON.
    meta = {
        "fps": data.get("fps"),
        "dominant_hand": data.get("dominant_hand"),
        "camera_angle": data.get("camera_angle"),
        "session_type": data.get("session_type"),
    }
    return entries, path, meta


def match_detections(gt_list, det_list, tolerance=MATCH_TOLERANCE,
                     require_class_match=False):
    """Match ground truth to detections using greedy nearest-neighbor.

    Args:
        tolerance: max seconds between GT and detection to count as TP.
        require_class_match: if True, only matches with same shot_type count
            as TP. A detection at the right time but wrong type becomes both
            an FN (missed type) and an FP (wrong-type detection).

    Returns (tp_pairs, fp_dets, fn_gts) where:
        tp_pairs: list of (gt_entry, det_entry, time_error) tuples
        fp_dets: unmatched detections (false positives)
        fn_gts: unmatched ground truths (false negatives)
    """
    gt_matched = set()
    det_matched = set()
    tp_pairs = []

    # Sort both by timestamp
    gt_sorted = sorted(enumerate(gt_list), key=lambda x: x[1]["timestamp"])
    det_sorted = sorted(enumerate(det_list), key=lambda x: x[1]["timestamp"])

    # Greedy matching: for each GT, find closest unmatched detection
    # (optionally restricted to same shot_type)
    for gi, gt in gt_sorted:
        best_di = None
        best_err = tolerance + 1
        for di, det in det_sorted:
            if di in det_matched:
                continue
            if require_class_match and det.get("shot_type") != gt.get("shot_type"):
                continue
            err = abs(gt["timestamp"] - det["timestamp"])
            if err < best_err:
                best_err = err
                best_di = di
        if best_di is not None and best_err <= tolerance:
            gt_matched.add(gi)
            det_matched.add(best_di)
            tp_pairs.append((gt, det_list[best_di], best_err))

    fp_dets = [det_list[i] for i in range(len(det_list)) if i not in det_matched]
    fn_gts = [gt_list[i] for i in range(len(gt_list)) if i not in gt_matched]

    return tp_pairs, fp_dets, fn_gts


def event_tolerance_seconds(det_data):
    """Resolve the event-level tolerance in seconds for a video.

    Reads fps from the detection JSON if available, else falls back to a
    fixed seconds value. Returns (tolerance_sec, fps_used).
    """
    fps = None
    if isinstance(det_data, dict):
        fps = det_data.get("fps")
    if fps and fps > 0:
        return EVENT_TOLERANCE_FRAMES / float(fps), float(fps)
    return EVENT_TOLERANCE_FALLBACK_SEC, None


def compute_metrics(tp, fp, fn):
    """Compute precision, recall, F1."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def validate_video(video_name, gt_override=None, det_path=None, verbose=True):
    """Run full validation for a single video.

    Returns dict with metrics, per-type breakdown, per-source breakdown.
    Reports both window-level (legacy 1.5s tolerance) and event-level
    (±6 frames, TenniSet convention) F1 — the second is what biomech sees.
    """
    gt_list, gt_meta = load_ground_truth(video_name, gt_override)

    # Resolve GT file path so we can exclude it from detection search
    gt_file_path = None
    if gt_override:
        gt_file_path = os.path.join(PROJECT_ROOT, gt_override) if not os.path.isabs(gt_override) else gt_override
    elif video_name in GT_VIDEOS:
        gt_file_path = os.path.join(PROJECT_ROOT, GT_VIDEOS[video_name]["gt_file"])

    det_list, det_file, det_meta = load_detections(video_name, det_path, gt_path=gt_file_path)

    if not gt_list:
        if verbose:
            print(f"  {video_name}: No ground truth available, skipping")
        return None

    if not det_list and det_file is None:
        if verbose:
            print(f"  {video_name}: No pipeline output found (run fused_detect.py first)")
        return None

    # Window-level (legacy): ±1.5s, class-agnostic. Backwards compatible — what
    # the regression baseline was trained on.
    tp_pairs, fp_dets, fn_gts = match_detections(gt_list, det_list)

    # Event-level: ±6 frames at video fps (≈100ms @ 60fps). Biomech-aligned.
    # Compute both class-agnostic ("did we find the shot?") and class-strict
    # ("did we find AND classify it correctly?"). Gap between them = pure
    # classification error within the temporal tolerance.
    event_tol_sec, event_fps = event_tolerance_seconds(det_meta or gt_meta)
    e_det_pairs, e_det_fp, e_det_fn = match_detections(
        gt_list, det_list, tolerance=event_tol_sec)
    e_cls_pairs, e_cls_fp, e_cls_fn = match_detections(
        gt_list, det_list, tolerance=event_tol_sec, require_class_match=True)

    # Overall metrics
    overall = compute_metrics(len(tp_pairs), len(fp_dets), len(fn_gts))
    overall["gt_total"] = len(gt_list)
    overall["det_total"] = len(det_list)

    event_detection = compute_metrics(len(e_det_pairs), len(e_det_fp), len(e_det_fn))
    event_detection["tolerance_sec"] = round(event_tol_sec, 4)
    event_detection["tolerance_frames"] = EVENT_TOLERANCE_FRAMES if event_fps else None
    event_detection["fps"] = event_fps

    event_classification = compute_metrics(len(e_cls_pairs), len(e_cls_fp), len(e_cls_fn))
    event_classification["tolerance_sec"] = round(event_tol_sec, 4)

    # Contact-time MAE in milliseconds (event-level matches only) — what biomech
    # actually feels.
    if e_det_pairs:
        ms_errors = [err * 1000.0 for _, _, err in e_det_pairs]
        event_detection["contact_mae_ms"] = round(sum(ms_errors) / len(ms_errors), 1)
        event_detection["contact_max_ms"] = round(max(ms_errors), 1)
    else:
        event_detection["contact_mae_ms"] = None
        event_detection["contact_max_ms"] = None

    # Time error stats
    time_errors = [err for _, _, err in tp_pairs]
    if time_errors:
        overall["mean_time_error"] = round(sum(time_errors) / len(time_errors), 3)
        overall["max_time_error"] = round(max(time_errors), 3)
    else:
        overall["mean_time_error"] = 0.0
        overall["max_time_error"] = 0.0

    # ── Per-shot-type breakdown ──────────────────────────────────
    type_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for gt, det, err in tp_pairs:
        gt_type = gt["shot_type"]
        det_type = det["shot_type"]
        if gt_type == det_type:
            type_stats[gt_type]["tp"] += 1
        else:
            # Correct detection but wrong type classification
            type_stats[gt_type]["tp"] += 1  # detection-level TP
            # Track classification mismatch separately

    for det in fp_dets:
        type_stats[det["shot_type"]]["fp"] += 1

    for gt in fn_gts:
        type_stats[gt["shot_type"]]["fn"] += 1

    per_type = {}
    for shot_type, stats in sorted(type_stats.items()):
        per_type[shot_type] = compute_metrics(stats["tp"], stats["fp"], stats["fn"])

    # ── Event-level per-type (class-strict, ±6 frame tolerance) ───
    event_type_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for gt, det, _ in e_cls_pairs:
        # Class-strict: gt_type == det_type by construction
        event_type_stats[gt["shot_type"]]["tp"] += 1
    for det in e_cls_fp:
        # FPs include "wrong-type-at-right-time" detections — those land in the
        # predicted-type bucket. Distinct from window-level FPs.
        event_type_stats[det["shot_type"]]["fp"] += 1
    for gt in e_cls_fn:
        event_type_stats[gt["shot_type"]]["fn"] += 1
    event_per_type = {
        st: compute_metrics(s["tp"], s["fp"], s["fn"])
        for st, s in sorted(event_type_stats.items())
    }

    # ── Per-source breakdown ─────────────────────────────────────
    source_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "total": 0})

    for _, det, _ in tp_pairs:
        src = det.get("source", "unknown")
        source_stats[src]["tp"] += 1
        source_stats[src]["total"] += 1

    for det in fp_dets:
        src = det.get("source", "unknown")
        source_stats[src]["fp"] += 1
        source_stats[src]["total"] += 1

    per_source = {}
    for src, stats in sorted(source_stats.items()):
        per_source[src] = {
            "tp": stats["tp"],
            "fp": stats["fp"],
            "total": stats["total"],
            "precision": round(stats["tp"] / stats["total"], 4) if stats["total"] > 0 else 0.0,
        }

    # ── Classification accuracy (among TPs) ──────────────────────
    type_correct = sum(1 for gt, det, _ in tp_pairs if gt["shot_type"] == det["shot_type"])
    classification_acc = round(type_correct / len(tp_pairs), 4) if tp_pairs else 0.0

    # ── Confusion matrix ─────────────────────────────────────────
    all_types = sorted(set(
        [gt["shot_type"] for gt in gt_list] +
        [det["shot_type"] for det in det_list]
    ))
    confusion = {t1: {t2: 0 for t2 in all_types} for t1 in all_types}
    for gt, det, _ in tp_pairs:
        confusion[gt["shot_type"]][det["shot_type"]] += 1

    # ── Temporal error analysis ──────────────────────────────────
    temporal_bins = []
    bin_size = 60.0  # 1-minute bins
    if gt_list:
        max_t = max(e["timestamp"] for e in gt_list)
        for bin_start in range(0, int(max_t) + 1, int(bin_size)):
            bin_end = bin_start + bin_size
            bin_gt = [g for g in gt_list if bin_start <= g["timestamp"] < bin_end]
            bin_det = [d for d in det_list if bin_start <= d["timestamp"] < bin_end]
            if bin_gt:
                tp_b, fp_b, fn_b = match_detections(bin_gt, bin_det)
                metrics_b = compute_metrics(len(tp_b), len(fp_b), len(fn_b))
                temporal_bins.append({
                    "bin_start": bin_start,
                    "bin_end": bin_end,
                    **metrics_b,
                })

    # Stratification metadata: prefer GT JSON's labels (set by shot_review.py),
    # fall back to det JSON. Fps for tolerance came from det_meta first.
    stratify = {
        "dominant_hand": (gt_meta or {}).get("dominant_hand")
                          or (det_meta or {}).get("dominant_hand"),
        "camera_angle": (gt_meta or {}).get("camera_angle")
                          or (det_meta or {}).get("camera_angle"),
        "session_type": (gt_meta or {}).get("session_type")
                          or (det_meta or {}).get("session_type"),
        "fps": (det_meta or {}).get("fps") or (gt_meta or {}).get("fps"),
    }

    result = {
        "video": video_name,
        "gt_file": os.path.basename(gt_file_path) if gt_file_path else None,
        "det_file": os.path.basename(det_file) if det_file else None,
        "overall": overall,
        "event_detection": event_detection,
        "event_classification": event_classification,
        "per_type": per_type,
        "event_per_type": event_per_type,
        "per_source": per_source,
        "classification_accuracy": classification_acc,
        "confusion_matrix": confusion,
        "temporal_bins": temporal_bins,
        "stratify": stratify,
        "false_positives": [
            {"timestamp": d["timestamp"], "shot_type": d["shot_type"],
             "source": d.get("source", ""), "confidence": d.get("confidence", 0)}
            for d in fp_dets
        ],
        "false_negatives": [
            {"timestamp": g["timestamp"], "shot_type": g["shot_type"]}
            for g in fn_gts
        ],
    }

    if verbose:
        _print_video_report(result)

    return result


def _print_video_report(result):
    """Print a formatted report for a single video."""
    v = result["video"]
    o = result["overall"]
    ed = result.get("event_detection", {})
    ec = result.get("event_classification", {})
    strat = result.get("stratify", {})

    print(f"\n{'='*60}")
    print(f"  {v}")
    print(f"{'='*60}")
    gt_f = result.get("gt_file", "?")
    det_f = result.get("det_file", "?")
    print(f"  GT file: {gt_f}")
    print(f"  Det file: {det_f}")
    if any(strat.get(k) for k in ("dominant_hand", "camera_angle", "session_type")):
        print(f"  Stratify: hand={strat.get('dominant_hand') or '-'}  "
              f"angle={strat.get('camera_angle') or '-'}  "
              f"session={strat.get('session_type') or '-'}")
    print(f"  GT: {o['gt_total']}  Det: {o['det_total']}")

    # Window-level (legacy 1.5s tolerance)
    print(f"  Window F1 (±1.5s): "
          f"TP={o['tp']} FP={o['fp']} FN={o['fn']}  "
          f"P={o['precision']:.1%}  R={o['recall']:.1%}  F1={o['f1']:.1%}")

    # Event-level (±6 frames). Two flavors: detection vs class-strict.
    tol_label = "?"
    if ed:
        tol_label = (f"±{ed['tolerance_frames']}f@{ed['fps']:.0f}fps"
                     if ed.get("tolerance_frames") and ed.get("fps")
                     else f"±{ed['tolerance_sec']:.2f}s")
        mae = ed.get("contact_mae_ms")
        mae_str = f"  contact MAE={mae:.0f}ms" if mae is not None else ""
        print(f"  Event detection ({tol_label}): "
              f"TP={ed['tp']} FP={ed['fp']} FN={ed['fn']}  "
              f"P={ed['precision']:.1%}  R={ed['recall']:.1%}  F1={ed['f1']:.1%}{mae_str}")
    if ec:
        print(f"  Event classification (class-strict, {tol_label}): "
              f"TP={ec['tp']} FP={ec['fp']} FN={ec['fn']}  "
              f"P={ec['precision']:.1%}  R={ec['recall']:.1%}  F1={ec['f1']:.1%}")

    print(f"  Mean time error (window): {o['mean_time_error']:.3f}s  Max: {o['max_time_error']:.3f}s")
    print(f"  Classification accuracy (window): {result['classification_accuracy']:.1%}")

    # Per-type, side by side: window (±1.5s, class-agnostic) vs event (class-strict)
    if result["per_type"] or result.get("event_per_type"):
        print(f"\n  Per shot type:")
        print(f"    {'type':<12s}  {'window F1 (±1.5s)':<28s}  {'event F1 (class-strict)':<28s}")
        all_types = sorted(set(list(result["per_type"].keys()) +
                                list((result.get("event_per_type") or {}).keys())))
        for st in all_types:
            wm = result["per_type"].get(st)
            em = (result.get("event_per_type") or {}).get(st)
            wstr = (f"TP={wm['tp']:3d} FP={wm['fp']:2d} FN={wm['fn']:2d} F1={wm['f1']:.1%}"
                    if wm else "-")
            estr = (f"TP={em['tp']:3d} FP={em['fp']:2d} FN={em['fn']:2d} F1={em['f1']:.1%}"
                    if em else "-")
            print(f"    {st:<12s}  {wstr:<28s}  {estr:<28s}")

    # Per-source
    if result["per_source"]:
        print(f"\n  Per detection source:")
        for src, m in sorted(result["per_source"].items()):
            print(f"    {src:<25s}  TP={m['tp']:3d}  FP={m['fp']:2d}  "
                  f"total={m['total']:3d}  P={m['precision']:.1%}")

    # Confusion matrix
    cm = result["confusion_matrix"]
    if cm:
        types = sorted(cm.keys())
        print(f"\n  Confusion matrix (rows=GT, cols=pred):")
        header = "    " + "".join(f"{t[:6]:>8s}" for t in types)
        print(header)
        for t1 in types:
            row = f"    {t1[:6]:<6s}" + "".join(f"{cm[t1][t2]:>8d}" for t2 in types)
            print(row)

    # FPs and FNs
    if result["false_positives"]:
        print(f"\n  False positives ({len(result['false_positives'])}):")
        for fp in result["false_positives"][:10]:
            print(f"    t={fp['timestamp']:.1f}s  type={fp['shot_type']}  "
                  f"source={fp['source']}  conf={fp['confidence']:.2f}")

    if result["false_negatives"]:
        print(f"\n  False negatives ({len(result['false_negatives'])}):")
        for fn in result["false_negatives"][:10]:
            print(f"    t={fn['timestamp']:.1f}s  type={fn['shot_type']}")


def save_baseline(results, path=None):
    """Save validation results as a regression baseline."""
    os.makedirs(BASELINES_DIR, exist_ok=True)
    if path is None:
        # Auto-name with timestamp
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(BASELINES_DIR, f"baseline_{ts}.json")

    # Extract just the key metrics for comparison
    summary = {}
    for r in results:
        if r is None:
            continue
        summary[r["video"]] = {
            "f1": r["overall"]["f1"],
            "precision": r["overall"]["precision"],
            "recall": r["overall"]["recall"],
            "tp": r["overall"]["tp"],
            "fp": r["overall"]["fp"],
            "fn": r["overall"]["fn"],
            "classification_accuracy": r["classification_accuracy"],
        }

    baseline = {
        "timestamp": str(Path(path).stem),
        "videos": summary,
    }

    with open(path, "w") as f:
        json.dump(baseline, f, indent=2)
    print(f"\nBaseline saved to {path}")
    return path


def check_regression(results, threshold=0.01):
    """Check current results against most recent baseline.

    Flags any video where F1 dropped by more than threshold.
    Returns (passed, regressions) tuple.
    """
    if not os.path.isdir(BASELINES_DIR):
        print("No baselines directory found. Run with --save-baseline first.")
        return True, []

    baselines = sorted(
        f for f in os.listdir(BASELINES_DIR) if f.endswith(".json")
    )
    if not baselines:
        print("No baseline files found. Run with --save-baseline first.")
        return True, []

    latest = os.path.join(BASELINES_DIR, baselines[-1])
    with open(latest) as f:
        baseline = json.load(f)

    print(f"\nChecking against baseline: {baselines[-1]}")
    print(f"{'─'*60}")

    regressions = []
    for r in results:
        if r is None:
            continue
        vid = r["video"]
        if vid not in baseline.get("videos", {}):
            print(f"  {vid}: NEW (no baseline)")
            continue

        bl = baseline["videos"][vid]
        f1_diff = r["overall"]["f1"] - bl["f1"]
        p_diff = r["overall"]["precision"] - bl["precision"]
        r_diff = r["overall"]["recall"] - bl["recall"]

        status = "OK" if f1_diff >= -threshold else "REGRESSION"
        marker = "  " if f1_diff >= -threshold else ">>"

        print(f"  {marker} {vid}: F1 {bl['f1']:.1%} → {r['overall']['f1']:.1%} "
              f"({f1_diff:+.1%})  P {p_diff:+.1%}  R {r_diff:+.1%}  [{status}]")

        if f1_diff < -threshold:
            regressions.append({
                "video": vid,
                "f1_baseline": bl["f1"],
                "f1_current": r["overall"]["f1"],
                "f1_diff": round(f1_diff, 4),
            })

    passed = len(regressions) == 0
    if passed:
        print(f"\n  All videos passed regression check (threshold={threshold:.0%})")
    else:
        print(f"\n  REGRESSIONS DETECTED in {len(regressions)} video(s)!")
        for reg in regressions:
            print(f"    {reg['video']}: F1 dropped {reg['f1_diff']:.1%}")

    return passed, regressions


def generate_temporal_plot(results, output_path=None):
    """Generate temporal error analysis plot (matplotlib)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping temporal plot")
        return

    videos_with_bins = [r for r in results if r and r.get("temporal_bins")]
    if not videos_with_bins:
        return

    if output_path is None:
        output_path = os.path.join(PROJECT_ROOT, "training", "temporal_error_analysis.png")

    n_videos = len(videos_with_bins)
    fig, axes = plt.subplots(n_videos, 1, figsize=(12, 4 * n_videos), squeeze=False)

    for i, r in enumerate(videos_with_bins):
        ax = axes[i][0]
        bins = r["temporal_bins"]
        times = [(b["bin_start"] + b["bin_end"]) / 2 / 60 for b in bins]
        f1s = [b["f1"] for b in bins]
        fps = [b["fp"] for b in bins]
        fns = [b["fn"] for b in bins]

        ax.plot(times, f1s, "b-o", label="F1", markersize=4)
        ax.bar([t - 0.15 for t in times], fps, width=0.3, alpha=0.5,
               color="red", label="FP")
        ax.bar([t + 0.15 for t in times], fns, width=0.3, alpha=0.5,
               color="orange", label="FN")
        ax.set_title(f"{r['video']} — Temporal Error Analysis")
        ax.set_xlabel("Time (minutes)")
        ax.set_ylabel("F1 / Count")
        ax.set_ylim(-0.1, 1.1)
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"\nTemporal plot saved to {output_path}")


def _aggregate_metric_block(results, key):
    """Sum TP/FP/FN across results for the given metric block ('overall',
    'event_detection', 'event_classification') and compute aggregate."""
    tp = sum(r.get(key, {}).get("tp", 0) for r in results)
    fp = sum(r.get(key, {}).get("fp", 0) for r in results)
    fn = sum(r.get(key, {}).get("fn", 0) for r in results)
    m = compute_metrics(tp, fp, fn)
    return m, tp, fp, fn


def _print_aggregate(results):
    """Print overall aggregate (window + event-level)."""
    total_gt = sum(r["overall"]["gt_total"] for r in results)
    print(f"\n{'='*60}")
    print(f"  AGGREGATE ({len(results)} videos, {total_gt} GT shots)")
    print(f"{'='*60}")

    w, wtp, wfp, wfn = _aggregate_metric_block(results, "overall")
    ed, etp, efp, efn = _aggregate_metric_block(results, "event_detection")
    ec, ctp, cfp, cfn = _aggregate_metric_block(results, "event_classification")

    print(f"  Window  (±1.5s, class-agnostic):  TP={wtp:>4d} FP={wfp:>3d} FN={wfn:>3d}  "
          f"P={w['precision']:.1%}  R={w['recall']:.1%}  F1={w['f1']:.1%}")
    print(f"  Event   (±6f, class-agnostic):    TP={etp:>4d} FP={efp:>3d} FN={efn:>3d}  "
          f"P={ed['precision']:.1%}  R={ed['recall']:.1%}  F1={ed['f1']:.1%}")
    print(f"  Event   (±6f, class-strict):      TP={ctp:>4d} FP={cfp:>3d} FN={cfn:>3d}  "
          f"P={ec['precision']:.1%}  R={ec['recall']:.1%}  F1={ec['f1']:.1%}")

    # Aggregate contact-time MAE across all event-level matches.
    all_mae_ms = []
    for r in results:
        m = r.get("event_detection", {}).get("contact_mae_ms")
        n = r.get("event_detection", {}).get("tp", 0)
        if m is not None and n:
            all_mae_ms.append((m, n))
    if all_mae_ms:
        weighted = sum(m * n for m, n in all_mae_ms) / sum(n for _, n in all_mae_ms)
        print(f"  Contact-time MAE (event TPs): {weighted:.1f}ms")

    # Per-type aggregate, side by side
    window_type = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    event_type = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for r in results:
        for st, m in r.get("per_type", {}).items():
            window_type[st]["tp"] += m["tp"]; window_type[st]["fp"] += m["fp"]; window_type[st]["fn"] += m["fn"]
        for st, m in r.get("event_per_type", {}).items():
            event_type[st]["tp"] += m["tp"]; event_type[st]["fp"] += m["fp"]; event_type[st]["fn"] += m["fn"]

    print(f"\n  Per shot type (aggregate):")
    print(f"    {'type':<12s}  {'window F1 (±1.5s)':<32s}  {'event F1 (class-strict, ±6f)':<32s}")
    all_st = sorted(set(window_type.keys()) | set(event_type.keys()))
    for st in all_st:
        wm = compute_metrics(**window_type[st]) if st in window_type else None
        em = compute_metrics(**event_type[st]) if st in event_type else None
        ws = (f"TP={wm['tp']:3d} FP={wm['fp']:3d} FN={wm['fn']:3d} F1={wm['f1']:.1%}"
              if wm else "-")
        es = (f"TP={em['tp']:3d} FP={em['fp']:3d} FN={em['fn']:3d} F1={em['f1']:.1%}"
              if em else "-")
        print(f"    {st:<12s}  {ws:<32s}  {es:<32s}")


def _print_per_video_breakdown(results):
    """Per-video breakdown — verifies no video is dragging the aggregate
    down silently. Stratification by video is the FIRST split (no shot-level
    leakage by construction since each video is its own row)."""
    print(f"\n  Per-video breakdown (sorted by event-strict F1, lowest first):")
    print(f"    {'video':<12s}  {'GT':>4s}  {'win F1':>7s}  {'evt F1':>7s}  "
          f"{'cls F1':>7s}  {'MAE ms':>7s}  {'hand':<6s}  {'angle':<10s}")
    rows = []
    for r in results:
        ed = r.get("event_detection", {})
        ec = r.get("event_classification", {})
        rows.append((
            ec.get("f1", 0.0), r["video"],
            r["overall"]["gt_total"], r["overall"]["f1"],
            ed.get("f1", 0.0), ec.get("f1", 0.0),
            ed.get("contact_mae_ms"),
            (r.get("stratify") or {}).get("dominant_hand") or "-",
            (r.get("stratify") or {}).get("camera_angle") or "-",
        ))
    rows.sort(key=lambda x: x[0])
    for _, vid, n, w_f1, ed_f1, ec_f1, mae, hand, angle in rows:
        mae_s = f"{mae:.0f}" if mae is not None else "-"
        print(f"    {vid:<12s}  {n:>4d}  {w_f1:>7.1%}  {ed_f1:>7.1%}  "
              f"{ec_f1:>7.1%}  {mae_s:>7s}  {hand:<6s}  {angle:<10s}")


def _print_stratified_breakdown(results):
    """Aggregate event-level F1 broken out by handedness and camera angle.
    Tells you whether lefty / left-side-camera videos systematically
    underperform — the failure mode the adversarial review flagged."""

    def aggregate_for(predicate):
        subset = [r for r in results if predicate(r)]
        if not subset:
            return None
        wm, *_ = _aggregate_metric_block(subset, "overall")
        em, *_ = _aggregate_metric_block(subset, "event_detection")
        cm, *_ = _aggregate_metric_block(subset, "event_classification")
        gt = sum(r["overall"]["gt_total"] for r in subset)
        return {
            "n_videos": len(subset), "gt": gt,
            "window_f1": wm["f1"], "event_f1": em["f1"], "cls_f1": cm["f1"],
        }

    # By handedness
    print(f"\n  Stratified by handedness:")
    for hand in ("right", "left", None):
        label = hand if hand else "(unknown)"
        agg = aggregate_for(lambda r, h=hand: (r.get("stratify") or {}).get("dominant_hand") == h)
        if agg:
            print(f"    {label:<10s} videos={agg['n_videos']:>2d}  GT={agg['gt']:>5d}  "
                  f"win F1={agg['window_f1']:.1%}  evt F1={agg['event_f1']:.1%}  "
                  f"cls F1={agg['cls_f1']:.1%}")

    # By camera angle
    print(f"\n  Stratified by camera angle:")
    angles = sorted({(r.get("stratify") or {}).get("camera_angle") or "(unknown)"
                     for r in results})
    for ang in angles:
        target = ang if ang != "(unknown)" else None
        agg = aggregate_for(lambda r, t=target:
                            ((r.get("stratify") or {}).get("camera_angle") or None) == t)
        if agg:
            print(f"    {ang:<14s} videos={agg['n_videos']:>2d}  GT={agg['gt']:>5d}  "
                  f"win F1={agg['window_f1']:.1%}  evt F1={agg['event_f1']:.1%}  "
                  f"cls F1={agg['cls_f1']:.1%}")


def main():
    parser = argparse.ArgumentParser(
        description="Validate tennis shot detection pipeline")
    parser.add_argument("--video", help="Validate specific video (e.g. IMG_6703)")
    parser.add_argument("--gt", help="Ground truth file override")
    parser.add_argument("--det", help="Detection file override")
    parser.add_argument("--save-baseline", action="store_true",
                        help="Save current results as regression baseline")
    parser.add_argument("--check-regression", action="store_true",
                        help="Check against saved baseline")
    parser.add_argument("--threshold", type=float, default=0.01,
                        help="F1 regression threshold (default: 0.01 = 1%%)")
    parser.add_argument("--plot", action="store_true",
                        help="Generate temporal error analysis plot")
    parser.add_argument("--json-output", help="Save full results to JSON")
    parser.add_argument("--include-all-gt", action="store_true",
                        help="Validate against every detections/*_fused.json (not just "
                             "the 26 enrolled in GT_VIDEOS dict). Useful after adding "
                             "new labeled videos.")
    args = parser.parse_args()

    results = []

    if args.video:
        # Single video validation
        r = validate_video(args.video, gt_override=args.gt, det_path=args.det)
        if r:
            results.append(r)
    elif args.include_all_gt:
        # Discover all GT files in detections/, treat each as ground truth
        gt_videos = []
        for f in sorted(os.listdir(DETECTIONS_DIR)):
            if not f.endswith("_fused.json"):
                continue
            if any(s in f for s in ("_detections", "_v2", "_v5", "_ml", "_pre", "_baseline")):
                continue
            name = f.replace("_fused.json", "")
            if name in {"IMG_0993", "IMG_0995", "IMG_6712"}:  # known-bad GT
                continue
            gt_videos.append((name, f"detections/{f}"))
        print(f"Validating {len(gt_videos)} GT videos (--include-all-gt)...")
        for name, gt_path in gt_videos:
            # Skip if GT file matches the only candidate detection file (no fresh inference)
            r = validate_video(name, gt_override=gt_path)
            if r:
                results.append(r)
    else:
        # Validate all GT videos enrolled in the dict (regression-baseline scope)
        print(f"Validating {len(GT_VIDEOS)} ground truth videos...")
        for video_name in sorted(GT_VIDEOS.keys()):
            r = validate_video(video_name)
            if r:
                results.append(r)

    if not results:
        print("No validation results produced.")
        return

    # ── Aggregate summary ─────────────────────────────────────────
    if len(results) > 1:
        _print_aggregate(results)
        _print_per_video_breakdown(results)
        _print_stratified_breakdown(results)

    # ── Regression check ──────────────────────────────────────────
    if args.check_regression:
        check_regression(results, threshold=args.threshold)

    # ── Save baseline ─────────────────────────────────────────────
    if args.save_baseline:
        save_baseline(results)

    # ── Temporal plot ─────────────────────────────────────────────
    if args.plot:
        generate_temporal_plot(results)

    # ── JSON output ───────────────────────────────────────────────
    if args.json_output:
        os.makedirs(os.path.dirname(args.json_output) or ".", exist_ok=True)
        with open(args.json_output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nFull results saved to {args.json_output}")


if __name__ == "__main__":
    main()
