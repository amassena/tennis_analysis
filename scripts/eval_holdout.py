#!/usr/bin/env python3
"""Evaluate a shot-detection model against the frozen holdout.

One command in, JSON results out. Used as the input to compare_models.py
deploy gate, and to populate the holdout_eval_results field on a model
sidecar.

Usage:
    python scripts/eval_holdout.py models/sequence_detector.pt
    python scripts/eval_holdout.py path/to/candidate.pt --output eval_results/foo.json
    python scripts/eval_holdout.py path/to/candidate.pt --manifest eval/holdout/manifest.json

Output:
    eval_results/{model_sha256_short}_{date}.json with the schema documented
    in the verification-system spec.

Reuses match_detections + event_tolerance_seconds from validate_pipeline.py
and detect_video from detect_shots_sequence.py — same metric machinery the
overall pipeline already uses.
"""

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

from scripts.validate_pipeline import (
    match_detections,
    event_tolerance_seconds,
    compute_metrics,
    IGNORE_TYPES,
    MATCH_TOLERANCE,  # legacy 1.5s shot-level tolerance — the basis of the F1
                      # number quoted in CLAUDE.md / docs
)
from scripts.detect_shots_sequence import detect_video
from scripts.sequence_model import load_model

DEFAULT_MANIFEST = PROJECT_ROOT / "eval" / "holdout" / "manifest.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "eval_results"

# Detection knobs — match production worker config
DETECT_THRESHOLD = 0.90
DETECT_NMS_GAP = 1.5
DETECT_STEP_SEC = 0.1


def sha256_of(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def short(h):
    return h[:8]


def _path_for_sidecar(p):
    """Return path-string suitable for embedding in a sidecar.

    Prefers a path relative to PROJECT_ROOT for portability across machines.
    Falls back to absolute string for paths outside the project (e.g. /tmp).
    """
    p = Path(p)
    try:
        return str(p.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(p)


def normalize_gt_shots(gt_data):
    """Strip ignored types (practice, offscreen, unknown_shot) from GT."""
    out = []
    for d in gt_data.get("detections", []):
        st = d.get("shot_type", "")
        if st in IGNORE_TYPES:
            continue
        out.append(d)
    return out


def normalize_det_shots(det_data):
    """Strip ignored types from detector output too."""
    return normalize_gt_shots(det_data)


def per_class_metrics(gt_shots, det_shots, tolerance):
    """For each class present in GT, compute precision/recall/F1 using
    class-strict event matching."""
    classes = sorted({d["shot_type"] for d in gt_shots})
    out = {}
    for cls in classes:
        gt_cls = [g for g in gt_shots if g["shot_type"] == cls]
        # Class-strict: detector must agree on shot_type within tolerance
        tp_pairs, fp_dets, fn_gts = match_detections(
            gt_cls,
            [d for d in det_shots if d["shot_type"] == cls],
            tolerance=tolerance,
            require_class_match=True,
        )
        m = compute_metrics(len(tp_pairs), len(fp_dets), len(fn_gts))
        m["support"] = len(gt_cls)
        out[cls] = m
    return out


def compute_confusion(gt_shots, det_shots, tolerance):
    """Class-agnostic match first (did we find the shot?), then check
    class agreement. Plus a 'not_shot' column for unmatched detections
    (FPs at the right time but in space where no GT exists)."""
    classes = sorted({d["shot_type"] for d in gt_shots} | {"not_shot"})
    confusion = {gt_cls: {pred_cls: 0 for pred_cls in classes}
                 for gt_cls in classes if gt_cls != "not_shot"}
    confusion["not_shot"] = {pred_cls: 0 for pred_cls in classes}

    tp_pairs, fp_dets, fn_gts = match_detections(
        gt_shots, det_shots, tolerance=tolerance, require_class_match=False)
    for gt, det, _err in tp_pairs:
        gt_cls = gt["shot_type"]
        pred_cls = det["shot_type"]
        if gt_cls in confusion and pred_cls in confusion[gt_cls]:
            confusion[gt_cls][pred_cls] += 1
    for gt in fn_gts:
        gt_cls = gt["shot_type"]
        if gt_cls in confusion:
            confusion[gt_cls]["not_shot"] += 1
    for det in fp_dets:
        pred_cls = det["shot_type"]
        if pred_cls in confusion["not_shot"]:
            confusion["not_shot"][pred_cls] += 1
    return confusion


def calibration_curve(gt_shots, det_shots, tolerance, n_bins=10):
    """Return reliability-diagram bins. ECE is computed from these."""
    tp_pairs, fp_dets, _fn_gts = match_detections(
        gt_shots, det_shots, tolerance=tolerance, require_class_match=False)
    matched_dets = {id(d): True for _g, d, _e in tp_pairs}
    items = []  # (confidence, is_correct)
    for _g, d, _e in tp_pairs:
        items.append((d.get("confidence", 0.0), 1))
    for d in fp_dets:
        items.append((d.get("confidence", 0.0), 0))
    if not items:
        return [], 0.0
    bins = []
    ece = 0.0
    n = len(items)
    for b in range(n_bins):
        lo = b / n_bins
        hi = (b + 1) / n_bins
        bin_items = [x for x in items if lo <= x[0] < hi or (b == n_bins - 1 and x[0] == 1.0)]
        if not bin_items:
            bins.append({"bin_low": lo, "bin_high": hi, "frac_positive": None,
                         "avg_confidence": None, "n": 0})
            continue
        frac_pos = sum(c for _, c in bin_items) / len(bin_items)
        avg_conf = sum(c for c, _ in bin_items) / len(bin_items)
        bins.append({"bin_low": lo, "bin_high": hi,
                     "frac_positive": round(frac_pos, 4),
                     "avg_confidence": round(avg_conf, 4),
                     "n": len(bin_items)})
        ece += (len(bin_items) / n) * abs(frac_pos - avg_conf)
    return bins, round(ece, 4)


def evaluate(model_path, manifest_path, output_path):
    if not os.path.exists(model_path):
        sys.exit(f"[ERR] model not found: {model_path}")
    if not os.path.exists(manifest_path):
        sys.exit(f"[ERR] manifest not found: {manifest_path}")

    model_sha = sha256_of(model_path)
    manifest_sha = sha256_of(manifest_path)
    manifest = json.loads(open(manifest_path).read())

    print(f"[eval_holdout] model: {model_path} ({short(model_sha)})")
    print(f"[eval_holdout] manifest: {manifest_path} ({short(manifest_sha)})")
    print(f"[eval_holdout] {len(manifest['videos'])} holdout videos")

    model, device = load_model(model_path)
    if model is None:
        sys.exit("[ERR] could not load model — torch unavailable or bad checkpoint")
    print(f"[eval_holdout] loaded model: num_classes={getattr(model, 'num_classes', '?')} "
          f"on {device}")

    # Aggregate accumulators
    all_tp_pairs = []
    all_fp_dets = []
    all_fn_gts = []
    all_det_shots = []
    all_gt_shots = []
    videos_with_any_miss = 0
    per_video = []

    t0 = time.time()
    for v in manifest["videos"]:
        vid = v["video_id"]
        video_path = PROJECT_ROOT / v["video_path"]
        gt_path = PROJECT_ROOT / v["gt_path"]
        if not video_path.exists():
            print(f"  [WARN] missing video file for {vid}: {video_path}")
            continue
        if not gt_path.exists():
            print(f"  [WARN] missing GT file for {vid}: {gt_path}")
            continue

        print(f"\n  -- {vid} ({v.get('camera_angle','?')}) --")
        gt_data = json.loads(gt_path.read_text())
        gt_shots = normalize_gt_shots(gt_data)

        det_data = detect_video(
            str(video_path), model, device,
            threshold=DETECT_THRESHOLD,
            nms_gap=DETECT_NMS_GAP,
            step_sec=DETECT_STEP_SEC,
        )
        if det_data is None:
            print(f"  [WARN] detection failed for {vid}")
            continue
        det_shots = normalize_det_shots(det_data)

        tol_tight_sec, fps_used = event_tolerance_seconds(det_data)
        # Headline F1 uses the legacy 1.5s shot-level tolerance — this is the
        # number CLAUDE.md and stakeholders track. Also report a tighter event
        # tolerance (6 frames ≈ 0.1s) to expose contact-time precision.
        tp_any, fp_any, fn_any = match_detections(
            gt_shots, det_shots, tolerance=MATCH_TOLERANCE, require_class_match=False)
        tp_strict, fp_strict, fn_strict = match_detections(
            gt_shots, det_shots, tolerance=MATCH_TOLERANCE, require_class_match=True)
        # Tight event-level (better signal for contact-time work)
        tp_tight, fp_tight, fn_tight = match_detections(
            gt_shots, det_shots, tolerance=tol_tight_sec, require_class_match=True)

        per_video.append({
            "video_id": vid,
            "camera_angle": v.get("camera_angle"),
            "shots_gt": len(gt_shots),
            "shots_predicted": len(det_shots),
            "tol_legacy_sec": MATCH_TOLERANCE,
            "tol_tight_sec": round(tol_tight_sec, 4),
            "fps_used": fps_used,
            "shot_any_f1": compute_metrics(len(tp_any), len(fp_any), len(fn_any))["f1"],
            "shot_strict_f1": compute_metrics(len(tp_strict), len(fp_strict), len(fn_strict))["f1"],
            "event_strict_f1": compute_metrics(len(tp_tight), len(fp_tight), len(fn_tight))["f1"],
            "false_negatives": len(fn_strict),
            "false_positives": len(fp_strict),
        })
        if len(fn_strict) > 0 or len(fp_strict) > 0:
            videos_with_any_miss += 1

        all_tp_pairs.extend(tp_strict)
        all_fp_dets.extend(fp_strict)
        all_fn_gts.extend(fn_strict)
        all_gt_shots.extend(gt_shots)
        all_det_shots.extend(det_shots)

        print(f"     gt={len(gt_shots)} pred={len(det_shots)} "
              f"shot_F1={per_video[-1]['shot_strict_f1']:.3f} (1.5s) "
              f"event_F1={per_video[-1]['event_strict_f1']:.3f} (0.1s) "
              f"FN={len(fn_strict)} FP={len(fp_strict)}")

    elapsed = time.time() - t0

    # Aggregate metrics. Headline at MATCH_TOLERANCE (1.5s legacy);
    # also compute a tight 0.1s event-level F1 for contact-time signal.
    overall = compute_metrics(len(all_tp_pairs), len(all_fp_dets), len(all_fn_gts))
    pc = per_class_metrics(all_gt_shots, all_det_shots, tolerance=MATCH_TOLERANCE)
    confusion = compute_confusion(all_gt_shots, all_det_shots, tolerance=MATCH_TOLERANCE)
    calib_bins, ece = calibration_curve(all_gt_shots, all_det_shots, tolerance=MATCH_TOLERANCE)
    # Tight event-level
    tp_tight_all, fp_tight_all, fn_tight_all = match_detections(
        all_gt_shots, all_det_shots, tolerance=0.1, require_class_match=True)
    event_tight = compute_metrics(len(tp_tight_all), len(fp_tight_all), len(fn_tight_all))

    out = {
        "schema_version": 1,
        "model_sha256": model_sha,
        "model_path": _path_for_sidecar(model_path),
        "manifest_sha256": manifest_sha,
        "manifest_path": _path_for_sidecar(manifest_path),
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "evaluated_on": socket.gethostname(),
        "elapsed_seconds": round(elapsed, 1),
        "detect_config": {
            "threshold": DETECT_THRESHOLD,
            "nms_gap": DETECT_NMS_GAP,
            "step_sec": DETECT_STEP_SEC,
        },
        "event_level_f1": overall["f1"],          # legacy 1.5s shot-level — HEADLINE
        "event_level_precision": overall["precision"],
        "event_level_recall": overall["recall"],
        "tolerances": {
            "headline_sec": MATCH_TOLERANCE,
            "tight_sec": 0.1,
        },
        "event_tight_f1": event_tight["f1"],       # 0.1s tolerance — contact-time precision
        "event_tight_precision": event_tight["precision"],
        "event_tight_recall": event_tight["recall"],
        "per_class": pc,
        "ece": ece,
        "confusion": confusion,
        "videos_with_any_miss": videos_with_any_miss,
        "videos_total": len(per_video),
        "total_shots_gt": len(all_gt_shots),
        "total_shots_predicted": len(all_det_shots),
        "false_negatives": len(all_fn_gts),
        "false_positives": len(all_fp_dets),
        "calibration_curve": calib_bins,
        "per_video": per_video,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, indent=2))
    print(f"\n[eval_holdout] event_level_F1={overall['f1']:.3f} (legacy 1.5s, headline) "
          f"event_tight_F1={event_tight['f1']:.3f} (0.1s) "
          f"FN={len(all_fn_gts)} FP={len(all_fp_dets)} "
          f"videos_with_miss={videos_with_any_miss}/{len(per_video)} "
          f"ECE={ece:.3f}")
    print(f"[eval_holdout] wrote {output_path}")

    # Update sidecar if it exists
    sidecar_path = Path(model_path).with_suffix(Path(model_path).suffix + ".sidecar.json")
    if sidecar_path.exists():
        try:
            sc = json.loads(sidecar_path.read_text())
            sc["holdout_eval_results"] = _path_for_sidecar(output_path)
            sc["holdout_eval_at"] = out["evaluated_at"]
            sidecar_path.write_text(json.dumps(sc, indent=2))
            print(f"[eval_holdout] updated sidecar: {sidecar_path.name}")
        except Exception as e:
            print(f"[eval_holdout] sidecar update failed: {e}")

    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("model", help="Path to .pt model")
    p.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    p.add_argument("--output", default=None,
                   help="Output JSON path (default: eval_results/<model_sha256_short>_<date>.json)")
    args = p.parse_args()

    model_sha = sha256_of(args.model)
    if args.output:
        out_path = Path(args.output)
    else:
        date = datetime.now(timezone.utc).strftime("%Y%m%d")
        out_path = DEFAULT_OUTPUT_DIR / f"{short(model_sha)}_{date}.json"

    evaluate(args.model, args.manifest, out_path)


if __name__ == "__main__":
    main()
