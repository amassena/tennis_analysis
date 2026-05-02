#!/usr/bin/env python3
"""Backfill <model>.sidecar.json for an existing .pt that wasn't trained
through the new path. The sidecar carries the model identity — without
one, the worker hash check refuses to start.

Usage:
    scripts/backfill_sidecars.py models/baseline_bbe8a42b_20260502.pt \
        --classes backhand,forehand,not_shot,serve --deploy-status approved

    scripts/backfill_sidecars.py models/baseline_28814eeb_BROKEN_20260502.pt \
        --classes backhand,backhand_volley,forehand,forehand_volley,not_shot,serve \
        --deploy-status broken
"""
import argparse
import hashlib
import json
import os
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

VALID_DEPLOY = {"candidate", "approved", "retired", "broken"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model_path", help="Path to .pt file")
    p.add_argument("--classes", required=True,
                   help="Comma-separated class list, e.g. backhand,forehand,not_shot,serve")
    p.add_argument("--deploy-status", default="candidate", choices=sorted(VALID_DEPLOY))
    p.add_argument("--note", help="Optional human-readable note appended to sidecar")
    p.add_argument("--force", action="store_true",
                   help="Overwrite sidecar if it already exists")
    args = p.parse_args()

    model_path = Path(args.model_path).resolve()
    if not model_path.exists():
        sys.exit(f"[ERR] no model at {model_path}")

    sidecar_path = model_path.with_suffix(model_path.suffix + ".sidecar.json")
    if sidecar_path.exists() and not args.force:
        sys.exit(f"[ERR] sidecar already exists: {sidecar_path}\nPass --force to overwrite.")

    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    if not classes:
        sys.exit("[ERR] --classes empty")

    model_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()

    # Try to read companion meta.json if it sits next to the .pt as
    # `<stem>_meta.json` (the trainer wrote those before sidecars existed).
    meta_path = model_path.with_name(model_path.stem + "_meta.json")
    legacy_meta = {}
    if meta_path.exists():
        try:
            legacy_meta = json.loads(meta_path.read_text())
        except Exception:
            pass

    sidecar = {
        "schema_version": 1,
        "model_sha256": model_sha,
        "model_path": str(model_path.relative_to(PROJECT_ROOT))
                       if str(model_path).startswith(str(PROJECT_ROOT))
                       else str(model_path),
        "trained_at": legacy_meta.get("trained_at"),
        "trained_on": legacy_meta.get("trained_on"),
        "trained_commit": legacy_meta.get("trained_commit"),
        "trained_command": legacy_meta.get("trained_command"),
        "training_data_sha256": legacy_meta.get("training_data_sha256"),
        "training_data_manifest": legacy_meta.get("training_data_manifest"),
        "classes": classes,
        "architecture": legacy_meta.get("architecture"),
        "loocv_accuracy": legacy_meta.get("loocv_accuracy"),
        "holdout_eval_results": None,
        "holdout_eval_at": None,
        "deploy_status": args.deploy_status,
        "backfilled_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "backfilled_on": socket.gethostname(),
    }
    if args.note:
        sidecar["note"] = args.note

    sidecar_path.write_text(json.dumps(sidecar, indent=2))
    print(f"[backfill] {sidecar_path}")
    print(f"  sha256: {model_sha}")
    print(f"  classes: {classes}")
    print(f"  deploy_status: {args.deploy_status}")


if __name__ == "__main__":
    main()
