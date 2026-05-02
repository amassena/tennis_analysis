#!/usr/bin/env python3
"""Compare two shot-detection models on the holdout, apply deploy gate.

Reads each model's sidecar to get its eval result JSON, runs eval if missing,
prints a Markdown comparison table, exits 0 (PASS) or 1 (BLOCK) per the
gate rules in:

  ~/.claude/projects/-Users-andrewhome/memory/project_model_deploy_gate_rules.md

The exit code IS the gate. Wire it into pre-deploy / pre-merge checks.

Usage:
    # Default: read eval results from sidecars, auto-run if missing
    python scripts/compare_models.py models/baseline.pt models/candidate.pt

    # Override eval JSONs explicitly
    python scripts/compare_models.py BASE.pt CAND.pt \\
        --baseline-eval eval_results/base.json --candidate-eval eval_results/cand.json

    # Strict: any per-class regression (within or above soft threshold) blocks
    python scripts/compare_models.py --strict BASE.pt CAND.pt
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_MANIFEST = PROJECT_ROOT / "eval" / "holdout" / "manifest.json"

# ── Deploy gate thresholds (memorized — see deploy_gate_rules.md) ──
HEADLINE_F1_REGRESSION_LIMIT = 0.003     # > this regression on event_level_f1 → BLOCK
PER_CLASS_F1_REGRESSION_LIMIT = 0.005    # > this regression on any class → BLOCK unless offset
PER_CLASS_OFFSET_GAIN = 0.010            # gain of this much on another class permits a soft regression
ECE_INCREASE_LIMIT = 0.010               # absolute ECE increase that triggers BLOCK


# ── Plumbing ──

def load_sidecar(model_path: Path) -> dict:
    sidecar_path = model_path.with_suffix(model_path.suffix + ".sidecar.json")
    if not sidecar_path.exists():
        sys.exit(
            f"FATAL: no sidecar for {model_path}\n"
            f"  expected at: {sidecar_path}\n"
            f"  Fix: scripts/backfill_sidecars.py {model_path} --classes <...> --deploy-status <...>"
        )
    return json.loads(sidecar_path.read_text())


def resolve_eval_path(p: str | None) -> Path | None:
    if not p:
        return None
    path = Path(p)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path if path.exists() else None


def run_eval_if_needed(model_path: Path, sidecar: dict, override: Path | None,
                       manifest: Path) -> Path:
    """Return the path to the eval-results JSON for this model. Runs
    eval_holdout.py if neither the override nor the sidecar's
    holdout_eval_results points to an existing file."""
    if override and override.exists():
        return override
    sidecar_eval = resolve_eval_path(sidecar.get("holdout_eval_results"))
    if sidecar_eval and sidecar_eval.exists():
        return sidecar_eval

    print(f"[compare] no eval result for {model_path.name} — running eval_holdout.py...",
          file=sys.stderr)
    eval_script = PROJECT_ROOT / "scripts" / "eval_holdout.py"
    cmd = [sys.executable, str(eval_script), str(model_path), "--manifest", str(manifest)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"[compare] eval_holdout.py failed:\n{r.stderr[-1000:]}")

    # eval_holdout writes to eval_results/<sha8>_<date>.json and updates the sidecar.
    # Re-read sidecar to find the path.
    fresh_sidecar = load_sidecar(model_path)
    eval_p = resolve_eval_path(fresh_sidecar.get("holdout_eval_results"))
    if not eval_p:
        sys.exit(f"[compare] eval_holdout.py ran but sidecar has no holdout_eval_results")
    return eval_p


def load_eval(p: Path) -> dict:
    d = json.loads(p.read_text())
    return d


# ── Gate logic ──

def evaluate_gate(baseline: dict, candidate: dict,
                  baseline_classes: list[str], candidate_classes: list[str],
                  strict: bool = False) -> tuple[bool, list[str], list[str]]:
    """Return (passes, block_reasons, warnings)."""
    blocks: list[str] = []
    warns: list[str] = []

    # Rule 5: schema additions require human override
    base_classes = set(baseline_classes or [])
    cand_classes = set(candidate_classes or [])
    extra = cand_classes - base_classes
    missing = base_classes - cand_classes
    if missing:
        blocks.append(
            f"candidate is missing baseline classes: {sorted(missing)} "
            f"(rule 5 — schema must be a superset of baseline)"
        )
    if extra:
        blocks.append(
            f"candidate adds classes not in baseline: {sorted(extra)} "
            f"(rule 5 — schema additions require explicit human override, not auto-pass)"
        )

    # Rule 1: headline F1 regression
    base_f1 = baseline.get("event_level_f1", 0.0)
    cand_f1 = candidate.get("event_level_f1", 0.0)
    delta_f1 = cand_f1 - base_f1
    if -delta_f1 > HEADLINE_F1_REGRESSION_LIMIT:
        blocks.append(
            f"event_level_f1 regressed by {delta_f1:+.4f} "
            f"(> {HEADLINE_F1_REGRESSION_LIMIT} threshold) — rule 1"
        )

    # Rule 2: per-class regressions w/ offset rule
    base_pc = baseline.get("per_class", {}) or {}
    cand_pc = candidate.get("per_class", {}) or {}
    per_class_deltas: dict[str, float] = {}
    for cls in sorted(set(base_pc) | set(cand_pc)):
        b = base_pc.get(cls, {}).get("f1", 0.0) if cls in base_pc else None
        c = cand_pc.get(cls, {}).get("f1", 0.0) if cls in cand_pc else None
        if b is None or c is None:
            continue
        per_class_deltas[cls] = round(c - b, 4)

    largest_gain = max((d for d in per_class_deltas.values()), default=0.0)
    for cls, d in per_class_deltas.items():
        if d < 0 and -d > PER_CLASS_F1_REGRESSION_LIMIT:
            # Hard regression — needs offsetting gain on another class
            other_gains = [v for k, v in per_class_deltas.items() if k != cls and v > 0]
            best_offset = max(other_gains, default=0.0)
            if best_offset >= PER_CLASS_OFFSET_GAIN:
                warns.append(
                    f"{cls} F1 regressed by {d:+.4f} (> {PER_CLASS_F1_REGRESSION_LIMIT}), "
                    f"but offset by {best_offset:+.4f} on another class — rule 2 (passes)"
                )
            else:
                blocks.append(
                    f"{cls} F1 regressed by {d:+.4f} (> {PER_CLASS_F1_REGRESSION_LIMIT}) "
                    f"with no offsetting gain ≥ {PER_CLASS_OFFSET_GAIN} on another class — rule 2"
                )
        elif d < 0 and -d <= PER_CLASS_F1_REGRESSION_LIMIT:
            warns.append(
                f"{cls} F1 regressed by {d:+.4f} (within {PER_CLASS_F1_REGRESSION_LIMIT} soft threshold)"
            )

    # Rule 3: ECE increase
    base_ece = baseline.get("ece", 0.0)
    cand_ece = candidate.get("ece", 0.0)
    delta_ece = cand_ece - base_ece
    if delta_ece > ECE_INCREASE_LIMIT:
        blocks.append(
            f"ECE increased by {delta_ece:+.4f} (> {ECE_INCREASE_LIMIT}) — rule 3"
        )

    # Rule 4: videos_with_any_miss must not increase
    base_misses = baseline.get("videos_with_any_miss", 0)
    cand_misses = candidate.get("videos_with_any_miss", 0)
    if cand_misses > base_misses:
        blocks.append(
            f"videos_with_any_miss increased: {base_misses} → {cand_misses} — rule 4"
        )

    if strict and warns and not blocks:
        blocks.append("--strict mode: warnings promoted to blocks")

    return (len(blocks) == 0), blocks, warns


# ── Output ──

def fmt_delta(d: float, fmt: str = "{:+.4f}") -> str:
    if d == 0:
        return "  0     "
    return fmt.format(d)


def status_glyph(d: float, lower_better: bool = False) -> str:
    if d == 0:
        return "✓"
    if lower_better:
        return "✓" if d < 0 else "⚠"
    return "✓" if d > 0 else "⚠"


def print_table(baseline_path: Path, baseline_sidecar: dict,
                candidate_path: Path, candidate_sidecar: dict,
                baseline_eval: dict, candidate_eval: dict,
                manifest: Path):
    base_sha = baseline_sidecar.get("model_sha256", "?")[:8]
    cand_sha = candidate_sidecar.get("model_sha256", "?")[:8]
    base_status = baseline_sidecar.get("deploy_status", "?")
    cand_status = candidate_sidecar.get("deploy_status", "?")

    manifest_sha = baseline_eval.get("manifest_sha256", "?")[:8]
    n_videos = baseline_eval.get("videos_total", "?")
    total_gt = baseline_eval.get("total_shots_gt", "?")

    print()
    print("=== Model comparison ===")
    print(f"Baseline:  {baseline_path}  (sha256: {base_sha}…, deploy_status: {base_status})")
    print(f"Candidate: {candidate_path}  (sha256: {cand_sha}…, deploy_status: {cand_status})")
    print(f"Holdout:   {manifest}  ({n_videos} videos, {total_gt} GT shots, manifest_sha256: {manifest_sha}…)")
    print()

    rows = []

    def row(label, b, c, lower_better=False, fmt="{:.4f}", delta_fmt="{:+.4f}"):
        if b is None or c is None:
            rows.append((label, str(b), str(c), "—", "—"))
            return
        delta = c - b
        glyph = status_glyph(delta, lower_better=lower_better)
        rows.append((label, fmt.format(b), fmt.format(c), delta_fmt.format(delta), glyph))

    row("event_level_f1 (1.5s)", baseline_eval.get("event_level_f1"), candidate_eval.get("event_level_f1"))
    row("event_tight_f1 (0.1s)", baseline_eval.get("event_tight_f1"), candidate_eval.get("event_tight_f1"))

    pc_b = baseline_eval.get("per_class") or {}
    pc_c = candidate_eval.get("per_class") or {}
    for cls in sorted(set(pc_b) | set(pc_c)):
        bf = pc_b.get(cls, {}).get("f1") if cls in pc_b else None
        cf = pc_c.get(cls, {}).get("f1") if cls in pc_c else None
        if bf is None and cf is None:
            continue
        row(f"  {cls} F1", bf, cf)

    row("ECE (lower better)", baseline_eval.get("ece"), candidate_eval.get("ece"), lower_better=True)
    row("videos_with_any_miss",
        baseline_eval.get("videos_with_any_miss"), candidate_eval.get("videos_with_any_miss"),
        lower_better=True, fmt="{}", delta_fmt="{:+d}")
    row("false_negatives",
        baseline_eval.get("false_negatives"), candidate_eval.get("false_negatives"),
        lower_better=True, fmt="{}", delta_fmt="{:+d}")
    row("false_positives",
        baseline_eval.get("false_positives"), candidate_eval.get("false_positives"),
        lower_better=True, fmt="{}", delta_fmt="{:+d}")

    base_classes = baseline_sidecar.get("classes", [])
    cand_classes = candidate_sidecar.get("classes", [])
    classes_match = set(base_classes) == set(cand_classes)
    rows.append((
        "classes match baseline",
        ",".join(base_classes) if base_classes else "?",
        ",".join(cand_classes) if cand_classes else "?",
        "—", "✓" if classes_match else "⚠",
    ))

    # Pretty print fixed-width
    headers = ("metric", "baseline", "candidate", "Δ", "")
    cols = list(zip(*([headers] + rows)))
    widths = [max(len(str(c)) for c in col) for col in cols]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print(fmt.format(*r))
    print()


# ── Entry ──

def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("baseline", help="Baseline model .pt")
    p.add_argument("candidate", help="Candidate model .pt")
    p.add_argument("--baseline-eval", help="Override baseline eval JSON path")
    p.add_argument("--candidate-eval", help="Override candidate eval JSON path")
    p.add_argument("--manifest", default=str(DEFAULT_MANIFEST),
                   help=f"Holdout manifest (default {DEFAULT_MANIFEST})")
    p.add_argument("--strict", action="store_true",
                   help="Soft warnings promote to BLOCK")
    args = p.parse_args()

    baseline_path = Path(args.baseline).resolve()
    candidate_path = Path(args.candidate).resolve()
    manifest_path = Path(args.manifest).resolve()

    if not baseline_path.exists():
        sys.exit(f"FATAL: baseline not found: {baseline_path}")
    if not candidate_path.exists():
        sys.exit(f"FATAL: candidate not found: {candidate_path}")
    if not manifest_path.exists():
        sys.exit(f"FATAL: manifest not found: {manifest_path}")

    baseline_sidecar = load_sidecar(baseline_path)
    candidate_sidecar = load_sidecar(candidate_path)

    baseline_eval_path = run_eval_if_needed(
        baseline_path, baseline_sidecar,
        Path(args.baseline_eval).resolve() if args.baseline_eval else None,
        manifest_path,
    )
    candidate_eval_path = run_eval_if_needed(
        candidate_path, candidate_sidecar,
        Path(args.candidate_eval).resolve() if args.candidate_eval else None,
        manifest_path,
    )

    baseline_eval = load_eval(baseline_eval_path)
    candidate_eval = load_eval(candidate_eval_path)

    print_table(baseline_path, baseline_sidecar,
                candidate_path, candidate_sidecar,
                baseline_eval, candidate_eval, manifest_path)

    passes, blocks, warns = evaluate_gate(
        baseline_eval, candidate_eval,
        baseline_sidecar.get("classes", []),
        candidate_sidecar.get("classes", []),
        strict=args.strict,
    )

    if warns:
        print("Warnings:")
        for w in warns:
            print(f"  ⚠ {w}")
        print()

    if passes:
        print("DEPLOY DECISION: PASS")
        print("  All gate rules passed. Candidate may be promoted to "
              "deploy_status: approved (human flips the sidecar).")
        sys.exit(0)
    else:
        print("DEPLOY DECISION: BLOCK")
        print(f"Reason{'s' if len(blocks) > 1 else ''}:")
        for b in blocks:
            print(f"  ✗ {b}")
        sys.exit(1)


if __name__ == "__main__":
    main()
