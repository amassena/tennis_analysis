# 3D-lifting investigation — frozen criteria

**Status:** Phase 1 complete. Phase 2 not warranted (see REPORT.md for reasoning).

**Predecessor brief:** `.handoffs/inbox/20260509-0140-world-landmarks-investigation-from-main.md`

## The question (per scoping brief)

Is MediaPipe's `world_landmarks` (its synthesized 2D→3D lifting) the
load-bearing weakness for biomech outputs? If yes, test alternative lifting
heads (MotionBERT, HMR2). If no, redirect.

## Phase 1: heuristic audit (~half day, no GPU) — DONE

For each GT shot in 3 representative videos (IMG_0996, IMG_6874, IMG_6851 — covering BH-heavy / FH-heavy / serve-heavy), run `biomechanical_analysis.py` and flag if any of:

- `knee_angle_at_contact` outside [120°, 180°]
- `trunk_rotation_at_contact` magnitude > 90°
- `arm_extension_at_contact` outside [60°, 180°]
- `kinetic_chain_timing_ms` with reversed order (any segment arriving > 10 ms before its predecessor in hip → shoulder → elbow → wrist)
- Any of those scalar values reported as exactly 0.0 or 180.0 (degenerate)

For each flagged shot: render a PNG showing the original frame, MediaPipe 2D
keypoints overlaid, and a side panel of biomech values + flag reasons.
Manually inspect a sample.

## Phase 1 locked verdict criteria

(Frozen at experiment start; do not edit post-hoc.)

- **CONFIRMED weakness if** ≥ 60 % of flagged shots show plausible 2D pose AND implausible 3D body-frame.
- **NOT THE BOTTLENECK if** ≤ 30 % of flagged shots fit that pattern.
- **AMBIGUOUS** in between → look harder; may need ground-truth 3D annotations.

## Phase 2: alternative 2D→3D lifting (gated on Phase 1)

Only run if Phase 1 confirmed weakness. Verdict: **not run** — Phase 1 found
the bottleneck wasn't `world_landmarks` quality.

## Phase 3: decide

Rolled into REPORT.md as the closing recommendations section.
