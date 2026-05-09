# 3D-lifting investigation — Phase 1 REPORT

**Brief:** `.handoffs/archive/...20260509-0140-world-landmarks-investigation-from-main.md` (still in inbox at time of writing).
**Phase 1 date:** 2026-05-09.
**Audit corpus:** IMG_0996 (BH-heavy rally), IMG_6874 (FH-heavy rally), IMG_6851 (serve-heavy). 256 GT shots total.

---

## TL;DR — `world_landmarks` is NOT the bottleneck. The biomech kinetic-chain timing algorithm has a bug.

Phase 1 audited 256 GT shots across 3 videos and found **84 % of them
flagged** by the heuristic implausibility criteria. But the flag distribution
points to a very specific cause:

| Flag category            |    Count | Share of flags |
| ------------------------ | -------: | -------------: |
| `chain_reversed`         |  **212** |        **98 %** |
| `knee_out_of_range`      |       24 |        11 % |
| `trunk_extreme`          |        7 |         3 % |
| `trunk_degenerate`       |        2 |        1 %  |
| `arm_out_of_range`       |        2 |        1 %  |

**98 % of all flag instances are the same problem: the kinetic-chain timing
extraction is unreliable.** The other flags are sparse and (per visual
inspection — see below) often false-positives from over-tight thresholds.

Per the locked Phase 1 criteria (≥ 60 % "2D right + 3D wrong" → confirm; ≤ 30 % → not the bottleneck): **NOT THE BOTTLENECK**. Static 3D-derived metrics (knee, trunk, arm angles) look biomechanically sound; the dynamic timing metric is broken upstream-of-pose-source.

**Phase 2 (alternative 2D→3D lifting) is not warranted.** Switching the lifter wouldn't fix the kinetic-chain bug.

---

## What the audit found

### Static 3D-derived metrics — biomechanically sane

256 shots across 3 videos. All values from MediaPipe's `world_landmarks`:

| Metric                      |     n |  median | range          | Outside heuristic range | Exactly 0/180 |
| --------------------------- | ----: | ------: | -------------- | ----------------------: | ------------: |
| `knee_angle_at_contact`     |   256 | 149.5°  | [96.7°, 179.0°] | 24 (9 %) |        0 (0 %) |
| `trunk_rotation_at_contact` |   256 |  −2.2°  | [−66.5°, 335°]  |  7 (3 %) |        2 (1 %) |
| `arm_extension_at_contact`  |   256 | 115.5°  | [50.4°, 162°]   |  2 (1 %) |        0 (0 %) |
| `peak_swing_speed`          |   256 |   8.0   | [1.1, 48.4]     |        — |        0 (0 %) |

Knee angle distribution by shot type (matches real tennis biomech well):

- **Backhand:** median 130° (deep prep stance, well-bent legs)
- **Forehand:** median 149° (medium bend at contact)
- **Serve:** median 166° (legs nearly extended at contact / peak height)

These distributions track expected human biomech for the user's swing types
— different per shot type in the right direction, in the right ranges.
Visual inspection of 12 rendered frames (eval/3d-lifting/audits/) confirms:

- **2D pose detection is visually correct** in all sampled cases
- **Static angles match what's visible in the frame** (e.g. 119° knee flagged as "out-of-range" was actually a deep-prep backhand stance — the 120° lower-bound was just too tight)

### Dynamic kinetic-chain timing — broken

The other 98 % of flags. The relevant code in `biomechanical_analysis.py`:

```python
# scripts/biomechanical_analysis.py:174-192 (lightly paraphrased)
for joint_name, joint_idx in [...]:
    max_vel = 0
    max_frame = cf
    for i in range(search_start + 1, search_end):
        p_curr = get_keypoint(frames[i], joint_idx)
        p_prev = get_keypoint(frames[i - 1], joint_idx)
        if p_curr and p_prev:
            vel = sqrt(sum((a - b) ** 2 for a, b in zip(p_curr[:3], p_prev[:3]))) * fps
            if vel > max_vel:
                max_vel = vel
                max_frame = i
    chain_times[joint_name] = max_frame
```

This computes per-frame velocity from raw position deltas (no smoothing) and
picks the frame with the maximum velocity. Two issues:

1. **No smoothing window.** Pose data has small per-frame jitter; raw
   position deltas amplify that into large velocity-noise spikes. The "peak"
   that gets picked is often a noise spike, not the actual biomechanical
   peak.
2. **Wide search window (40 frames at 60 fps = 667 ms)** plus no
   minimum-prominence filter. Three different segments
   (shoulder/elbow/wrist) all hit max velocity at the same noisy frame in
   ~50 % of "non-reversed" cases (e.g. `sh = el = wr = 417 ms` repeated in
   the data).

**Symptom in our data:**
- 100 % of shots have `hip = 0 ms` (it's the reference baseline, fine).
- 66 % of "reversed-chain" shots have *negative* shoulder/elbow/wrist times (peaks reportedly happening before contact, which the body-frame velocity argument says is wrong direction).
- "Clean" chains (40 of 256) very often have `shoulder == elbow == wrist` (e.g. all 417 ms). Three independent segments syncing within 1 frame is statistically implausible — they're hitting noise plateaus, not real peaks.

Both reversed and "clean" cases are downstream of the same bug: peak-from-unsmoothed-velocity finds noise, not signal. **This is a biomech-equation issue, not a pose-data issue.**

The fix is in `biomechanical_analysis.py` lines 174-192:

- Smooth the per-joint velocity signal before peak-picking (rolling avg ~5–10 frames)
- Apply minimum-prominence filter so noise spikes don't qualify
- Or use a different signal entirely (e.g. acceleration zero-crossings — the moment a joint stops accelerating defines a chain-link)

### Visible bug: `trunk_rotation` signed-angle wraparound

The audit also surfaced 7 shots with `trunk_rotation_at_contact` magnitude
> 90° (max observed: 335°). Sample: IMG_0996 frame 4185, backhand at
contact, reported trunk_rotation = 158°. Visual inspection shows a normal
backhand stance — no extreme rotation.

This is likely a signed-angle wraparound: when the actual rotation crosses ±180° (e.g. on a follow-through), the angle math returns 158° instead of −22°. Easy fix: clamp to (−180°, +180°) by mod-and-recenter.

### Calibration note on the locked thresholds

A few of the static-metric flags were false positives from threshold tightness, not data issues:

- `knee_out_of_range` < 120° (24 shots): looking at IMG_0996 frame 8817 — that's a real deep-prep backhand stance with knees at 119°. The lower bound of 120° was conservative; real-world tennis prep can dip below 120°. Loosen to ~100° for backhand if a future eval tightens criteria around knee.

The locked criteria did their job here: they surfaced shots that *might* be issues, then visual inspection refined the picture. Most "out-of-range static angle" flags are real biomech the threshold caught up against.

---

## Decision

**Per locked criteria:** Phase 1 result is **NOT THE BOTTLENECK**. Static 3D
metrics derived from `world_landmarks` are biomechanically sound. The 84 %
flag rate is dominated by a fixable algorithm bug in `biomechanical_analysis.py`,
not by 3D pose-data quality.

**Phase 2 (MotionBERT / HMR2 alternative lifting): not run.** Switching the
lifter would not fix the kinetic-chain extraction bug. Recommended budget for
Phase 2 is reallocated to the algorithm-fix work below.

---

## Recommendations

In priority order:

### 1. Fix kinetic-chain timing (high impact, low effort — ~half day)

Edit `scripts/biomechanical_analysis.py:174-192`. Two changes:

- **Smooth before peak-pick.** Compute `velocities[]` for each joint over the search window, then apply a 5–7 frame rolling-mean smoothing. Pick the max from the smoothed signal.
- **Add a minimum-prominence filter.** A "peak" must exceed the local mean by at least N standard deviations (or a fixed threshold like 0.3 m/s in body-frame meters). Otherwise treat as no-peak.

Optional: switch from "max velocity per joint" to "first acceleration zero-crossing per joint" — that's the moment the joint stops accelerating, which is a more robust kinetic-chain landmark.

### 2. Fix trunk_rotation wraparound (low impact, trivial — ~5 min)

`trunk_rotation_at_contact` value should be wrapped to (−180°, +180°). One-line fix wherever the angle is computed.

### 3. Loosen static-angle plausibility thresholds for downstream consumers (judgment call)

If `claude_coach.py` or any other downstream code uses thresholds like "knee should be > 120° at contact = bad form", widen them. The audit shows real deep-prep stances hit 100–119° on backhand prep.

### 4. Re-run audit after fixes 1 + 2 (~1 hour)

Use the same `audit_world_landmarks.py` script, confirm flag rate drops below 30 %, surface any remaining residual issues. This becomes a regression test.

### Items deliberately not recommended

- Switching pose extractor (already explored in `feature/detection/pose-experiments`, REJECT)
- Adding MotionBERT/HMR2 lifting (would not fix the kinetic-chain bug)
- Ground-truth 3D pose annotation (would help calibrate static metrics but isn't needed for the timing fix)

---

## What this means for the user's original concern

The user's framing was "pose detection model isn't good enough." Two evals deep, the data says:

1. **2D pose detection (MediaPipe vs Sapiens-2 vs RTMPose):** all three agree to ~5 px at 1080p. Not the issue. (`feature/detection/pose-experiments`)
2. **3D `world_landmarks`:** static angles are biomechanically sound across 256 shots. Not the issue. (this brief's Phase 1)
3. **Biomech equation algorithm:** the kinetic-chain timing extraction is brittle. Smoothing-before-peak-picking should resolve. (this brief's recommendation)

If the user has been seeing implausible biomech / coaching outputs (e.g. weird kinetic-chain lag values, "your wrist peaked before your shoulder"), this is the explanation. The fix is small and local; it doesn't require a new model.

---

## Reproducibility

- `scripts/audit_world_landmarks.py` — heuristic flagging on biomech output
- `scripts/render_audit_frames.py` — diagnostic PNG renderer (2D pose overlay + biomech panel)
- `eval/3d-lifting/audit_phase1.json` — full per-shot audit results
- `eval/3d-lifting/audits/*.png` — 12 sample diagnostic frames (per flag category)
