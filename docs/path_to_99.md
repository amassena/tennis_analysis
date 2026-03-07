# Path to 99% F1: Shot Detection Pipeline

## Current State

**After meta-ensemble (Mar 7 2026):** F1=94.8% (P=94.9%, R=94.7%) on 12 GT videos, 546 shots.

| Metric | Value | Previous |
|--------|-------|----------|
| TP | 517 | 498 |
| FP | 28 | 27 |
| FN | 29 | 48 |
| Total errors | 57 | 75 |

**99% F1 requires:** FP+FN ≤ 11. Must eliminate 46 of 57 errors (81% reduction).

## Error Concentration

| Video | FP | FN | Total | % of Errors | Root Cause |
|-------|----|----|-------|-------------|------------|
| IMG_6713 | 8 | 12 | 20 | 35% | Left-side camera. 2D features are camera-angle dependent. Meta-ensemble recovered 6 TPs. |
| IMG_0929 | 4 | 11 | 15 | 26% | Wall drill backhands at fast pace. Meta-ensemble recovered 7 TPs. |
| Other 10 | 16 | 6 | 22 | 39% | Scattered edge cases. |

## What We've Tried and Learned

### The Model-Threshold Co-Optimization Trap

The pipeline has 9 detection stages with 22 hardcoded thresholds, all co-tuned with a 25-feature RF model. Changing the model shifts probability distributions and breaks the thresholds. This was proven across 11 experiments:

- **More features (46 vs 25):** LOOCV 88.9% (+5.5%), pipeline F1 88.8% (-5.1%). More features lower not_shot probabilities → FP explosion.
- **More training data:** Adding IMG_0929 data always regresses pipeline regardless of curation strategy. Distribution shift causes backhand over-classification.
- **Different algorithms:** ExtraTrees, CNN ensemble — all produce different probability scales that break hardcoded thresholds.
- **Auto-calibration:** LOOCV-derived thresholds ignore multi-stage cascade interactions. Regressed F1 by 1.2%.
- **End-to-end threshold grid search:** Best result with 46-feature model still 4.2% below baseline. Cannot overcome fundamental distribution shift.

**Lesson:** Cannot improve the model in isolation. Model + thresholds must be co-optimized, or the pipeline must be redesigned to be threshold-insensitive.

### What Actually Worked (+0.9% total)

Only post-processing filters that don't interact with the cascade:

1. **Post-dedup weak filters** (+0.3%) — surgically remove specific FP patterns after cascade completes
2. **unknown_shot promotion** (+0.2%) — trust ML when heuristic has no opinion
3. **Racket visibility filter** (+0.4%) — YOLOv8 racket detection to filter off-camera FPs

### Window Detector: Complementary but Unusable via Simple Ensemble

Sliding window detector (115 features, F1=90.0%) finds 28 shots the baseline misses. Baseline finds 22 the window misses. Ensemble ceiling: 96.9% recall. But:

- Window-only detections have ~30-40% precision at any threshold
- Confidence scores don't discriminate TPs from FPs
- Best weighted union: F1=93.8% (below baseline 93.9%)
- Needs a meta-classifier, not confidence thresholding

### 3D Pose Lifting: Right Idea, Incomplete Execution

MotionBERT infrastructure is built (script, 32 lifted pose files, `--poses-3d-dir` flag). But:

- 3D-only: 0 detections (smoothing kills velocity features)
- Dual-pose: F1=73.9% (probability shift without threshold co-tuning)
- Production model was accidentally destroyed during experiments (now tracked in git)

Camera-invariant features would fix IMG_6713's 24 errors. Requires threshold co-tuning with new model.

## Requirements to Reach 99%

### R1: Camera Angle Invariance

**Problem:** IMG_6713 (left-side camera) contributes 20 errors (35% of total, down from 24 after R4). 2D pose features are completely different from back-court training data.

**Required outcome:** Features that produce consistent classification regardless of camera position.

**Attempted (Mar 7):** MotionBERT Lite 3D pose lifting. Trained RF on 3D features (LOOCV=83.9%, +0.5% vs 2D). With threshold sweep (ns=-0.15, mc=+0.05), best 3D F1=91.9% vs 2D baseline 95.3% on 3 key videos. IMG_6713 improved only marginally (61.3% → 62.9%) while back-court videos regressed. **Uniform 3D replacement is a dead end** with MotionBERT Lite — the 3D output is not sufficiently camera-invariant.

**Remaining approaches:**
- **Better 3D lifting model.** MotionBERT Full (vs Lite), or MotionBERT fine-tuned on sports data. The Lite model may lack capacity for tennis-specific poses.
- **Hybrid features.** Use 3D features only for camera-dependent measurements (x-offset, shoulder rotation) while keeping 2D for camera-independent ones (velocity, wrist height relative to body).
- **Per-angle models.** Train separate models per camera position, auto-detect angle from pose geometry. Limited by having only 1 left-side video.
- **Camera angle normalization.** Detect camera angle from pose geometry, then rotate 2D features to canonical "back-court" view.

**Expected recovery:** 10-15 of 20 remaining IMG_6713 errors.

### R2: Fast-Sequence Detection

**Problem:** IMG_0929 (wall drills) contributes 24 errors (37% of total). 21 FNs are rapid repetitive backhands rejected because `not_shot > 0.30-0.40`.

**Required outcome:** Detect rapid repetitive strokes during confirmed rally/drill sequences.

**Approaches:**
- **Rally-aware threshold relaxation.** During confirmed rally clusters, lower the not_shot gate (e.g., from 0.40 to 0.55). Risk: may add FPs in non-rally sections.
- **Temporal context model.** Current model sees ±45 frames around contact. A temporal model (TCN/Transformer) over full video could learn drill rhythm and detect shots from sequence context.
- **Rhythm inference.** Once shots 1, 3, 5 are detected in a fast sequence, infer shots 2 and 4 from timing pattern. Rally rhythm fill (stage 6) attempts this but is too conservative.

**Expected recovery:** 12-18 of 24 IMG_0929 errors.

### R3: Audio Shape Features

**Problem:** Pipeline uses audio amplitude as a binary trigger (0.09 RMS threshold). Window detector proved audio_peak_ratio (9.7% importance) and audio_rise_rate (6.8%) are top-3 discriminators — shape matters more than amplitude.

**Required outcome:** Replace rigid amplitude gates with learned audio envelope features in the main classifier.

**Expected recovery:** 5-8 errors (fewer audio-triggered FPs, recover FNs with distinctive audio shape but low amplitude).

### R4: Meta-Classifier Ensemble [DONE]

**Problem:** 28 shots only the window detector finds. 22 shots only the baseline finds. Simple confidence thresholding can't distinguish window-only TPs from FPs.

**Result:** Meta-classifier (RF, 17 features) learns which window-only detections to trust. At threshold=0.3: recovered 19 FNs with 1 additional FP. F1: 93.0% → 94.8%.

**Key features:** min_baseline_dist (31%), has_baseline_match (25%), w_conf (16%) — the model learned that window detections far from any baseline detection are likely genuine new finds.

**Actual recovery:** 18 net errors recovered (19 TP - 1 FP). Exceeded expected 8-12.

**Scripts:** `scripts/meta_ensemble.py` (train + `--apply` mode), `models/meta_ensemble.pkl`.

### R5: Remaining Edge Cases

**Problem:** 17 scattered errors across 10 well-performing videos. Serve toss FPs, soft volley FNs, camera transition errors, occluded contacts.

**Required outcome:** Targeted post-filters and expanded computer vision coverage.

**Approaches:**
- Expand racket detection to all GT videos (currently 8 of 12)
- Ball tracking integration at contact point (TrackNet exists, 40-45% detection rate)
- Temporal pose smoothing (Savitzky-Golay) to reduce jitter-induced FPs

**Expected recovery:** 4-6 errors.

### R6: Pipeline Architecture (if incremental improvements plateau)

**Problem:** The multi-stage cascade with hardcoded thresholds is the fundamental bottleneck. Every model improvement requires re-tuning 22 thresholds.

**Required outcome:** Replace multi-stage cascade with a learned temporal action detection model.

**Architecture:**
- Temporal pose tubes (±30 frames) with learned temporal pooling
- Joint action proposal + classification (single model, no cascading thresholds)
- Audio shape features as first-class input (not just a trigger)

**When to pursue:** If R1-R5 plateau at ~97% F1.

## Expected Progression

| After | Errors | F1 | Status |
|-------|--------|-----|--------|
| Backup baseline | 75 | 93.0% | Current model |
| R4: Meta-ensemble | 57 | 94.8% | **DONE** |
| R1: Camera invariance | ~42 | ~96.1% | 3D uniform failed, need alternative |
| R2: Fast-sequence detection | ~32 | ~97.0% | |
| R3: Audio shape features | ~27 | ~97.5% | |
| R5: Edge case filters | ~23 | ~97.9% | |
| R6: Temporal architecture | ~11 | ~99.0% | |

## Execution Order

1. ~~R4 — Meta-classifier ensemble~~ **DONE** (F1: 93.0% → 94.8%)
2. R1 — Camera angle invariance (3D uniform replacement failed; try per-angle models or better 3D lifting)
3. R3 — Audio shape features (independent, can parallelize)
4. R2 — Fast-sequence detection (benefits from R3's audio features)
5. R5 — Edge case filters (mop up remaining errors)
6. R6 — Temporal architecture (only if R1-R5 plateau below 99%)

## Key Constraints

- **GPU work on andrew-pc or tmassena only** — Mac is for orchestration, labeling, lightweight scripts
- **Archive before modifying production artifacts** — model tracked in git, always archive before overwriting
- **Auto-commit meaningful changes** — don't wait for user to ask
- **Validate against full 12-video baseline after every change** — regression baselines in `training/regression_baselines/`
