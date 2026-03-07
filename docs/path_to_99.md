# Path to 99% F1: Shot Detection Pipeline

## Current State

**Production baseline (Mar 3 2026):** F1=93.9% (P=96.2%, R=91.8%) on 12 GT videos, 546 shots.

| Metric | Value |
|--------|-------|
| TP | 501 |
| FP | 20 |
| FN | 45 |
| Total errors | 65 |

**99% F1 requires:** FP+FN ≤ 11. Must eliminate 54 of 65 errors (83% reduction).

## Error Concentration

Two videos account for 74% of all errors:

| Video | FP | FN | Total | % of Errors | Root Cause |
|-------|----|----|-------|-------------|------------|
| IMG_6713 | 7 | 17 | 24 | 37% | Left-side camera. 2D features are camera-angle dependent. Model is blind to this viewing angle. |
| IMG_0929 | 3 | 21 | 24 | 37% | Wall drill backhands at fast pace. Features overlap with not_shot. Rapid repetitive strokes. |
| Other 10 | 10 | 7 | 17 | 26% | Scattered edge cases. These videos average F1=97.4%. |

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

**Problem:** IMG_6713 (left-side camera) contributes 24 errors (37% of total). 2D pose features are completely different from back-court training data.

**Required outcome:** Features that produce consistent classification regardless of camera position.

**Approaches:**
- **3D pose lifting + threshold co-tuning.** Infrastructure exists. Retrain model on 3D features, then systematic grid search over the 6 most sensitive thresholds (ns_permissive, ns_moderate, mc_strong, mc_weak, mc_floor_heuristic_only, mc_floor_jerk) against full 12-video set.
- **Per-angle models.** Train separate models per camera position, auto-detect angle from pose geometry. Limited by having only 1 left-side video.

**Expected recovery:** 15-20 of 24 IMG_6713 errors.

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

### R4: Meta-Classifier Ensemble

**Problem:** 28 shots only the window detector finds. 22 shots only the baseline finds. Simple confidence thresholding can't distinguish window-only TPs from FPs.

**Required outcome:** A meta-classifier that takes both systems' features and predictions as input, and learns which disagreements to trust.

**Input features for meta-classifier:**
- Baseline confidence, detection source, not_shot probability
- Window detector confidence, top feature values
- Agreement/disagreement signal
- Temporal context (nearby detections)

**Expected recovery:** 8-12 errors (capture most of the 28 window-only TPs while filtering the 74 window FPs).

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

| After | Expected Errors | Expected F1 |
|-------|----------------|-------------|
| Current baseline | 65 | 93.9% |
| R1: Camera invariance | ~47 | ~95.7% |
| R2: Fast-sequence detection | ~35 | ~96.7% |
| R3: Audio shape features | ~30 | ~97.2% |
| R4: Meta-classifier ensemble | ~22 | ~97.9% |
| R5: Edge case filters | ~18 | ~98.3% |
| R6: Temporal architecture | ~11 | ~99.0% |

## Execution Order

1. R1 — Camera angle invariance (3D + threshold co-tuning)
2. R3 — Audio shape features (independent of R1, can parallelize)
3. R2 — Fast-sequence detection (benefits from R3's audio features)
4. R4 — Meta-classifier ensemble (benefits from R1-R3 improving both systems)
5. R5 — Edge case filters (mop up remaining errors)
6. R6 — Temporal architecture (only if R1-R5 plateau below 99%)

## Key Constraints

- **GPU work on andrew-pc or tmassena only** — Mac is for orchestration, labeling, lightweight scripts
- **Archive before modifying production artifacts** — model tracked in git, always archive before overwriting
- **Auto-commit meaningful changes** — don't wait for user to ask
- **Validate against full 12-video baseline after every change** — regression baselines in `training/regression_baselines/`
