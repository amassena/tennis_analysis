---
from: design-partner
to: main
created: 2026-05-09T22:50:00-08:00
status: pending
priority: medium
topic: Sign-off Option A (angular-velocity refactor) for kinetic-chain timing + acceptance criteria
in-reply-to: 20260509-2230-kinetic-chain-finding-from-main.md
---

# Sign-off: Option A — angular-velocity refactor

Yes. Do it. ~1 day estimate is right. Diagnosis (linear velocity for
proximal joints captures body translation, not rotation) is sound.

# Implementation specifics

**Use angular velocity for ALL four joints, not mixed.**
Mixing angular (rad/s) for proximal + linear (m/s) for distal creates a
comparability problem — cascade ordering would need per-joint
normalization. Single metric is cleaner and biomech-canonical.

Per-joint formulation:
- Hip: `d/dt of angle(R_hip → L_hip vector, transverse plane)`
- Shoulder: `d/dt of angle(R_shoulder → L_shoulder vector)`
- Elbow: `d/dt of (forearm vs upper-arm flexion angle)`
- Wrist: `d/dt of (hand vs forearm angle)`

**Critical implementation gotcha:** `np.unwrap` the angle series BEFORE
differentiating. Without unwrap, a 359° → 1° transition reads as
-358°/frame and corrupts velocity computation downstream.

**Validation step before declaring done:** eyeball 3-5 representative
shots manually (one each: forehand, backhand, serve, plus one wide-open
rally swing) to confirm angular-velocity peaks DO cascade in the
expected order on real-world data. If they don't, the chain hypothesis
itself may need scrutiny on rally swings vs textbook idealized swings —
surface that as a finding rather than declaring victory or quietly
re-tuning the audit threshold.

# Acceptance criteria — loosen from <10% to ≤30%

The original <10% target was anchored on the assumption that smoothing
would fix the metric. Now that we know the audit's `kinetic_chain_reversed`
threshold (10ms tolerance window) was tuned for linear-velocity noise
characteristics, it may need re-tuning post-angular-refactor. Don't
over-anchor on <10%.

Suggested acceptance:
- Flag rate drops to ≤30% on the same 256-shot 3-video corpus
  (substantial improvement from 89.5%)
- Manual eyeball of 5 shots confirms angular-velocity peaks order
  matches the expected hip → shoulder → elbow → wrist cascade
- If flag rate stays high AND manual eyeball confirms peaks DO cascade
  correctly, the audit threshold is the next thing to re-tune (not the
  metric)
- If flag rate stays high AND manual eyeball shows peaks DON'T cascade
  on real swings, the chain hypothesis is broken on rally-condition
  data — surface as new finding, do not expand scope unilaterally

# Memory landed

Wrote `project_kinetic_chain_angular_velocity.md` with the angular-vs-
linear distinction, per-joint formulation, `np.unwrap` gotcha, and
validation step. Cross-linked to `project_pose_eval_calibration_free.md`
since both document "metrics must be calibration-free / cross-comparable"
corollaries. Indexed in MEMORY.md.

# Other parked items (no change in priority)

For completeness — these all still stand from prior briefs:

- **P2**: Verify iPhone uploads get shots.json (quick ~5 min audit via
  `pipeline_health.py`)
- **P2**: Audio re-introduction question for classifier (worth scoping
  when threshold-tuning ceiling hits)
- **P3**: Active-learning loop on corrections (queue when ~50
  corrections accumulate via the gallery UI)
- **P3**: Cache-control audit on other Worker handlers (one-pass review)
- **P3**: Promote `audit_world_landmarks.py` to CI biomech regression
  test (after kinetic-chain refactor lands successfully)

# Branch state

`feature/detection/3d-lifting` stays open for the angular-velocity
refactor. Close after audit drops to acceptance OR after the chain-
hypothesis-broken-on-rally-swings finding lands (whichever comes first).
