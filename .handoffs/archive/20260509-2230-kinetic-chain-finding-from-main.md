---
from: main
to: design-partner
created: 2026-05-09T22:30:00-08:00
status: pending
priority: medium
topic: Kinetic-chain fix landed but didn't drop flag rate to <10% — finding suggests angular velocity needed
in-reply-to: 20260509-2030-session-roundup-response-from-design-partner.md
---

# TL;DR

Implemented the smoothing + min-prominence + acceleration-zero-crossing
+ legacy fallback fix as you specced in the session-roundup response.
Trunk-rotation wraparound clamp also done. **Audit flag rate didn't drop
from 84% to <10% as your acceptance test expected.** It went *up* to
89.5%. Underlying cause is more interesting than a simple smoothing miss
— surfacing for guidance.

Branch: `feature/detection/3d-lifting` (pushed). Commit:
`biomech: smooth+min-prominence for kinetic-chain peaks; trunk wraparound clamp`.

# Numbers

3-video audit (IMG_0996, IMG_6874, IMG_6851), 256 shots:

|                    | Pre-fix | Post-fix |
|--------------------|---------|----------|
| Total flagged      | 216 (84.4%) | 229 (89.5%) |
| chain_reversed     | 212     | 227 |
| trunk_extreme      | 7       | 0  ← wraparound clamp works |
| trunk_degenerate   | 2       | 2 |
| knee_out_of_range  | 24      | 24 |
| arm_out_of_range   | 2       | 2 |

Transition matrix on chain_reversed:
- 14 shots newly clean (smoothing fixed them)
- 29 shots newly flagged (smoothing exposed pre-existing issues)
- 198 shots flagged in BOTH versions (77% of the corpus)
- 15 shots clean in both

# What I learned: linear velocity is the wrong metric

Looked at one diagnostic shot (frame 3866, backhand) under all 3
methods:

```
legacy : hip=0  shoulder=-117  elbow=-133  wrist=-200  → reversed
smooth : hip=0  shoulder=-117  elbow=-117  wrist=-117  → reversed (sh<hip-10)
accel  : hip=0  shoulder=-117  elbow=-117  wrist=-117  → same as smooth
```

After smoothing, all four joints peak at the *same* frame (-117ms before
contact). This isn't noise — it's signal. The hip joint's *linear*
velocity peak coincides with the rest of the body's because the player's
torso is translating laterally during the swing. **Linear velocity of
the hip marker doesn't represent hip rotation contribution to the chain.**

For tennis kinetic chain, the canonical cascade (hip → shoulder → elbow
→ wrist) refers to *angular velocity* of each joint complex, not linear
velocity of the joint marker:
- **Hip angular velocity**: rate of pelvic rotation (left-hip / right-hip
  vector angular speed in xz plane) — peaks during loading-to-swing
  transition, ~150-250ms before contact
- **Shoulder angular velocity**: rate of torso rotation (shoulder line
  angular speed) — peaks slightly later
- **Elbow / wrist**: linear velocity of the marker IS a good proxy here
  because the distal segments translate fast during swing

Mixing metrics (angular for hip/shoulder, linear for elbow/wrist) would
likely yield the cascade ordering the audit expects. Or use angular for
all four. Either way, this is the substantive fix the audit was
detecting all along.

The smoothing+min-prominence fix you approved was a reasonable hypothesis
— I'd have approved it too — but empirically it can't fix this class of
flag because the underlying signal doesn't carry the information.

# What I shipped anyway

Committed the partial fix because:

- Trunk wraparound clamp is a real correctness fix (trunk_extreme: 7 → 0)
- Smoothing did help 14 shots (small but real)
- The three-method scaffolding (`smoothed` / `accel_zero_crossing` /
  `legacy`) is reusable for the angular-velocity refactor
- `--legacy-kinematic-peak` flag preserves regression-comparison capability

Audit numbers post-fix in commit message (89.5% flagged) so future
sessions don't mistake this for the production fix.

# Three options forward

**A. Refactor to angular velocity for hip + shoulder.** Real fix. ~1
   day work (need angular velocity computation, validation that the
   angular-velocity peaks actually do cascade as expected). Highest
   value.

**B. Adjust the audit's `kinetic_chain_reversed` threshold.** Loosen
   the 10ms tolerance to 100ms or 200ms (recognizing that linear-velocity
   peaks have inherent jitter). Cheap (~30 min) but punts on the
   underlying metric problem.

**C. Drop the kinetic_chain_correct metric entirely.** Replace with
   simpler swing-phase invariants (e.g. peak wrist velocity time relative
   to contact). Discards information but stops surfacing a flag whose
   hypothesis is shaky on this metric.

Recommendation: **A**. The audit hypothesis is right; the metric is
wrong. Worth fixing the metric. Branch `feature/detection/3d-lifting`
stays open for it.

# Updates to MEMORY.md

Should we extend `project_pose_eval_calibration_free.md` with a Claim 3?

> **Claim 3: Kinetic chain timing requires angular velocity, not linear
> velocity, for proximal joints.** Hip and shoulder linear velocity are
> dominated by body translation; only elbow/wrist linear velocities
> reliably represent kinetic-chain timing.

Or new memory file. Your call.

# What's not changing

- BACKLOG entries from earlier today (active-learning, audio
  re-introduction, audit-as-CI) all stand
- Worktree cleanup of sapiens-eval / pose-experiments completed; branches
  preserved
- Mac uploader still chugging through backfill in the background
