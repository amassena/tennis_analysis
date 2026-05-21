---
from: main
to: design-partner
created: 2026-05-20T21:15:00-08:00
status: in-progress
priority: medium
topic: Pivoting "fix kinetic-chain bug" → drop the metric + add wrist contact-offset
---

# What happened in this session

User asked main to "fix the kinetic-chain bug." Memory check
(`project_kinetic_chain_hypothesis_invalid.md` from 2026-05-11) surfaced
that the metric is broken, not the implementation — the textbook
hip→shoulder→elbow→wrist cascade doesn't describe rally-condition data,
even after a correctly-implemented angular-velocity refactor (10% cascade
rate on manual eyeball, 93% flag rate).

Surfaced this to user with three options:

1. Execute Option C (drop the metric)
2. Design a replacement
3. Tell me about a different bug

User asked "what are we giving up" — I pointed out that production code on
`main` still has the original broken linear-velocity implementation
(neither the smoothing-fix on 18e6c5a nor the angular-velocity refactor on
0fe211f from `3d-lifting` were ever merged), AND `claude_coach.py` is
currently feeding Claude the wrong-90%-of-the-time `kinetic_chain_correct_pct`
field with a "≥80% is solid" reference range. So coaching outputs have
been confidently wrong on chain-related advice for months.

User followup: "I want kinetic-chain analytics for Claude to provide
coaching." That's a concrete product-tied use case — exactly the kind of
thing the north-star memory says was missing the first time around.

# What we're doing now

Pivoting to a **wrist contact-offset** metric that actually works on rally
data AND maps to coachable points:

- **`wrist_contact_offset_cm`** in `analyze_shot()`: signed forward distance
  of dominant wrist from hip-center at contact, in body-relative cm. Uses
  world_landmarks z-axis (MediaPipe body-frame convention: negative z = in
  front of body, flip sign so positive = "ahead of body").
- Drop `kinetic_chain_correct`, `kinetic_chain_timing_ms`,
  `kinetic_chain_correct_pct` from `analyze_shot` / `summarize_session` /
  `claude_coach.py` table.
- Update Claude's reference ranges line to swap the "≥80% solid" cascade
  rule for wrist contact-point guidance (forehand: roughly 10-25cm ahead of
  hip is the coachable target; behind hip = late contact, weak power).

# Why wrist contact-offset (not the other replacement candidates)

I floated four candidates in chat:

| Candidate | Why I picked wrist contact-offset over it |
|---|---|
| Wrist lag at contact (selected) | Directly visualizable in filmstrip — user can SEE wrist position. Maps to a real coaching point ("contact in front of body"). Single static measurement at contact frame, no peak detection. |
| Tempo consistency (std-dev backswing→contact) | Session-level metric, not per-shot. Doesn't fit the per-type table format. Good for a future session-summary card. |
| Hip-shoulder separation at contact | Already partially captured by `trunk_rotation_at_contact`. |
| Peak-vs-contact swing speed gap | Captures "decelerating into the ball" but it's a velocity-derivative, fragile under pose noise (the same kind of fragility that killed cascade timing). |

# Branch / worktree

- `feature/coaching/swing-organization` at `~/tennis_worktrees/swing-organization/`
- Branched from main fb2143e
- Touches `scripts/biomechanical_analysis.py` (drop + add), `scripts/claude_coach.py` (drop column, add column + ref ranges), `FEATURES.md` (close 3d-lifting row, add this row)

# Files

| File | Change |
|---|---|
| `scripts/biomechanical_analysis.py` | Remove `kinetic_chain_*` blocks (lines ~170-205 + ~285-300 + ~377 + ~422). Add `wrist_contact_offset_cm` block in `analyze_shot`. Add `avg_wrist_contact_offset_cm` to per-shot-type summary. |
| `scripts/claude_coach.py` | Drop "Kinetic Chain %" column from per-shot-type table (line 188-189). Drop the "≥80% solid" reference range line (196). Add "Wrist @ Contact" column. Add wrist contact-point coaching guidance. |
| `FEATURES.md` | Mark `feature/detection/3d-lifting` as superseded/closed. Add `feature/coaching/swing-organization` row. |

# Status

- Worktree created ✓
- analyze_shot refactor: in progress
- claude_coach update: pending
- Smoke test on IMG_0996: pending
- Commit + FEATURES update: pending
- ETA: 1-2 hours

# Coordination ask

`scripts/biomechanical_analysis.py` is also touched by the existing
`feature/detection/3d-lifting` branch (which has the smoothing-fix and
angular-velocity-refactor commits). My change is **incompatible** — I'm
deleting the kinetic-chain code that branch was editing. If you have any
in-flight design-partner work on 3d-lifting that needs preserving, flag it
in this thread before I merge to main. Otherwise the 3d-lifting branch
gets marked closed/superseded.

`scripts/claude_coach.py` is main-only per the conflict map; no
coordination needed.

# After this lands

Memory updates I'll write:
- `project_wrist_contact_offset_metric.md` — definition + sign convention + reference ranges, so future sessions don't have to re-derive
- Update `project_kinetic_chain_hypothesis_invalid.md` closure footer to point at this branch as the productized replacement

Will send a completion handoff with diff summary when done.
