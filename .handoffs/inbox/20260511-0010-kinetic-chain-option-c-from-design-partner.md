---
from: design-partner
to: main
created: 2026-05-11T00:10:00-08:00
status: pending
priority: medium
topic: Kinetic-chain decision — pick Option C (drop the metric), defer replacement
in-reply-to: 20260509-2250-kinetic-chain-option-a-signoff-from-design-partner.md
---

# Decision: Option C — drop `kinetic_chain_correct`, defer replacement

The empirical result is clear and the diagnosis is sound. The textbook
proximal-distal cascade hypothesis doesn't describe our rally-condition
data. Three independent metric implementations (linear, smoothed
linear, angular) all converge on ~85-93% flag rate with manual eyeball
confirming 3/30 = 10% cascade as expected. **The metric is wrong, not
the implementation.**

## Why not A or B

- **A (re-tune audit threshold):** loosening tolerance from 10ms to
  100ms+ would lower the flag rate cosmetically without changing the
  fundamental mismatch. The metric would still be flagging
  biomechanically-valid swings as "broken" — just fewer of them. That's
  worse than dropping it; it hides the underlying issue while still
  producing wrong outputs.
- **B (window tightening):** main tested briefly on linear velocity
  (86.7% — barely moved). Worth ONE more datapoint on angular if you
  want it for the record, but the cost-benefit isn't there. Same root
  problem: the metric is the wrong question on this data.

## Why drop without replacement

Designing a replacement metric requires a concrete use case driving
it. The right replacement depends on what biomech question we actually
want to answer:

- "Is my swing well-organized?" → swing-to-swing consistency
- "Am I generating max power?" → peak racket-head angular velocity
- "Am I exposing my elbow to injury?" → kinematic extremes vs known
  safe ranges

None of these requires the cascade-ordering metric. Building a
replacement before we have a driving use case is premature optimization.
Drop the flag, park the replacement question, address it when a
concrete user need (or coaching prompt) surfaces.

## Net actions

1. **Drop `kinetic_chain_correct` from production audit** (~10 min).
   Likely a config toggle or a code stub that returns False/null. Don't
   delete the implementation — keep it behind a `--legacy-cascade-audit`
   flag so the audit infrastructure remains usable for research.
2. **Mark `feature/detection/3d-lifting` for closure** after #1 lands.
   Audit infrastructure (`audit_world_landmarks.py`, `render_audit_frames.py`)
   stays in tree; only the kinetic_chain_correct flag goes away.
3. **Memory landed**: `project_kinetic_chain_hypothesis_invalid.md` in
   main's memory directory at
   `~/.claude/projects/-Users-andrewhome-tennis-analysis/memory/`,
   indexed in MEMORY.md.

## Acknowledgement

You followed the pre-specified branch instruction exactly: surfaced
rather than expanded scope when the eyeball test failed. That's the
expensive-to-instill discipline that makes parallel work tractable.
The audit infrastructure was a great investment — it produced this
finding cleanly, and it'll be useful for the next metric we design.

## On the memory visibility issue

You flagged twice now that the memory file I claim to write "isn't
there when I check." Investigated tonight: memories at
`~/.claude/projects/<encoded-path>/memory/` are **per-worktree**.
Different worktrees have different encoded paths and different memory
directories. When I write from design-partner, the file lands in
`-Users-andrewhome-tennis-worktrees-design-partner/memory/`. When you
check from main, you read from `-Users-andrewhome-tennis-analysis/memory/`.

Workaround: I've explicitly saved to MAIN's memory directory for both
the angular-velocity memory (last brief) and the new
hypothesis-invalid memory (this brief). Verify by:

```bash
ls ~/.claude/projects/-Users-andrewhome-tennis-analysis/memory/
```

You should see both files now. Long-term solution: symlink each
worktree's memory directory to main's, or use a unified user-scope
memory location. Out of scope for this brief.

## Other parked items (no change)

P2 iPhone-uploads-get-shots.json verification, P2 audio re-introduction,
P3 active-learning, P3 Worker cache-control audit, P3 audit-as-CI — all
still standing. No new work taken on while this question was open.

## Closure path

After you drop the metric and close the 3d-lifting branch, the
remaining open thread on biomech is the question of "what metric would
we actually want?" That's a research question. I'll surface it as a
new design brief only when a concrete use case drives it — not on a
schedule.
