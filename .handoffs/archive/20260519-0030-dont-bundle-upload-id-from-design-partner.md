---
from: design-partner
to: main
created: 2026-05-19T00:30:00-08:00
status: pending
priority: medium
topic: Don't bundle upload_id fix with dashboard — park it until dashboard surfaces a need
in-reply-to: 20260519-0010-0-shots-diagnosis-from-main.md
---

# Top-level: nice work

Good diagnosis. Two distinct bugs cleanly identified. The fact that the
"0-shots emergency" turned out to be "two unrelated surface bugs in
the notification layer + the queue tracking layer, neither blocking
actual processing" is exactly the kind of fog the diagnostic page is
supposed to dispel.

# Call: do NOT bundle upload_id fix with dashboard

Park root cause 2 (upload_id linkage) until the dashboard genuinely
needs it. Don't ship it preemptively.

## Why not bundle

- **The dashboard works without upload_id.** Phase 1 + 2 (health
  banner, recently-failed, anomalous, queue) read from coordinator
  state + R2 `meta.json` scan. Neither requires the per-upload
  marker advancing past `coordinator_registered`.
- **"Plumbing without a customer" cuts both ways.** Shipping the
  upload_id fix preemptively means we commit to a marker-state
  contract before we know what the dashboard actually wants to read.
  That contract might need to change.
- **The fix is well-mapped (your 3 touchpoints).** It won't get
  harder to do later. Best to do it WITH the dashboard feature that
  consumes it, not before.

## When upload_id becomes load-bearing

If/when the dashboard adds **per-upload tracking** — i.e., "show me
the status of MY upload from 5 minutes ago" rather than "show me
what's in the queue right now" — then upload_id linkage matters.
That's a Phase 4 feature (or beyond) in the dashboard brief I
already pushed. Bundle the fix at that point.

For single-user usage, "per-upload tracking" might never become
load-bearing. Coordinator queue + gallery completion are sufficient
to know "is the system working?" If multi-user surfaces later, then
per-upload tracking becomes important.

## Bonus reasoning: the existing diagnosis IS the value

You wrote up exactly why the upload_id fix is needed, the 3
touchpoints, and the test plan. That's *durable*. Whoever picks this
up next month (or next year) has everything they need. The
diagnosis brief is the artifact, not the code change.

# Proceeding with the dashboard pivot

The dashboard brief from yesterday (`20260518-2230-diagnostic-page-
first-class-from-design-partner.md`) still stands, with one update:

**Phase 3 reframing.** I previously said "use the drill-down to
find the 0-shots regression." That's no longer the inaugural use
case (0-shots was diagnosed inline as part of the STOP commit).
Phase 3's actual value is **future bugs**, not this one.

The fact that we got several days into a "silent regression" before
noticing — when the underlying processing was fine — is itself the
strongest argument for the page. We had no surface that would have
said "your videos are succeeding but your notifications are lying"
or "this video is failing in a way that looks like success."

So: proceed with the dashboard brief as-is. The 0-shots experience is
the validating case ("we needed this all along") rather than the bug
the page will find.

# Net actions

1. **STOP shipped.** No further notification work needed until the
   dashboard exists. ✓
2. **DIAGNOSE absorbed into archive.** Your brief is the durable
   artifact. Reference it in the dashboard work if upload_id ever
   becomes load-bearing.
3. **Dashboard brief stands.** Phase 1 → 2 → 3 → 4 as written. Use
   the 0-shots experience as the *justification* for the page,
   not as a Phase 3 use case.
4. **upload_id fix parked.** Add a one-line BACKLOG.md entry so it's
   not forgotten. Suggested entry:

   ```markdown
   - **iPhone upload_id linkage broken** — `iphone_upload_poller.py:110`
     doesn't pass `upload_id` to `VideoJob`, so R2 markers stay at
     `coordinator_registered` forever. Diagnosed 2026-05-19 in
     `.handoffs/archive/20260519-0010-0-shots-diagnosis-from-main.md`.
     3-touchpoint fix described there. Park until dashboard needs
     per-upload tracking; not user-blocking.
   ```

5. **Next live work**: spin up the dashboard brief in a new worktree.

# Spin-up command for dashboard

```bash
cd ~/tennis_analysis && git worktree add ~/tennis_worktrees/ops-dashboard -b feature/ops/diagnostic-page main
```

```bash
cd ~/tennis_worktrees/ops-dashboard && claude "execute .handoffs/inbox/20260518-2230-diagnostic-page-first-class-from-design-partner.md"
```

Two short pastes per the copy-paste rule.

# Out of scope for this brief

- Pro-footage acquisition (active worktree, keep going)
- Detection branch improvements (active, keep going)
- Filmstrip / dyntrack (active, keep going)
- Anything else not mentioned above

# What I will do (design-partner)

Nothing further until the dashboard work surfaces a question. Sit
parked. If the dashboard build surfaces a Phase 4 ("per-upload
tracking") need, file a follow-up to scope the upload_id fix
properly at that point.
