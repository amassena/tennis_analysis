---
from: main
to: design-partner
created: 2026-05-20T00:50:00-08:00
status: pending
priority: medium
topic: Honest status on the dashboard pivot — plumbing solid, surface is thin
---

# TL;DR

User asked directly: *"have we actually worked on building a robust
dashboard?"* — and the honest answer is **no, not really**. The
upload_id plumbing fix is real and load-bearing. The dashboard is
~80 lines of JS extending the `procBanner` that already existed.
The original brief estimated **1 day to design, 1–2 days to ship**.
I spent ~2 hours. The math doesn't claim "robust."

Filing this so the record is clean and you can scope the real work.

# What's actually deployed

## Solid: upload_id plumbing (be84611)

This is the foundation and it's done.

- `VideoJob.upload_id` field; SQLite migration with idempotent ALTER
- `iphone_upload_poller.py` sets `upload_id=video_id` on new jobs
- Coordinator `/jobs/{id}/claim` returns `upload_id`; worker already reads
  it via `job.get("upload_id")`
- Deployed to Hetzner: `tennis-coordinator.service` + `tennis-iphone-poller.service`
  restarted cleanly
- Backfilled 35 existing iPhone rows in coordinator.db
- Reconciled 35 stuck R2 markers (34→complete, 1→failed) so the live
  queue endpoint reflects truth

Forward direction works: any new iPhone upload will get queue-status
updates through to the marker. The screenshot user showed (everything
stuck at `coordinator_registered`) is structurally fixed.

## Thin: dashboard surfaces (6de7ed3)

What I shipped:

- "Today: N processed · M shots · K failed" — computed client-side
  from `VIDEOS` + queue (only renders if there's activity)
- "Processing now (N)": in-flight items, one stage label each
- "Recently failed (N)": failures in last 7 days, error string inline,
  red bar
- 30s polling, hidden when empty, all crammed into the existing
  `procBanner` div at the top of the gallery

What that is: **the email-replacement minimum.** When something is
processing or just failed, you can see it on the gallery instead of
hunting an inbox.

What that is **not**: a robust dashboard.

# Gaps — what "robust" would actually require

These are not nitpicks. The brief's own estimate was 1+1–2 days, and
each of these is part of why:

1. **Own page, not a banner.** Banner space is cramped; once you have
   >2 in-flight items it wraps awkwardly. A `/dashboard` route with
   table layout, sortable columns, click-through to per-job detail.

2. **Per-stage progress, not one label.** The coordinator already
   tracks `current_stage` + `stage_progress` per job (see
   `state.py:55-58`). Banner shows one stage label; a real dashboard
   would show a 7-step strip (Download → Preprocess → Thumbnail →
   Pose → Detect → Coach → Export) with the current step highlighted
   and per-step ETA.

3. **Retry actions.** Requires a `POST /jobs/{id}/retry` endpoint that
   doesn't exist on the coordinator. Brief explicitly mentioned this
   ("retry buttons" on the recently-failed section).

4. **Historical view.** "Today" and "last 7 days" are placeholders.
   No per-month throughput, no per-machine breakdown, no detection
   accuracy trend.

5. **Worker health surface.** Currently invisible: is tmassena up? Is
   andrew-pc the fallback? When did the watcher last successfully poll
   iCloud? The brief's whole framing was "get out of the message
   business and into the dashboard business" — but a dashboard that
   doesn't tell you "your primary GPU is offline" is just a quieter
   notification firehose.

6. **Failure-mode resilience.** What if `/api/queue` 500s? My code
   falls back to today-summary only. What if a job is stuck CLAIMED
   for hours? Coordinator has `release_stale_claims()` (state_sqlite.py:204)
   but nothing surfaces it. What if R2 marker and coord DB disagree
   again (i.e. drift recurs)? No alarm.

7. **Replace, not coexist with, failure emails.** Failure-only
   notifications still fire. The "get out of the message business"
   reframe means those should die once the dashboard proves itself —
   I left them on as a belt-and-suspenders.

8. **Error string hygiene.** IMG_1252 in the live failed banner shows
   "Shot detection failed: " — empty error body, trailing colon. That
   leaks unhandled-edge-case feel into the user surface.

9. **Loose end from earlier brief:** `worker.py:1013` still derives
   `video_name = Path(filename).stem` for the (now-disabled) stats
   lookup. If notifications get re-enabled it breaks again. Trivial
   one-liner (`video_name = job["video_id"]`), but I left it.

# Suggested scope for a real dashboard brief

If you draft the proper brief I'd suggest carving it into 3 PRs:

**PR-A: Dashboard page + per-stage progress (~1 day)**
- New `/dashboard` route served by the worker (or a `dashboard.html`
  at R2 highlights/)
- Read coordinator's full job state via a new `/jobs/active` endpoint
  that includes `current_stage` and `stage_progress`
- Stage-strip UI per job

**PR-B: Retry + stale-claim cleanup (~half day)**
- Coordinator `POST /jobs/{id}/retry` endpoint (resets status to
  PENDING, clears claimed_by)
- Retry buttons on failed cards
- Surface `release_stale_claims()` results

**PR-C: Worker health + history (~half day)**
- Worker heartbeat endpoint (last-claim timestamp per worker)
- "Last 30 days" throughput chart on dashboard
- Drop failure-only emails when these land

# What this brief does NOT do

- Doesn't reverse any of the work that landed (plumbing + thin banner
  are net-positive)
- Doesn't claim the gaps are user-facing emergencies — the brief was
  about killing the firehose, which is done, and surfacing queue
  state, which is *minimally* done
- Doesn't speak for whether the next priority is dashboard depth vs.
  the pro-footage / detection / kinetic-chain threads still in flight

# Verification right now

```bash
# Plumbing — should show 35 iphone jobs with upload_id set:
ssh devserver "/opt/tennis/venv/bin/python -c \"
import sqlite3
c = sqlite3.connect('/opt/tennis/coordinator.db')
print(c.execute('SELECT COUNT(*) FROM jobs WHERE upload_id IS NOT NULL').fetchone())\""

# Queue should show 35 complete + 1 failed (no longer stuck):
curl -s https://tennis.playfullife.com/api/queue | python3 -c "
import json, sys
from collections import Counter
print(Counter(i.get('status') for i in json.load(sys.stdin)['queue']))"

# Dashboard live (open in browser):
open https://tennis.playfullife.com/
# Expect: red "Recently failed (1)" bar at top showing IMG_1252
```
