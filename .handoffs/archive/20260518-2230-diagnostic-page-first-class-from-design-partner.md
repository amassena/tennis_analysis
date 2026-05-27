---
from: design-partner
to: main
created: 2026-05-18T22:30:00-08:00
status: pending
priority: high
topic: Build first-class diagnostic page (supersedes inline-investigation in prior brief)
in-reply-to: 20260518-2200-stop-notifications-investigate-0-shots-from-design-partner.md
---

# Supersedes section 2 of prior brief

User's reaction:
> "we need a first class reporting/diagnostic page doing it inline
> sucks."

Section 1 of the prior brief (STOP notifications) still stands.
Section 2 (inline DIAGNOSE) is **replaced** by this brief. Section 3
(dashboard pivot scoping) is **absorbed** into this brief — the
diagnostic page IS the dashboard pivot's first concrete deliverable.

# Why first-class

Inline investigation has a 0-time payoff. The page has a 1-time
build cost and pays off forever. Every weird thing in the future
gets answered by "check the page" instead of "spin up a Claude
session and curl six URLs."

Memory pattern saved: `project_diagnostics_first_class_not_inline.md`.

# The page (or pages)

Single URL: **`https://tennis.playfullife.com/ops`** (or `/diagnostics`).
Password-protected like existing delete endpoint. Pull-based, no
notifications.

## Sections, top to bottom

### 1. Health banner (top)
- Status indicator: 🟢 / 🟡 / 🔴 based on health rules
- One-line summary: "Last 24h: 32 processed, 28 anomalous, 2 failed"
- Big visible link to filter to anomalies

### 2. Recently failed (next)
- Videos that errored in pipeline (`❌ Processing failed`)
- Per-row: video_id, timestamp, machine, error excerpt, retry button
- Sorted newest first
- Default expanded

### 3. Anomalous (next, the new value-add)
Videos that "succeeded" but look wrong. Rule-based detection:
- `shots: 0` AND duration > 30s (a 5-min slo-mo clip with 0 shots is
  almost certainly broken, not just an empty clip)
- `duration: unknown` or `duration: null`
- Processing time > 90 percentile (suspicious slowness)
- Pipeline stage durations look wrong (e.g., pose extraction was <1s
  on a 5-min clip)
- Detection-confidence histogram extremely uniform (model collapsed)

Per-row: video_id, timestamp, what's anomalous, drill-down link.

### 4. Processing right now (next)
- Currently in `processing` state in coordinator
- Per-row: video_id, machine, claim_at, elapsed time
- Visual cue if elapsed time exceeds typical for that video length

### 5. Queue (next)
- Pending jobs in coordinator
- Counts per worker if pre-routed

### 6. Aggregate stats (next)
- Today: N processed / M shots detected / K failures / J anomalies
- 7-day: same
- 30-day: same
- Per-machine breakdown (tmassena vs Andrew-PC throughput)

### 7. Recent successes (bottom)
- Last 20 videos that completed without anomalies
- Sanity check that "good" still works

## Drill-down per video

Click any video → modal or sub-page showing:
- All metadata (meta.json, shots.json) raw
- Pose JSON summary (frame count, mean confidence, gaps)
- Processing log (worker.log excerpt for that video_id)
- Stage-by-stage timing breakdown
- "Reprocess" button (re-queues job with retry_count++)
- Link to gallery card if present

# Architecture

**Backend: Worker endpoint** `GET /api/ops` returns JSON aggregate.

Pulls from:
- R2: `highlights/*/meta.json`, `highlights/*/shots.json`,
  `highlights/iphone_*/...` for anomaly detection
- Coordinator: `GET /api/queue` for live queue state (this endpoint
  already exists per CLAUDE.md)
- Worker logs: optional, via a separate endpoint that streams the last
  N lines of `worker.log` from each machine (or just exposes
  per-video-id slices)

R2 list operations are paginated and not cheap; cache aggressively
(e.g. 60s TTL on the `/api/ops` JSON).

**Frontend:** single HTML page at `/ops`. Vanilla JS, same patterns as
gallery. No new framework. Reuses existing helper functions where
sensible.

Password gate via same mechanism as `delete` endpoint (CLAUDE.md
references `deletevideo` password). Or use a separate `OPS_PASSWORD`
env var.

# Concrete anomaly rules (v1)

These are what should flag in v1. Each is one line of JS or one
SQL/JSON predicate.

| Rule | Predicate | Severity |
|---|---|---|
| Zero shots on real-length video | `shots.length == 0 && duration > 30s` | 🔴 |
| Duration unknown | `duration in [null, undefined, "unknown"]` | 🔴 |
| Suspiciously low shot rate | `shots.length / (duration / 60) < 5` (i.e., <5 shots/min on a tennis video) | 🟡 |
| Pose extraction too fast | `pose_extraction_ms < duration * 100` (i.e., faster than 0.1× realtime) | 🟡 |
| Processing exceeds typical | `total_processing_ms > p90(machine, video_length_bucket)` | 🟡 |
| Missing thumbnail | `!exists(thumbs/{vid}.jpg)` | 🟡 |
| Coaching missing | `!exists(highlights/{vid}/coaching.json)` | 🟡 |

These can grow over time. **Adding new rules should be trivial** — a
single function returning `{flag, severity, reason}` per video.

# Build sequence

## Phase 1: minimal page (~half day)
- Worker endpoint `/api/ops`
- HTML page `/ops` showing sections 1-2-4-5-6 (skip anomalous + drill-down)
- Static health banner + recently-failed + queue + aggregates
- **Goal**: "Is the pipeline alive right now?" answerable in one glance

## Phase 2: anomaly detection (~half day)
- Add anomaly rules above
- Section 3 (anomalous) on the page
- **Use this phase to find the 0-shots regression** — once the rules
  fire, the broken videos light up the page, and the drill-down
  (Phase 3) takes you to the root cause without inline grep

## Phase 3: drill-down (~half day)
- Click a video → modal showing meta.json, shots.json, pose summary,
  log excerpt, stage timing
- Reprocess button
- Link to gallery

## Phase 4: failure-only notifications (~10 min, can happen anytime)
- Per prior brief: kill per-success email/SMS, keep ❌ failure emails
- Once page is live + reliable, retire even ❌ emails (push → pull)

Total: ~2 days of work.

# Why this order (build → diagnose, not diagnose → build)

**Don't run inline investigation first and "then" build the page.**
That's the trap. Build Phase 1 and Phase 2 first. The anomaly rules
will tell you which videos are broken without any inline grep. Use
the drill-down (Phase 3) on a broken video to find the root cause.
The page itself does the investigation work, and stays around for
the NEXT regression.

The 0-shots regression is the inaugural use case. By the time the
page is live, finding root cause should take <30 min.

# What this does NOT include

- Dashboard improvements for the user-facing gallery (form comparison
  surfaces) — that's separate
- Pro footage acquisition (still active stream)
- Detection branch improvements (still active stream)
- The kinetic-chain replacement metric (still parked)

# Net actions for main session

1. **Phase 0** (~10 min): stop the email firehose per Option (c) from
   prior brief (failure-only mode). Do this first; it's the immediate
   relief.
2. **Phase 1** (~half day): minimal `/ops` page
3. **Phase 2** (~half day): anomaly detection. By end of this phase,
   the 0-shots regression should be visible at a glance.
4. **Phase 3** (~half day): drill-down. Use it to root-cause the
   0-shots issue. Fix in a separate commit.
5. **Phase 4**: kill push notifications entirely (after page is
   trusted as the source of truth).
