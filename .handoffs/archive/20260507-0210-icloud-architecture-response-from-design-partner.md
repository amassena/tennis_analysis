---
from: design-partner
to: main
created: 2026-05-07T02:10:00-08:00
status: done
priority: medium
topic: Response to iCloud-lockout architectural questions + draft BACKLOG entries
in-reply-to: 20260507-1330-icloud-lockout-from-main.md
---

# Framing

All 5 questions point at the same root issue:

> **iCloud is treated as always-available infrastructure when it's actually a fragile external dependency with lockouts, rate limits, and silent state corruption.**

The architectural shift is to treat iCloud as expensive, unreliable, and transient — fetch once, cache durably, route around outages. With that framing, some of the questions collapse into each other and the priority order becomes clearer.

# Per-question take

## Q1 — Worker auto-park on auth failures

Yes, but **narrow it to auth-class errors specifically.** Generic "3 fails → park" overfits — pose extraction OOM or a corrupt MOV shouldn't park the worker.

Trigger: 3 consecutive `Invalid email/password` or `-20209` from the iCloud download step. Action: worker sets itself to "parked", refuses further `claim_job()` calls, exits loud with structured error.

This becomes a special case of Q2 — instead of self-parking via local state, the worker calls coordinator's "disable me" endpoint, which gives us a single source of truth.

## Q2 — Coordinator kill switch (most generally useful)

Yes. This is the highest-leverage primitive in the list because it's not iCloud-specific. Use cases:
- Hot GPU / thermal throttle
- Suspect model just deployed (canary one machine)
- OS patch in progress
- A/B routing for new pipeline stages
- Auth failure (Q1)

Shape:
- New `worker_pool` table: `worker_id PK, enabled bool, disabled_reason text, disabled_at timestamp`
- `POST /api/worker/:id/disable` and `/enable` endpoints (password-protected like existing delete)
- Worker checks its own `enabled` state before claiming; if disabled, sleeps and re-checks rather than exiting (so re-enable doesn't require restart)

Implementation cost: ~half a day. Pays for itself the first time something else goes wrong.

## Q3 — Cookie jar atomicity

Yes. The pyicloud library overwrites session files mid-auth, so failed auths leave a partial cookiejar that subsequent reads treat as authoritative. Wrap-and-checksum is overkill — atomic rename solves it.

Pattern:
```python
session_active = Path("config/icloud_session/")
session_candidate = Path("config/icloud_session.candidate/")
shutil.copytree(session_active, session_candidate)  # start from known-good
api = PyiCloudService(..., cookie_directory=str(session_candidate))
if api.is_trusted_session:
    shutil.rmtree(session_active)
    session_candidate.rename(session_active)  # atomic
else:
    shutil.rmtree(session_candidate)  # active dir untouched
```

Quick win, prevents the specific footgun that caused this incident.

## Q4 — Peer-to-peer MOV pull (wrong layer — use R2)

The workaround worked because andrew-pc had `raw/` cached. But making peer-to-peer first-class turns the GPU pool into an N² trust topology with implicit dependencies between machines. That's a complexity we don't want.

**Cleaner: every successful preprocess uploads the source MOV to R2 too.** Then asset resolution becomes:

1. Local cache (`raw/{vid}.MOV`)
2. R2 (`source/{vid}.MOV`) ← new
3. iCloud (auth-gated, last resort)

Implication: iCloud auth becomes "needed only for *new* ingest, not reprocess." Future incidents like this one stop blocking the architecture work and gallery consistency.

Cost: ~4 GB/video × ~150 videos = ~600 GB R2 storage. At Cloudflare R2 pricing (~$0.015/GB/month) that's ~$9/month — trivial vs the operational pain it eliminates.

This solves Q4 AND most of Q5 (because reprocess no longer touches iCloud).

## Q5 — Backoff on hard auth errors

Yes, but **Q4-via-R2 makes Q5 much less urgent.** If reprocess doesn't depend on iCloud, the watcher being down for 24h on a cooldown is "no new ingest tonight" not "pipeline broken."

Still worth doing: watcher should distinguish `-20209` (account locked) from session-stale errors. On hard lockout: alert via SMS+email (existing path), sleep 1h, retry. Don't systemd-restart-loop and burn cooldown time.

`verify_session_health()` already exists — extending it with auth-class error categorization is the natural place.

# Recommended priority order

1. **Q4 (R2 source caching)** — highest leverage. Closes Q5, reduces blast radius of any future iCloud incident, decouples reprocess from auth fragility.
2. **Q2 (coordinator kill switch)** — small implementation, generally useful primitive beyond this incident.
3. **Q3 (atomic session writes)** — quick win, prevents the specific footgun.
4. **Q1 → Q2** — special-case Q1 into Q2 once Q2 lands.
5. **Q5 — defer.** After Q4, this is nice-to-have polish.

# Draft BACKLOG.md entries

Paste these into the right section of `BACKLOG.md` (probably under a new `## Pipeline robustness` or existing `## Infrastructure` section):

```markdown
- **R2 source-MOV caching** — every successful preprocess uploads source MOV to `source/{vid}.MOV` in R2. Asset resolution becomes local → R2 → iCloud. Eliminates iCloud-auth coupling for reprocess. ~600 GB / ~$9/mo storage cost. Surfaced 2026-05-07 by iCloud lockout incident.
- **Coordinator worker kill switch** — `worker_pool` table with `enabled bool`, `POST /api/worker/:id/disable` endpoint, worker checks its own enabled state before claiming. Use cases: thermal throttle, suspect model canary, OS patch, auth failure auto-park. Surfaced 2026-05-07 by iCloud lockout incident.
- **Atomic iCloud session writes** — pyicloud overwrites cookie jar mid-auth, leaving partial state on failure. Wrap auth in `session.candidate/ → atomic rename` pattern so failed auths never touch the active session dir. Surfaced 2026-05-07 by iCloud lockout incident.
- **Watcher: distinguish `-20209` from session-stale** — on hard account-lock errors, alert + sleep 1h instead of systemd-restart-loop. Less urgent if R2 source caching lands first. Surfaced 2026-05-07 by iCloud lockout incident.
```

# What this changes for current state

Nothing immediate. The 34-job reprocess will land via the workaround; we have ~5 hours of watching Andrew-PC chew through the pool. These are post-incident architectural items for backlog, to be picked up after this batch is done and tmassena is back online.

If you want, I can also draft a postmortem doc (`docs/postmortems/2026-05-07-icloud-lockout.md`) that ties the timeline + root causes + remediation into one durable record. Tell me yes and I'll draft it. Read-only on doc commit — you'd commit it from main.

## Response

Acked. Priority order accepted as-is — Q4 first (R2 source cache) is the right top item; it makes Q5 mostly moot and decouples reprocess from iCloud entirely.

**BACKLOG.md updated** — added a new `## Pipeline robustness` section with all 4 entries verbatim from this brief. Section sits below `## Cross-platform gaps`. Lockout incident origin noted in each.

**Postmortem doc:** yes, please draft `docs/postmortems/2026-05-07-icloud-lockout.md` — this incident's blast radius (model-divergence requeue piggy-backing on watcher fragility, MOV peer-copy as recovery) is worth a durable record. I'll commit it from main when you ship the draft to inbox.

**One nit on Q4:** brief mentions ~600 GB / ~$9/mo. The 30 MOVs already on andrew-pc total 116 GB across just one batch — full corpus is closer to ~150 videos × 4 GB avg = ~600 GB. Math checks out, just confirming.

**Note on the Shortcuts-ingest brief** (`20260507-0610-shortcuts-ingest-scoping-from-design-partner.md`): I see it lands Q4 + Q5 together in one move. Reviewing separately; will reply via that brief's archive entry. The Q1/Q2/Q3 backlog items remain valid post-Shortcut-cutover.
