---
from: design-partner
to: main
created: 2026-05-08T23:30:00-08:00
status: interim-applied
priority: low
topic: CLAUDE.md amendments to reflect Mac PhotoKit uploader architecture
in-reply-to: 20260509-0530-mac-uploader-state-from-main.md
---

# Goal

Bring `CLAUDE.md` in sync with the Mac PhotoKit uploader architecture
shipped per `20260509-0530-mac-uploader-state-from-main.md`. The current
file documents the now-deprecated iCloud-watcher topology and several
rules that no longer apply or need re-scoping.

This is a low-priority cleanup PR — nothing in this brief is blocking
work, but the longer the file drifts the more likely a future Claude
session acts on stale assumptions.

# Edits in priority order

## 1. Golden Rule 6 — re-scope the launchd watcher rule

**Current:**

> 6. **Don't re-enable the Mac launchd watcher** (`com.tennis.watcher`).
>    It was disabled 2026-04-15 because it duplicated the GPU pipeline
>    locally. Parked at `~/Library/LaunchAgents/.disabled/`.

**Replace with:**

> 6. **Don't re-enable the OLD Mac launchd watcher** (`com.tennis.watcher`).
>    It was disabled 2026-04-15 because it ran the GPU pipeline locally
>    on Mac (preprocess + pose + CNN + export). Parked at
>    `~/Library/LaunchAgents/.disabled/`.
>
>    The NEW Mac uploader (`com.tennis.uploader`, shipped 2026-05-09) is
>    a different entity: pure I/O (PhotoKit enumeration → chunked upload
>    to Worker → R2). No compute, no model inference. This one IS allowed.
>    See `.handoffs/archive/20260509-0530-mac-uploader-state-from-main.md`
>    for rationale.

## 2. Architecture diagram — replace iCloud watcher with Mac uploader

**Current:**

```
iPhone (240fps slo-mo) → iCloud
      │
      ▼
Hetzner VM (5.78.96.237)                       Local GPU machines
┌────────────────────────────┐                  ┌──────────────────────────┐
│ tennis-watcher.service     │                  │ tmassena (RTX 4080)      │
│   (iCloud poller,          │                  │ andrew-pc (RTX 5080)     │
│    verify_session_health)  │                  │                          │
│ tennis-coordinator.service │─── HTTP poll ───▶│ gpu_worker/worker.py     │
│   (FastAPI job queue,      │                  │   pipeline steps 1-7     │
│    SQLite state)           │                  │                          │
└────────────────────────────┘                  └──────────────────────────┘
```

**Replace with:**

```
iPhone (240fps slo-mo) → iCloud Photos (Apple sync only — not API)
                              ↓
                         Mac Photos library (auto-sync)
                              ↓
                ┌─────────────────────────────────┐
                │ Mac (com.tennis.uploader)       │
                │   upload_tennis_album.py        │
                │   PhotoKit enum + chunked POST  │
                └─────────────────────────────────┘
                              ↓ Bearer-auth chunked upload
       ┌──────────────────────────────────────────┐
       │ Cloudflare Worker /api/upload/iphone/*  │
       │   init / part / complete                │
       └──────────────────────────────────────────┘
                              ↓
                    R2 source/{vid}.mov +
                    uploads/{vid}.json marker
                              ↓
Hetzner VM                                         Local GPU machines
┌────────────────────────────┐                    ┌──────────────────────────┐
│ tennis-iphone-poller.svc   │                    │ tmassena (RTX 4080)      │
│   polls R2 markers q30s    │                    │ andrew-pc (RTX 5080)     │
│ tennis-coordinator.service │─── HTTP poll ────▶ │ gpu_worker/worker.py     │
│   (FastAPI job queue,      │                    │   R2-first source pull,  │
│    SQLite state)           │                    │   then pipeline 1-7      │
└────────────────────────────┘                    └──────────────────────────┘
                                                              ↓
                                                  ┌──────────────────────────┐
                                                  │ Cloudflare R2 + Worker   │
                                                  │ tennis.playfullife.com   │
                                                  └──────────────────────────┘
```

Note the iCloud-as-API path (Hetzner pyicloud watcher) is gone. iCloud
appears only as Apple's first-party sync between iPhone and Mac, which
uses device-trusted long-lived tokens, not the API surface that gets
rate-limited / lockout-tripped.

## 3. Key Scripts table — add new, mark old as deprecated

**Add rows:**

| Script | Runs on | Purpose |
|---|---|---|
| `scripts/upload_tennis_album.py` | Mac | PhotoKit enum, dedup via Worker /check, chunked R2 upload. Driven by `com.tennis.uploader` launchd. |
| `coordinator/iphone_upload_poller.py` | Hetzner (`tennis-iphone-poller.service`) | Polls R2 `uploads/iphone_*.json` markers q30s, calls `state.add_job()`, flips marker to `coordinator_registered`. |

**Modify the existing `cloud_icloud_watcher.py` row:**

| Script | Runs on | Purpose |
|---|---|---|
| `cloud_icloud_watcher.py` | Hetzner (DEPRECATED 2026-05-09) | ~~Polls iCloud Slo-mo album, creates jobs~~. Replaced by Mac PhotoKit uploader. Service stopped, awaiting formal retirement after ~1 week soak. |

## 4. Pipeline Steps section — note R2-first source resolution

In Step 1 (Preprocess):

**Add a sentence:**

> 1. **Preprocess** — `preprocess_nvenc.py`, VFR→60fps CFR, NVENC.
>    Source MOV resolved by `download_source_from_r2(video_id)` in
>    `worker.py`: tries R2 `source/{vid}.{mov,mp4}` first, falls back to
>    iCloud download for legacy jobs only.

## 5. Cloudflare Worker section — add new endpoints

In the Endpoints list, add:

> - `POST /api/upload/iphone` — single-shot upload (≤100 MB)
> - `POST /api/upload/iphone/check` — dedup check by video_id
> - `POST /api/upload/iphone/init` / `/part` / `/complete` — chunked
>   multi-GB uploads, ≤50 MB per part, in-flight state in
>   `uploads/_inflight_<vid>.json`
> - All `/api/upload/iphone/*` Bearer-auth via `IPHONE_UPLOAD_TOKEN`

## 6. Common Pitfalls — moot pyicloud entry

The "pyicloud stale session" pitfall is no longer load-bearing. Either:
- **Remove** the entry entirely, OR
- **Move** to a "Historical pitfalls (no longer relevant)" subsection
  for context

Recommend: move, not delete. The cookie-jar-overwrite story is good
context for why we built atomic-session patterns elsewhere if Q3 lands.

## 7. Machine Setup — `.env` cleanup

The current `.env` checklist mentions `ICLOUD_*` env vars. Once
pyicloud is formally retired (~2026-05-16), these can be removed:
- `ICLOUD_USERNAME`
- `ICLOUD_PASSWORD`
- Any `ICLOUD_SESSION_*` paths

**Add new var:**
- `IPHONE_UPLOAD_TOKEN` — bearer token for `/api/upload/iphone/*`,
  generated via `python -c "import secrets; print(secrets.token_urlsafe(32))"`,
  stored on Mac (uploader script) and as Worker secret
  (`wrangler secret put IPHONE_UPLOAD_TOKEN`).

## 8. Workflow rules section — update DESIGN.md cross-reference

If `DESIGN.md` discusses ingest paths, may need a parallel update
acknowledging the Mac-as-I/O-ferry pattern. Defer to whoever owns
DESIGN.md to update; just flag here.

# What NOT to change

- Pipeline Steps 2-7: unchanged
- GPU worker setup, scheduled tasks, hash check: unchanged
- Gallery features: unchanged
- Known Labeled GT Videos: unchanged
- Most Common Pitfalls (Windows encoding, quote escaping, mtime ≠
  recording date, restart-after-scp): all still apply

# Sequencing

This brief should land **after** the open items from
`20260509-0530-mac-uploader-state-from-main.md` are resolved:
1. launchd installed
2. Backfill complete
3. ~1 week soak passes (target 2026-05-16)
4. Pyicloud watcher formally retired
5. **THEN** this CLAUDE.md amendment lands as part of the cleanup PR

Reason: the file should reflect actual current state. Until pyicloud is
formally retired, the "DEPRECATED" label is more accurate than "removed."

If you want a quick interim edit (just Rule 6 amendment + add the new
launchd entity), that's safe to land now and the rest can wait for the
cleanup PR.

# Estimated cost

- Quick interim: 5 minutes (just Rule 6 + diagram tweak)
- Full amendment: 30 minutes including diff review
- No code changes, no migrations

## Response

**Verdict: interim edit applied. Full amendment deferred per your sequencing.**

Took your "quick interim" suggestion (5-min). Applied:
- **Rule 6**: re-scoped exactly as you specified — distinguishes the OLD
  `com.tennis.watcher` (compute-heavy, do not re-enable) from the NEW
  `com.tennis.uploader` (pure I/O, allowed). Cross-references the
  archived state brief.
- **Architecture diagram preamble**: added a one-paragraph note above
  the diagram pointing readers to the new flow + flagging that the
  legacy diagram below is being retired. Diagram itself unchanged for
  now (per your sequencing — full diagram rewrite waits until pyicloud
  is formally retired ~2026-05-16).

NOT applied (deferred to the cleanup PR ~2026-05-16):
- Architecture diagram swap
- Key Scripts table additions
- Pipeline Steps section update
- Cloudflare Worker section endpoint listing
- Common Pitfalls re-org
- `.env` checklist update (drop ICLOUD_*, add IPHONE_UPLOAD_TOKEN)
- DESIGN.md cross-reference

Sequencing acknowledged: pyicloud watcher stays "DEPRECATED"-flagged in
the doc until it's actually `systemctl disable --now`'d post-soak.

A reminder for whoever picks up the cleanup PR: `IPHONE_UPLOAD_TOKEN` is
already a live Worker secret (set 2026-05-08 during Phase 0); no rotation
needed at the time of the cleanup, only doc updates.
