---
from: main
to: design-partner
created: 2026-05-09T05:30:00-08:00
status: pending
priority: medium
topic: Mac PhotoKit uploader replaced iCloud watcher + Shortcut path; state, choices, open items
---

# Where we are

The iCloud-lockout incident triggered a rethink of the ingest path. After
trying the iOS Shortcut approach (Brief `20260507-0610-shortcuts-ingest-scoping`)
and hitting iOS automation timeouts on multi-GB videos, we pivoted to a
**Mac-side PhotoKit uploader** that streams new slo-mo videos from the local
Photos library through the Worker into R2.

This brief captures the resulting architecture, the decisions we made along
the way, and what's still open.

## Architecture (live)

```
iPhone records slo-mo → iCloud Photos
                         ↓ (auto-sync)
                  Mac Photos library
                         ↓
   scripts/upload_tennis_album.py (PhotoKit enumeration)
                         ↓
   Worker /api/upload/iphone/{init,part,complete}  (chunked, Bearer)
                         ↓
   R2 source/{video_id}.mov + uploads/{video_id}.json marker
                         ↓
   tennis-iphone-poller.service (Hetzner, polls every 30s)
                         ↓
                 coordinator.add_job()
                         ↓
   GPU worker claims, R2-first source pull, normal pipeline
                         ↓
                       gallery
```

End-to-end smoke-tested with a 917 MB IMG_1213.MOV: 19-chunk upload in ~75s,
poller picked up in <30s, tmassena claimed and ran preprocessing → pose →
detection successfully.

## Components & locations

- **`scripts/upload_tennis_album.py`** (Mac) — enumerates slo-mo via
  PhotoKit, dedups via Worker `/check`, uploads single-shot or chunked.
  CLI: `--limit N`, `--watch N`, `--since YYYY-MM-DD` (default `2026-04-01`).
- **`worker/upload-worker.js`** — `/api/upload/iphone` (single-shot ≤100 MB),
  `/api/upload/iphone/check`, `/api/upload/iphone/{init,part,complete}` (chunked).
  Bearer auth via `IPHONE_UPLOAD_TOKEN`.
- **`coordinator/iphone_upload_poller.py`** + `tennis-iphone-poller.service`
  on Hetzner — polls R2 `uploads/iphone_*.json` markers every 30s, calls
  `state.add_job()`, flips marker `status: coordinator_registered`.
- **`gpu_worker/worker.py`** — `download_source_from_r2(video_id)` tries R2
  `source/{vid}.{mov,mp4}` before iCloud download. Both GPU machines updated.
- **`docs/IPHONE_SHORTCUT_SETUP.md`** — user-facing setup doc (now mostly
  irrelevant since we abandoned the Shortcut path; could trim).

## Decisions & rationale

### 1. Slo-mo subtype as the filter, not a Photos album

Originally specced an explicit "Tennis" album the user would manually curate.
After building the iOS Shortcut version, switched to filtering by slo-mo
subtype because every tennis recording is already slo-mo and album curation
adds friction. Detection: presence of a type-6 resource (the auto-generated
30fps "flat" version that Photos creates from 240fps originals) — more
reliable than `mediaSubtypes` flag, which sometimes doesn't survive iCloud
sync from iPhone → Mac.

Risk: non-tennis slo-mo videos (family recordings) would also flow into the
pipeline. Mitigated with `--since 2026-04-01` default — older slo-mo content
predates tennis-only-slo-mo behavior.

### 2. Marker-file pattern instead of Worker→Coordinator HTTP

We tried having the Worker call coordinator directly after upload. CF Workers
can't fetch their own origin IP (error 1003 "Direct IP access not allowed");
needed a DNS-only subdomain that we don't have permission to create
(API token lacks DNS edit scope). Pivoted to:
- Worker writes `uploads/{video_id}.json` marker to R2
- Hetzner-local poller reads markers, calls `state.add_job()` directly

Pros: no public auth surface on coordinator, decoupled, resilient to brief
Hetzner downtime. Cons: ~30s registration latency vs <1s for direct call.
For single-user volume this is fine.

The HTTP fast-track code is still in `upload-worker.js` (no-op currently);
will activate if/when `coordinator.playfullife.com` DNS-only record exists.

### 3. Chunked upload via R2 multipart (not presigned URL)

CF edge has a 100 MB body cap on Workers — single-shot path fails on any
real tennis video (1-7 GB for 240fps slo-mo). Built `init/part/complete`
endpoints that:
- Stash in-flight upload state in `uploads/_inflight_<vid>.json` (Worker stays stateless)
- Each part ≤50 MB, well under the edge cap
- `complete` writes the marker for the poller

Considered: R2 presigned URLs (client uploads directly to R2 bypassing Worker).
Rejected for now — adds AWS-style URL signing complexity in a Worker, and
chunked-via-Worker works.

### 4. iOS Shortcut path abandoned (kept for reference, not active)

iOS automations have a hard ~30-60s timeout. Multi-GB uploads over LTE/wifi
exceed it. Manual share-sheet works but the user has to remember to share
each video. Mac-side automation has no such constraint and uses the same
Worker endpoints anyway.

`docs/IPHONE_SHORTCUT_SETUP.md` should probably be archived to a
`docs/deprecated/` folder rather than deleted — it documents the Worker
endpoints clearly and a future iOS app would use the same protocol.

### 5. Mac ≠ "no automation" — narrowed to "no compute"

The old `com.tennis.watcher.plist` launchd agent was disabled because it ran
preprocess/pose/CNN/export on Mac (compute that belongs on GPU). The new
script is pure I/O — PhotoKit enumeration, /check, chunked upload. No GPU
work, no ffmpeg, no model inference. Memory rule "Mac watcher disabled, do
not re-enable" was specifically about the old compute-heavy watcher; this is
a different entity.

## Open: how should the new Mac script run continuously?

User asked the right question: "as long as Mac is up, will new videos flow?"
Currently the script is one-shot; needs scheduling. Three options:

| Option | Pros | Cons |
|---|---|---|
| `--watch 300` foreground | Trivial. No system config. | Closes on terminal exit / reboot |
| **launchd agent** | Set-and-forget, persists across reboots, stdout/stderr logged | New plist file; have to be careful not to confuse with the disabled old one |
| Cron | Simple if you already use cron | Less robust on macOS; no Notification Center integration |

Recommendation: launchd. Plist suggestion at the bottom of this brief.

## Open items / handoffs

1. **Decide on continuous-run mechanism.** Tell main to install the launchd
   agent or commit to manual `--watch` invocation.
2. **Retire the iOS Shortcut path explicitly.** Either delete or move
   `docs/IPHONE_SHORTCUT_SETUP.md` to `docs/deprecated/`. Delete the
   `Tennis Sync` shortcut on iPhone (cosmetic).
3. **Pyicloud watcher** still installed on Hetzner, currently failing in a
   crash loop because the iCloud account is locked. Plan: leave disabled
   until the new Mac path is soaked for ~1 week; then formally retire.
   `tennis-watcher.service` should be `systemctl disable --now`'d once we're
   confident.
4. **iCloud account password regen** — still parked on Apple's post-lockout
   cooldown. May not even matter once pyicloud is retired. Decide whether to
   fix proactively or let it stay broken.

## Suggested launchd plist

For `~/Library/LaunchAgents/com.tennis.uploader.plist` — distinct name from
the disabled `com.tennis.watcher.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.tennis.uploader</string>
  <key>ProgramArguments</key>
  <array>
    <string>/Users/andrewhome/tennis_analysis/.venv/bin/python</string>
    <string>/Users/andrewhome/tennis_analysis/scripts/upload_tennis_album.py</string>
    <string>--watch</string>
    <string>300</string>
  </array>
  <key>WorkingDirectory</key>
  <string>/Users/andrewhome/tennis_analysis</string>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>StandardOutPath</key>
  <string>/Users/andrewhome/Library/Logs/tennis_uploader.log</string>
  <key>StandardErrorPath</key>
  <string>/Users/andrewhome/Library/Logs/tennis_uploader.err.log</string>
</dict>
</plist>
```

Load with `launchctl load -w ~/Library/LaunchAgents/com.tennis.uploader.plist`.
Unload with `launchctl unload ~/Library/LaunchAgents/com.tennis.uploader.plist`.

Use `KeepAlive: true` plus `--watch 300` so the script self-loops and
launchd just restarts on crash. Alternative: `StartInterval: 300` with no
`--watch` and `KeepAlive: false` — runs as one-shot every 5 min. Either
works; KeepAlive+watch has lower latency and one less process spawn cycle.

## What's NOT in scope of this brief

- Detection accuracy / model work — separate stream
- iOS app for live coaching — separate stream
- Multi-user / token-per-user — flagged in Shortcut brief, still post-MVP

## Backfill in flight

While drafting this brief, `upload_tennis_album.py` is running the initial
backfill of 13 missed videos from the lockout window (2026-05-01 →
2026-05-07). ~50 GB total. Each video pulled from PhotoKit, chunk-uploaded,
poller registers, GPU processes. Self-monitoring; no action required.
