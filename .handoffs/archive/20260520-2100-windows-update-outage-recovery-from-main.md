---
from: main
to: design-partner
created: 2026-05-20T21:00:00-08:00
status: pending
priority: low
topic: Recovered from 2-day GPU outage (Windows update), iPhone backlog fully drained
---

# What happened

Both GPU machines (Andrew-PC and tmassena) went offline 2026-05-16 evening,
stayed dark through 2026-05-18. Tailscale unreachable, LAN ping silent.
Cause: Windows Update auto-reboot. The user confirmed it on inspection;
TennisGPUWorker scheduled task and Tailscale daemon don't auto-start
fast enough after some update cycles.

During the outage:
- `com.tennis.uploader` (Mac launchd) kept running normally
- `tennis-iphone-poller.service` (Hetzner) kept registering jobs
- Coordinator queue accumulated 9 pending iPhone jobs
- Gallery showed ~39 `coordinator_registered` badges (registered but
  awaiting GPU pickup)

# Recovery

1. Confirmed both machines unreachable (Tailscale + LAN).
2. Sent WoL magic packets via `scripts/wake_machines.py all`.
3. Both came back, TennisGPUWorker auto-started, claims resumed within
   minutes.
4. Backlog cleared overnight.

# Current state — iPhone jobs

```
completed:  34
failed:      1
pending:     0
processing:  0
```

Last completion 2026-05-19T04:43Z (IMG_1260). Queue is clean.

# Open: the 1 failed job

Haven't dug in yet. If you want me to investigate (single video_id, ~5
min to triage), say so. Otherwise leaving it parked.

# Suggestion (low priority)

Worth a one-pass thought: should the watcher on Hetzner SMS/email when
GPU workers haven't claimed in N hours despite pending markers? Today
the failure mode was silent — gallery just accumulated `coordinator_registered`
badges with no surfaced alert. Adding a "stale pending jobs >2h" alarm
on the watcher would catch the next Windows-update-induced outage in
hours instead of days.

Not proposing scope right now, just flagging.

# No code changes this session

Pure ops recovery. No commits to main beyond the docs already auto-updated.
