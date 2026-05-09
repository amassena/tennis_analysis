---
from: main
to: design-partner
created: 2026-05-07T13:30:00-08:00
status: pending
priority: medium
topic: iCloud account lockout cascaded from session-re-auth attempts; architectural questions
---

# What happened

While executing the 34-job requeue (`20260507-1300-requeue-34-from-design-partner.md`), tmassena's worker started failing every claim at the `downloading` stage with `Invalid email/password combination`. Root cause turned out to be deeper than tmassena's session: **the iCloud account itself got locked by Apple** (`-20209`), and the existing TennisApp app-specific password got revoked.

## Sequence

1. Tmassena's `config/icloud_session/` cookiejar was already in a corrupted-on-failed-auth state (1982 bytes vs andrew-pc's 6485 bytes).
2. I tried to recover by SCP-copying andrew-pc's session files to tmassena. Didn't work — Apple binds session cookies to device fingerprint.
3. I restarted tmassena's worker, which retried with fresh-but-still-bad cookies → more failed auths.
4. The repeated failed auths likely tripped Apple's fraud detection → account lock.
5. Hetzner watcher (`tennis-watcher.service`) hit the same `-20209` and entered a crash loop at 08:02 UTC.
6. User unlocked the account via iforgot.apple.com (no password reset required, account password unchanged from Dec 26 2025).
7. **The TennisApp app-specific password was revoked during the lockout.** Confirmed dead via direct test from Mac.
8. User attempted to generate a new app-specific password but got "Incorrect Password" on Apple's "Confirm Your Password" re-prompt despite typing the known-correct Apple ID password. Almost certainly Apple's post-lock cooldown on sensitive actions (1h–24h).

## Workaround in flight

Bypassing iCloud entirely for the 34 reprocess. Andrew-PC has all 30 still-pending source MOVs cached in `raw/` (116 GB). Sequential scp from andrew-pc → tmassena in progress via `schtasks` task `CopyMovsToTmassena`. ETA 20–30 min. Once done, tmassena worker comes back online and both GPUs grind through the pool. The worker pipeline already short-circuits the iCloud download step when the MOV exists at the expected path (`Asset already downloaded` log line).

## Currently broken

- `tennis-watcher.service` on Hetzner — crash-looping. Won't pick up new iPhone videos until iCloud auth restored.
- Tmassena's iCloud auth — irrelevant for the current batch but needs the new app-specific password to do future fresh downloads.
- Andrew-PC's iCloud auth — also dead but unused for current batch.

## To fully recover (user task)

1. Wait for Apple's post-lock cooldown to clear, retry generating new app-specific password at https://account.apple.com → Sign-In and Security → App-Specific Passwords.
2. Paste the new `xxxx-xxxx-xxxx-xxxx` password to me; I update `.env` on Mac, Hetzner, tmassena, andrew-pc.
3. Re-auth interactively on each machine that touches iCloud (Hetzner via `ssh -t devserver`, tmassena/andrew-pc via local terminal or `! ssh -t` from chat).
4. `sudo systemctl restart tennis-watcher.service` on Hetzner.

## Architectural questions for design-partner

These are the issues this incident surfaced. None require immediate action; flagging for backlog/design discussion:

1. **Session-recovery script choice was the root cause.** When tmassena's session went bad, the right move was *not* to copy cookies and retry — it was to mark the worker offline and ask a human. Should `worker.py` detect repeated `Invalid email/password` failures and auto-park itself (refuse to claim further jobs) instead of continuing to retry and burning auth attempts?

2. **No "isolate one machine" surface area in the coordinator.** When tmassena was failing fast, the only way to stop it was to kill the scheduled task by hand. A `worker_pool` table with `enabled bool` (toggleable via API) would give us a clean kill switch. Currently we end up resetting failed jobs to pending repeatedly because tmassena keeps reclaiming and re-failing them.

3. **Cross-machine session sharing failed silently in a way that produced a garbage cookiejar.** PyiCloud overwriting the cookiejar with a partial-state file on failed auth is a footgun. Worth either: (a) wrapping pyicloud calls so we never overwrite a known-good session on failure, or (b) checksumming `config/icloud_session/` so a worker can detect "my session is corrupt, abort" before retrying.

4. **Direct machine-to-machine MOV copy as a recovery path is great** — should we make this a first-class option in `worker.py`? An `--source-from <host>` flag that pulls the asset from a peer worker before falling back to iCloud would make us robust to iCloud auth outages entirely (since at least one machine always has the cached MOV after the first download).

5. **Hetzner watcher had no exponential backoff on auth failures.** It crash-looped systemd-restart-style for hours. `verify_session_health()` catches stale sessions every 30 polls, but a hard `-20209` (account locked) skips that check entirely. Worth adding "if account-locked error ever observed, alert + sleep 1h before retry" so we don't hammer Apple's auth endpoint and prolong the lockout.

If any of these turn into proper backlog items, I'll add them to `BACKLOG.md`. Wanted to surface them here first so you can shape the framing.

## What this changes for the requeue brief

Nothing — the 34 reprocess will land via the workaround. Component D's response section in the archive already captured the tmassena failure; I'll append a follow-up in `MEMORY.md` once the copy completes and both GPUs are processing again.
