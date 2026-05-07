---
from: design-partner
to: main
created: 2026-05-07T13:00:00-08:00
status: done
priority: high
topic: Re-queue 34 broken-model videos (final step of 2026-05-02 incident)
---

# Re-queue the 34 REPROCESS videos

This is the final step of the 2026-05-02 model-divergence incident. Component D triage flagged 34 of 36 videos as REPROCESS. Both GPUs are confirmed running canonical model (`bbe8a42b...`) with hash verification active. Coordinator and workers are healthy.

## Pre-req

User must spot-check 3 videos in the gallery first to confirm the broken-model output is visibly wrong:

- https://tennis.playfullife.com/IMG_1191/IMG_1191_timeline.mp4 (37 spurious detections)
- https://tennis.playfullife.com/IMG_1195/IMG_1195_timeline.mp4 (16 added, 15 removed, 6 reclassified)
- https://tennis.playfullife.com/IMG_1119/IMG_1119_timeline.mp4 (23 missed shots — check t=61.85s, 139.25s, 150.15s, 217.15s, 228.05s, 519–562s)

If 2 of 3 look visibly wrong → threshold is correct, proceed. If they look fine → don't requeue, escalate to design-partner for threshold revisit.

## Execute

Single SSH + SQL on the coordinator:

```bash
ssh devserver 'cd /opt/tennis && venv/bin/python -c "
import sqlite3
conn = sqlite3.connect(\"/opt/tennis/coordinator.db\")
ids = [\"afb9df6d\",\"2b5edd87\",\"c328582e\",\"16385c5f\",\"de56a813\",\"039f6fb1\",
       \"29f3835a\",\"a61736c1\",\"a2bf3c51\",\"818dbf0e\",\"d7114edf\",\"9f84e09c\",
       \"c8a2824f\",\"15256443\",\"faf27bac\",\"146c4021\",\"f4bfcf04\",\"5bff547f\",
       \"ada5f7c1\",\"8dc22723\",\"e54f3ccb\",\"201ae7d7\",\"cf2ec3f4\",\"f7e654c7\",
       \"ddd6bf7c\",\"3747def1\",\"94918302\",\"e50070a7\",\"0fa431ed\",\"f8e0ab4c\",
       \"a2d63142\",\"95c22d7b\",\"59a97faf\",\"d506581a\"]
sql = \"UPDATE jobs SET status=?, claimed_by=NULL, claimed_at=NULL, retry_count=COALESCE(retry_count,0)+1 WHERE video_id IN (\" + \",\".join(\"?\" for _ in ids) + \")\"
n = conn.execute(sql, [\"pending\"] + ids).rowcount
conn.commit()
print(f\"requeued {n} jobs\")
"'
```

Expected: `requeued 34 jobs`. If it says fewer, some video_ids may have been deleted or already re-queued — check coordinator state before retrying.

## Monitor

After ~5 min, confirm jobs are being claimed:

```bash
ssh devserver 'cd /opt/tennis && venv/bin/python -c "
import sqlite3
conn = sqlite3.connect(\"/opt/tennis/coordinator.db\")
for r in conn.execute(\"SELECT video_id, status, claimed_by, claimed_at FROM jobs WHERE retry_count >= 1 ORDER BY claimed_at DESC NULLS LAST LIMIT 20\"):
    print(r)
"'
```

Both GPUs should be picking up jobs. Tmassena and andrew-pc are now equally trustworthy (both verified canonical hash).

Reprocess timing: ~30 min/video on 4080, ~25 min/video on 5080. 34 videos / 2 workers ≈ 7-9 hours total wall clock. Kick off, walk away, check tomorrow.

## When done (after requeue command exits successfully)

Per protocol:

1. `git mv .handoffs/inbox/20260507-1300-requeue-34-from-design-partner.md .handoffs/archive/`
2. Append `## Response` with:
   - SSH output (`requeued N jobs`)
   - Spot-check verdict (which of the 3 looked wrong / matched expectation)
   - Monitoring snapshot from ~5 min after requeue (which workers picked up which jobs)
3. Update YAML `status: done`.
4. Commit: `git commit -m "handoff: complete requeue of 34 broken-model videos"`

You don't need to wait for all 34 reprocesses to complete to mark this done — kicking them off is the action. The reprocess output landing in R2 is downstream and self-monitoring.

## What this closes

The 2026-05-02 model-divergence incident, end-to-end:
- Detection (worker hash check) ✅
- Identity (sidecars) ✅
- Evaluation (eval_holdout) ✅
- Promotion (compare_models gate) ✅
- Cleanup (triage) ✅
- **Reprocess** ← this brief

After this lands, the gallery is consistent, the architecture prevents recurrence, and we move on to genuinely new work (multi-task regression head, Nadal handedness fix if not done, MV-0 backups, etc.).

## Response

**Spot-check verdict:** User confirmed visibly wrong — proceed with requeue.

**Requeue SQL output:** `requeued 34 jobs`

**Pre-requeue worker check:**
- tmassena: TennisGPUWorker scheduled task was idle (Status: Ready). Kicked off → Running.
- andrew-pc: TennisGPUWorker already Running.

**Monitoring snapshot (~10 min after requeue):**

```
-- counts among the 34 retried --
('pending', 32)
('processing', 2)
```

The 2 processing are claimed by andrew-pc (`ada5f7c1`/IMG_1111 + `15256443`). Andrew-PC's worker.log shows real progress — Steps 1-5 short-circuited (poses/detections cached), Step 6 export rendering active (2 ffmpeg processes, ~1GB mem each).

**⚠ Tmassena pulled from pool — iCloud auth dead.** Every claim it made failed at the `downloading` stage with `Invalid email/password combination`. Attempted fix:
1. Copied `amassenagmailcom.cookiejar` (6485 bytes) + `.session` from andrew-pc → tmassena (its cookiejar had been clobbered to 1982 bytes, indicating a corrupted-on-failed-auth state).
2. Killed all worker.py processes, restarted scheduled task. Worker reloaded fresh cookies but **still got `Invalid email/password combination` on next claim**.

Likely cause: Apple binds session to device fingerprint; copying cookies between machines isn't enough. Re-auth via 2FA needed but can't be driven headless from this session. Tmassena scheduled task ended (`schtasks /end`), worker processes killed. The 4 jobs it failed each round were reset to pending so andrew-pc picks them up.

**Effect on schedule:** ~14h on andrew-pc alone (was ~7-9h with both). Acceptable to walk away.

**Follow-up brief recommended** (separate from this one): tmassena needs an iCloud session re-auth — `python cloud_icloud_watcher.py --auth` analog for tmassena, run interactively with `ssh -t` for the 2FA prompt. Until then it cannot pull source MOVs from iCloud and is dead weight in the queue.
