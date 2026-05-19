---
from: main
to: design-partner
created: 2026-05-19T00:10:00-08:00
status: pending
priority: high
topic: 0-shots diagnosis — pipeline isn't broken; two distinct surface bugs
---

# TL;DR

The brief's premise — *"silent pipeline failure, every recent iPhone
upload produces 0 shots"* — turned out to be wrong. Detection IS
running and producing real output for iPhone-prefixed videos. Two
unrelated surface bugs were combining to look like a pipeline
regression. Both are now diagnosed; one is moot post-STOP, the other
is a real UX bug worth bundling with the dashboard pivot.

# STOP — done

`gpu_worker/worker.py` committed (1e822c5): both `_send_notification(
"notify_upload_complete", ...)` calls in `worker_loop` (line 1038) and
`process_local_video` (line 1113) are replaced with no-ops. Failure
path (`notify_processing_failed`) kept intact.

SCP'd to tmassena + Andrew-PC, both workers restarted at 12:02 AM,
hash matches (`71DD9B23F9D07A7F1B547B814358A007CF2160866804F6D64A53CDE727986BF6`).
Queue was empty at restart — no in-flight jobs interrupted.

The "ready (0 shots) — Duration: Unknown" emails will stop
immediately. ❌ failure emails still fire (rare → manageable signal).

# DIAGNOSE — what was actually happening

## Evidence that detection is working

- `iphone_9ca0a615_fused_detections.json` 42758 bytes (5/18 9:19 PM) on tmassena
- `iphone_1d46ea54_fused_detections.json` 91984 bytes (5/18 8:20 PM)
- `iphone_f0bce125_fused_detections.json` 27929 bytes (5/18 5:56 PM)
- `https://tennis.playfullife.com/highlights/iphone_fe04574a/meta.json`
  → `{"duration": 519.22, "shots": 67, "breakdown": {"forehand": 48, "backhand": 19}}`
- Gallery shows `iphone_*_forehands.mp4`, `_backhands.mp4`, etc. for
  recent uploads — those exports require non-zero shots upstream.

So the model + pose + detection chain on iPhone uploads is *fine*.
The "5-min audit" P2 item from the 2026-05-09 brief was correct that
the pipeline produces output; what nobody had checked was whether the
**notification stats lookup** finds it.

## Root cause 1 — wrong key for stats lookup (caused the "0 shots" emails)

`gpu_worker/worker.py:1013` in `worker_loop`:

```python
video_name = Path(job.get("filename", "")).stem  # → "IMG_1260" (iCloud filename)
...
stats = _get_detection_stats(video_name)          # looks for IMG_1260_fused_detections.json
```

But everywhere *inside* `run_pipeline_with_stages`, `video_name =
video_path.stem` (line 605) — and `video_path` comes from
`download_source_from_r2()` which writes to `RAW_DIR / f"{video_id}.{ext}"`.
So the detection JSON gets saved as `iphone_<hash>_fused_detections.json`,
keyed by `video_id`, not by the iCloud filename.

The two names diverge only on the iPhone path (legacy iCloud jobs use
`IMG_*.MOV` end-to-end and happen to match). `_get_detection_stats()`
falls through to the empty-dict path → 0 shots, duration 0 →
"Duration: Unknown" in `email_notify.py:265`.

**This is moot now** that the per-success notification is off, but if
notifications are ever re-enabled, change `worker_loop` line 1013 to
use `job["video_id"]` (or skip the filename derivation entirely and
read from `_get_detection_stats(job["video_id"])`).

## Root cause 2 — broken upload_id linkage (the "stuck at coordinator_registered" you saw)

Your screenshot (IMG_1252..IMG_1260, all `coordinator_registered`) is
a *different* bug from the email one.

State machine for iPhone uploads:

1. Cloudflare Worker writes `uploads/iphone_<hash>.json` marker with
   status `awaiting_coordinator` (upload-worker.js:735)
2. Hetzner `iphone_upload_poller.py` polls R2, creates a coordinator
   `VideoJob`, flips marker to `coordinator_registered`
   (iphone_upload_poller.py:122)
3. **Expected**: GPU worker claims job → `report_queue_status(upload_id,
   status="processing", ...)` → eventually `complete`
4. **Actual**: step 3 never fires. The marker stays at
   `coordinator_registered` forever, even though the video gets
   processed and gallery cards appear.

The break is at `iphone_upload_poller.py:110`:

```python
job = VideoJob(
    video_id=video_id,
    icloud_asset_id=asset_id,
    filename=filename,
    status=VideoStatus.PENDING,
    album_name=marker.get("source", "iphone_shortcut"),
)
```

No `upload_id` field is set. So when the GPU worker later does
`job_upload_id = job.get("upload_id")` (worker.py:1015), it gets
`None`, and every subsequent `if upload_id: report_queue_status(...)`
silently no-ops. The website-queue marker in R2 never advances.

The fix has three touchpoints:
1. Cloudflare Worker: the marker `uploads/iphone_<hash>.json` already
   has `video_id` — for iPhone uploads `video_id == upload_id == iphone_<hash>`
   (worker.js:678 confirms this). So no schema change needed there.
2. `iphone_upload_poller.py`: pass `upload_id=video_id` to `VideoJob(...)`
3. `coordinator/state.py` + `coordinator/api.py`: ensure `VideoJob` has
   an `upload_id` field that round-trips through claim/return JSON
4. GPU worker should then receive `job["upload_id"]` from the
   coordinator and start updating the queue marker.

**Not trivial — three services. Filing this back to you rather than
shipping in the STOP commit.**

# SCOPE — dashboard pivot

Pre-existing memory note from earlier today: *"Diagnostics are
first-class, not inline. Push notifications deprecated except failure
alerts."* That principle and the dashboard direction in your brief
are the same thing.

Recommendation: **bundle the upload_id fix with the dashboard pivot**,
since the dashboard surface is exactly where users will see whether
something processed. Fixing the queue state machine without a place to
read it from is just plumbing without a customer.

Proposed split:
- **Now (no follow-up needed):** STOP committed + deployed. Failure
  alerts retained. Per the brief, success emails die. Done.
- **Next (when you scope the dashboard brief):** the "Processing now"
  and "Recently failed" surfaces on the gallery. Wire the upload_id
  linkage as part of that work so the markers can actually drive the
  dashboard. Specifically, the table in your brief becomes:

  | Question | Dashboard answer | Requires |
  |---|---|---|
  | Did my video process? | Card appears w/ thumbnail + shot count | (already works) |
  | How many shots? | On the card | (already works) |
  | Did it fail? | "Recently failed" section | Read coord's `failed` jobs + R2 marker `failed` status |
  | What's in the queue? | "Processing now" indicator | **Requires upload_id plumbing fix above** |

# What I did NOT do

- Did not fix root cause 1 in code (moot post-STOP)
- Did not fix root cause 2 in code (non-trivial, want your call on
  whether it bundles with dashboard or ships standalone)
- Did not touch pro-footage / detection / filmstrip work
- Did not draft the dashboard brief itself — that's still on your plate

# Verification commands

```bash
# Confirm pipeline still runs end-to-end on next upload:
curl -s "https://tennis.playfullife.com/highlights/iphone_9ca0a615/meta.json" | jq .
# Expect: duration > 0, shots > 0, breakdown populated.

# Confirm no more success emails:
grep -nE "notify_upload_complete" gpu_worker/worker.py
# Expect: only the comments/_= stub lines, no live call sites.

# Confirm failure path still wired:
grep -nE "notify_processing_failed" gpu_worker/worker.py
# Expect: two live call sites (worker_loop + process_local_video).
```
