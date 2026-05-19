---
from: design-partner
to: main
created: 2026-05-18T22:00:00-08:00
status: pending
priority: high
topic: Stop email/SMS firehose AND investigate why iPhone uploads produce 0-shot outputs
---

# Two urgent items, one brief

User flagged tonight: emails and SMS from the pipeline are out of
control. Looking at the volume + content reveals a worse problem —
**every recent iPhone upload is reporting "ready (0 shots) — N/A
Duration: Unknown."** That's not a notification problem; that's a
silent pipeline failure.

User's strategic reframe also worth capturing: **"get out of the
message business and into the dashboard business."** Gallery is
already a dashboard surface; the email/SMS firehose was a 2024-era
hack we should retire.

This brief has three actions, in priority order: **STOP**, **DIAGNOSE**,
**SCOPE**.

# 1. STOP — disable email + SMS notifications (~10 min, do first)

The firehose is masking real failures and adding cognitive load. Kill
it now.

**Find:** `gpu_worker/worker.py` has notification calls — look for
`[SMS] Sent` and `[EMAIL] Sent` log lines, trace back to the call site.
Probably in the post-pipeline-success path after Step 7 (gallery index
regen) or in a `notify_completion()` helper.

**Fix options** (pick one):

a) **Env-var gate** (recommended — least destructive):
   - Wrap notification calls with `if os.getenv("NOTIFY_ENABLED", "0") == "1":`
   - Default off. Future me can flip it back via `.env` if needed.

b) **Hard remove**: comment out the calls + log a one-liner like
   `print("[notify] disabled per 2026-05-18 design decision")`

c) **Failure-only mode**: keep notification on `Processing failed` path
   (the ❌ emails are still useful — they're the only honest signal in
   the current state), kill the per-success "ready" emails. This is
   actually the smartest interim — failures are rare enough that the
   alert volume becomes manageable.

**Recommend option (c).** Failure emails stay. "Ready" emails die.
Same change for SMS.

Commit message suggestion:
```
ops(notify): kill per-success email/SMS firehose; keep failure alerts only

User reported notification volume is unworkable. Inspection revealed
every recent iPhone upload reports "ready (0 shots)" anyway — emails
were masking a silent pipeline regression (see investigate brief).
Failure path retained as it's the only honest signal until the
0-shots issue is diagnosed.
```

# 2. DIAGNOSE — why are iPhone uploads producing 0 shots?

Critical. Every "ready (0 shots) — N/A Duration: Unknown" is a real
processing failure surfaced as a fake success. User has been getting
these since at least May 13 (probably earlier).

## What we know

- Detection code hasn't changed since 2026-05-11 (verified via git
  log on `gpu_worker/`, `scripts/detect_shots_sequence.py`, etc.)
- Mac iPhone uploader went live around 2026-05-09. **Timing matches.**
- iPhone-prefixed videos (`iphone_*`) flow through the same downstream
  pipeline as legacy `IMG_*` videos
- Failure mode is silent: pipeline reports "ready" (no error), but
  `shots.json` has 0 entries AND `meta.json` shows `duration: unknown`
- Earlier in this session (2026-05-09 session-roundup brief from main)
  there was a P2 item: "Verify iPhone-uploaded videos get shots.json
  produced by the GPU pipeline." That was treated as a 5-min audit.
  **It clearly didn't run, OR it ran and missed this.**

## Investigation order (cheapest first)

1. **Pick one broken video** (e.g. IMG_1260 or whichever iphone_*
   maps to it). Pull its actual `shots.json` and `meta.json` from R2.

   ```bash
   curl -s "https://tennis.playfullife.com/highlights/iphone_<hash>/shots.json" | jq .
   curl -s "https://tennis.playfullife.com/highlights/iphone_<hash>/meta.json" | jq .
   ```

   What's actually in them? Empty array? Object with `duration: null`?
   Some unexpected schema?

2. **Pull the pose JSON for that video**. If pose extraction itself
   produced nothing, that's the upstream cause. If pose has data but
   detection still produced 0, the detector is choking on the input.

3. **Check the worker log on whichever GPU machine processed it.**
   ssh into Andrew-PC or tmassena, grep worker.log for the video_id.
   Look for: did Step 4 (pose extraction) complete? Did Step 5
   (sequence CNN) run? What did it print?

4. **Diff against a known-working video.** Pick one from before the
   regression (look for any video with non-zero shot count). Compare
   meta.json schema, pose JSON length, etc.

5. **Suspect list, in decreasing likelihood:**
   - iPhone MOV preprocessing produces frames the pose extractor doesn't
     handle (codec, resolution, framerate mismatch)
   - `download_source_from_r2()` is fetching the file but it's corrupt
     or zero-length
   - The iPhone upload chunked-multipart-complete step is producing
     a valid-looking R2 object that's missing data
   - Preprocessing step finishes but produces an unwatchable output
     (silently)
   - Some `iphone_*` filename pattern trips a path-handling assumption
     downstream (e.g. shot_classifier expects specific name format)
   - Model hash check is passing but the model is producing all-zero
     predictions on this specific input distribution

## Acceptance for the diagnose phase

- Root cause identified (one or more from the suspect list, or
  something not listed)
- Either FIXED in same PR (if small) or NEW brief filed to design-
  partner for fix scoping (if non-trivial)
- One re-processed iPhone video confirmed working end-to-end (non-zero
  shot count, real duration, gallery rendering)

# 3. SCOPE — dashboard pivot (lower priority, but capture the direction)

User's reframe: emails/SMS are noise. The gallery is already a
dashboard. The dashboard should answer all the questions emails were
trying to answer:

| Question email tried to answer | Dashboard answer |
|---|---|
| "Did my video process?" | Gallery card appears with thumbnail + shot count |
| "How many shots?" | On the gallery card, no email needed |
| "Did it fail?" | New "Recently failed" section on gallery (didn't exist before) |
| "What's in the queue?" | New "Processing now" indicator on gallery (didn't exist before) |

**This is a separate workstream.** Don't try to bundle it with the
0-shots fix. After diagnose+stop landing, draft a "gallery as
operations dashboard" brief covering:

- Processing-status surface on the gallery home (queue + in-flight)
- Recently failed section with retry buttons
- Aggregate counts (today: N processed, M shots, K failed)
- No notification logic anywhere — pull, not push

Estimated effort for that brief: ~1 day to design, ~1-2 days to ship.
Not now.

# What this does NOT do

- Pro-footage acquisition (still active in `pro-footage` worktree —
  unrelated, keep going)
- Detection branch improvements (still active — unrelated)
- Anything related to filmstrip / dyntrack streams
- Kinetic-chain replacement metric (still parked as research)

# Sequencing

1. (~10 min) STOP notifications — Option (c), failure-only mode
2. (~30 min - 2 hours) DIAGNOSE root cause for 0-shots
3. (depends on root cause) FIX or file a design-partner brief
4. (separately, after) Dashboard pivot scoping
