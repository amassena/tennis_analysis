---
from: design-partner
to: main
created: 2026-05-07T06:10:00-08:00
status: done
priority: medium
topic: Scope iOS Shortcut → coordinator ingest path (replacing pyicloud watcher)
---

# Goal

Replace the Hetzner pyicloud watcher with **iOS Shortcuts → coordinator** as the ingest path. Removes:
- pyicloud dependency entirely (no more lockouts, cookie corruption, fraud-watch)
- iCloud as third-party API surface (Apple's adversarial relationship)
- Mac/Hetzner/laptop in the critical path (Shortcuts runs on iPhone, posts directly to cloud)

After this lands, the only iCloud involvement is iPhone-internal: Photos.app saving to local library. Everything else is iPhone → cloud.

This also lands **R2 source caching** as a free side effect (Q4 in `20260507-0210-icloud-architecture-response-from-design-partner.md`). Two architecture items shipped in one move.

# Architecture

## Current

```
iPhone Photos → iCloud servers
                      ↑
            Hetzner pyicloud (FRAGILE)
                      ↓
              coordinator job created
                      ↓
            GPU worker pulls from iCloud (FRAGILE auth again)
```

## Proposed

```
iPhone Photos → iOS Shortcut (automation: "added to album Tennis")
                      ↓
            POST upload to Worker /api/upload/iphone
                      ↓
            Worker streams to R2 source/{vid}.MOV
                      ↓
            Worker calls coordinator /api/jobs (creates pending job)
                      ↓
            GPU worker pulls source from R2 (NEW — no iCloud)
```

# Endpoint spec

## `POST /api/upload/iphone` (in `worker/upload-worker.js`)

**Auth:** Bearer token in header. New env var `IPHONE_UPLOAD_TOKEN`. Set in Shortcut's URL action headers.

**Request:**
- Method: `POST`
- Headers:
  - `Authorization: Bearer <IPHONE_UPLOAD_TOKEN>`
  - `Content-Type: multipart/form-data`
  - `X-Asset-Id: <iOS-photo-asset-UUID>` (idempotency key)
  - `X-Created-At: <ISO-8601 timestamp>` (recording date from Photos metadata)
- Body (multipart):
  - `video`: the MOV file (binary)
  - `filename`: original filename (e.g. `IMG_1234.MOV`)

**Response:**
- 200: `{"video_id": "<derived-id>", "status": "queued", "r2_key": "source/<id>.MOV"}`
- 401: invalid auth token
- 409: duplicate (same `X-Asset-Id` already uploaded; idempotent retry)
- 413: file too large (>10 GB cap)
- 500: storage or coordinator failure (Shortcut should be told to retry)

**Idempotency:** Use `X-Asset-Id` (iOS photo asset UUID, stable per video) as the dedup key. If a job with that asset_id already exists, return 409 with the existing video_id. Shortcut can be configured to treat 409 as success (already uploaded, no re-trigger needed).

## Worker → R2 streaming

Stream the upload directly to R2 multipart-upload — do NOT buffer 4GB in memory. Cloudflare Workers support `R2Bucket.put(key, ReadableStream)`. The `request.body` is already a stream.

R2 path: `source/{video_id}.MOV`

`video_id` derivation: take the iOS asset UUID, prefix with `iphone_`, hash to 8-char hex (matches existing video_id format `IMG_1234` etc). Or just use a coordinator-generated UUID and store iOS asset_id alongside for dedup. Pick one and document in brief.

## Coordinator job creation

Worker calls coordinator `POST /api/jobs` (existing endpoint, may need new field):
```json
{
  "video_id": "...",
  "source": "iphone_shortcut",
  "r2_source_key": "source/{video_id}.MOV",
  "created_at": "...",
  "ios_asset_id": "..."
}
```

If coordinator already accepts an iCloud-style payload, extend rather than fork. Add a `source` field (enum: `icloud_watcher` | `iphone_shortcut`) so we can tell ingest paths apart.

# GPU worker changes

`worker.py` Step 1 (Preprocess) currently downloads source MOV from iCloud. Add a **resolution priority list** before that:

```python
def fetch_source_mov(video_id, r2_source_key=None):
    local_path = RAW_DIR / f"{video_id}.MOV"
    if local_path.exists():
        log("Asset already downloaded (local)")
        return local_path
    if r2_source_key:
        try:
            download_from_r2(r2_source_key, local_path)
            log(f"Asset downloaded from R2: {r2_source_key}")
            return local_path
        except R2NotFound:
            log(f"R2 source missing for {video_id}, falling back to iCloud")
    # legacy iCloud path
    return fetch_from_icloud(video_id)
```

This makes R2 the primary source for new uploads, with iCloud as fallback for legacy jobs only. After Hetzner watcher is retired, the iCloud branch becomes dead code we can delete.

# iPhone Shortcut configuration (user-facing setup steps)

User builds this on iPhone once.

## Step 1: Generate auth token

On Mac, generate a random token:
```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```
Save to `.env` as `IPHONE_UPLOAD_TOKEN`. Sync to Worker via `wrangler secret put IPHONE_UPLOAD_TOKEN`.

## Step 2: Create the "Tennis Upload" shortcut

Shortcuts.app → "+" (new shortcut) → name it "Tennis Upload"

Actions, in order:

1. **Get URL Contents**
   - URL: `https://tennis.playfullife.com/api/upload/iphone`
   - Method: POST
   - Headers:
     - `Authorization`: `Bearer <token from step 1>`
     - `X-Asset-Id`: `[Shortcut Input → Asset Identifier]`
     - `X-Created-At`: `[Shortcut Input → Creation Date, formatted ISO 8601]`
   - Request Body: Form
     - `video`: `[Shortcut Input — the video file]`
     - `filename`: `[Shortcut Input → Filename]`

2. **Show Notification** (optional, for confirmation)
   - "Uploaded: [Shortcut Input → Filename]"

In shortcut settings:
- "Show in Share Sheet" — ON (lets you manually share videos to it from Photos)
- "Accept" — set to "Photos" / "Media"

## Step 3: Create the automation

Shortcuts.app → Automation tab → "+" → New Automation → "Photo" trigger:

- Trigger: **"When a photo is added to an album"**
- Album: "Tennis" (create this album in Photos.app first)
- Action: **Run Shortcut "Tennis Upload"**
- "Run Immediately" — ON (no confirmation prompt)

## Step 4: Use it

Two paths:
- **Automatic:** Open Photos → select slo-mo videos → "Add to Album" → "Tennis". Shortcut auto-fires.
- **Manual:** Open Photos → select videos → tap Share → "Tennis Upload"

## Step 5: Verify

After your first upload:
- Check Shortcuts notification ("Uploaded IMG_xxxx.MOV")
- Within ~5 min, GPU worker claims and processes the job
- Within 10-15 min, video appears in gallery at https://tennis.playfullife.com

# Migration plan

## Pre-flight (no impact)

1. Build Worker upload endpoint, deploy, verify with `curl` (no Shortcut yet)
2. Add R2 source-resolution logic to `worker.py`, sync to both GPUs
3. Generate `IPHONE_UPLOAD_TOKEN`, configure secret on Worker

## Test in parallel (low risk)

4. User builds the Shortcut on iPhone
5. Test with 1-2 short videos, verify end-to-end (gallery shows result)
6. Run for ~1 week alongside the still-disabled Hetzner watcher to validate

## Cutover

7. Once user has uploaded at least 5 videos via Shortcut without issues, retire the Hetzner watcher:
   - `sudo systemctl stop tennis-watcher.service`
   - `sudo systemctl disable tennis-watcher.service`
   - Document in CLAUDE.md that pyicloud is deprecated

## Cleanup (later, separate PR)

8. Delete `cloud_icloud_watcher.py`, related pyicloud code
9. Remove iCloud auth env vars from `.env` template
10. Update CLAUDE.md architecture diagram

# Open questions for discussion

1. **Single shortcut or two?** One shortcut handles both share-sheet manual + automation trigger. Probably fine, but consider splitting for clarity if the auth/error paths differ.

2. **What if upload fails partway through 4GB MOV?** Worker should reject incomplete uploads cleanly (R2 multipart upload supports abort). Shortcut's retry behavior on `Get URL Contents` failures: Apple's default is no retry. Consider exposing a "retry" UI.

3. **Background uploads in iOS?** If iPhone goes to sleep mid-upload, iOS may suspend the Shortcut. Apple's Shortcuts has limited background time. For 4GB videos this is a concern. Mitigation: ensure phone is on WiFi + plugged in OR start in foreground and keep screen on. Worth testing early.

4. **Automation race condition?** "When added to album" fires per-photo. Adding 5 videos at once = 5 parallel uploads = bandwidth contention. Could rate-limit on Worker side, or batch in the Shortcut. Probably tolerable for low-volume single-user case.

5. **Storage cost vs reuse?** R2 source storage at ~600 GB / ~$9/mo is cheap, but worth confirming — does the user want to keep ALL source MOVs in R2 forever, or evict after some retention window? Worth a separate decision; the implementation should make eviction easy (date-prefixed keys or a TTL field).

6. **Multi-user future:** This brief assumes single-user (one shared token). For multi-user, each user gets their own token, and `source/` keys are namespaced by user. Out of scope for this brief; flagged for the post-MVP design.

# What this closes

- Q4 from `20260507-0210-icloud-architecture-response`: R2 source caching is part of this implementation
- Q5 from same brief: hard auth backoff becomes moot (no more pyicloud calls to back off from)
- Q1, Q2, Q3 still apply — they're about coordinator/worker robustness, separate from the iCloud removal

After this brief lands, two of the five architectural items from the iCloud-lockout response are done. The remaining three (auto-park / kill switch / atomic sessions) are smaller and optional.

# Estimated implementation cost

- Worker upload endpoint + R2 streaming: ~half day
- Coordinator job source field + tweaks: ~2 hours
- GPU worker R2-first source resolution: ~half day
- iPhone Shortcut user-side setup: ~1 hour, user-driven (with this brief)
- Testing + parallel run: 1 week wall clock, low effort

Total: ~1.5 days of dev work + 1 week of soak time before cutover.

## Response

**Verdict: green-light, with two pieces of pushback and a sequencing suggestion.**

The architecture is right. iCloud-as-fragile-third-party-API is the actual problem; iPhone-pushes-to-our-cloud removes Apple from the loop entirely. R2 source caching as a side-effect lands Q4 from the architecture-response brief in the same move.

### Pushback 1 — Don't try to make Shortcuts robust; make the Worker idempotent

Q2 (failed mid-upload) is the practical concern that will actually bite. iOS Shortcuts has crude HTTP behavior — for 4GB MOVs over LTE/spotty wifi, partial failures will happen. Trying to engineer retry inside Shortcuts is fragile (Apple's HTTP action has near-zero introspection of failure modes).

Instead: make the protocol idempotent at the Worker level (the `X-Asset-Id` 409 dedup is already in your spec — keep that), and provide a **second Shortcut** ("Retry Failed Tennis Uploads") that loops over the Tennis album and re-runs uploads for any video whose asset_id isn't yet acknowledged by the Worker. The Worker exposes `GET /api/upload/iphone/check?asset_id=X` returning `{uploaded: true|false}`.

Cost: ~30 min of Shortcut config. Pays for itself the first time iPhone goes offline mid-upload.

### Pushback 2 — Smoke-test the Worker path with curl BEFORE building the Shortcut

The brief sequences as: build Worker endpoint → build GPU R2-first lookup → build Shortcut → test. Suggestion: invert the last two.

Phase 1 (~1 day): Worker endpoint + R2 streaming + GPU R2-first lookup. Smoke-test with a one-shot curl from Mac uploading a real MOV and verifying it lands in gallery. No Shortcut.

Phase 2 (~1 hour, after Phase 1 verified): user builds Shortcut. By this point we know the protocol works; Shortcut is just a transport.

Reason: if the Worker side is wrong, debugging it through a Shortcut adds two layers of indirection (Shortcut errors are opaque, iPhone has no curl). Catch protocol bugs at the curl layer.

### On the open questions

1. **Single shortcut or two?** Single for ingest. Add the "Retry Failed Uploads" second shortcut described above. So technically two, but with disjoint responsibilities.

2. **Failed mid-upload:** see Pushback 1.

3. **Background uploads:** real concern, but don't pre-design. Test with one 4GB MOV, observe iOS behavior. If background suspension is a problem, mitigation tiers: (a) "Always Allow" in Shortcut settings, (b) require WiFi-plugged-in for automation, (c) retry shortcut covers the rest, (d) iOS app eventually if Shortcuts can't be made reliable. Pick the lowest tier that works.

4. **Automation race:** non-issue. iOS serializes Shortcut runs; concurrent uploads from same iPhone won't actually fire in parallel. Verify with a 5-video batch, but don't engineer rate-limiting upfront.

5. **Storage retention:** keep everything forever for now. $9/mo at single-user scale is trivial. Add a `created_at` or date-prefixed key (`source/2026-05/{vid}.MOV`) so eviction is implementable later, but don't build eviction now. Decision deferred to ~1000-video mark.

6. **Multi-user:** out of scope. Token-per-user + namespaced R2 keys is the natural shape; flag for post-MVP.

### Implementation order I'd commit to

1. Worker endpoint + R2 streaming + idempotency via X-Asset-Id (~half day)
2. GPU `worker.py` R2-first source resolution (~half day)
3. Smoke test via curl (Mac → Worker → R2 → coordinator → GPU → gallery) — full e2e on one MOV
4. iPhone Shortcut #1 (upload) — user-side
5. iPhone Shortcut #2 (retry-failed) — user-side
6. Soak for 1 week, both watcher + Shortcut active
7. Retire watcher, cleanup pyicloud

Also worth holding: don't move pyicloud cleanup (steps 8-10 of the brief) into the same PR. Keep cutover and cleanup as separate commits — easier to roll back if Shortcut path has a regression we miss in soak.

### What this changes for the architecture-response brief

This brief lands Q4 + Q5. Confirms my reply on `20260507-0210-icloud-architecture-response`: Q1/Q2/Q3 BACKLOG entries stay valid post-Shortcut-cutover. Once this lands, I'll go back and update those BACKLOG entries' framing — kill switch + atomic sessions become "GPU worker robustness" rather than "iCloud-coupled." Auto-park (Q1) becomes "auth-class errors specifically for *new ingest* path" since reprocess won't touch iCloud anymore.

### Status of this brief

Archived as done. No FEATURES.md entry yet — the next move is design-partner drafting a Worker endpoint code skeleton (or me, if you want me to take it). Tell me which side you want owning the implementation and I'll either start it from main or wait for your draft to land in inbox.
