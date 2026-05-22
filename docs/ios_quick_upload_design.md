# iOS Quick Upload — Design

**Status:** plan-mode draft v3 (2026-05-21)
**Branch:** `feature/ios-live/quick-upload`
**Worktree:** `~/tennis_worktrees/ios-upload`
**Brief:** `.handoffs/inbox/20260521-0100-ios-direct-upload-design-from-design-partner.md`

## Goal (after user clarification)

Build a **TestFlight- and App-Store-shippable** iOS app whose only
job is: record or pick a video → upload directly to our server.
Designed to be installable by friends and colleagues, not just the
primary user. **No live coaching, no on-device pose, no inline
review.** Those exist in the scaffold but are out of scope here.

After upload, the user can browse the gallery in a secondary tab
(WebView to the existing `tennis.playfullife.com`). The primary
surface is the upload flow itself.

## User-confirmed decisions

| # | Decision | User choice (2026-05-21) |
|---|---|---|
| D1 | **Auth** | **Sign in with Apple.** Server mints a per-user JWT after verifying Apple identity token. JWT stored in iOS Keychain. |
| D2 | **Gallery scope** | **Shared gallery for v1.** All uploads land in the existing `tennis.playfullife.com`. Every upload is stamped with `uploaded_by: <user_hash>` so we can migrate to per-user later without backfill ambiguity. |
| D3 | **WebView** | **Keep as secondary tab.** Tab bar: "Upload" (primary) + "Gallery" (WebView to `tennis.playfullife.com`). |
| D4 | **Scaffold scope** | **Strip to just upload.** Drop pose overlay, shot counter, `SessionReviewView`, inline grading. `CameraManager` retained in recording-only mode. |

---

## Audit of existing assets (from the worktree as of 2026-05-21)

The brief assumed `ios/CourtIQ/` already had a working R2 uploader.
**It does not.** Findings, in priority order:

### 1. `R2Uploader.swift` does not upload to R2

Despite the name, `Models/R2Uploader.swift` (47 lines) only calls
`PHAssetChangeRequest.creationRequestForAssetFromVideo` — it **saves
the recorded clip to the Photos library** and lets the existing
iCloud→Mac→Worker path take over. Comment on line 18 admits it. No
HTTP client, no chunked upload, no Bearer auth, no Worker calls.
We rewrite this file (preserving the type name as a thin facade
over the real `UploadManager`).

### 2. The app does not currently compile

`CourtIQ.xcodeproj/project.pbxproj` lists `TennisSession.swift` and
`SessionRecording.swift` under `CourtIQ/Models/`, and
`Views/ContentView.swift` + `Views/SessionReviewView.swift`
reference types `TennisSession`, `SessionRecording`, `RecordedShot`,
`ShotReplayView`, `GalleryView`. **None of those .swift files exist
on disk.** The repo also has no `Info.plist`.

Per D4 we **replace `ContentView` outright** with a clean tab-bar
root, delete `SessionReviewView.swift` from the target, and drop
`TennisSession`/`SessionRecording` references entirely. No stubs
needed — we just don't compile what we don't ship. `Pose/`, `Shot/`,
`Overlay/`, `CameraPreviewView` remain on disk (live-coaching
stream may resurrect them later) but are removed from the Xcode
target membership for v1.

### 3. `CameraManager` is reusable in recording-only mode

`Camera/CameraManager.swift` already configures `AVCaptureSession`
with 240 fps when available, `AVCaptureMovieFileOutput`, and a
`startRecording(to:)` / `stopRecording()` pair. The Vision pose
work runs on every 8th frame inside the same delegate — we add a
`configure(includePoseProcessing: Bool = false)` flag so quick-upload
does not pay that cost.

### 4. The Worker chunked endpoints are ready, auth needs a JWT path

`worker/upload-worker.js:601-771` implements:

```
POST /api/upload/iphone/init      { asset_id, filename, created_at? }
PUT  /api/upload/iphone/<upid>/<n>   raw chunk bytes (5-100 MB)
POST /api/upload/iphone/<upid>/complete  { parts: [{partNumber, etag}] }
GET  /api/upload/iphone/check?asset_id=…
```

All currently authed with `Authorization: Bearer
<IPHONE_UPLOAD_TOKEN>` — a single shared secret. **That model
cannot ship to multiple users.** We add a parallel JWT auth path
(see §S1) and migrate the iPhone routes to accept either: shared
token (Mac uploader keeps working) OR a user-scoped JWT (the iOS
app uses this). **Both auth paths still write to the existing flat
`source/<vid>` path** — the JWT path additionally stamps
`uploaded_by: <user_hash>` into the marker JSON.

### 5. The Mac uploader is the reference client

`scripts/upload_tennis_album.py:182-282` uses 50 MB chunks, single-
shot below that. We match. Worker has a 100 MB-per-part hard cap
(CF edge body limit) — 50 MB leaves headroom.

---

## Architecture decisions

### A1. App shape — two-tab native shell

```
TabView
 ├── Tab 1: Upload  (default, opens on launch)
 │     ├─ "Your uploads" list (in-progress + recent)
 │     └─ Big "+" → choice: Record new / Pick from Photos
 │
 └── Tab 2: Gallery  (WKWebView → tennis.playfullife.com)
```

Settings is reachable from the upload-tab nav bar (gear icon):
sign-out, WiFi-only toggle, account deletion, "About / privacy".

### A2. Authentication — Sign in with Apple → server JWT

**Client flow:**
1. First launch: full-screen "Sign in with Apple" gate. Friend taps,
   gets Apple's native auth sheet, returns an `ASAuthorizationAppleIDCredential`.
2. App `POST`s `credential.identityToken` (a JWT signed by Apple) to
   `https://tennis.playfullife.com/api/auth/apple`.
3. Worker verifies the token against Apple's public JWKs
   (`https://appleid.apple.com/auth/keys`), pulls `sub` (the
   stable user id), upserts a `users/<sub>.json` record in R2, and
   mints **our** JWT (HS256 signed by a Worker secret, 30-day TTL).
4. iOS stashes our JWT in Keychain
   (`kSecAttrAccessibleAfterFirstUnlock`).
5. All subsequent requests carry
   `Authorization: Bearer <our_jwt>`.

**JWT claims (ours, not Apple's):**
```json
{
  "sub": "u_<8-hex-sha-of-apple-sub>",
  "apple_sub": "<full apple sub>",
  "iat": 1716240000,
  "exp": 1718832000,
  "scope": "upload"
}
```

We use the short `u_<hex>` id as the stable "who uploaded this"
attribution — never expose `apple_sub` outside the Worker.

**Token refresh:** Apple's identity token is short-lived (10 min)
and one-shot — we don't re-use it. Our 30-day JWT is the only
session credential. On 401, the app routes back to sign-in.

### A3. R2 layout — flat, with uploader attribution

```
source/<video_id>.{mov,mp4}                ← raw upload (unchanged from today)
uploads/<video_id>.json                    ← coordinator marker (NEW field: uploaded_by)
processed/<video_id>/timeline.mp4         ← pipeline output (unchanged)
processed/<video_id>/meta.json            ← (NEW field: uploaded_by)
processed/<video_id>/coaching.json
users/<apple_sub>.json                     ← {user_hash, created_at, last_seen, video_count}
```

`uploaded_by: <user_hash>` is the **only** new field across the
pipeline. It rides through `uploads/<vid>.json` → coordinator job
row → `processed/<vid>/meta.json` so the gallery and any future
per-user view can attribute correctly. The Mac uploader leaves
this field absent (or sets `uploaded_by: "andrew"` as a sentinel)
— Andrew's existing 1252-shot corpus is unaffected.

### A4. Chunk upload protocol — unchanged from §1.4 audit

50 MB chunks, 3 concurrent parts, `URLSession.background`, per-part
state persisted to Application Support, resume on launch. Bearer
header is now our JWT, not the shared `IPHONE_UPLOAD_TOKEN`.

For new in-app recordings (no PHAsset yet), `asset_id` is
`<user_hash>_<UUID>_<ISO timestamp>` so the SHA-256 fold to
`iphone_<hex>` stays unique across users.

### A5. App Review compliance checklist

Hard requirements Apple will look for:

| Requirement | Where in this design |
|---|---|
| Privacy policy URL | Add `tennis.playfullife.com/privacy` (static page, ~200 words). Linked from Settings + App Store listing. |
| In-app account deletion (mandatory since iOS 16) | Settings → "Delete account" → confirm → `DELETE /api/account` → Worker tombstones user + deletes all videos with `uploaded_by == this user_hash`. |
| `NSCameraUsageDescription` | "Record tennis videos to upload." |
| `NSMicrophoneUsageDescription` | "Capture audio with your videos." |
| `NSPhotoLibraryUsageDescription` | Required because PHPicker uses it. "Pick existing tennis videos to upload." |
| Sign in with Apple **as the only login** | No other login options offered → we don't need to add any other ID-provider buttons for parity. |
| User-generated content concerns (1.2) | Shared gallery has UGC visibility. Mitigations: (a) only signed-in users see any content (no public unauthenticated access), (b) Settings → Report button per video (deferred, but flagged for Apple), (c) Andrew is the implicit moderator + can revoke a friend's `user_hash` server-side. |
| Reviewer test credentials | Submission notes: "Sign in with Apple works with any Apple ID; no app-specific credential needed." |
| Minimum functionality | Native record + native upload + native progress UI = sufficient. WebView is one tab of two, not the whole app. |
| No private API use | None planned. PHPicker, AVFoundation, URLSession, AuthenticationServices, Keychain — all public. |
| Background modes justified | `UIBackgroundModes`: `["fetch"]` for upload completion only. We avoid `processing` (BGTaskScheduler) which Apple flags more carefully. |

### A6. Onboarding flow

```
First launch
  ↓
[Welcome screen] ─── "Sign in with Apple" button
  ↓
Apple's native sheet
  ↓
POST /api/auth/apple → JWT
  ↓
[Onboarding step 1] "What to expect" — 2-line copy + screenshot of upload tab
  ↓
[Permission asks] Camera + Mic + Photos (one at a time, on-tap, contextual copy)
  ↓
Land on Upload tab (empty state: "No uploads yet. Tap + to start.")
```

Permissions are **lazy**: we don't ask for Camera until they tap
"Record", don't ask for Photos until they tap "Choose existing".
Avoids the "denied all permissions at first launch" failure mode
Apple discourages.

---

## Server-side changes (Worker only — no Hetzner/GPU changes)

These are not iOS work but they're required for the iOS app to
function. **PRs for these land in `worker/`.** Critically: the
Hetzner poller, GPU worker, and gallery regen are **not** touched
in v1 — they just see new uploads at the existing flat path.

### S1. `POST /api/auth/apple`
- Body: `{ identity_token: "<apple JWT>", nonce?: "..." }`
- Verifies token signature against Apple JWKs (cache for 24h).
- Validates `aud == bundle id`, `iss == https://appleid.apple.com`.
- Computes `user_hash = sha256(apple_sub).slice(0,8)`.
- Upserts `users/<apple_sub>.json`.
- Mints HS256 JWT (secret: new Worker var `JWT_SIGNING_SECRET`).
- Returns `{ jwt, user_hash, gallery_url: "https://tennis.playfullife.com" }`.

### S2. Make iPhone routes JWT-aware
- `validateAuth(request, env)` returns `{ kind: 'shared' | 'user', user_hash? }`.
- Existing `IPHONE_UPLOAD_TOKEN` keeps working (Mac uploader, unchanged).
- JWT auth → still writes to `source/<vid>.{mov,mp4}` (same flat
  path as today), but the marker `uploads/<vid>.json` includes
  `uploaded_by: <user_hash>`.
- All four iPhone endpoints (`init`, `part`, `complete`, `check`)
  gain this branch.

### S3. `GET /api/me`
- Returns `{ user_hash, video_count, gallery_url, created_at }`.
- iOS app calls this on launch to confirm session is still valid
  and to count "your uploads".
- `video_count` is computed by scanning `uploads/*.json` for
  matching `uploaded_by` — cached in `users/<sub>.json` and
  incremented at `/complete` time to avoid full-bucket scans.

### S4. `DELETE /api/account`
- Tombstones `users/<apple_sub>.json` with `deleted_at`.
- Async-queues delete of all videos with `uploaded_by ==
  this_user_hash`:
  - For each match in `uploads/*.json`, delete:
    - `source/<vid>.{mov,mp4}`
    - `uploads/<vid>.json`
    - `processed/<vid>/*` (timeline, rally, bytype, meta,
      coaching, thumbs)
  - Trigger gallery regen so the deleted videos drop from the
    public `index.html`.
- Returns `204` immediately; deletion completes within ~1-2 min for
  a typical friend's footprint.

### S5. Gallery attribution (minor)
- `scripts/update_r2_index.py` reads `uploaded_by` from each
  `processed/<vid>/meta.json` and displays a small uploader badge
  per video card ("you" / "andrew" / "alex"). Optional in v1; we
  can ship without it and add later.

---

## File / module layout under `ios/CourtIQ/CourtIQ/`

```
CourtIQApp.swift                       (rewrite: BG-session wiring, auth gate)
Camera/
  CameraManager.swift                  (small refactor: includePoseProcessing flag)
  CameraPreviewView.swift              (keep)
Models/
  R2Uploader.swift                     (REWRITE — chunked URLSession + JWT)
Upload/                                ← NEW
  UploadManager.swift
  UploadState.swift
  UploadResumer.swift
  StatusPoller.swift
Auth/                                  ← NEW
  AuthCoordinator.swift                (Sign in with Apple flow)
  TokenStore.swift                     (Keychain wrapper for JWT)
  APIClient.swift                      (Auth-aware HTTP helper)
Views/
  RootView.swift                       ← NEW (tab bar root, replaces ContentView)
  AuthGateView.swift                   ← NEW (Sign in with Apple screen)
  UploadTabView.swift                  ← NEW (your-uploads list + + button)
  UploadComposerSheet.swift            ← NEW (Record/Pick choice)
  RecordView.swift                     ← NEW (stripped: record button only)
  PickerView.swift                     ← NEW (PHPicker wrapper)
  UploadRowView.swift                  ← NEW (per-upload progress card)
  GalleryTabView.swift                 ← NEW (WKWebView to tennis.playfullife.com)
  SettingsView.swift                   ← NEW (sign-out, WiFi-only, delete account)
  WebViewWrapper.swift                 (keep — used by GalleryTabView)
Resources/
  Info.plist                           ← NEW
  PrivacyInfo.xcprivacy                ← NEW (Apple's privacy manifest)

DELETED FROM TARGET (files remain on disk for live-coaching stream):
  Views/ContentView.swift              (replaced by RootView)
  Views/SessionReviewView.swift
  Pose/PoseProcessor.swift
  Shot/ShotDetector.swift
  Overlay/PoseOverlayView.swift
```

`Info.plist` must declare:
- `NSCameraUsageDescription`, `NSMicrophoneUsageDescription`,
  `NSPhotoLibraryUsageDescription`, `NSPhotoLibraryAddUsageDescription`
- `UIBackgroundModes`: `["fetch"]`
- `NSAppTransportSecurity`: default (HTTPS only)
- App entitlement: **`com.apple.developer.applesignin`** (added in
  `CourtIQ.entitlements`, currently absent)

`PrivacyInfo.xcprivacy` enumerates data-collection types Apple
requires us to declare (camera input → uploaded media; coarse
device info via Apple ID; no third-party analytics SDKs).

---

## Component diagram

```
┌──────────────────────────────────────────────────────────┐
│ CourtIQApp                                               │
│   └─ RootView (TabView, gated by AuthCoordinator)        │
│       ├─ AuthGateView (if no JWT)                        │
│       ├─ Tab 1: UploadTabView                            │
│       │    ├─ UploadComposerSheet                        │
│       │    │    ├─ RecordView (light camera screen)      │
│       │    │    └─ PickerView (PHPickerVC)               │
│       │    └─ UploadRowView × N                          │
│       └─ Tab 2: GalleryTabView (WKWebView)               │
│       (Settings reachable via gear icon)                 │
└──────────────────────────────────────────────────────────┘
            │                       │
            │                       │
            ▼                       ▼
    AuthCoordinator           UploadManager
    + APIClient               + UploadState (Codable, persisted)
    + TokenStore (Keychain)   + UploadResumer (launch-time scan)
                              + StatusPoller (5/10/20/40/80/120s)
            │                       │
            └───────── JWT ─────────┤
                                    ▼
          ┌──────────────────────────────────────────────┐
          │ Cloudflare Worker (extended)                 │
          │   POST /api/auth/apple              (NEW)    │
          │   GET  /api/me                      (NEW)    │
          │   DELETE /api/account               (NEW)    │
          │   POST /api/upload/iphone/init        (+JWT) │
          │   PUT  /api/upload/iphone/<upid>/<n>  (+JWT) │
          │   POST /api/upload/iphone/<upid>/complete    │
          │   GET  /api/upload/iphone/check?asset_id=…   │
          │   GET  /api/status/<vid>                     │
          │   GET  /                                     │
          │   (existing routes unchanged)                │
          └──────────────────────────────────────────────┘
                                    │
                                    ▼
          Hetzner poller → coordinator → GPU
          (UNCHANGED — flat-path-only, treats friend uploads
           identically to Mac uploads)
```

## Sequence — happy path (first-time user, pick a video, upload)

```
User       AuthGate    Apple    Worker    UploadComposer  PHPicker   UploadManager
 │           │           │         │           │             │            │
 │ tap "Sign in with Apple"        │           │             │            │
 │──────────►│           │         │           │             │            │
 │           │── auth ──►│         │           │             │            │
 │           │◄─ identityToken ────│           │             │            │
 │           │── POST /api/auth/apple ────────►│             │            │
 │           │◄─ {jwt, user_hash, gallery_url} │             │            │
 │  → land on Upload tab           │           │             │            │
 │── tap +  ─────────────────────────────────► │             │            │
 │── tap Pick existing  ──────────────────────►│             │            │
 │                                              │── present ►│            │
 │── select video ────────────────────────────────────────►  │            │
 │                                              │◄ URL + assetId          │
 │                                              │── enqueue ─────────────►│
 │                                              │            │── POST init (JWT) ─►│
 │                                              │            │◄─ {upload_id}       │
 │                                              │            │── PUT part 1..N ───►│
 │                                              │            │── POST complete ───►│
 │                                              │            │◄─ {queued}          │
 │                                              │            │── poll /status/<vid>►│
 │                                              │            │◄─ {completed}       │
 │ ← banner "✅ Ready"                          │            │                     │
 │── tap Gallery tab → WKWebView loads tennis.playfullife.com#<vid> ──────────────►│
```

## Crash recovery

`UploadResumer.scan()` at launch reads `upload_*.json` from
Application Support, checks `GET /api/upload/iphone/check` to skip
ones that completed while we were dead, and re-enqueues the rest.
JWT in the file is re-loaded from Keychain at scan time (do not
persist secrets in the state JSON).

---

## Implementation plan — PRs in order

PR 0 unblocks the build, PRs 1-4 ship the bare-minimum app, PRs 5-7
make it TestFlight-ready, PR 8 is App Store submission. Server-side
work (mostly in PR 1, small additions in PR 6) is bounded to
`worker/upload-worker.js`.

### PR 0 — Project cleanup [BLOCKER, no functional change]
- Add `Info.plist`, `CourtIQ.entitlements`, `PrivacyInfo.xcprivacy`.
- Remove `ContentView.swift`, `SessionReviewView.swift`, `Pose/`,
  `Shot/`, `Overlay/` from target membership in `project.pbxproj`.
  (Files stay on disk for the live-coaching stream.)
- Add new groups: `Auth/`, `Upload/`, `Views/`.
- Add `RootView.swift` as a "Hello, world" placeholder so the app
  launches on the simulator.
- **Ship gate**: `xcodebuild` clean build success.

### PR 1 — Sign in with Apple end-to-end
- `Auth/TokenStore.swift` (Keychain wrapper, get/set/clear).
- `Auth/AuthCoordinator.swift` — runs Apple's `ASAuthorizationController`,
  exchanges identity token with Worker.
- `Auth/APIClient.swift` — auth-aware HTTP helper that injects
  `Authorization: Bearer <jwt>` on every request.
- `Views/AuthGateView.swift` — full-screen welcome + Sign in button.
- **Server side (in this PR)**: `worker/upload-worker.js` adds
  `POST /api/auth/apple`, `JWT_SIGNING_SECRET` Worker secret, Apple
  JWK fetch+cache, JWT mint, `users/<sub>.json` upsert.
  Adds `GET /api/me`.
- **Ship gate**: install on a phone with my Apple ID, sign in, see
  `user_hash` round-trip. Verify `users/<sub>.json` written in R2.

### PR 2 — Tab shell + PHPicker upload
- `Views/RootView.swift` — TabView, "Upload" + "Gallery".
- `Views/UploadTabView.swift` — list view + "+" toolbar button.
- `Views/UploadComposerSheet.swift` — choice screen.
- `Views/PickerView.swift` — PHPicker wrapper.
- `Upload/UploadManager.swift`, `UploadState.swift` — chunked
  uploader (50 MB parts, 3 concurrent), `URLSession.default` for
  this PR (background URLSession lands in PR 4).
- `Models/R2Uploader.swift` rewritten as a `@MainActor` facade
  exposing the `@Published` state UploadRowView needs.
- **Worker side**: extend iPhone upload routes to accept JWT in
  addition to shared token. JWT path stamps `uploaded_by:
  <user_hash>` into the marker JSON; flat `source/<vid>` path is
  unchanged.
- `Views/UploadRowView.swift` — progress bar, status text.
- **Ship gate**: pick a 200 MB clip, see it land at
  `source/iphone_xxxxxxxx.mov`, marker has `uploaded_by`,
  status row turns green.

### PR 3 — Record-new flow
- `Views/RecordView.swift` — minimal camera screen. Big red
  record button, tap to stop. Show recorded duration. On stop,
  routes to a confirm screen: "Upload now / Re-record / Discard".
- `CameraManager` refactor: `configure(includePoseProcessing:
  false)` path bypasses Vision delegate; recording-only.
- New recordings get an `asset_id` of
  `"<user_hash>_<UUID>_<ISO timestamp>"`.
- Optional toggle in Settings: "Save to Photos after upload".
- **Ship gate**: record → stop → confirm → upload → see in
  gallery once pipeline finishes.

### PR 4 — Background URLSession + crash recovery
- Move `UploadManager` to `URLSessionConfiguration.background`.
- Wire `application(_:handleEventsForBackgroundURLSession:completionHandler:)`
  via `UIApplicationDelegateAdaptor`.
- `Upload/UploadResumer.swift` — scans Application Support at
  launch, re-enqueues incomplete uploads. Calls
  `GET /api/upload/iphone/check` to skip uploads that completed
  while we were terminated.
- **Ship gate**: kill the app mid-upload, relaunch, upload resumes
  from last completed part on real device.

### PR 5 — Gallery tab + status poller
- `Views/GalleryTabView.swift` — WKWebView to
  `tennis.playfullife.com`. Uses existing `WebViewWrapper.swift`.
  No URL parameters — the gallery is public-by-default for v1.
- `Upload/StatusPoller.swift` — polls `/api/status/<vid>` at
  5/10/20/40/80/120s; on `completed`, fires in-app banner with
  "View in gallery" button → switches to Gallery tab, navigates to
  `#<vid>` (the gallery already supports the anchor).
- **Ship gate**: full record → upload → process → ready → tap →
  Gallery tab scrolls to the new video.

### PR 6 — Settings + account deletion + privacy
- `Views/SettingsView.swift` — sign-out (clears Keychain + drops
  back to AuthGateView), WiFi-only toggle, "Delete my account"
  with double-confirm, About / Privacy link, support email.
- **Worker side**: `DELETE /api/account` → tombstones
  `users/<sub>.json`, iterates `uploads/*.json` for matching
  `uploaded_by`, deletes each `source/<vid>` + `uploads/<vid>` +
  `processed/<vid>/*`. Triggers gallery regen at the end.
- **Static content**: ship `tennis.playfullife.com/privacy`
  (markdown rendered to HTML, ~200 words, plain-language
  description of what we collect + how to delete).
- **Ship gate**: delete account, verify all that user's videos
  removed within 60s, re-sign-in works as a brand-new user (new
  `user_hash` if Apple gives new `sub`, same if it's the same
  Apple ID — which is fine).

### PR 7 — Polish + TestFlight beta
- Empty-state illustrations on Upload tab.
- Loading + error states throughout (no JWT → AuthGate; expired
  JWT → re-auth; offline → "Will retry when online").
- App Store screenshots (3 per device size, in `marketing/`).
- TestFlight build, invite friends to test.
- **Ship gate**: 3 friends successfully install, sign in, upload a
  video, see it process.

### PR 8 — App Store submission
- App Store Connect metadata: app name, description, keywords,
  support URL, privacy URL, age rating.
- App Privacy nutrition label.
- Reviewer notes: "App is a tennis video uploader for me and my
  friends. Sign in with Apple works with any Apple ID. After
  signing in, tap + to record or pick a video. Uploads complete
  in the background; processed clips appear in the Gallery tab.
  Gallery content is shared among invited users only."
- Bundle ID, signing identity, provisioning profile.
- Submit for review.
- **Ship gate**: app is in the App Store (or in Pending Developer
  Release).

---

## Risks / open items

| Risk | Mitigation |
|---|---|
| **Friends see Andrew's existing videos and each other's.** That's the design choice — shared gallery. | Worth a heads-up to each invitee. Per-video delete already exists in the Worker (`POST /api/video/:vid/delete`) for surgical removal. |
| Apple rejects on UGC visibility / lack of moderation (1.2). | Sign in with Apple gates all access (no anonymous viewing). Account deletion removes that user's content. Andrew can revoke a friend's `user_hash` server-side (1-line in `users/<sub>.json`). "Report" button per video is a fast follow if Apple insists. |
| Apple rejects on "minimum functionality" (4.2). | Native upload flow + native progress + native auth = defensible. WebView is one tab of two. If rejected, we move Gallery behind a button (SFSafariViewController per-video) and re-submit. |
| Apple rejects on missing Sign in with Apple (4.8). | We **only** offer Sign in with Apple. No third-party auth at all. Guideline is satisfied. |
| Sign in with Apple identity token verification is fiddly (JWKS rotation, algorithm checks). | Add a Worker unit test that signs a fake token with a fake key and verifies the failure path. |
| Background URLSession on iOS 17/18 may throttle large uploads. | PR 4 includes manual on-device test. Worst case we surface a foreground-keepalive screen for the first upload until it completes. |
| Multi-tenant abuse: a friend uploads 50 GB of non-tennis content. | Rate-limit at the Worker level (per `user_hash`, 5 uploads/hr, 10 GB/day). Out of scope for v1 but flagged before TestFlight invite list grows. |
| Account deletion = N HTTP deletes against R2 (no prefix-delete primitive). | Worker iterator pages 1000 keys, deletes, repeats. Acceptable for v1 scale. Cron-style cleanup task can finish stragglers. |
| `tennis.playfullife.com/privacy` doesn't exist yet. | PR 6 ships it as static HTML served via the Worker from R2. |
| `IPHONE_UPLOAD_TOKEN` is the same secret as the Mac uploader uses today. If revealed via APK, attackers could spam the legacy flat path. | The Mac uploader path stays Bearer-only and is **not** exposed in the iOS app. iOS uses JWT. We **never** ship `IPHONE_UPLOAD_TOKEN` in the iOS bundle. |
| Apple `sub` is per-team — if we ever change the bundle ID or team, users are forced to re-sign-in but their server-side `users/<sub>.json` no longer matches. | Acceptable for v1 (we are not changing teams). |

## Future migration to per-user galleries

If/when shared visibility becomes a problem, we migrate cleanly
because every upload from this app already carries `uploaded_by`:

1. Add the `/u/<hash>` Worker route + per-user gallery regen path
   (deferred work from this design's v2 draft).
2. **One-time backfill script**: iterate
   `processed/*/meta.json`, read `uploaded_by`, R2 server-side-copy
   each user's videos under `processed/u_<hash>/`. R2 native copy
   is cheap (no bandwidth charge); a few minutes of script.
3. Ship a new iOS build that flips `GalleryTabView` from
   `tennis.playfullife.com` to `tennis.playfullife.com/u/<hash>`.
4. (Optional) Keep the shared gallery alive at the root as an
   opt-in "all friends" view, or retire it.

No code from v1 needs to be rewritten — only added to. The only
risk in waiting too long is **abuse cleanup**: if a friend uploads
a year of objectionable content into the shared gallery before
migration, that content needs sorting before per-user splits make
sense. Stays manageable as long as the friend group is small.

## FEATURES.md coordination

Add to active features:
```
| Quick Upload iOS app | feature/ios-live/quick-upload | ~/tennis_worktrees/ios-upload | ios/CourtIQ/**, worker/upload-worker.js | active — design v3 landed 2026-05-21 (shared gallery, Sign in with Apple, App-Store target) |
```

Add to file-conflict map:
```
| ios/CourtIQ/** | quick-upload |
| worker/upload-worker.js | quick-upload (auth additions) |
```

`worker/upload-worker.js` is the only shared file we touch — coordinate
with main before PR 1 and PR 6 land. No Hetzner / GPU / index-regen
changes in v1.

## Out of scope (deferred)

- On-device live coaching (the pose+shot+overlay scaffold) — its own
  stream.
- Multi-select picker / batch upload.
- Per-user galleries (see future-migration section above).
- Friend invitations from within the app (friends get a TestFlight
  link or App Store URL out-of-band).
- Per-video Report button (fast follow if Apple insists).
- iPad-specific layouts (use iPhone layouts at iPad sizes).
- Notifications (push or local). Status appears in-app only.
- Pre-upload trim (PHPicker offers trim already; in-app trim is
  v2 work).
- Gallery uploader-attribution badges (S5) — optional in v1.
