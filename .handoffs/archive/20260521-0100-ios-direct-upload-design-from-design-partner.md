---
from: design-partner
to: ios-live
created: 2026-05-21T01:00:00-08:00
status: pending
priority: high
topic: iOS "Quick Upload" app — design + architecture pass (plan-mode first, code after approval)
---

# Goal

A minimal iOS app that lets the user **record a video OR pick an
existing one from Photos**, and **upload it directly to our servers
over WiFi** with no iCloud Photos roundtrip. Optimize for upload
speed and minimal taps.

This **replaces** (or supplements) the current Mac PhotoKit uploader
path for sessions where the user wants the video on the server
immediately after recording, rather than waiting for iCloud Photos
sync → Mac → upload.

# Existing assets (use these, don't rebuild)

The repo already has an iOS app at `ios/CourtIQ/`:

```
ios/CourtIQ/
  CourtIQ/
    CourtIQApp.swift          ← app entry
    Camera/CameraManager.swift, CameraPreviewView.swift
    Models/R2Uploader.swift   ← R2 upload code already exists!
    Models/SessionRecording.swift
    Models/TennisSession.swift
    Pose/PoseProcessor.swift  ← on-device MediaPipe
    Shot/ShotDetector.swift   ← on-device detector
    Overlay/PoseOverlayView.swift
    Views/ContentView.swift, SessionReviewView.swift, WebViewWrapper.swift
```

**First action: audit what's already wired.** Don't write new code
until you've confirmed which existing pieces do/don't do what we need.
Specifically:

- `R2Uploader.swift` — what auth model? Does it chunk? Does it use the
  existing Worker `/api/upload/iphone/{init,part,complete}` endpoints
  (the chunked path we built for the Mac uploader)?
- `CameraManager.swift` — can it record video to a file, or only
  preview?
- `ContentView.swift` — what's the current entry-flow UX?

Also reference the Worker side: `worker/upload-worker.js`. The
chunked-upload endpoints there are the production path; reuse them.
Bearer auth via `IPHONE_UPLOAD_TOKEN` env var.

# Requirements

## Functional

1. **Record a new video in the app.** Tap "Record" → camera → tap stop
   → preview/trim → tap "Upload" → upload starts immediately.
2. **Pick from Photos library.** Tap "Choose existing" → PHPicker
   (privacy-safe PhotoKit picker, no full library access needed) →
   pick one video → tap "Upload."
3. **Upload directly to our Worker.** Use the existing
   `/api/upload/iphone/{init,part,complete}` chunked endpoints. Bearer
   auth via the existing `IPHONE_UPLOAD_TOKEN`.
4. **Surface upload progress in-app.** % complete, estimated time
   remaining, chunk N of M.
5. **Confirm server registration.** After upload completes, poll the
   Worker to confirm a job was created in the coordinator (R2 marker
   → `coordinator_registered`).
6. **Background-uploads when possible.** iOS BGTaskScheduler /
   URLSessionConfiguration.background allows uploads to continue if
   the user leaves the app. Test on real device.

## Non-functional

- **Fast on WiFi.** Aim for upload throughput within 70% of raw
  user-WiFi bandwidth. Implementation: large multipart chunks
  (existing Worker caps at 50 MB per part), parallel uploads (2-4
  concurrent parts), HTTP/2 + keepalive.
- **No iCloud Photos roundtrip.** Video stays on-device until uploaded
  to our R2. Optional: after successful upload, delete local copy or
  prompt user.
- **Minimal taps.** Record→stop→upload should be ≤3 taps for the
  common case. Picker flow ≤4 taps.
- **WiFi-only enforced by default.** Cellular upload of 1-4 GB MOV is
  user-hostile. Setting toggleable.

# Architecture decisions to nail (in plan mode)

Make these explicit BEFORE writing code:

## 1. Existing app or new app?

`ios/CourtIQ` exists but its current scope (live AR coaching with
on-device pose+detection) is larger than this brief calls for. Decide:

- **Option A**: Extend CourtIQ with a "Quick Upload" tab. Pros:
  reuses R2Uploader, Camera code. Cons: bigger surface to maintain.
- **Option B**: Build a separate `QuickUpload.app` (smaller, focused).
  Pros: clean, fast to ship. Cons: duplicates code.
- **Option C**: Split CourtIQ into a SwiftPM workspace with a shared
  package (`UploadKit`) used by both QuickUpload (lite) and CourtIQ
  (full). Pros: clean architecture, no dupe. Cons: more setup work.

Recommendation lean: **Option A** (extend CourtIQ) for the MVP; refactor
to **C** later if the app grows.

## 2. Single-user or multi-user auth model?

Today: single shared `IPHONE_UPLOAD_TOKEN`. For an iOS app:

- **Option A**: Hardcode the token in the app (single user, your
  device). Pros: zero auth UX. Cons: token leaks if APK reverse-
  engineered; bad multi-user story.
- **Option B**: Per-user token, configured once via in-app QR-scan or
  paste. Pros: multi-user-ready, no hardcode. Cons: one-time setup.
- **Option C**: OAuth/auth flow via the Worker. Pros: most secure.
  Cons: most work, overkill for now.

Recommendation lean: **Option B** for the MVP. Token entered once,
stored in Keychain.

## 3. PHPicker vs PHAsset full access?

- **PHPicker** (iOS 14+): system-mediated picker, no library
  permission required, user picks specific assets.
- **PHAsset** (full PhotoKit access): requires PhotoKit permission
  prompt, allows enumeration of whole library.

For "pick a video" use case, **PHPicker is correct** — minimal
permission, simpler UX. PHAsset is only needed for the bulk-watcher
flow (which the Mac side already covers).

## 4. Chunked upload protocol on iOS

The Worker's existing `/api/upload/iphone/{init,part,complete}` flow:
- `POST /init` returns an `upload_id` + part-size hint
- `POST /part?upload_id=X&part_number=N` with the chunk body
- `POST /complete` finalizes + creates R2 marker

iOS implementation:
- `URLSession.background` with `uploadTask(with:from:)` for each part
- Manage queue: 2-4 parts in flight at a time
- Persist `upload_id` + part status to disk so crash-recovery works

## 5. Background upload limits

iOS background URL sessions can continue uploads after the app is
backgrounded, but with caveats:
- App can be terminated; session continues but app must be relaunched
  to handle completion
- Network reachability changes (WiFi → cellular) honored per session
  configuration
- iOS may throttle based on system load

Test on a real device (not simulator) for a 2-4 GB file before
declaring this done.

## 6. Where to put the in-app UX

CourtIQ's existing ContentView has tabs/views (need to inspect).
The Quick Upload entry needs to be obviously discoverable. Probably
a prominent button on the home screen.

# Open design questions for plan-mode session to resolve

a) Should the recorded video be saved to the user's Photos library
   AFTER successful upload, or kept only on-device-then-deleted? User
   preference TBD.

b) How does upload-in-progress survive an app crash? Need a state
   file describing in-flight uploads that can be resumed at launch.

c) For the picker flow: should we support multi-select (pick 5 videos
   to batch upload) or single-select for v1?

d) Network-quality detection: warn user if they're about to upload a
   large file on cellular (even with WiFi-only toggle off).

e) After upload, when do we tell the user "your video is ready"?
   Options: (1) push notification when GPU pipeline finishes, (2)
   in-app polling against the Worker, (3) link to gallery page.
   Note: per memory `project_diagnostics_first_class_not_inline.md`,
   push notifications are deprecated in favor of pull-based dashboard.

# Deliverables of the plan-mode session

1. **Design doc** at `docs/ios_quick_upload_design.md` covering:
   - Architecture decision per each open question (1-6 above + a-e)
   - Component diagram
   - Sequence diagram for record→upload→confirm flow
   - Background upload + crash recovery design
   - File/module structure within `ios/CourtIQ/`

2. **Implementation plan** as a list of PRs/commits, each scoped
   small enough to ship + test independently. e.g.:
   - PR 1: Audit existing `R2Uploader.swift`, wire to Worker chunked
     endpoints if not already
   - PR 2: PHPicker-based "Choose existing" flow + upload trigger
   - PR 3: Record-flow refactor with upload trigger on stop
   - PR 4: Background URLSession + crash recovery
   - PR 5: Token onboarding UX (QR/paste, Keychain)
   - PR 6: Upload progress UI + post-upload confirmation poll
   - PR 7: Real-device WiFi throughput benchmark + tuning

3. **Open questions / risks** at the bottom of the design doc — things
   that need user input or that are explicit unknowns at design time.

# Plan-mode discipline

Start the session in plan mode. **Don't write production code until
the design doc above is approved.** Plan mode = read, analyze, propose;
not implement.

When you do start writing code, the existing `ios/CourtIQ/` structure
is your starting point. Don't rewrite from scratch.

# How to spin up

In a fresh terminal tab (the design-partner can't launch interactive
sessions from inside its own):

```bash
cd ~/tennis_analysis && git worktree add ~/tennis_worktrees/ios-upload -b feature/ios-live/quick-upload main
```

```bash
cd ~/tennis_worktrees/ios-upload && claude --permission-mode plan "execute .handoffs/inbox/20260521-0100-ios-direct-upload-design-from-design-partner.md"
```

The `--permission-mode plan` flag tells Claude to start in plan mode
— no code changes until you explicitly approve via ExitPlanMode.
