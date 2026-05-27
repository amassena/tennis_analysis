---
from: design-partner
to: ios-live
created: 2026-05-27T22:45:00-08:00
status: pending
priority: high
topic: iOS app — additional gaps beyond the personal-experience set (user trust + Apple-required surfaces)
addendum-to: 20260527-2230-ios-app-personal-experience-gaps-from-design-partner.md
---

# Why a second brief

The first brief covered "personal experience" gaps — Recent view,
Mine-vs-Shared, per-video management, etc. This one is everything
else I'd surface walking through the app cold:

- Apple-required surfaces missing
- Trust/credibility surfaces missing
- The web→WKWebView translation has known friction
- Common iOS upload-app failure modes the app doesn't gracefully handle

Same priority frame: P0 = must-fix-now, P3 = nice-to-have.

---

## P1: Account deletion — Apple-required

Apple App Store Review Guideline 5.1.1(v) requires apps with account
creation to provide **in-app account deletion**. We have Sign-in-with-
Apple → account creation. So we need:

- Settings → "Delete my account" button
- Confirmation dialog ("This deletes your account and all uploaded
  videos. Cannot be undone.")
- API call: `DELETE /api/u/<hash>` (worker handles: kill JWT,
  invalidate, delete R2 user-prefix data, mark user `status: deleted`
  in users.json)

Without this, future App Store updates can be rejected. v1.1 may have
already shipped without it — worth checking your submission status.

## P1: First-run onboarding — empty-state UX

A user installs the app, opens it, signs in with Apple, lands on…
what? Probably Upload tab with empty state. Gallery tab empty.
Camera tab works but they don't know what to do.

**Proposed first-run flow:**
1. Sign in → welcome screen "Hi <name>. Tennis Uploader records
   your tennis sessions and gives you slow-motion analysis"
2. Permission grants in sequence:
   - Camera access — "to record sessions" (only request when they tap Record)
   - Photos access — "to pick existing videos" (only on first pick)
   - Notifications — "to tell you when analysis is ready" (skip if you're
     not using push)
3. "Try your first upload" CTA — opens Upload tab with prominent
   "Choose a video" button
4. After first successful upload + processing: "Your first session is
   ready! Tap to view"

Without this, first impression = "what am I supposed to do?"

## P2: Local notifications — "your video is ready" without server push

Per memory `project_diagnostics_first_class_not_inline.md`, server-
side push notifications are deprecated in favor of pull-based dashboard.
But the user shouldn't have to open the app every 10 min to see if
their video is done.

**Proposed fix**: app-local notifications scheduled at upload
completion. The app already knows when an upload finishes (it owns the
URLSession). After upload complete:

- Schedule a local notification: "Video processing started. We'll
  notify you when ready."
- Poll the worker every ~30s while app is foregrounded
- When state flips to `ready`, fire a local notification: "<video_id>
  is ready to view"

No server push required. iOS handles local notifications without
APNs entitlement.

Caveat: if app is killed before processing completes, polling stops
and local notification won't fire. Acceptable tradeoff for v1.

## P2: In-app video playback — currently delegated to WebView

When user taps a video card in the Gallery WebView, what happens?
Probably the gallery's existing player overlay opens inside the
WebView. On phone screen, that's:

- Player overlay may not size correctly
- Frame-step buttons (« ‹ › ») are tiny on touch
- Pinch-to-zoom inside a WebView player isn't natural
- Slow-mo speed picker requires multi-tap

**Two options**:

a) **Hand off to native AVPlayer** for full-screen playback. iOS app
   intercepts the WebView URL tap and presents an AVPlayer
   controller. Better touch UX, native scrubbing, native AirPlay.

b) **Polish the WebView player for mobile** — bigger touch targets,
   simplified controls, swipe-to-scrub.

(a) is more work but a noticeably better experience. (b) ships sooner.

## P2: Upload control — cancel/pause/retry visibility

Today: chunked upload is in-flight via `URLSession`. User can probably
swipe-to-delete a row in Upload tab (need to verify). But:

- Mid-upload, can they cancel? (Should call init-then-abort cleanup
  on the worker, free the R2 multipart upload.)
- Mid-upload, can they pause? (Less critical; cancel + restart is fine.)
- Failed upload — is "retry" surfaced clearly? Does retry resume from
  last successful part or restart from 0?

**Proposed fix**: per-row actions on `UploadRowView`:

- **Cancel** (in-progress) — calls `POST /api/upload/iphone/abort` with
  upload_id; row disappears
- **Retry** (failed) — re-runs chunked upload from where it failed if
  resume state exists; from 0 otherwise

## P3: WKWebView ↔ JS bridge gap — gallery features broken on iOS

The web gallery has features that assume desktop:

- **Right-click correction**: long-press on shot chip → 4-button picker.
  Does long-press work in WebView? Touch events vs contextmenu events.
- **vs pro modal**: opens a PNG. Probably works on phone but PNG might
  be too wide for the screen.
- **Filmstrip horizontal scroll**: fights iOS swipe-back navigation
  gesture. User tries to scroll filmstrip, app goes back instead.
- **Filter chips**: filter row may not fit phone width

Audit each WebView feature for mobile-touch usability. Fix in the web
gallery (so web users also benefit) or hide the feature in iOS via
user-agent detection.

## P3: Error surfaces

What does the user see when:

- JWT expires mid-session → 401 from worker → app blank? Show "Sign in
  expired, please re-authenticate" with Sign-in button.
- Network drops during upload → retry-with-backoff invisible, or
  surfaced as "Upload paused — waiting for network"?
- R2 upload fails (worker returns 5xx) → "Server error, tap to retry"
  with link to a status page?
- Per-user quota exceeded (if we add quotas later) → "You've used
  X GB of Y GB. Delete videos or upgrade."
- Camera permission denied → "Camera access needed to record. Open
  Settings to enable." with deep-link to system settings.

Each of these should have a designed surface, not a silent failure.
Most aren't there today.

## P3: Storage / data transparency

Apple App Store privacy labels require declaring what data the app
collects. We collect:

- Apple ID identifier (for Sign in)
- Upload metadata (file size, dimensions, timestamps)
- Video content itself
- Anything else?

**Proposed fix**: ensure App Store Connect "Privacy" section is
accurate. Settings → "About / Privacy" surface in-app linking to a
privacy policy page (served by worker at `/privacy`).

## P3: Cancel/account-switching edge cases

- User signs out → uploads in-flight: do we cancel or let them
  finish? Lean: cancel, since auth context goes away.
- User signs in to a different Apple ID on the same device → does
  the app recognize and switch context? Or does it still show old
  user's content cached locally?
- Apple ID is revoked or password changed → what happens to existing
  JWTs on the device?

These are rare but if any surface in App Store review they'll bounce
the build.

---

# Consolidated additional PR list

| PR | Title | Effort | Apple-required? |
|---|---|---|---|
| G | Account deletion (Settings → Delete Account → Worker DELETE) | ~1 day | **Yes** |
| H | First-run onboarding (welcome + permission gates + first-upload CTA) | ~1 day | No |
| I | Local notifications for "ready to view" | ~half day | No |
| J | Native AVPlayer for tapped videos | ~1 day | No (but big UX win) |
| K | Upload cancel/retry actions on `UploadRowView` | ~half day | No |
| L | WebView feature audit + mobile touch fixes | ~1 day | No |
| M | Error surfaces (JWT expired, network, etc.) | ~1 day | No |
| N | Privacy + Settings polish | ~half day | Partial Apple-req |

Adds to A–F from the first brief = total 14 PRs, ~10 days of work.
Order of attack:

**Must-do for App Store hygiene**: PR-G (account deletion).
**Biggest user impact**: PR-A (auth fix), PR-B (Recent view), PR-H
(onboarding).
**Polish, ship as bug reports come in**: PR-I through PR-N.

---

# Asking back

5. **Has v1.1 already shipped with account deletion or without?** If
   without, prioritize PR-G immediately to avoid future-rejection risk.

6. **Native AVPlayer worth the investment, or WebView player is good
   enough?** Lean native — touch UX is the #1 difference between
   feels-like-an-app and feels-like-a-website.

7. **What's your tolerance for first-launch friction?** Heavy
   onboarding (4 screens) vs light (2 screens) vs zero (cold-start to
   Upload tab).

8. **Privacy policy**: do you already have one published, or do we need
   to draft + host at `/privacy`?

---

# This brief + the prior one in priority order

1. **PR-A** (WKWebView cookie fix) — finish in-flight, immediate
2. **PR-G** (Account deletion) — Apple-required, must ship by next App
   Store update
3. **PR-B** (Recent uploads view) — biggest user-facing UX gap
4. **PR-H** (First-run onboarding) — first impression
5. **PR-M** (Error surfaces) — trust
6. **PR-J** (Native AVPlayer) — feels-like-an-app
7. **PR-C/D/E** (Delete/rename via JWT) — management
8. **PR-I** (Local notifications) — engagement
9. **PR-K** (Upload cancel/retry) — control
10. **PR-L** (WebView feature audit) — polish
11. **PR-F** (Share-via-link) — virality
12. **PR-N** (Privacy/settings polish) — compliance polish

Twelve PRs. ~2 weeks of work for a single implementer.
