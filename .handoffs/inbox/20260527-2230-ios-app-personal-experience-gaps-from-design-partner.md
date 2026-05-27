---
from: design-partner
to: ios-live
created: 2026-05-27T22:30:00-08:00
status: pending
priority: high
topic: iOS app personal-experience gaps — the leftover single-user web assumptions that need to die
---

# The frame

The web gallery was built when there was exactly one user. The iOS app
inherited those assumptions. Now that each user has a per-user gallery
at `/u/<hash>` and a Sign-in-with-Apple identity, the app needs to
answer questions the web never had to:

- *Whose* video is this?
- What's the status of *my* upload from 30 seconds ago?
- Can I delete *my* old test videos?
- What separates "my stuff" from "what everyone else sees"?

These questions don't have good surfaces today. This brief lists the
specific gaps in priority order and proposes the fixes.

# The gaps, in priority order

## P0: The in-flight WKWebView auth bug (you're already debugging this)

Uncommitted code in `ios/CourtIQ/CourtIQ/Views/WebViewWrapper.swift`
pre-seeds the `tennis_jwt` cookie into `WKHTTPCookieStore` before
load — fixes the race where subresources (img/video src) requested
during the 302 redirect can fail auth. Companion changes in
`worker/upload-worker.js` and `scripts/update_r2_index.py`.

**Action**: finish, commit, test on real device. If verified working,
this alone unblocks a clean per-user gallery experience.

## P1: "My recent uploads" status surface — completely missing

The app currently has an Upload tab (in-flight uploads), a Gallery tab
(WebView pointed at `/u/<hash>`), and... no surface for the *interesting
in-between state*. When a user has just uploaded:

- Where do they see "your video is queued — 2nd in line"?
- Where do they see "Andrew-PC is processing IMG_xxxx, 3 min elapsed"?
- Where do they see "failed — invalid codec; tap to retry"?
- Where do they see "ready! tap to view in gallery"?

**Today**: Upload tab shows mid-upload progress and then the row
disappears (probably). Gallery tab eventually shows the video as a
card. Between "upload completed" and "gallery card appears" there's
a multi-minute black box.

**Proposed fix**: A "Recent" view in the Upload tab (or as a new tab)
that lists the user's last N uploads with live status:

```
┌─────────────────────────────────────────┐
│ Recent uploads                          │
├─────────────────────────────────────────┤
│ ● 2 min ago    IMG_5523 · queued       │
│ ● 5 min ago    IMG_5520 · processing 4/7│
│ ✓ 12 min ago   IMG_5510 · ready →      │
│ ⚠ 28 min ago   IMG_5499 · failed (retry)│
└─────────────────────────────────────────┘
```

Driven by the existing R2 markers + coordinator queue endpoint we
built earlier (`/api/queue` filtered by user_hash). Worker already
serves `/u/<hash>` — extend to expose `/api/u/<hash>/recent` for the
app to poll.

## P2: Personal vs shared gallery — conceptual confusion

Today the app's Gallery tab loads `/u/<hash>` — which is supposed to be
the user's personal gallery. But:

- Is there a "shared" gallery elsewhere? (Maybe the old root `/`?)
- Does anything indicate to the user *whose* gallery they're looking at?
- Can the user navigate between "mine" and "shared" if both exist?
- If a user uploads a video, where does it go — only their personal
  gallery, or also somewhere shared?

**Proposed fix**: Three explicit modes in the iOS app, surfaced as
either a segmented control at the top of Gallery tab or as separate
tabs:

| Mode | What it shows | URL |
|---|---|---|
| **Mine** (default) | Only videos uploaded by this user | `/u/<hash>` |
| **Shared** (opt-in) | Friends' videos + your own | `/shared` (TBD; may not exist) |
| **Public** (later, maybe never) | Globally browsable | — |

For v1: probably just "Mine" is implemented; remove any UX implying
"shared" exists unless the server-side surface actually exists.

## P3: No per-video management

The web gallery has delete-via-password (`POST /api/video/:vid/delete`)
but no rename, no privacy toggle, no share-this-specific-video,
no tag, no album. On iOS, even delete is hidden behind a password the
user doesn't remember (`deletevideo` per CLAUDE.md).

**Per-user model makes this trickier and easier simultaneously:**

- Easier: when a user clicks delete on their own video, the auth context
  (their JWT) is the password. No more `deletevideo` magic word.
- Trickier: need to differentiate "delete my own" from "request takedown
  of someone else's content I see in shared" (if shared exists).

**Proposed fix (v1):**

- Long-press on a video card in their gallery → menu:
  - **Delete** — confirms, calls `DELETE /api/u/<hash>/video/<vid>` with
    user's JWT in Authorization header. Worker validates ownership
    (`uploaded_by == user_hash`) before deleting.
  - **Rename** — let user set a display name; stored in `meta.json` as
    `display_name`. Falls back to video_id if not set.
  - **Share link** — generates a public link to view this specific
    video (no auth required to view, but only this one video). Worker
    serves `/v/<share_token>` mapping.
  - (Deferred) tags, albums, privacy toggle

## P4: "Just uploaded" video doesn't surface in gallery for several minutes

After upload completes (Mac uploader or iOS app), the video enters the
coordinator queue. GPU picks it up. Processing takes 5-15 min. The
gallery only shows it once `processed/u_<hash>/.../meta.json` exists.

In the app, the user has nowhere to see this "uploading and ready"
journey except the Recent view in P1. If P1 isn't built, the user
opens the app, doesn't see their just-uploaded video, and thinks it
didn't upload.

**Proposed fix**: P1 covers this. Once Recent view exists, users see
the in-flight items clearly.

## P5: Upload tab UX after upload completes

Today's Upload tab shows in-flight items via `UploadRowView`. What
happens to a row when upload completes? Does it:

a) Disappear immediately (user wonders if it actually happened)
b) Stick around showing "uploaded — server processing"
c) Move to a "completed" section

(b) is the right behavior; (a) is what probably happens today
because the row was tied to in-flight URLSession tasks only.

**Proposed fix**: Upload tab shows two sections:
- **In progress** (currently exists — chunked uploads in flight)
- **Completed** (new — last 10 successfully uploaded, with link to
  their gallery entry once processing finishes)

# Per-app-area inventory of remaining single-user assumptions

Quick audit. For each, note whether the assumption is broken in the
multi-user world:

| Area | Single-user assumption | Multi-user reality |
|---|---|---|
| Gallery root path | `/` shows everything | should show user's `/u/<hash>` by default |
| Video card delete | Password-protected with shared password | Should be JWT-auth, only owner can delete |
| Pro library | All users see same `pros/index.json` | OK — pros are shared reference content |
| Coach summary | All users see same summary | OK — generated per-video, doesn't leak across users |
| Filmstrip thumbnails | At `highlights/thumbs/<vid>.jpg` | Should be at `highlights/<hash>/thumbs/<vid>.jpg` — already partially done per `upload_thumbnail()` |
| `VIDEOS` JSON in gallery HTML | All videos | Per-user filtered — needs verification |
| Pro comparison PNGs | At `compare/<vid>/shot_<n>.png` | OK (compare-against-pro doesn't need per-user prefix) |
| Push notifications | None (deprecated per memory) | Same; rely on Recent view for status |
| Background upload state | URLSession per upload | Needs persistent state per-user-account in case user signs out + back in mid-upload |

# Suggested PR sequence (each ships independently)

1. **PR-A**: Commit the in-flight WKWebView cookie fix + companion
   worker/`update_r2_index.py` changes. Real-device test. (~half day)
2. **PR-B**: Worker endpoint `GET /api/u/<hash>/recent` returning
   user's last N upload markers + their status from coordinator. Plus
   iOS `RecentUploadsView` consuming it. Adds the "Recent" surface.
   (~1 day)
3. **PR-C**: Long-press on video card in gallery → "Delete" action
   that calls `DELETE /api/u/<hash>/video/<vid>` with JWT. Worker
   validates ownership. iOS-side this is a JS bridge call from the
   WKWebView (since gallery is web). (~half day)
4. **PR-D**: Rename action (sets `display_name` in meta.json). Same
   JWT-auth approach. (~half day)
5. **PR-E**: Audit + remove leftover password-protected delete from
   the web gallery (since iOS users use JWT, and the shared password
   is dead). (~half day)
6. **PR-F**: Share-via-link (`/v/<share_token>`). Generates a public
   per-video URL the user can paste anywhere. (~1 day)

Order matters because PR-B unblocks Recent visibility which is the
biggest UX gap. PR-A is a strict prerequisite (without it, anything
running in the WebView is unauthenticated).

# Out of scope for this brief

- Pro Comparison view native in iOS (currently web-only; separate
  workstream)
- Social features (friend invites, feed) — out of v1 scope
- Pre-upload trim — already deferred per design v3
- Multi-select picker — already deferred
- Push notifications — deprecated per memory
- v1.2 feature scoping — separate exercise

# Asking back

1. **Does "shared gallery" exist as a real surface, or only conceptually?**
   If it doesn't exist, drop it from the Gallery tab UX. If it does (or
   should), where does the data come from?

2. **What's the right auth UX for delete?** Confirm dialog "Delete this
   video? This can't be undone" — minimal friction since the user's
   already JWT-auth'd. Or do we want a re-auth gate for destructive
   actions?

3. **For Recent surface**: is "last 10 uploads" enough, or do users
   want time-based ("today / this week / older")? Lean: 10 items, by
   timestamp desc.

4. **Branch**: continue on `feature/gallery/per-user`, or fork a new
   `feature/ios-live/personal-management` branch? Lean: continue on
   per-user since the work is intertwined.
