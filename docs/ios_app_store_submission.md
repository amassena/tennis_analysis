# Tennis Uploader — App Store Submission Checklist

Branch: `feature/ios-live/quick-upload`
Bundle ID: `com.amassena.courtiq.CourtIQ`
Display name: **Tennis Uploader**

This is the runbook for taking the work in PRs 0-7 from a green
`xcodebuild` to "approved on the App Store". Everything below is a
manual step — Claude cannot do them for you because they require your
Apple ID, your Cloudflare credentials, or interactive confirmation.

Order matters. Do the steps in sequence.

---

## Phase 1 — Server side (one-time)

### 1.1 Set the JWT signing secret

Generates a strong random secret, sets it as a Worker secret, never
written to disk:

```bash
cd ~/tennis_worktrees/ios-upload/worker
openssl rand -base64 48 | wrangler secret put JWT_SIGNING_SECRET
```

(Paste the random string at the prompt. The Worker is the only thing
that needs to know it.)

### 1.2 Deploy the Worker

```bash
cd ~/tennis_worktrees/ios-upload
./scripts/deploy_gallery.sh worker
```

This ships PRs 1, 2, 6 server-side: `/api/auth/apple`, `/api/me`,
`/api/account` DELETE, allowlist check, JWT-aware iPhone upload routes,
`/privacy` route, etc.

### 1.3 Upload the privacy policy to R2

The Worker serves `static/privacy.html` at `tennis.playfullife.com/privacy`.
Upload it once:

```bash
cd ~/tennis_worktrees/ios-upload
npx wrangler r2 object put tennis-videos/static/privacy.html \
    --file worker/static/privacy.html \
    --content-type "text/html; charset=utf-8"
```

Verify:

```bash
curl -I https://tennis.playfullife.com/privacy
# expect: HTTP/2 200, content-type: text/html
```

### 1.4 (Optional — but **recommended before** wide TestFlight)

Lock down sign-in to an allowlist. First sign in once with your Apple
ID via the app (see Phase 3) so you know your `apple_sub`. Then:

```bash
cat > /tmp/allowlist.json <<JSON
{
  "open": false,
  "subs": ["YOUR_APPLE_SUB_HERE", "FRIEND_1_APPLE_SUB", "..."]
}
JSON

npx wrangler r2 object put tennis-videos/users/_allowlist.json \
    --file /tmp/allowlist.json \
    --content-type application/json
```

Your `apple_sub` is the `apple_sub` field in `users/<sub>.json` after
your first successful sign-in. Or fetch it directly:

```bash
npx wrangler r2 object list tennis-videos --prefix users/ \
  | grep -v _allowlist | head
```

Friends installing the app will see "Not approved" 403 until their sub
is added.

---

## Phase 2 — Apple Developer Portal (one-time)

### 2.1 Enable Sign In with Apple capability

1. Go to <https://developer.apple.com/account/resources/identifiers/list>
2. Find the App ID `com.amassena.courtiq.CourtIQ` (create one if
   missing — App IDs ➝ + ➝ App ➝ explicit bundle id, enable Sign In
   with Apple under capabilities).
3. Click the app ID, enable "Sign In with Apple" capability if not
   already, Configure ➝ "Enable as a primary App ID" ➝ Save.

### 2.2 Confirm provisioning auto-renewed

Xcode handles this when you open the project. If it warns about
entitlements not matching the App ID, click "Try Again" or open
*Signing & Capabilities* and re-add Sign In with Apple from the +
button.

---

## Phase 3 — Local device install (sanity check)

Plug in your iPhone, then:

```bash
open ios/CourtIQ/CourtIQ.xcodeproj
```

In Xcode:
1. Select your iPhone as the run destination.
2. *Signing & Capabilities*: confirm Team = your Apple ID, Bundle ID
   = `com.amassena.courtiq.CourtIQ`, Automatically manage signing on.
3. Cmd-R to build + install.

Smoke test:
- Tap Sign in with Apple. Apple sheet appears. Use your Apple ID. App
  lands on the empty Upload tab.
- Tap +, Choose existing, pick a small video (~200 MB). Watch the row
  fill the progress bar. It should finish, then flip to "Uploaded —
  waiting for processing", then through processing states, then "Ready
  in gallery". Tap View, Gallery tab opens at the right anchor.
- Tap +, Record new, capture a 15s clip, tap Upload. Same flow.
- Settings ➝ verify privacy link opens the page, "Delete account"
  double-confirm works.

If you've enabled the allowlist (1.4), you must already be in it.

---

## Phase 4 — TestFlight

### 4.1 Archive

In Xcode: Product ➝ Destination ➝ "Any iOS Device" ➝ Product ➝ Archive.
Organizer opens.

### 4.2 Distribute App ➝ App Store Connect ➝ Upload

Follow the prompts. The app needs to validate successfully — common
gotchas:

- **Missing app icon**: PR 7 ships a 1024x1024 marketing icon.
  Apple may ask for additional sizes if your Xcode is old — newer
  Xcode (15+) accepts the single universal size.
- **Missing privacy manifest**: PR 7 ships
  `PrivacyInfo.xcprivacy`. Should be accepted.
- **Sign In with Apple entitlement**: enabled in Phase 2.

### 4.3 In App Store Connect (web)

<https://appstoreconnect.apple.com>

Create a new app:
- Name: **Tennis Uploader**
- Primary Language: English (U.S.)
- Bundle ID: `com.amassena.courtiq.CourtIQ`
- SKU: `tennis-uploader-001`
- User Access: Full Access

### 4.4 TestFlight tab

Add your build (from 4.2), wait ~10 min for processing.

Then add internal testers (your Apple ID — automatic) and invite
external testers by Apple ID email.

---

## Phase 5 — App Store submission

### 5.1 App Information

- **Subtitle**: Upload your tennis videos
- **Category**: Sports (primary), Photo & Video (secondary)
- **Content Rights**: I don't use third-party content

### 5.2 Pricing — Free

### 5.3 Privacy

**Privacy Policy URL**: `https://tennis.playfullife.com/privacy`

**App Privacy** (the nutrition label):

Data collected:
- **Identifiers ➝ User ID**: Linked to the user. Not used for tracking.
  Purpose: App functionality.
- **User Content ➝ Photos or Videos**: Linked to the user. Not used
  for tracking. Purpose: App functionality.
- **User Content ➝ Audio Data**: Linked to the user. Not used for
  tracking. Purpose: App functionality (recording-flow microphone).

(No analytics, no ads, no third-party SDK collecting anything.)

### 5.4 App Review Information

Reviewer notes (paste this):

> Tennis Uploader is a personal/private utility for me and a small
> group of friends. The app does two things: (1) lets the user record
> or pick a video from Photos, and (2) uploads it to a Cloudflare R2
> bucket for automated tennis-shot analysis. After processing, the
> video appears in a shared Gallery tab.
>
> To test:
> 1. Tap "Sign in with Apple" on the welcome screen. Any Apple ID
>    works — the app uses Apple as the identity provider; we do not
>    store passwords.
> 2. Tap the + button on the Upload tab. Choose "Pick existing" and
>    select any video from your library. The upload progress bar fills.
> 3. Tap Settings ➝ "Delete my account" to confirm account deletion
>    works.
>
> No demo account credentials required. Sign In with Apple is the only
> auth method, satisfying guideline 4.8.
>
> The Gallery tab embeds a private web view of
> tennis.playfullife.com — this is content shared among invited
> users only. It is not the app's primary surface; the native record
> and upload flows are.

### 5.5 Build

Choose the build you uploaded via TestFlight (4.2).

### 5.6 Version Release

Manual release after approval (recommended for v1).

### 5.7 Submit for Review

Click *Submit for Review*. First-app review takes 1–7 days. If rejected,
common reasons + fixes:

| Likely rejection | Fix |
|---|---|
| Guideline 4.2 (Minimum Functionality): "your app is just a web view" | Increase native surface — already mostly mitigated by record + upload native flows. If pressed, move Gallery tab behind a per-row "View" button (SFSafariViewController). |
| Guideline 5.1.1 (Data Collection): "your app collects user info but doesn't tell us how to delete" | Already mitigated — privacy page + in-app Settings ➝ Delete account. Point reviewer to the path. |
| Sign In with Apple verification failure | Confirm Phase 2 entitlement is enabled and the Worker has `JWT_SIGNING_SECRET` set. |

---

## Manual blockers summary

These are the only things Claude couldn't automate:

| Step | Why it's manual |
|---|---|
| `wrangler secret put JWT_SIGNING_SECRET` | Interactive prompt; secret never written to disk. |
| Apple Developer portal capability toggle | Requires your Apple ID auth. |
| App Store Connect metadata + submit | Requires your Apple ID auth. |
| Local device install + smoke test | Requires your phone + Apple ID. |
| Creating `users/_allowlist.json` after first sign-in | Needs your real `apple_sub` value. |

Everything else is committed in this branch.

---

## Known v1 limitations (acceptable for App Review)

- **Foreground uploads only.** We use `URLSession.default` + a
  `beginBackgroundTask` grace; uploads pause if the user keeps the
  app backgrounded for >30s. True background URLSession is a v1.1
  improvement (the design doc `docs/ios_quick_upload_design.md` lays
  this out under "Risks / open items").
- **Shared gallery.** All invited users see each others' uploads.
  Per-user galleries are a future migration; every upload from this
  build already carries `uploaded_by: <user_hash>` so the migration
  is purely additive (see design doc § Future migration).
- **Cached gallery HTML may briefly still show deleted videos.**
  Account deletion removes the R2 source/processed files immediately;
  the cached index.html refreshes on the next pipeline run.
- **The app icon is a placeholder.** Solid green with a tennis ball
  and upload arrow. Functional for review; swap in a designed icon
  before going wider.
