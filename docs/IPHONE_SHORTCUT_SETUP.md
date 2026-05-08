# iPhone Shortcut: Tennis Upload

User-facing setup guide for the iOS Shortcut that replaces the pyicloud watcher
ingest path. Build this once on iPhone; new tennis videos auto-upload to R2,
get registered with the coordinator, and flow through the GPU pipeline without
any iCloud auth involved.

Architecture (as deployed):

```
iPhone Photos → iOS Shortcut "Tennis Upload"
   ↓ (POST raw bytes + 3 headers)
Cloudflare Worker /api/upload/iphone
   ↓ (streams to R2)
R2 source/{video_id}.mov  +  R2 uploads/{video_id}.json (marker)
   ↓ (polled every 30s)
Hetzner tennis-iphone-poller.service → state.add_job() → coordinator pending
   ↓
GPU worker claims, download_source_from_r2() pulls source, normal pipeline
```

The video_id is derived deterministically from the iOS PHAsset id (SHA-256
prefix → `iphone_<8-hex>`), so a second tap of the Shortcut on the same video
is a no-op (the Worker returns 409 / `status: duplicate`).

---

## Step 1 — Find your auth token

The token is on the Mac at `/tmp/iphone_upload_token.txt` (mode 600). It's
also stored as the Cloudflare Worker secret `IPHONE_UPLOAD_TOKEN`. If the
file is gone, regenerate:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))" > /tmp/iphone_upload_token.txt
chmod 600 /tmp/iphone_upload_token.txt
cd worker && cat /tmp/iphone_upload_token.txt | npx wrangler secret put IPHONE_UPLOAD_TOKEN
```

Open the file on Mac, copy the value. You'll paste it into the Shortcut once.

## Step 2 — Create the upload Shortcut

On iPhone, open Shortcuts.app:

1. Tap **+** (new shortcut), name it **Tennis Upload**.
2. In shortcut Settings (top right gear): turn ON **Show in Share Sheet**.
   Set **Accept** to **Photos / Videos** only.
3. Add these actions in order:

### Action 1: Get Details of Photos

- Action: **Get Details of Photos**
- Get: **Asset Identifier** (this is the iOS PHAsset localIdentifier)
- Save it as a Magic Variable named `AssetID`

### Action 2: Get Details of Photos (creation date)

- Action: **Get Details of Photos**
- Get: **Date Taken**
- Save as Magic Variable `CreatedAt`

### Action 3: Format Date

- Action: **Format Date**
- Date: `CreatedAt`
- Date Format: **ISO 8601**
- Include Time: **ON**
- Save as `CreatedAtISO`

### Action 4: Get Details of Photos (filename)

- Action: **Get Name**
- Of: the photo input
- Save as `Filename`

### Action 5: Get Contents of URL

- Action: **Get Contents of URL**
- URL: `https://tennis.playfullife.com/api/upload/iphone`
- Method: **POST**
- Request Body: **File** → set to the photo input
- Headers (tap "Headers", add three):
  - `Authorization` → `Bearer <paste token from step 1>`
  - `X-Asset-Id` → Magic Variable `AssetID`
  - `X-Filename` → Magic Variable `Filename`
  - `X-Created-At` → Magic Variable `CreatedAtISO`

### Action 6: Show Notification (optional but recommended)

- Action: **Show Notification**
- Title: `Tennis Upload`
- Body: `Uploaded ` + `Filename` + ` (id: ` + `Get Dictionary Value` from
  the previous step's response → key `video_id` + `)`

If the Worker returns a `status: duplicate` (409, second tap), the
notification will show `(id: iphone_…)` confirming the dedup worked.

## Step 3 — Create the automation (auto-fire on adding to album)

1. In Photos, create an album called **Tennis** if you don't have one.
2. Shortcuts.app → **Automation** tab → **+** → **Photo** trigger:
   - Trigger: **When a photo is added to an album**
   - Album: **Tennis**
   - Action: **Run Shortcut "Tennis Upload"**
   - **Run Immediately** ON (no confirmation prompt).

## Step 4 — Use it

Two paths:

- **Automatic:** select tennis videos in Photos → "Add to Album" → "Tennis".
  The shortcut auto-fires per video.
- **Manual:** select videos → tap Share → "Tennis Upload" (because the
  shortcut is in the share sheet).

## Step 5 — Verify

After the first upload (give it ~1-2 minutes for the marker to be picked up):

- Coordinator should have the job:
  ```bash
  ssh devserver 'cd /opt/tennis && venv/bin/python -c "
  import sqlite3
  c = sqlite3.connect(\"/opt/tennis/coordinator.db\")
  for r in c.execute(\"SELECT video_id,filename,status FROM jobs WHERE video_id LIKE \\\"iphone_%\\\" ORDER BY created_at DESC LIMIT 5\"):
      print(r)
  "'
  ```
- Watch poller logs:
  ```bash
  ssh devserver 'sudo journalctl -u tennis-iphone-poller.service -n 20 -f'
  ```
- Within ~25-30 minutes (one full pipeline run on a normal session), the
  video lands at https://tennis.playfullife.com under its `iphone_<hash>`
  card.

## Failure recovery — "Retry Failed Tennis Uploads" Shortcut

For when iOS suspends the upload mid-stream (4 GB MOVs over LTE are at risk).

Build a second Shortcut, **Retry Failed Tennis Uploads**:

1. Action: **Find Photos where Album is Tennis**
2. Action: **Repeat with Each** photo
3. Inside the loop:
   - Get Asset Identifier → `AssetID`
   - **Get Contents of URL** to
     `https://tennis.playfullife.com/api/upload/iphone/check?asset_id=<AssetID>`
     with `Authorization: Bearer <token>`
   - **If** `Get Dictionary Value: uploaded` from response is `false`:
     - Run the same upload action chain as in the main shortcut

Run this manually after travel / spotty wifi sessions. It re-uploads only the
videos the Worker hasn't acked.

---

## Auth & rotation

`IPHONE_UPLOAD_TOKEN` is a single shared secret today (single-user MVP). To
rotate:

```bash
NEW_TOKEN=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
echo $NEW_TOKEN > /tmp/iphone_upload_token.txt
cd worker && echo $NEW_TOKEN | npx wrangler secret put IPHONE_UPLOAD_TOKEN
# Then update the Shortcut's Authorization header on iPhone.
```

The Shortcut stops working immediately on rotation; that's the point.

## Migration plan (parallel run)

The pyicloud watcher (`tennis-watcher.service` on Hetzner) is currently down
due to the iCloud lockout. Once it's restored, run both ingest paths in
parallel for ~1 week:

- Watcher continues to pick up videos in the **iCloud "Slo-mo"** album
- Shortcut picks up videos added to the **"Tennis"** album in Photos

Track that both produce gallery entries. After 5+ Shortcut uploads complete
without issues, retire the watcher:

```bash
ssh devserver 'sudo systemctl disable --now tennis-watcher.service'
```

The Worker `/api/upload/iphone` endpoint, the marker poller, and the GPU
worker R2-first source resolution are the production ingest path from then
on. pyicloud cleanup is a separate PR.
