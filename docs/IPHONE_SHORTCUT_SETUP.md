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

## Step 2 — App-trigger automation (iOS 26+)

This section is now in Step 3 below. iOS 26 dropped "added to album" triggers,
so we use an App-on-close trigger paired with a scan-and-upload shortcut.

## Step 3 — Create the automation (iOS 26+: use App trigger)

iOS 26 removed the "When a photo is added to an album" trigger. The replacement
is an **App** trigger on Photos closing — fires every time you exit Photos,
running a "Tennis Sync" shortcut that scans the Tennis album and uploads any
new videos.

1. In Photos, create an album called **Tennis** if you don't have one.
2. Shortcuts.app → **Automation** tab → **+** → **App** trigger:
   - App: **Photos**
   - When: **Is Closed**
   - Run Immediately: ON
   - Action: **Run Shortcut → Tennis Sync** (build it in Step 4 below)

## Step 4 — Build "Tennis Sync" (the scan-and-upload shortcut)

Library tab → **+** → name it **Tennis Sync**. Actions in order:

1. **Find Photos** — Album is **Tennis**, sort Creation Date Latest First, Get All Items
2. **Repeat with Each** — input is step 1's result; everything below goes inside this loop
3. **Get Details of Images** — Photos: **Repeat Item**, Detail: **File Path** (used as dedup key — iOS 26 dropped Asset Identifier from this action; File Path is a stable per-asset URL that works as the X-Asset-Id value)
4. **Get Details of Images** — Photos: **Repeat Item**, Detail: **Date Taken**
5. **Format Date** — Date: step 4's result, Format: **ISO 8601**, Include Time ON
6. **Get Details of Images** — Photos: **Repeat Item**, Detail: **Name**
7. **URL Encode** — encode step 3's File Path (it has special chars)
8. **Get Contents of URL** (the check call):
   - URL: `https://tennis.playfullife.com/api/upload/iphone/check?asset_id=` + step 7's encoded result
   - Method: **GET**
   - Headers: `Authorization: Bearer <token>`
9. **Get Dictionary Value** — get value for `uploaded` from step 8's response
10. **If** — condition: step 9's value **is** `false`
    - **10a. Get Contents of URL** (the upload):
      - URL: `https://tennis.playfullife.com/api/upload/iphone`
      - Method: **POST**, Request Body: **File** = **Repeat Item**
      - Headers: `Authorization: Bearer <token>`, `X-Asset-Id` = step 3's File Path (raw, not encoded — only the URL query param needs encoding), `X-Filename` = step 6's Name, `X-Created-At` = step 5's formatted date
    - **10b. Show Notification** (optional): title "Tennis Synced", body = step 6's Name
11. **End If**, **End Repeat** auto-close.

## Step 5 — Build "Tennis Upload" (single-video manual share)

Optional but useful as a backup. Library → **+** → **Tennis Upload**.

Settings (info icon at top): Show in Share Sheet ON, Accept: Photos / Videos.

Actions:
1. **Get Details of Images** — Detail: **File Path**, input: **Shortcut Input**
2. **Get Details of Images** — Detail: **Date Taken**, input: **Shortcut Input**
3. **Format Date** — ISO 8601 + time
4. **Get Details of Images** — Detail: **Name**
5. **Get Contents of URL** — same headers as Tennis Sync 10a, with `X-Asset-Id` = step 1's File Path

This lives in the share sheet. From Photos, select a video → Share → Tennis Upload.

## Step 6 — Use it

Two paths:

- **Automatic via Tennis Sync:** add tennis videos to the **Tennis** album in
  Photos. Close Photos. The App-Closed automation fires Tennis Sync, which
  scans the album and uploads anything new (skipping already-uploaded via
  the `/check` endpoint).
- **Manual via Tennis Upload:** select a single video in Photos → Share →
  **Tennis Upload**.

## Step 7 — Verify

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

## Failure recovery

Tennis Sync IS the recovery mechanism — every time you close Photos, it scans
the Tennis album and uploads anything not yet acknowledged by the Worker. So
if iOS suspends an upload mid-stream, just open and close Photos again and
Tennis Sync re-attempts the failed ones.

For travel / spotty wifi: tap **▶** on Tennis Sync manually after reconnecting
to good wifi. Same loop, runs on demand.

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
