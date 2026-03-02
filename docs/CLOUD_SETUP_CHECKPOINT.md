# Cloud Pipeline Checkpoint - 2026-02-18

## What's Working

### Hetzner Server (5.78.96.237)
- **tennis-coordinator**: Job queue API on port 8080
- **tennis-worker**: Dispatches jobs to RunPod GPUs (RTX 4090)
- **tennis-watcher**: Polls iCloud albums every 5 minutes

### Web UI
- **URL**: https://playfullife.com
- Video gallery with thumbnails (proxied via /thumb/)
- Video player with speed controls (keyboard: arrows for speed/frames)
- URL routing: /watch/raw/filename.mp4 for shareable links

### Cloudflare R2 Storage (tennis-videos bucket)
- raw/ - Original videos from iCloud
- poses/ - Extracted pose JSON files
- thumbs/ - Video thumbnails

### Services Status Commands
```bash
ssh root@5.78.96.237 'systemctl status tennis-coordinator tennis-worker tennis-watcher'
ssh root@5.78.96.237 'journalctl -u tennis-worker -f'  # Watch processing
```

## Files on Hetzner (/opt/tennis/)
- cloud_pipeline.py - RunPod worker script
- cloud_icloud_watcher.py - iCloud poller
- coordinator/ - FastAPI coordinator + gallery
- .env - Credentials (iCloud, RunPod, R2)
- icloud_cookies/ - iCloud session

## Credentials Location
- Local: /Users/andrewhome/tennis_analysis/.env
- Hetzner: /opt/tennis/.env

## To Resume Work
The pipeline is fully automated. New videos added to "Tennis Videos" or
"Tennis Videos Group By Shot Type" iCloud albums will be:
1. Detected within 5 minutes
2. Downloaded and uploaded to R2
3. Processed on RunPod GPU (pose extraction)
4. Viewable at playfullife.com

## Known Issues / TODO
- Thumbnails load via proxy (R2 presigned URLs had CORS issues)
- Could add shot detection step after pose extraction
- Could add highlight compilation step

## Deploy Commands
```bash
# Deploy gallery changes
scp scripts/video_gallery.py root@5.78.96.237:/opt/tennis/coordinator/gallery.py
ssh root@5.78.96.237 'systemctl restart tennis-coordinator'

# Deploy worker changes
scp scripts/cloud_pipeline.py root@5.78.96.237:/opt/tennis/cloud_pipeline.py
ssh root@5.78.96.237 'systemctl restart tennis-worker'

# Deploy watcher changes
scp scripts/cloud_icloud_watcher.py root@5.78.96.237:/opt/tennis/cloud_icloud_watcher.py
ssh root@5.78.96.237 'systemctl restart tennis-watcher'
```
