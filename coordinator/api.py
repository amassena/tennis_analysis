"""HTTP API for the distributed coordinator.

Runs on Raspberry Pi (local) or AWS Lambda/ECS (cloud).
GPU workers poll this API to claim and complete jobs.
"""

import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel

from .state import StateBackend, VideoJob, VideoStatus, ProcessingStage

# R2 Configuration
R2_PUBLIC_URL = os.environ.get("R2_PUBLIC_URL", "https://media.playfullife.com")
from .state_sqlite import SQLiteStateBackend

# Configuration
DB_PATH = os.environ.get("COORDINATOR_DB", "coordinator.db")
STALE_CLAIM_SECONDS = int(os.environ.get("STALE_CLAIM_SECONDS", "3600"))

# State backend (swap implementation for AWS)
state: StateBackend = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup state backend."""
    global state
    state = SQLiteStateBackend(DB_PATH)
    await state.init()
    yield
    await state.close()


app = FastAPI(
    title="Tennis Pipeline Coordinator",
    description="Distributed job queue for video processing",
    version="1.0.0",
    lifespan=lifespan,
)


# Request/Response Models
class AddJobRequest(BaseModel):
    icloud_asset_id: str
    filename: str
    album_name: Optional[str] = None


class ClaimResponse(BaseModel):
    success: bool
    job: Optional[dict] = None
    message: Optional[str] = None


class CompleteRequest(BaseModel):
    highlights_url: Optional[str] = None
    error_message: Optional[str] = None
    success: bool = True


class StageUpdateRequest(BaseModel):
    stage: str
    progress: Optional[float] = None
    message: Optional[str] = None


class PodUpdateRequest(BaseModel):
    pod_id: Optional[str] = None
    pod_status: Optional[str] = None


class StatsResponse(BaseModel):
    pending: int = 0
    claimed: int = 0
    processing: int = 0
    completed: int = 0
    failed: int = 0
    total: int = 0


# Endpoints

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.post("/jobs")
async def add_job(request: AddJobRequest):
    """
    Add a new video job to the queue.
    Called by iCloud watcher when new videos are detected.
    """
    # Check if already exists
    if await state.job_exists(request.icloud_asset_id):
        return {"status": "exists", "message": "Job already in queue"}

    video_id = str(uuid.uuid4())[:8]
    job = VideoJob(
        video_id=video_id,
        icloud_asset_id=request.icloud_asset_id,
        filename=request.filename,
        status=VideoStatus.PENDING,
        album_name=request.album_name,
    )
    await state.add_job(job)

    return {"status": "created", "video_id": video_id}


@app.get("/jobs/pending")
async def list_pending():
    """List all pending jobs available for claiming."""
    jobs = await state.get_pending_jobs()
    return {
        "count": len(jobs),
        "jobs": [
            {
                "video_id": j.video_id,
                "filename": j.filename,
                "icloud_asset_id": j.icloud_asset_id,
                "album_name": j.album_name,
                "created_at": j.created_at.isoformat() if j.created_at else None,
            }
            for j in jobs
        ],
    }


@app.post("/jobs/{video_id}/claim")
async def claim_job(video_id: str, worker_id: str):
    """
    Claim a job for processing.
    Worker must provide their ID (hostname).
    Returns job details if claim successful.
    """
    success = await state.claim_job(video_id, worker_id)

    if not success:
        return ClaimResponse(
            success=False,
            message="Job already claimed or does not exist",
        )

    job = await state.get_job(video_id)
    return ClaimResponse(
        success=True,
        job={
            "video_id": job.video_id,
            "filename": job.filename,
            "icloud_asset_id": job.icloud_asset_id,
            "album_name": job.album_name,
        },
    )


@app.post("/jobs/{video_id}/processing")
async def mark_processing(video_id: str, worker_id: str):
    """Mark job as actively processing (heartbeat)."""
    job = await state.get_job(video_id)

    if not job:
        raise HTTPException(404, "Job not found")

    if job.claimed_by != worker_id:
        raise HTTPException(403, "Job not claimed by this worker")

    await state.update_status(video_id, VideoStatus.PROCESSING)
    return {"status": "ok"}


@app.post("/jobs/{video_id}/complete")
async def complete_job(video_id: str, worker_id: str, request: CompleteRequest):
    """
    Mark job as completed or failed.
    Include YouTube URL on success.
    """
    job = await state.get_job(video_id)

    if not job:
        raise HTTPException(404, "Job not found")

    if job.claimed_by != worker_id:
        raise HTTPException(403, "Job not claimed by this worker")

    if request.success:
        await state.update_status(
            video_id,
            VideoStatus.COMPLETED,
            highlights_url=request.highlights_url,
        )
    else:
        await state.update_status(
            video_id,
            VideoStatus.FAILED,
            error_message=request.error_message,
        )

    return {"status": "ok"}


@app.post("/jobs/{video_id}/reset")
async def reset_job(video_id: str, force: bool = False):
    """
    Reset a job back to pending status for reprocessing.
    Clears claimed_by and error_message.

    Args:
        force: If True, reset even if job is claimed/processing (use for stuck jobs)
    """
    job = await state.get_job(video_id)

    if not job:
        raise HTTPException(404, "Job not found")

    if job.status == VideoStatus.COMPLETED and not force:
        raise HTTPException(400, "Cannot reset completed job (use force=true)")

    if job.status == VideoStatus.PENDING:
        raise HTTPException(400, "Job already pending")

    await state.update_status(
        video_id,
        VideoStatus.PENDING,
        error_message=None,
    )
    # Clear claimed_by
    await state.unclaim_job(video_id)

    return {"status": "ok", "message": f"Job {video_id} reset to pending"}


@app.get("/jobs/active")
async def list_active_jobs():
    """List all currently processing jobs with their stages."""
    jobs = await state.get_active_jobs()
    return {
        "count": len(jobs),
        "jobs": [
            {
                "video_id": j.video_id,
                "filename": j.filename,
                "status": j.status.value,
                "claimed_by": j.claimed_by,
                "claimed_at": j.claimed_at.isoformat() if j.claimed_at else None,
                "current_stage": j.current_stage.value if j.current_stage else None,
                "stage_progress": j.stage_progress,
                "stage_message": j.stage_message,
                "stage_updated_at": j.stage_updated_at.isoformat() if j.stage_updated_at else None,
                "pod_id": j.pod_id,
                "pod_status": j.pod_status,
            }
            for j in jobs
        ],
    }


@app.get("/jobs/{video_id}")
async def get_job(video_id: str):
    """Get job details by ID."""
    job = await state.get_job(video_id)

    if not job:
        raise HTTPException(404, "Job not found")

    return {
        "video_id": job.video_id,
        "filename": job.filename,
        "icloud_asset_id": job.icloud_asset_id,
        "status": job.status.value,
        "claimed_by": job.claimed_by,
        "claimed_at": job.claimed_at.isoformat() if job.claimed_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        "highlights_url": job.highlights_url,
        "error_message": job.error_message,
        "retry_count": job.retry_count,
        "album_name": job.album_name,
        "created_at": job.created_at.isoformat() if job.created_at else None,
    }


@app.get("/jobs")
async def list_jobs(status: Optional[str] = None, limit: int = 100):
    """List all jobs with optional status filter."""
    status_enum = VideoStatus(status) if status else None
    jobs = await state.list_jobs(status=status_enum, limit=limit)

    return {
        "count": len(jobs),
        "jobs": [
            {
                "video_id": j.video_id,
                "filename": j.filename,
                "status": j.status.value,
                "claimed_by": j.claimed_by,
                "highlights_url": j.highlights_url,
                "created_at": j.created_at.isoformat() if j.created_at else None,
            }
            for j in jobs
        ],
    }


@app.get("/stats")
async def get_stats():
    """Get job statistics."""
    stats = await state.get_stats()
    return StatsResponse(
        pending=stats.get("pending", 0),
        claimed=stats.get("claimed", 0),
        processing=stats.get("processing", 0),
        completed=stats.get("completed", 0),
        failed=stats.get("failed", 0),
        total=stats.get("total", 0),
    )


@app.post("/maintenance/release-stale")
async def release_stale_claims():
    """Release jobs that have been claimed too long without completing."""
    count = await state.release_stale_claims(STALE_CLAIM_SECONDS)
    return {"released": count}


# ── R2 Media Proxy ────────────────────────────────────────────


@app.get("/thumbs/{filename:path}")
async def proxy_thumbnail(filename: str):
    """Proxy thumbnail requests to R2."""
    return RedirectResponse(f"{R2_PUBLIC_URL}/thumbs/{filename}")


@app.get("/highlights/{filename:path}")
async def proxy_highlights(filename: str):
    """Proxy highlight video requests to R2."""
    return RedirectResponse(f"{R2_PUBLIC_URL}/highlights/{filename}")


@app.get("/raw/{filename:path}")
async def proxy_raw(filename: str):
    """Proxy raw video requests to R2."""
    return RedirectResponse(f"{R2_PUBLIC_URL}/raw/{filename}")


# ── Stage & Progress Tracking ────────────────────────────────


@app.post("/jobs/{video_id}/stage")
async def update_stage(video_id: str, worker_id: str, request: StageUpdateRequest):
    """Update the current processing stage for a job."""
    job = await state.get_job(video_id)

    if not job:
        raise HTTPException(404, "Job not found")

    if job.claimed_by != worker_id:
        raise HTTPException(403, "Job not claimed by this worker")

    try:
        stage = ProcessingStage(request.stage)
    except ValueError:
        raise HTTPException(400, f"Invalid stage: {request.stage}")

    await state.update_stage(video_id, stage, request.progress, request.message)
    return {"status": "ok", "stage": stage.value}


@app.post("/jobs/{video_id}/pod")
async def update_pod(video_id: str, worker_id: str, request: PodUpdateRequest):
    """Update RunPod pod information for a job."""
    job = await state.get_job(video_id)

    if not job:
        raise HTTPException(404, "Job not found")

    if job.claimed_by != worker_id:
        raise HTTPException(403, "Job not claimed by this worker")

    await state.update_pod_info(video_id, request.pod_id, request.pod_status)
    return {"status": "ok"}


# ── Web Dashboard ────────────────────────────────────────────


from fastapi.responses import HTMLResponse

DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Tennis Pipeline Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            padding: 20px;
        }
        h1 { color: #00d9ff; margin-bottom: 20px; }
        h2 { color: #00d9ff; margin: 20px 0 10px; font-size: 1.2em; }
        .stats {
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
            margin-bottom: 20px;
        }
        .stat {
            background: #16213e;
            padding: 15px 25px;
            border-radius: 8px;
            text-align: center;
        }
        .stat-value { font-size: 2em; font-weight: bold; }
        .stat-label { color: #888; font-size: 0.9em; }
        .stat.pending .stat-value { color: #ffd93d; }
        .stat.processing .stat-value { color: #6bcb77; }
        .stat.completed .stat-value { color: #4d96ff; }
        .stat.failed .stat-value { color: #ff6b6b; }
        .jobs { margin-top: 20px; }
        .job {
            background: #16213e;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 10px;
        }
        .job-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        .job-filename { font-weight: bold; color: #00d9ff; }
        .job-status {
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: 500;
        }
        .job-status.processing { background: #2d5a27; color: #6bcb77; }
        .job-status.claimed { background: #5a4a27; color: #ffd93d; }
        .job-status.completed { background: #274a5a; color: #4d96ff; }
        .job-status.failed { background: #5a2727; color: #ff6b6b; }
        .job-status.pending { background: #3a3a4a; color: #aaa; }
        .progress-bar {
            height: 8px;
            background: #0a0a15;
            border-radius: 4px;
            overflow: hidden;
            margin: 10px 0;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #00d9ff, #6bcb77);
            transition: width 0.3s ease;
        }
        .job-details {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 10px;
            font-size: 0.9em;
            color: #888;
        }
        .job-details span { color: #ccc; }
        .stage { color: #ffd93d; font-weight: 500; }
        .worker { color: #6bcb77; }
        .pod { color: #ff9f43; }
        .updated { font-size: 0.8em; color: #666; margin-top: 10px; }
        .refresh-info { color: #666; font-size: 0.9em; margin-top: 20px; }
        .youtube-link { color: #ff0000; text-decoration: none; }
        .youtube-link:hover { text-decoration: underline; }
        .empty { color: #666; font-style: italic; padding: 20px; text-align: center; }
    </style>
</head>
<body>
    <h1>🎾 Tennis Pipeline</h1>

    <div class="stats" id="stats">
        <div class="stat pending"><div class="stat-value" id="pending">-</div><div class="stat-label">Pending</div></div>
        <div class="stat processing"><div class="stat-value" id="processing">-</div><div class="stat-label">Processing</div></div>
        <div class="stat completed"><div class="stat-value" id="completed">-</div><div class="stat-label">Completed</div></div>
        <div class="stat failed"><div class="stat-value" id="failed">-</div><div class="stat-label">Failed</div></div>
    </div>

    <h2>Active Jobs</h2>
    <div class="jobs" id="active-jobs"></div>

    <h2>Recent Jobs</h2>
    <div class="jobs" id="recent-jobs"></div>

    <div class="refresh-info">Auto-refreshes every 5 seconds</div>

    <script>
        const STAGES = ['queued', 'downloading', 'preprocessing', 'prescan', 'poses', 'detection', 'clips', 'slowmo', 'combined', 'uploading', 'done'];

        function stageIndex(stage) {
            const idx = STAGES.indexOf(stage);
            return idx >= 0 ? idx : 0;
        }

        function stageProgress(stage, progress) {
            if (!stage) return 0;
            const idx = stageIndex(stage);
            const stageWeight = 100 / STAGES.length;
            const base = idx * stageWeight;
            const inStage = (progress || 0) / 100 * stageWeight;
            return Math.min(base + inStage, 100);
        }

        function timeAgo(isoString) {
            if (!isoString) return '';
            const date = new Date(isoString);
            const seconds = Math.floor((new Date() - date) / 1000);
            if (seconds < 60) return seconds + 's ago';
            if (seconds < 3600) return Math.floor(seconds / 60) + 'm ago';
            if (seconds < 86400) return Math.floor(seconds / 3600) + 'h ago';
            return Math.floor(seconds / 86400) + 'd ago';
        }

        function renderJob(job) {
            const progress = stageProgress(job.current_stage, job.stage_progress);
            const stageDisplay = job.current_stage ? job.current_stage.toUpperCase() : '';
            const progressText = job.stage_progress ? ` (${job.stage_progress.toFixed(0)}%)` : '';

            let details = '';
            if (job.claimed_by) details += `<div>Worker: <span class="worker">${job.claimed_by}</span></div>`;
            if (job.current_stage) details += `<div>Stage: <span class="stage">${stageDisplay}${progressText}</span></div>`;
            if (job.pod_id) details += `<div>Pod: <span class="pod">${job.pod_id}</span> (${job.pod_status || 'unknown'})</div>`;
            if (job.highlights_url) details += `<div>YouTube: <a class="youtube-link" href="${job.highlights_url}" target="_blank">${job.highlights_url}</a></div>`;
            if (job.error_message) details += `<div style="color: #ff6b6b;">Error: ${job.error_message}</div>`;

            return `
                <div class="job">
                    <div class="job-header">
                        <span class="job-filename">${job.filename}</span>
                        <span class="job-status ${job.status}">${job.status}</span>
                    </div>
                    ${job.status === 'processing' || job.status === 'claimed' ? `
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: ${progress}%"></div>
                        </div>
                    ` : ''}
                    <div class="job-details">${details}</div>
                    <div class="updated">${timeAgo(job.stage_updated_at || job.claimed_at || job.created_at)}</div>
                </div>
            `;
        }

        async function refresh() {
            try {
                // Fetch stats (relative to current origin)
                const base = window.location.origin;
                const statsRes = await fetch(base + '/stats');
                const stats = await statsRes.json();
                document.getElementById('pending').textContent = stats.pending || 0;
                document.getElementById('processing').textContent = (stats.processing || 0) + (stats.claimed || 0);
                document.getElementById('completed').textContent = stats.completed || 0;
                document.getElementById('failed').textContent = stats.failed || 0;

                // Fetch active jobs
                const activeRes = await fetch(base + '/jobs/active');
                const active = await activeRes.json();
                const activeEl = document.getElementById('active-jobs');
                if (active.jobs.length > 0) {
                    activeEl.innerHTML = active.jobs.map(renderJob).join('');
                } else {
                    activeEl.innerHTML = '<div class="empty">No active jobs</div>';
                }

                // Fetch recent jobs
                const recentRes = await fetch(base + '/jobs?limit=10');
                const recent = await recentRes.json();
                const recentEl = document.getElementById('recent-jobs');
                const recentJobs = recent.jobs.filter(j => j.status !== 'processing' && j.status !== 'claimed');
                if (recentJobs.length > 0) {
                    recentEl.innerHTML = recentJobs.slice(0, 5).map(renderJob).join('');
                } else {
                    recentEl.innerHTML = '<div class="empty">No recent jobs</div>';
                }
            } catch (e) {
                console.error('Refresh failed:', e);
            }
        }

        refresh();
        setInterval(refresh, 5000);
    </script>
</body>
</html>
"""


@app.get("/dash", response_class=HTMLResponse)
async def dashboard():
    """Web dashboard for pipeline monitoring."""
    return DASHBOARD_HTML


# ── Video Gallery ────────────────────────────────────────────


VIDEO_GALLERY_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Tennis Videos</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0f0f0f;
            color: #fff;
            min-height: 100vh;
        }
        header {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            padding: 30px 20px;
            text-align: center;
            border-bottom: 1px solid #333;
        }
        h1 {
            color: #00d9ff;
            font-size: 2em;
            margin-bottom: 5px;
        }
        .subtitle { color: #888; font-size: 1em; }
        .nav {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-top: 15px;
        }
        .nav a {
            color: #00d9ff;
            text-decoration: none;
            padding: 8px 16px;
            border: 1px solid #00d9ff;
            border-radius: 20px;
            font-size: 0.9em;
            transition: all 0.2s;
        }
        .nav a:hover {
            background: #00d9ff;
            color: #000;
        }
        main {
            max-width: 1400px;
            margin: 0 auto;
            padding: 30px 20px;
        }
        .stats-bar {
            display: flex;
            gap: 30px;
            justify-content: center;
            margin-bottom: 30px;
            flex-wrap: wrap;
        }
        .stats-bar .stat {
            text-align: center;
        }
        .stats-bar .value {
            font-size: 1.8em;
            font-weight: bold;
            color: #00d9ff;
        }
        .stats-bar .label {
            color: #666;
            font-size: 0.85em;
        }
        .video-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
            gap: 20px;
        }
        .video-card {
            background: #1a1a1a;
            border-radius: 12px;
            overflow: hidden;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .video-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 8px 30px rgba(0, 217, 255, 0.15);
        }
        .video-thumbnail {
            width: 100%;
            aspect-ratio: 16/9;
            background: #000;
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
        }
        .video-thumbnail img {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
        .play-icon {
            position: absolute;
            width: 60px;
            height: 60px;
            background: rgba(255, 0, 0, 0.9);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .play-icon::after {
            content: '';
            border-left: 20px solid white;
            border-top: 12px solid transparent;
            border-bottom: 12px solid transparent;
            margin-left: 5px;
        }
        .video-info {
            padding: 15px;
        }
        .video-title {
            font-weight: 600;
            font-size: 1em;
            margin-bottom: 8px;
            color: #fff;
        }
        .video-meta {
            display: flex;
            justify-content: space-between;
            color: #888;
            font-size: 0.85em;
        }
        .video-card a {
            text-decoration: none;
            color: inherit;
            display: block;
        }
        .empty-state {
            text-align: center;
            padding: 60px 20px;
            color: #666;
        }
        .empty-state h2 { color: #888; margin-bottom: 10px; }
        .processing-badge {
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(255, 217, 61, 0.9);
            color: #000;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 0.75em;
            font-weight: 600;
        }
    </style>
</head>
<body>
    <header>
        <h1>Tennis Videos</h1>
        <p class="subtitle">Auto-generated highlight reels from practice sessions</p>
        <nav class="nav">
            <a href="/dash">Pipeline Dashboard</a>
        </nav>
    </header>

    <main>
        <div class="stats-bar" id="stats"></div>
        <div class="video-grid" id="videos"></div>
    </main>

    <script>
        function formatDate(isoString) {
            if (!isoString) return '';
            const date = new Date(isoString);
            return date.toLocaleDateString('en-US', {
                month: 'short',
                day: 'numeric',
                year: 'numeric'
            });
        }

        function getBasename(filename) {
            // Remove extension from filename
            return filename ? filename.replace(/\\.[^.]+$/, '') : '';
        }

        function renderVideo(job) {
            const basename = getBasename(job.filename);
            // Use local proxy routes for thumbnails and highlights
            const thumbnail = `/thumbs/${basename}.jpg`;
            const hasHighlight = job.highlights_url && job.status === 'completed';
            const link = hasHighlight ? `/${job.highlights_url}` : '#';
            const target = hasHighlight ? '_blank' : '';

            return `
                <div class="video-card">
                    <a href="${link}" target="${target}">
                        <div class="video-thumbnail">
                            <img src="${thumbnail}" alt="${job.filename}" onerror="this.style.display='none'; this.nextElementSibling.style.display='flex';">
                            <div style="display:none; color: #444; font-size: 3em; justify-content: center; align-items: center; height: 100%;">🎾</div>
                            ${hasHighlight ? '<div class="play-icon"></div>' : ''}
                            ${job.status === 'processing' || job.status === 'claimed'
                                ? '<div class="processing-badge">Processing...</div>'
                                : ''
                            }
                        </div>
                        <div class="video-info">
                            <div class="video-title">${job.filename.replace(/\\.[^.]+$/, '')}</div>
                            <div class="video-meta">
                                <span>${formatDate(job.completed_at || job.created_at)}</span>
                                <span>${job.album_name || ''}</span>
                            </div>
                        </div>
                    </a>
                </div>
            `;
        }

        async function loadVideos() {
            try {
                // Load stats
                const base = window.location.origin;
                const statsRes = await fetch(base + '/stats');
                const stats = await statsRes.json();
                document.getElementById('stats').innerHTML = `
                    <div class="stat"><div class="value">${stats.completed || 0}</div><div class="label">Videos</div></div>
                    <div class="stat"><div class="value">${(stats.processing || 0) + (stats.claimed || 0)}</div><div class="label">Processing</div></div>
                    <div class="stat"><div class="value">${stats.pending || 0}</div><div class="label">Queued</div></div>
                `;

                // Load videos
                const res = await fetch(base + '/jobs?limit=50');
                const data = await res.json();

                // Sort: processing first, then completed by date
                const videos = data.jobs.sort((a, b) => {
                    if (a.status === 'processing' || a.status === 'claimed') return -1;
                    if (b.status === 'processing' || b.status === 'claimed') return 1;
                    return 0;
                });

                const videosEl = document.getElementById('videos');
                if (videos.length > 0) {
                    videosEl.innerHTML = videos.map(renderVideo).join('');
                } else {
                    videosEl.innerHTML = `
                        <div class="empty-state">
                            <h2>No videos yet</h2>
                            <p>Add videos to your iCloud album to start processing</p>
                        </div>
                    `;
                }
            } catch (e) {
                console.error('Failed to load videos:', e);
            }
        }

        loadVideos();
        setInterval(loadVideos, 30000); // Refresh every 30s
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def video_gallery():
    """Video gallery showing completed tennis videos."""
    return VIDEO_GALLERY_HTML


# Run with: uvicorn coordinator.api:app --host 0.0.0.0 --port 8080
