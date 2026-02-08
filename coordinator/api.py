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
from pydantic import BaseModel

from .state import StateBackend, VideoJob, VideoStatus
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
    youtube_url: Optional[str] = None
    error_message: Optional[str] = None
    success: bool = True


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
            youtube_url=request.youtube_url,
        )
    else:
        await state.update_status(
            video_id,
            VideoStatus.FAILED,
            error_message=request.error_message,
        )

    return {"status": "ok"}


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
        "youtube_url": job.youtube_url,
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
                "youtube_url": j.youtube_url,
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


# Run with: uvicorn coordinator.api:app --host 0.0.0.0 --port 8080
