"""Abstract state interface for coordinator.

Implementations:
- SQLite for Raspberry Pi / local deployment
- DynamoDB for AWS deployment
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional


class VideoStatus(str, Enum):
    PENDING = "pending"
    CLAIMED = "claimed"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class VideoJob:
    """Represents a video processing job."""
    video_id: str
    icloud_asset_id: str
    filename: str
    status: VideoStatus
    claimed_by: Optional[str] = None  # hostname of GPU machine
    claimed_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    youtube_url: Optional[str] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    album_name: Optional[str] = None  # "Tennis Videos" or "Tennis Videos Group By Shot Type"
    created_at: Optional[datetime] = None


class StateBackend(ABC):
    """Abstract interface for job state storage."""

    @abstractmethod
    async def init(self) -> None:
        """Initialize the storage backend (create tables, etc.)."""
        pass

    @abstractmethod
    async def add_job(self, job: VideoJob) -> None:
        """Add a new video job to the queue."""
        pass

    @abstractmethod
    async def get_job(self, video_id: str) -> Optional[VideoJob]:
        """Get a job by its video_id."""
        pass

    @abstractmethod
    async def get_pending_jobs(self) -> list[VideoJob]:
        """Get all jobs with status PENDING."""
        pass

    @abstractmethod
    async def claim_job(self, video_id: str, worker_id: str) -> bool:
        """
        Atomically claim a job for a worker.
        Returns True if claim succeeded, False if already claimed.
        """
        pass

    @abstractmethod
    async def update_status(
        self,
        video_id: str,
        status: VideoStatus,
        youtube_url: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """Update job status and optional metadata."""
        pass

    @abstractmethod
    async def release_stale_claims(self, max_age_seconds: int = 3600) -> int:
        """
        Release jobs that have been claimed for too long without completing.
        Returns number of jobs released.
        """
        pass

    @abstractmethod
    async def list_jobs(
        self,
        status: Optional[VideoStatus] = None,
        limit: int = 100,
    ) -> list[VideoJob]:
        """List jobs, optionally filtered by status."""
        pass

    @abstractmethod
    async def job_exists(self, icloud_asset_id: str) -> bool:
        """Check if a job with this iCloud asset ID already exists."""
        pass

    @abstractmethod
    async def get_stats(self) -> dict:
        """Get job statistics (counts by status)."""
        pass
