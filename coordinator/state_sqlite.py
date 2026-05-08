"""SQLite implementation of state backend for local/Pi deployment."""

import aiosqlite
import os
from datetime import datetime, timezone
from typing import Optional

from .state import StateBackend, VideoJob, VideoStatus, ProcessingStage


class SQLiteStateBackend(StateBackend):
    """SQLite-based state storage for Raspberry Pi or local deployment."""

    def __init__(self, db_path: str = "coordinator.db"):
        self.db_path = db_path
        self._db: Optional[aiosqlite.Connection] = None

    async def init(self) -> None:
        """Create database and tables if they don't exist."""
        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row

        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                video_id TEXT PRIMARY KEY,
                icloud_asset_id TEXT UNIQUE NOT NULL,
                filename TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                claimed_by TEXT,
                claimed_at TEXT,
                completed_at TEXT,
                highlights_url TEXT,
                error_message TEXT,
                retry_count INTEGER DEFAULT 0,
                album_name TEXT,
                created_at TEXT NOT NULL,
                current_stage TEXT,
                stage_progress REAL,
                stage_message TEXT,
                stage_updated_at TEXT,
                pod_id TEXT,
                pod_status TEXT
            )
        """)

        await self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)
        """)

        await self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_jobs_icloud ON jobs(icloud_asset_id)
        """)

        # Migration: add new columns if they don't exist
        try:
            await self._db.execute("ALTER TABLE jobs ADD COLUMN current_stage TEXT")
        except:
            pass
        try:
            await self._db.execute("ALTER TABLE jobs ADD COLUMN stage_progress REAL")
        except:
            pass
        try:
            await self._db.execute("ALTER TABLE jobs ADD COLUMN stage_message TEXT")
        except:
            pass
        try:
            await self._db.execute("ALTER TABLE jobs ADD COLUMN stage_updated_at TEXT")
        except:
            pass
        try:
            await self._db.execute("ALTER TABLE jobs ADD COLUMN pod_id TEXT")
        except:
            pass
        try:
            await self._db.execute("ALTER TABLE jobs ADD COLUMN pod_status TEXT")
        except:
            pass

        await self._db.commit()

    async def close(self) -> None:
        """Close database connection."""
        if self._db:
            await self._db.close()

    def _row_to_job(self, row: aiosqlite.Row) -> VideoJob:
        """Convert a database row to a VideoJob."""
        # Handle optional columns that may not exist in older databases
        row_dict = dict(row)
        return VideoJob(
            video_id=row["video_id"],
            icloud_asset_id=row["icloud_asset_id"],
            filename=row["filename"],
            status=VideoStatus(row["status"]),
            claimed_by=row["claimed_by"],
            claimed_at=datetime.fromisoformat(row["claimed_at"]) if row["claimed_at"] else None,
            completed_at=datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None,
            highlights_url=row["highlights_url"],
            error_message=row["error_message"],
            retry_count=row["retry_count"],
            album_name=row["album_name"],
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
            current_stage=ProcessingStage(row_dict["current_stage"]) if row_dict.get("current_stage") else None,
            stage_progress=row_dict.get("stage_progress"),
            stage_message=row_dict.get("stage_message"),
            stage_updated_at=datetime.fromisoformat(row_dict["stage_updated_at"]) if row_dict.get("stage_updated_at") else None,
            pod_id=row_dict.get("pod_id"),
            pod_status=row_dict.get("pod_status"),
        )

    async def add_job(self, job: VideoJob) -> None:
        """Add a new video job."""
        now = datetime.now(timezone.utc).isoformat()
        await self._db.execute(
            """
            INSERT INTO jobs (video_id, icloud_asset_id, filename, status, album_name, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (job.video_id, job.icloud_asset_id, job.filename, job.status.value, job.album_name, now),
        )
        await self._db.commit()

    async def get_job(self, video_id: str) -> Optional[VideoJob]:
        """Get a job by video_id."""
        async with self._db.execute(
            "SELECT * FROM jobs WHERE video_id = ?", (video_id,)
        ) as cursor:
            row = await cursor.fetchone()
            return self._row_to_job(row) if row else None

    async def get_pending_jobs(self) -> list[VideoJob]:
        """Get all pending jobs."""
        async with self._db.execute(
            "SELECT * FROM jobs WHERE status = ? ORDER BY created_at ASC",
            (VideoStatus.PENDING.value,),
        ) as cursor:
            rows = await cursor.fetchall()
            return [self._row_to_job(row) for row in rows]

    async def claim_job(self, video_id: str, worker_id: str) -> bool:
        """Atomically claim a job. Returns True if successful."""
        now = datetime.now(timezone.utc).isoformat()

        # SQLite doesn't have row-level locking, but single-writer ensures atomicity
        cursor = await self._db.execute(
            """
            UPDATE jobs
            SET status = ?, claimed_by = ?, claimed_at = ?
            WHERE video_id = ? AND status = ?
            """,
            (VideoStatus.CLAIMED.value, worker_id, now, video_id, VideoStatus.PENDING.value),
        )
        await self._db.commit()

        return cursor.rowcount > 0

    async def unclaim_job(self, video_id: str) -> None:
        """Clear claimed_by and claimed_at for a job."""
        await self._db.execute(
            """
            UPDATE jobs
            SET claimed_by = NULL, claimed_at = NULL
            WHERE video_id = ?
            """,
            (video_id,),
        )
        await self._db.commit()

    async def update_status(
        self,
        video_id: str,
        status: VideoStatus,
        highlights_url: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """Update job status."""
        updates = ["status = ?"]
        params = [status.value]

        if status == VideoStatus.COMPLETED:
            updates.append("completed_at = ?")
            params.append(datetime.now(timezone.utc).isoformat())

        if highlights_url is not None:
            updates.append("highlights_url = ?")
            params.append(highlights_url)

        if error_message is not None:
            updates.append("error_message = ?")
            params.append(error_message)

        if status == VideoStatus.FAILED:
            updates.append("retry_count = retry_count + 1")

        params.append(video_id)

        await self._db.execute(
            f"UPDATE jobs SET {', '.join(updates)} WHERE video_id = ?",
            params,
        )
        await self._db.commit()

    async def release_stale_claims(self, max_age_seconds: int = 3600) -> int:
        """Release jobs claimed more than max_age_seconds ago."""
        cutoff = datetime.now(timezone.utc)

        cursor = await self._db.execute(
            """
            UPDATE jobs
            SET status = ?, claimed_by = NULL, claimed_at = NULL
            WHERE status IN (?, ?)
            AND claimed_at IS NOT NULL
            AND (julianday('now') - julianday(claimed_at)) * 86400 > ?
            """,
            (
                VideoStatus.PENDING.value,
                VideoStatus.CLAIMED.value,
                VideoStatus.PROCESSING.value,
                max_age_seconds,
            ),
        )
        await self._db.commit()

        return cursor.rowcount

    async def list_jobs(
        self,
        status: Optional[VideoStatus] = None,
        limit: int = 100,
    ) -> list[VideoJob]:
        """List jobs with optional status filter."""
        if status:
            async with self._db.execute(
                "SELECT * FROM jobs WHERE status = ? ORDER BY created_at DESC LIMIT ?",
                (status.value, limit),
            ) as cursor:
                rows = await cursor.fetchall()
        else:
            async with self._db.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ) as cursor:
                rows = await cursor.fetchall()

        return [self._row_to_job(row) for row in rows]

    async def job_exists(self, icloud_asset_id: str) -> bool:
        """Check if job with this iCloud asset already exists."""
        async with self._db.execute(
            "SELECT 1 FROM jobs WHERE icloud_asset_id = ?", (icloud_asset_id,)
        ) as cursor:
            row = await cursor.fetchone()
            return row is not None

    async def get_job_by_asset_id(self, icloud_asset_id: str) -> Optional[VideoJob]:
        """Look up a job by iCloud asset id (or iOS PHAsset id)."""
        async with self._db.execute(
            "SELECT * FROM jobs WHERE icloud_asset_id = ? LIMIT 1",
            (icloud_asset_id,),
        ) as cursor:
            row = await cursor.fetchone()
            return self._row_to_job(row) if row else None

    async def get_stats(self) -> dict:
        """Get job statistics."""
        stats = {}
        async with self._db.execute(
            "SELECT status, COUNT(*) as count FROM jobs GROUP BY status"
        ) as cursor:
            rows = await cursor.fetchall()
            for row in rows:
                stats[row["status"]] = row["count"]

        async with self._db.execute("SELECT COUNT(*) as total FROM jobs") as cursor:
            row = await cursor.fetchone()
            stats["total"] = row["total"]

        return stats

    async def update_stage(
        self,
        video_id: str,
        stage: ProcessingStage,
        progress: Optional[float] = None,
        message: Optional[str] = None,
    ) -> None:
        """Update the current processing stage and progress."""
        now = datetime.now(timezone.utc).isoformat()
        await self._db.execute(
            """
            UPDATE jobs
            SET current_stage = ?, stage_progress = ?, stage_message = ?, stage_updated_at = ?
            WHERE video_id = ?
            """,
            (stage.value, progress, message, now, video_id),
        )
        await self._db.commit()

    async def update_pod_info(
        self,
        video_id: str,
        pod_id: Optional[str] = None,
        pod_status: Optional[str] = None,
    ) -> None:
        """Update RunPod pod information for a job."""
        updates = []
        params = []

        if pod_id is not None:
            updates.append("pod_id = ?")
            params.append(pod_id)

        if pod_status is not None:
            updates.append("pod_status = ?")
            params.append(pod_status)

        if updates:
            params.append(video_id)
            await self._db.execute(
                f"UPDATE jobs SET {', '.join(updates)} WHERE video_id = ?",
                params,
            )
            await self._db.commit()

    async def get_active_jobs(self) -> list[VideoJob]:
        """Get all jobs that are currently being processed."""
        async with self._db.execute(
            """
            SELECT * FROM jobs
            WHERE status IN (?, ?)
            ORDER BY claimed_at DESC
            """,
            (VideoStatus.CLAIMED.value, VideoStatus.PROCESSING.value),
        ) as cursor:
            rows = await cursor.fetchall()
            return [self._row_to_job(row) for row in rows]
