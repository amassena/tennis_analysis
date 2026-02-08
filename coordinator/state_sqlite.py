"""SQLite implementation of state backend for local/Pi deployment."""

import aiosqlite
import os
from datetime import datetime, timezone
from typing import Optional

from .state import StateBackend, VideoJob, VideoStatus


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
                youtube_url TEXT,
                error_message TEXT,
                retry_count INTEGER DEFAULT 0,
                album_name TEXT,
                created_at TEXT NOT NULL
            )
        """)

        await self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)
        """)

        await self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_jobs_icloud ON jobs(icloud_asset_id)
        """)

        await self._db.commit()

    async def close(self) -> None:
        """Close database connection."""
        if self._db:
            await self._db.close()

    def _row_to_job(self, row: aiosqlite.Row) -> VideoJob:
        """Convert a database row to a VideoJob."""
        return VideoJob(
            video_id=row["video_id"],
            icloud_asset_id=row["icloud_asset_id"],
            filename=row["filename"],
            status=VideoStatus(row["status"]),
            claimed_by=row["claimed_by"],
            claimed_at=datetime.fromisoformat(row["claimed_at"]) if row["claimed_at"] else None,
            completed_at=datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None,
            youtube_url=row["youtube_url"],
            error_message=row["error_message"],
            retry_count=row["retry_count"],
            album_name=row["album_name"],
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
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

    async def update_status(
        self,
        video_id: str,
        status: VideoStatus,
        youtube_url: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """Update job status."""
        updates = ["status = ?"]
        params = [status.value]

        if status == VideoStatus.COMPLETED:
            updates.append("completed_at = ?")
            params.append(datetime.now(timezone.utc).isoformat())

        if youtube_url is not None:
            updates.append("youtube_url = ?")
            params.append(youtube_url)

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
