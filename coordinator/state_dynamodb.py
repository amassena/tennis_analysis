"""DynamoDB implementation of state backend for AWS deployment.

This is a stub - implement when deploying to AWS.
Uses boto3 for DynamoDB access.

Table schema:
- PK: video_id
- GSI1: status-created_at (for listing by status)
- GSI2: icloud_asset_id (for deduplication)
"""

from datetime import datetime, timezone
from typing import Optional

from .state import StateBackend, VideoJob, VideoStatus


class DynamoDBStateBackend(StateBackend):
    """DynamoDB-based state storage for AWS deployment.

    Environment variables:
    - AWS_REGION: AWS region (default: us-west-2)
    - DYNAMODB_TABLE: Table name (default: tennis-pipeline-jobs)
    """

    def __init__(self, table_name: str = None, region: str = None):
        import os

        self.table_name = table_name or os.environ.get("DYNAMODB_TABLE", "tennis-pipeline-jobs")
        self.region = region or os.environ.get("AWS_REGION", "us-west-2")
        self._table = None

    async def init(self) -> None:
        """Initialize DynamoDB client."""
        import boto3

        dynamodb = boto3.resource("dynamodb", region_name=self.region)
        self._table = dynamodb.Table(self.table_name)

        # Note: Table creation should be done via CloudFormation/Terraform
        # This just connects to an existing table

    async def close(self) -> None:
        """No cleanup needed for DynamoDB."""
        pass

    def _item_to_job(self, item: dict) -> VideoJob:
        """Convert DynamoDB item to VideoJob."""
        return VideoJob(
            video_id=item["video_id"],
            icloud_asset_id=item["icloud_asset_id"],
            filename=item["filename"],
            status=VideoStatus(item["status"]),
            claimed_by=item.get("claimed_by"),
            claimed_at=datetime.fromisoformat(item["claimed_at"]) if item.get("claimed_at") else None,
            completed_at=datetime.fromisoformat(item["completed_at"]) if item.get("completed_at") else None,
            youtube_url=item.get("youtube_url"),
            error_message=item.get("error_message"),
            retry_count=item.get("retry_count", 0),
            album_name=item.get("album_name"),
            created_at=datetime.fromisoformat(item["created_at"]) if item.get("created_at") else None,
        )

    async def add_job(self, job: VideoJob) -> None:
        """Add a new job."""
        now = datetime.now(timezone.utc).isoformat()

        self._table.put_item(
            Item={
                "video_id": job.video_id,
                "icloud_asset_id": job.icloud_asset_id,
                "filename": job.filename,
                "status": job.status.value,
                "album_name": job.album_name,
                "created_at": now,
                "retry_count": 0,
            },
            ConditionExpression="attribute_not_exists(video_id)",
        )

    async def get_job(self, video_id: str) -> Optional[VideoJob]:
        """Get job by ID."""
        response = self._table.get_item(Key={"video_id": video_id})
        item = response.get("Item")
        return self._item_to_job(item) if item else None

    async def get_pending_jobs(self) -> list[VideoJob]:
        """Get all pending jobs using GSI."""
        from boto3.dynamodb.conditions import Key

        response = self._table.query(
            IndexName="status-created_at-index",
            KeyConditionExpression=Key("status").eq(VideoStatus.PENDING.value),
        )

        return [self._item_to_job(item) for item in response.get("Items", [])]

    async def claim_job(self, video_id: str, worker_id: str) -> bool:
        """Atomically claim a job using conditional update."""
        from botocore.exceptions import ClientError

        now = datetime.now(timezone.utc).isoformat()

        try:
            self._table.update_item(
                Key={"video_id": video_id},
                UpdateExpression="SET #status = :claimed, claimed_by = :worker, claimed_at = :now",
                ConditionExpression="#status = :pending",
                ExpressionAttributeNames={"#status": "status"},
                ExpressionAttributeValues={
                    ":claimed": VideoStatus.CLAIMED.value,
                    ":pending": VideoStatus.PENDING.value,
                    ":worker": worker_id,
                    ":now": now,
                },
            )
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
                return False
            raise

    async def update_status(
        self,
        video_id: str,
        status: VideoStatus,
        youtube_url: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """Update job status."""
        update_expr = "SET #status = :status"
        expr_names = {"#status": "status"}
        expr_values = {":status": status.value}

        if status == VideoStatus.COMPLETED:
            update_expr += ", completed_at = :completed"
            expr_values[":completed"] = datetime.now(timezone.utc).isoformat()

        if youtube_url is not None:
            update_expr += ", youtube_url = :url"
            expr_values[":url"] = youtube_url

        if error_message is not None:
            update_expr += ", error_message = :error"
            expr_values[":error"] = error_message

        if status == VideoStatus.FAILED:
            update_expr += " ADD retry_count :inc"
            expr_values[":inc"] = 1

        self._table.update_item(
            Key={"video_id": video_id},
            UpdateExpression=update_expr,
            ExpressionAttributeNames=expr_names,
            ExpressionAttributeValues=expr_values,
        )

    async def release_stale_claims(self, max_age_seconds: int = 3600) -> int:
        """Release stale claims - scan and update."""
        from boto3.dynamodb.conditions import Key, Attr

        cutoff = datetime.now(timezone.utc)
        released = 0

        # Query claimed jobs
        for status in [VideoStatus.CLAIMED.value, VideoStatus.PROCESSING.value]:
            response = self._table.query(
                IndexName="status-created_at-index",
                KeyConditionExpression=Key("status").eq(status),
            )

            for item in response.get("Items", []):
                if not item.get("claimed_at"):
                    continue

                claimed_at = datetime.fromisoformat(item["claimed_at"])
                age = (cutoff - claimed_at).total_seconds()

                if age > max_age_seconds:
                    self._table.update_item(
                        Key={"video_id": item["video_id"]},
                        UpdateExpression="SET #status = :pending REMOVE claimed_by, claimed_at",
                        ExpressionAttributeNames={"#status": "status"},
                        ExpressionAttributeValues={":pending": VideoStatus.PENDING.value},
                    )
                    released += 1

        return released

    async def list_jobs(
        self,
        status: Optional[VideoStatus] = None,
        limit: int = 100,
    ) -> list[VideoJob]:
        """List jobs with optional status filter."""
        from boto3.dynamodb.conditions import Key

        if status:
            response = self._table.query(
                IndexName="status-created_at-index",
                KeyConditionExpression=Key("status").eq(status.value),
                Limit=limit,
                ScanIndexForward=False,  # Most recent first
            )
        else:
            response = self._table.scan(Limit=limit)

        return [self._item_to_job(item) for item in response.get("Items", [])]

    async def job_exists(self, icloud_asset_id: str) -> bool:
        """Check if job exists using GSI."""
        from boto3.dynamodb.conditions import Key

        response = self._table.query(
            IndexName="icloud_asset_id-index",
            KeyConditionExpression=Key("icloud_asset_id").eq(icloud_asset_id),
            Limit=1,
        )

        return len(response.get("Items", [])) > 0

    async def get_stats(self) -> dict:
        """Get job statistics - expensive scan operation."""
        response = self._table.scan(
            Select="COUNT",
        )

        # This is inefficient - for production, use CloudWatch metrics
        # or maintain counters in a separate item
        stats = {"total": response.get("Count", 0)}

        for status in VideoStatus:
            from boto3.dynamodb.conditions import Key

            response = self._table.query(
                IndexName="status-created_at-index",
                KeyConditionExpression=Key("status").eq(status.value),
                Select="COUNT",
            )
            stats[status.value] = response.get("Count", 0)

        return stats
