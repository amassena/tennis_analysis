"""Cloudflare R2 storage client for video files.

Uses boto3 with S3-compatible API. Zero egress fees make R2 ideal for
video storage where files are frequently downloaded to GPU workers.

Usage:
    from storage import R2Client

    client = R2Client()

    # Upload a video
    client.upload("raw/video.mov", "raw/video.mov")

    # Download to worker
    client.download("raw/video.mov", "/tmp/video.mov")

    # Generate presigned URL for RunPod
    url = client.presign("raw/video.mov", expires_in=3600)
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import CLOUD


class R2Client:
    """Cloudflare R2 storage client using boto3."""

    def __init__(self, bucket_name: str = None):
        """Initialize R2 client.

        Args:
            bucket_name: Override bucket from settings
        """
        self.config = CLOUD["r2"]
        self.bucket_name = bucket_name or self.config["bucket_name"]
        self._client = None

    @property
    def client(self):
        """Lazy-load boto3 client."""
        if self._client is None:
            try:
                import boto3
                from botocore.config import Config
            except ImportError:
                raise ImportError(
                    "boto3 required for R2 storage. Install with: pip install boto3"
                )

            self._client = boto3.client(
                "s3",
                endpoint_url=self.config["endpoint_url"],
                aws_access_key_id=self.config["access_key_id"],
                aws_secret_access_key=self.config["secret_access_key"],
                config=Config(
                    signature_version="s3v4",
                    retries={"max_attempts": 3, "mode": "adaptive"},
                ),
            )
        return self._client

    def upload(
        self,
        local_path: str,
        remote_key: str,
        content_type: str = None,
        metadata: dict = None,
    ) -> str:
        """Upload a file to R2.

        Uses multipart upload for files larger than threshold.

        Args:
            local_path: Local file path
            remote_key: S3 key (path in bucket)
            content_type: MIME type (auto-detected if None)
            metadata: Optional metadata dict

        Returns:
            The remote key
        """
        local_path = Path(local_path)
        file_size = local_path.stat().st_size
        threshold = self.config["multipart_threshold_mb"] * 1024 * 1024

        # Auto-detect content type
        if content_type is None:
            ext = local_path.suffix.lower()
            content_types = {
                ".mov": "video/quicktime",
                ".mp4": "video/mp4",
                ".json": "application/json",
                ".csv": "text/csv",
                ".h5": "application/x-hdf5",
            }
            content_type = content_types.get(ext, "application/octet-stream")

        extra_args = {"ContentType": content_type}
        if metadata:
            extra_args["Metadata"] = metadata

        if file_size > threshold:
            self._multipart_upload(local_path, remote_key, extra_args)
        else:
            self.client.upload_file(
                str(local_path),
                self.bucket_name,
                remote_key,
                ExtraArgs=extra_args,
            )

        print(f"Uploaded {local_path.name} -> r2://{self.bucket_name}/{remote_key}")
        return remote_key

    def _multipart_upload(self, local_path: Path, remote_key: str, extra_args: dict):
        """Upload large file using multipart."""
        from boto3.s3.transfer import TransferConfig

        chunk_size = self.config["multipart_chunk_mb"] * 1024 * 1024
        config = TransferConfig(
            multipart_threshold=chunk_size,
            multipart_chunksize=chunk_size,
            max_concurrency=4,
            use_threads=True,
        )

        self.client.upload_file(
            str(local_path),
            self.bucket_name,
            remote_key,
            ExtraArgs=extra_args,
            Config=config,
        )

    def download(self, remote_key: str, local_path: str) -> Path:
        """Download a file from R2.

        Args:
            remote_key: S3 key (path in bucket)
            local_path: Local destination path

        Returns:
            Path to downloaded file
        """
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)

        self.client.download_file(self.bucket_name, remote_key, str(local_path))
        print(f"Downloaded r2://{self.bucket_name}/{remote_key} -> {local_path}")
        return local_path

    def presign(self, remote_key: str, expires_in: int = 3600) -> str:
        """Generate a presigned URL for temporary access.

        Useful for giving RunPod workers direct access to files.

        Args:
            remote_key: S3 key
            expires_in: URL validity in seconds (default 1 hour)

        Returns:
            Presigned URL string
        """
        return self.client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket_name, "Key": remote_key},
            ExpiresIn=expires_in,
        )

    def delete(self, remote_key: str) -> bool:
        """Delete a file from R2.

        Args:
            remote_key: S3 key to delete

        Returns:
            True if deleted
        """
        self.client.delete_object(Bucket=self.bucket_name, Key=remote_key)
        print(f"Deleted r2://{self.bucket_name}/{remote_key}")
        return True

    def list(self, prefix: str = "", max_keys: int = 10000) -> list:
        """List objects with optional prefix. Paginates automatically.

        Args:
            prefix: Filter by key prefix (e.g., "raw/")
            max_keys: Maximum total objects to return

        Returns:
            List of object keys
        """
        all_keys = []
        continuation = None
        while len(all_keys) < max_keys:
            kwargs = {
                "Bucket": self.bucket_name,
                "Prefix": prefix,
                "MaxKeys": min(1000, max_keys - len(all_keys)),
            }
            if continuation:
                kwargs["ContinuationToken"] = continuation
            response = self.client.list_objects_v2(**kwargs)
            all_keys.extend(obj["Key"] for obj in response.get("Contents", []))
            if not response.get("IsTruncated"):
                break
            continuation = response.get("NextContinuationToken")
        return all_keys

    def exists(self, remote_key: str) -> bool:
        """Check if an object exists.

        Args:
            remote_key: S3 key to check

        Returns:
            True if exists
        """
        try:
            self.client.head_object(Bucket=self.bucket_name, Key=remote_key)
            return True
        except self.client.exceptions.ClientError:
            return False

    def ensure_bucket(self):
        """Create bucket if it doesn't exist."""
        try:
            self.client.head_bucket(Bucket=self.bucket_name)
        except self.client.exceptions.ClientError:
            self.client.create_bucket(Bucket=self.bucket_name)
            print(f"Created bucket: {self.bucket_name}")

    def set_lifecycle_rules(self):
        """Apply lifecycle rules from config to auto-delete old files."""
        rules = []
        lifecycle_days = self.config["lifecycle_days"]

        for prefix, days in lifecycle_days.items():
            if days > 0:  # 0 means keep forever
                rules.append({
                    "ID": f"expire-{prefix}",
                    "Filter": {"Prefix": f"{prefix}/"},
                    "Status": "Enabled",
                    "Expiration": {"Days": days},
                })

        if rules:
            self.client.put_bucket_lifecycle_configuration(
                Bucket=self.bucket_name,
                LifecycleConfiguration={"Rules": rules},
            )
            print(f"Applied {len(rules)} lifecycle rules to {self.bucket_name}")


def main():
    """CLI for R2 operations."""
    import argparse

    parser = argparse.ArgumentParser(description="R2 storage operations")
    parser.add_argument("action", choices=["upload", "download", "list", "delete", "setup"])
    parser.add_argument("--local", help="Local file path")
    parser.add_argument("--remote", help="Remote key (path in bucket)")
    parser.add_argument("--prefix", default="", help="Prefix for list operation")
    args = parser.parse_args()

    client = R2Client()

    if args.action == "upload":
        if not args.local or not args.remote:
            parser.error("upload requires --local and --remote")
        client.upload(args.local, args.remote)

    elif args.action == "download":
        if not args.local or not args.remote:
            parser.error("download requires --local and --remote")
        client.download(args.remote, args.local)

    elif args.action == "list":
        for key in client.list(prefix=args.prefix):
            print(key)

    elif args.action == "delete":
        if not args.remote:
            parser.error("delete requires --remote")
        client.delete(args.remote)

    elif args.action == "setup":
        client.ensure_bucket()
        client.set_lifecycle_rules()
        print("R2 bucket configured")


if __name__ == "__main__":
    main()
