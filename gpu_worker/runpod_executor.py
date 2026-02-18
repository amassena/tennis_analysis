"""RunPod GPU executor for burst processing.

Manages pod lifecycle via RunPod's GraphQL API:
- Create pods with specified GPU type
- Execute pipeline commands
- Monitor job status
- Terminate pods when done

Usage:
    from gpu_worker.runpod_executor import RunPodExecutor

    executor = RunPodExecutor()

    # Process a video
    pod_id = executor.create_pod()
    executor.run_pipeline(pod_id, "video.mov")
    executor.terminate_pod(pod_id)
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import CLOUD

RUNPOD_API_URL = "https://api.runpod.io/graphql"


class RunPodExecutor:
    """Execute GPU jobs on RunPod infrastructure."""

    def __init__(self, api_key: str = None):
        """Initialize RunPod executor.

        Args:
            api_key: Override API key from settings
        """
        self.config = CLOUD["runpod"]
        self.api_key = api_key or self.config["api_key"]
        if not self.api_key:
            raise ValueError(
                "RUNPOD_API_KEY not set. Get one at https://runpod.io/console/user/settings"
            )

    def _graphql(self, query: str, variables: dict = None) -> dict:
        """Execute GraphQL query against RunPod API."""
        try:
            import requests
        except ImportError:
            raise ImportError("requests required. Install with: pip install requests")

        response = requests.post(
            RUNPOD_API_URL,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            json={"query": query, "variables": variables or {}},
            timeout=30,
        )
        response.raise_for_status()

        data = response.json()
        if "errors" in data:
            raise RuntimeError(f"RunPod API error: {data['errors']}")
        return data["data"]

    def list_gpus(self) -> list:
        """List available GPU types and their pricing."""
        query = """
        query {
            gpuTypes {
                id
                displayName
                memoryInGb
                secureCloud
                communityCloud
                lowestPrice(input: { gpuCount: 1 }) {
                    minimumBidPrice
                    uninterruptablePrice
                }
            }
        }
        """
        data = self._graphql(query)
        return data["gpuTypes"]

    def create_pod(
        self,
        name: str = "tennis-pipeline",
        gpu_type: str = None,
        volume_size_gb: int = None,
    ) -> str:
        """Create a new GPU pod.

        Args:
            name: Pod name
            gpu_type: GPU type ID (default from settings)
            volume_size_gb: Storage volume size

        Returns:
            Pod ID
        """
        gpu_type = gpu_type or self.config["gpu_type"]
        volume_size = volume_size_gb or self.config["volume_size_gb"]
        cloud_type = self.config["cloud_type"]

        query = """
        mutation createPod($input: PodFindAndDeployOnDemandInput!) {
            podFindAndDeployOnDemand(input: $input) {
                id
                name
                desiredStatus
                imageName
                machineId
                machine {
                    gpuDisplayName
                }
            }
        }
        """

        variables = {
            "input": {
                "name": name,
                "imageName": self.config["docker_image"],
                "gpuTypeId": gpu_type,
                "cloudType": cloud_type,
                "volumeInGb": volume_size,
                "containerDiskInGb": 20,
                "minVcpuCount": 4,
                "minMemoryInGb": 16,
                "gpuCount": 1,
                "volumeMountPath": "/workspace",
                "startSsh": True,
                "env": [
                    {"key": "PYTHONUNBUFFERED", "value": "1"},
                ],
            }
        }

        data = self._graphql(query, variables)
        pod = data["podFindAndDeployOnDemand"]
        print(f"Created pod {pod['id']} ({pod['machine']['gpuDisplayName']})")
        return pod["id"]

    def get_pod(self, pod_id: str) -> dict:
        """Get pod status and details."""
        query = """
        query getPod($podId: String!) {
            pod(input: { podId: $podId }) {
                id
                name
                desiredStatus
                lastStatusChange
                imageName
                machineId
                machine {
                    gpuDisplayName
                }
                runtime {
                    uptimeInSeconds
                    ports {
                        ip
                        isIpPublic
                        privatePort
                        publicPort
                        type
                    }
                    gpus {
                        id
                        gpuUtilPercent
                        memoryUtilPercent
                    }
                }
            }
        }
        """
        data = self._graphql(query, {"podId": pod_id})
        return data["pod"]

    def wait_for_ready(self, pod_id: str, timeout: int = 300) -> dict:
        """Wait for pod to be ready with SSH access.

        Args:
            pod_id: Pod ID to wait for
            timeout: Max wait time in seconds

        Returns:
            Pod details when ready
        """
        start = time.time()
        while time.time() - start < timeout:
            pod = self.get_pod(pod_id)

            if pod["desiredStatus"] == "RUNNING" and pod.get("runtime"):
                # Check for SSH port
                ports = pod["runtime"].get("ports", [])
                ssh_port = next(
                    (p for p in ports if p["privatePort"] == 22 and p["isIpPublic"]),
                    None,
                )
                if ssh_port:
                    pod["ssh_host"] = ssh_port["ip"]
                    pod["ssh_port"] = ssh_port["publicPort"]
                    print(f"Pod ready: ssh root@{ssh_port['ip']} -p {ssh_port['publicPort']}")
                    return pod

            print(f"Waiting for pod {pod_id}... status={pod['desiredStatus']}")
            time.sleep(10)

        raise TimeoutError(f"Pod {pod_id} not ready after {timeout}s")

    def terminate_pod(self, pod_id: str) -> bool:
        """Terminate a pod.

        Args:
            pod_id: Pod to terminate

        Returns:
            True if terminated
        """
        query = """
        mutation terminatePod($podId: String!) {
            podTerminate(input: { podId: $podId })
        }
        """
        self._graphql(query, {"podId": pod_id})
        print(f"Terminated pod {pod_id}")
        return True

    def stop_pod(self, pod_id: str) -> bool:
        """Stop (but don't delete) a pod. Saves volume for later.

        Args:
            pod_id: Pod to stop

        Returns:
            True if stopped
        """
        query = """
        mutation stopPod($podId: String!) {
            podStop(input: { podId: $podId })
        }
        """
        self._graphql(query, {"podId": pod_id})
        print(f"Stopped pod {pod_id}")
        return True

    def run_command(self, pod_id: str, command: str, timeout: int = None) -> tuple:
        """Run a command on the pod via SSH.

        Args:
            pod_id: Pod ID
            command: Shell command to run
            timeout: Command timeout in seconds

        Returns:
            (return_code, stdout, stderr)
        """
        import subprocess

        pod = self.get_pod(pod_id)
        if not pod.get("runtime"):
            raise RuntimeError(f"Pod {pod_id} not running")

        ports = pod["runtime"].get("ports", [])
        ssh_port = next(
            (p for p in ports if p["privatePort"] == 22 and p["isIpPublic"]),
            None,
        )
        if not ssh_port:
            raise RuntimeError(f"No SSH port for pod {pod_id}")

        ssh_cmd = [
            "ssh",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-p", str(ssh_port["publicPort"]),
            f"root@{ssh_port['ip']}",
            command,
        ]

        result = subprocess.run(
            ssh_cmd,
            capture_output=True,
            text=True,
            timeout=timeout or self.config["timeout_seconds"],
        )
        return result.returncode, result.stdout, result.stderr

    def run_pipeline(
        self,
        pod_id: str,
        video_url: str,
        output_key: str,
    ) -> str:
        """Run the full tennis pipeline on a video.

        Args:
            pod_id: Pod to run on
            video_url: Presigned R2 URL for input video
            output_key: R2 key for output highlights

        Returns:
            R2 key of uploaded highlights
        """
        # Setup script that runs on pod
        setup_script = """
        cd /workspace
        if [ ! -d tennis_analysis ]; then
            git clone https://github.com/yourusername/tennis_analysis.git
        fi
        cd tennis_analysis
        pip install -q -r requirements.txt
        """

        # Pipeline script
        pipeline_script = f"""
        cd /workspace/tennis_analysis

        # Download video from R2
        curl -o raw/video.mov "{video_url}"

        # Run pipeline
        python preprocess_nvenc.py video.mov
        python scripts/extract_poses.py preprocessed/video.mp4
        python scripts/detect_shots.py preprocessed/video.mp4
        python scripts/extract_clips.py preprocessed/video.mp4 --highlights

        # Upload results
        python -c "
from storage import R2Client
client = R2Client()
client.upload('highlights/video_highlights.mp4', '{output_key}')
"
        """

        print(f"Setting up pod {pod_id}...")
        self.run_command(pod_id, setup_script)

        print(f"Running pipeline...")
        returncode, stdout, stderr = self.run_command(pod_id, pipeline_script)

        if returncode != 0:
            raise RuntimeError(f"Pipeline failed: {stderr}")

        return output_key


def main():
    """CLI for RunPod operations."""
    import argparse

    parser = argparse.ArgumentParser(description="RunPod GPU operations")
    parser.add_argument(
        "action",
        choices=["list-gpus", "create", "status", "terminate", "run"],
    )
    parser.add_argument("--pod-id", help="Pod ID for status/terminate")
    parser.add_argument("--gpu", help="GPU type for create")
    parser.add_argument("--video-url", help="R2 presigned URL for run")
    parser.add_argument("--output-key", help="R2 output key for run")
    args = parser.parse_args()

    executor = RunPodExecutor()

    if args.action == "list-gpus":
        gpus = executor.list_gpus()
        for gpu in gpus:
            price = gpu.get("lowestPrice", {})
            spot = price.get("minimumBidPrice", "N/A")
            ondemand = price.get("uninterruptablePrice", "N/A")
            print(
                f"{gpu['displayName']:30} {gpu['memoryInGb']:3}GB  "
                f"spot=${spot}  ondemand=${ondemand}"
            )

    elif args.action == "create":
        pod_id = executor.create_pod(gpu_type=args.gpu)
        executor.wait_for_ready(pod_id)

    elif args.action == "status":
        if not args.pod_id:
            parser.error("status requires --pod-id")
        pod = executor.get_pod(args.pod_id)
        print(json.dumps(pod, indent=2))

    elif args.action == "terminate":
        if not args.pod_id:
            parser.error("terminate requires --pod-id")
        executor.terminate_pod(args.pod_id)

    elif args.action == "run":
        if not all([args.pod_id, args.video_url, args.output_key]):
            parser.error("run requires --pod-id, --video-url, --output-key")
        executor.run_pipeline(args.pod_id, args.video_url, args.output_key)


if __name__ == "__main__":
    main()
