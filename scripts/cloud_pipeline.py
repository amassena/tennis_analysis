#!/usr/bin/env python3
"""Cloud pipeline for tennis video processing.

Runs on Hetzner coordinator. Watches for new jobs and dispatches to RunPod.

Flow:
1. Poll coordinator API for pending jobs
2. For each job: upload raw video to R2
3. Spin up RunPod GPU
4. Run processing (pose extraction, shot detection, clip extraction)
5. Upload highlights to R2
6. Mark job complete
7. Terminate RunPod
"""

import os
import sys
import json
import time
import requests
import subprocess
from pathlib import Path
from datetime import datetime

# Load environment
from dotenv import load_dotenv
load_dotenv('/opt/tennis/.env')

RUNPOD_API_KEY = os.environ.get('RUNPOD_API_KEY')
CF_ACCOUNT_ID = os.environ.get('CF_ACCOUNT_ID')
CF_R2_ACCESS_KEY_ID = os.environ.get('CF_R2_ACCESS_KEY_ID')
CF_R2_SECRET_ACCESS_KEY = os.environ.get('CF_R2_SECRET_ACCESS_KEY')

COORDINATOR_URL = "http://localhost:8080"  # Local on Hetzner
R2_ENDPOINT = f"https://{CF_ACCOUNT_ID}.r2.cloudflarestorage.com"
R2_BUCKET = "tennis-videos"

# RunPod processing script (runs on GPU pod)
PROCESSING_SCRIPT = '''
#!/bin/bash
set -e

VIDEO_KEY="$1"
OUTPUT_KEY="$2"

echo "=== Installing dependencies ==="
pip install -q boto3 matplotlib opencv-contrib-python
pip install -q mediapipe==0.10.18 --no-deps
pip install -q protobuf==4.25.5 absl-py attrs flatbuffers numpy

echo "=== Downloading video from R2 ==="
python3 << ENDPY
import boto3
from botocore.config import Config

client = boto3.client('s3',
    endpoint_url='${R2_ENDPOINT}',
    aws_access_key_id='${CF_R2_ACCESS_KEY_ID}',
    aws_secret_access_key='${CF_R2_SECRET_ACCESS_KEY}',
    config=Config(signature_version='s3v4'))

client.download_file('${R2_BUCKET}', '${VIDEO_KEY}', '/workspace/input.mp4')
print("Downloaded video")
ENDPY

echo "=== Extracting thumbnail ==="
python3 << ENDPY
import cv2

cap = cv2.VideoCapture('/workspace/input.mp4')
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Seek to 10% into video for a good frame
cap.set(cv2.CAP_PROP_POS_FRAMES, min(total_frames // 10, 100))
ret, frame = cap.read()

if ret:
    # Resize to max 480px width for thumbnail
    h, w = frame.shape[:2]
    if w > 480:
        scale = 480 / w
        frame = cv2.resize(frame, (480, int(h * scale)))
    cv2.imwrite('/workspace/thumb.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    print("Thumbnail extracted")
else:
    print("Failed to extract thumbnail")

cap.release()
ENDPY

echo "=== Running pose extraction ==="
python3 << ENDPY
import cv2
import mediapipe as mp
import json
import time

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(model_complexity=2)

cap = cv2.VideoCapture('/workspace/input.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video: {fps:.0f} fps, {total_frames} frames")

poses = []
frame_idx = 0
start = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:
        landmarks = []
        for lm in results.pose_landmarks.landmark:
            landmarks.append({'x': lm.x, 'y': lm.y, 'z': lm.z, 'v': lm.visibility})
        poses.append({'frame': frame_idx, 'landmarks': landmarks})

    frame_idx += 1
    if frame_idx % 500 == 0:
        print(f"Processed {frame_idx}/{total_frames} frames")

cap.release()
elapsed = time.time() - start
print(f"Done: {len(poses)} poses in {elapsed:.1f}s ({frame_idx/elapsed:.1f} fps)")

with open('/workspace/poses.json', 'w') as f:
    json.dump({'fps': fps, 'total_frames': total_frames, 'poses': poses}, f)
ENDPY

echo "=== Uploading results to R2 ==="
python3 << ENDPY
import boto3
from botocore.config import Config
import json

client = boto3.client('s3',
    endpoint_url='${R2_ENDPOINT}',
    aws_access_key_id='${CF_R2_ACCESS_KEY_ID}',
    aws_secret_access_key='${CF_R2_SECRET_ACCESS_KEY}',
    config=Config(signature_version='s3v4'))

with open('/workspace/poses.json', 'rb') as f:
    client.put_object(Bucket='${R2_BUCKET}', Key='${OUTPUT_KEY}', Body=f.read())
print(f"Uploaded poses to ${OUTPUT_KEY}")

# Upload thumbnail
import os
if os.path.exists('/workspace/thumb.jpg'):
    thumb_key = '${VIDEO_KEY}'.replace('raw/', 'thumbs/').rsplit('.', 1)[0] + '.jpg'
    with open('/workspace/thumb.jpg', 'rb') as f:
        client.put_object(Bucket='${R2_BUCKET}', Key=thumb_key, Body=f.read(), ContentType='image/jpeg')
    print(f"Uploaded thumbnail to {thumb_key}")
ENDPY

echo "=== Done ==="
'''


def log(msg):
    """Log with timestamp."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def runpod_query(query):
    """Execute RunPod GraphQL query."""
    response = requests.post(
        "https://api.runpod.io/graphql",
        headers={
            "Authorization": f"Bearer {RUNPOD_API_KEY}",
            "Content-Type": "application/json"
        },
        json={"query": query},
        timeout=30
    )
    return response.json()


def create_pod():
    """Create a RunPod GPU instance."""
    query = '''
    mutation {
      podFindAndDeployOnDemand(input: {
        name: "tennis-cloud"
        imageName: "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04"
        gpuTypeId: "NVIDIA GeForce RTX 4090"
        cloudType: SECURE
        volumeInGb: 20
        containerDiskInGb: 20
        minVcpuCount: 4
        minMemoryInGb: 16
        gpuCount: 1
        volumeMountPath: "/workspace"
        ports: "22/tcp"
        startSsh: true
      }) {
        id
        machine { gpuDisplayName }
      }
    }
    '''
    result = runpod_query(query)
    if "data" in result and result["data"]["podFindAndDeployOnDemand"]:
        pod = result["data"]["podFindAndDeployOnDemand"]
        log(f"Created pod {pod['id']} ({pod['machine']['gpuDisplayName']})")
        return pod["id"]
    else:
        log(f"Failed to create pod: {result}")
        return None


def wait_for_ssh(pod_id, timeout=300):
    """Wait for pod to be ready with SSH."""
    query = '''
    query {
      pod(input: { podId: "%s" }) {
        runtime {
          ports { ip isIpPublic privatePort publicPort }
        }
      }
    }
    ''' % pod_id

    start = time.time()
    while time.time() - start < timeout:
        result = runpod_query(query)
        pod = result.get("data", {}).get("pod", {})
        runtime = pod.get("runtime")

        if runtime and runtime.get("ports"):
            for port in runtime["ports"]:
                if port["privatePort"] == 22 and port["isIpPublic"]:
                    return port["ip"], port["publicPort"]

        time.sleep(10)

    return None, None


def terminate_pod(pod_id):
    """Terminate a RunPod instance."""
    query = f'mutation {{ podTerminate(input: {{ podId: "{pod_id}" }}) }}'
    runpod_query(query)
    log(f"Terminated pod {pod_id}")


def run_on_pod(ssh_host, ssh_port, script):
    """Run a script on the pod via SSH."""
    cmd = [
        "ssh", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
        "-p", str(ssh_port), f"root@{ssh_host}",
        "bash", "-s"
    ]
    result = subprocess.run(cmd, input=script, capture_output=True, text=True, timeout=3600)
    return result.returncode, result.stdout, result.stderr


def process_job(job):
    """Process a single video job."""
    video_id = job["video_id"]
    filename = job["filename"]

    log(f"Processing job {video_id}: {filename}")

    # Create RunPod
    pod_id = create_pod()
    if not pod_id:
        return False, "Failed to create pod"

    try:
        # Wait for SSH
        log("Waiting for pod SSH...")
        ssh_host, ssh_port = wait_for_ssh(pod_id)
        if not ssh_host:
            return False, "Pod SSH timeout"

        log(f"Pod ready: {ssh_host}:{ssh_port}")

        # Run processing
        video_key = f"raw/{filename}"
        output_key = f"poses/{filename.replace('.mov', '.json').replace('.mp4', '.json')}"

        script = PROCESSING_SCRIPT.replace('${VIDEO_KEY}', video_key)
        script = script.replace('${OUTPUT_KEY}', output_key)
        script = script.replace('${R2_ENDPOINT}', R2_ENDPOINT)
        script = script.replace('${R2_BUCKET}', R2_BUCKET)
        script = script.replace('${CF_R2_ACCESS_KEY_ID}', CF_R2_ACCESS_KEY_ID)
        script = script.replace('${CF_R2_SECRET_ACCESS_KEY}', CF_R2_SECRET_ACCESS_KEY)

        log("Running processing on pod...")
        returncode, stdout, stderr = run_on_pod(ssh_host, ssh_port, script)

        if returncode != 0:
            log(f"Processing failed: {stderr}")
            return False, stderr

        log("Processing complete!")
        return True, output_key

    finally:
        # Always terminate pod
        terminate_pod(pod_id)


def get_pending_jobs():
    """Get pending jobs from coordinator."""
    try:
        response = requests.get(f"{COORDINATOR_URL}/jobs/pending", timeout=10)
        data = response.json()
        return data.get("jobs", [])
    except Exception as e:
        log(f"Error getting jobs: {e}")
        return []


def claim_job(video_id):
    """Claim a job for processing."""
    try:
        response = requests.post(
            f"{COORDINATOR_URL}/jobs/{video_id}/claim",
            params={"worker_id": "cloud-worker"},
            timeout=10
        )
        return response.json().get("success", False)
    except Exception as e:
        log(f"Error claiming job: {e}")
        return False


def complete_job(video_id, success, result=None, error=None):
    """Mark job as complete."""
    try:
        requests.post(
            f"{COORDINATOR_URL}/jobs/{video_id}/complete",
            params={"worker_id": "cloud-worker"},
            json={"success": success, "youtube_url": result, "error_message": error},
            timeout=10
        )
    except Exception as e:
        log(f"Error completing job: {e}")


def main():
    """Main worker loop."""
    log("Cloud pipeline worker starting...")
    log(f"Coordinator: {COORDINATOR_URL}")
    log(f"R2 Bucket: {R2_BUCKET}")

    while True:
        jobs = get_pending_jobs()

        if jobs:
            job = jobs[0]
            video_id = job["video_id"]

            if claim_job(video_id):
                log(f"Claimed job {video_id}")
                success, result = process_job(job)
                complete_job(video_id, success, result if success else None, result if not success else None)
            else:
                log(f"Failed to claim job {video_id}")

        time.sleep(60)  # Poll every minute


if __name__ == "__main__":
    main()
