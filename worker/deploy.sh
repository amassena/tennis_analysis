#!/bin/bash
set -e
cd "$(dirname "$0")"

echo "=== Deploying tennis-media worker ==="

# Set the upload password secret (prompts for value)
echo "Setting UPLOAD_PASSWORD secret..."
npx wrangler secret put UPLOAD_PASSWORD

# Deploy the worker
echo "Deploying worker..."
npx wrangler deploy

echo ""
echo "Done! Worker deployed to media.playfullife.com/api/*"
echo "Test: curl -X POST https://media.playfullife.com/api/upload/init -H 'Content-Type: application/json' -d '{\"password\":\"...\",\"filename\":\"test.mov\"}'"
