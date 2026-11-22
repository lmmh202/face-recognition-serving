#!/usr/bin/env bash
# Runs on the instance in /opt/stacks/face-recognition-serving:  ./deploy/deploy.sh <image-tag>
# Starts the given locally built tag, waits for /health, and records the tag in .deployed-tag.
# Redeploy a previous tag still on the host to roll back.
set -euo pipefail
cd "$(dirname "$0")/.."

tag=${1:?usage: deploy.sh <image-tag>}
docker image inspect "face-recognition-serving:$tag" >/dev/null

# Compose reads .env for interpolation, and so does Dockge.
printf 'IMAGE_TAG=%s\n' "$tag" >.env
docker compose up -d --remove-orphans --wait --wait-timeout 240
echo "$tag" >.deployed-tag
echo "deployed $tag"
