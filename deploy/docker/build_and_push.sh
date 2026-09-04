#!/bin/bash
# Build and push the OpenFinGym RL training image.
#
# Usage:
#   bash deploy/docker/build_and_push.sh <registry/repo:tag> [--skip-push]
#
# Examples:
#   bash deploy/docker/build_and_push.sh ghcr.io/deepintostreams/openfinai-rl:v1
#   bash deploy/docker/build_and_push.sh <your-registry>/openfinai-rl:v1
#
# Build context: this directory (deploy/docker/). The Dockerfile does not
# COPY any local repo files — code is mounted at runtime via PVC.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <registry/repo:tag> [--skip-push]" >&2
    exit 1
fi

IMAGE="$1"
SKIP_PUSH="${2:-}"

cd "$(dirname "$0")"

echo "=== Building ${IMAGE} ==="
# Quick daemon sanity check — surfaces "Docker Desktop not running" before
# the long silent base-image pull.
if ! docker version >/dev/null 2>&1; then
    echo "error: docker daemon not reachable. Is Docker Desktop running?" >&2
    exit 1
fi

# BuildKit gives parallel pulls but its early progress can look silent in
# some terminals. Set DOCKER_BUILDKIT=0 to get the classic layered output.
: "${DOCKER_BUILDKIT:=1}"
echo "DOCKER_BUILDKIT=${DOCKER_BUILDKIT} (set =0 for classic verbose output)"

DOCKER_BUILDKIT="${DOCKER_BUILDKIT}" docker build \
    --pull \
    --progress=plain \
    -t "${IMAGE}" \
    -f Dockerfile \
    .

if [[ "${SKIP_PUSH}" == "--skip-push" ]]; then
    echo "=== Skipped push (--skip-push). Image is local only: ${IMAGE} ==="
    exit 0
fi

echo "=== Pushing ${IMAGE} ==="
docker push "${IMAGE}"

echo "=== Done. Pulled-image digest: ==="
docker inspect "${IMAGE}" --format='{{index .RepoDigests 0}}' 2>/dev/null || true
