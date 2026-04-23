#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_TAG="${PAT3D_DOCKER_IMAGE_TAG:-pat3d-install:cuda13-ubuntu24.04}"
DOCKERFILE="${REPO_ROOT}/docker/Dockerfile.install"
BUILD_CONTEXT="${REPO_ROOT}/docker"
PULL=0
NO_CACHE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      cat <<'EOF'
Usage: bash scripts/build_docker_image.sh [options]

Options:
  --image-tag <tag>   Docker image tag to build
  --pull              Always pull the latest base image layers
  --no-cache          Disable the Docker build cache
EOF
      exit 0
      ;;
    --image-tag)
      IMAGE_TAG="$2"
      shift 2
      ;;
    --pull)
      PULL=1
      shift
      ;;
    --no-cache)
      NO_CACHE=1
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required to build the PAT3D install image." >&2
  exit 2
fi

build_args=(build -f "${DOCKERFILE}" -t "${IMAGE_TAG}")
if [[ "${PULL}" == "1" ]]; then
  build_args+=(--pull)
fi
if [[ "${NO_CACHE}" == "1" ]]; then
  build_args+=(--no-cache)
fi
build_args+=("${BUILD_CONTEXT}")

echo "[docker-build] docker ${build_args[*]}"
docker "${build_args[@]}"
echo "[docker-build] built image: ${IMAGE_TAG}"
