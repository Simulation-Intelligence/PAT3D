#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTAINER_REPO_ROOT="/workspace/PAT3D_private"
IMAGE_TAG="${PAT3D_DOCKER_IMAGE_TAG:-pat3d-install:cuda13-ubuntu24.04}"
DEFAULT_CACHE_BASE="${XDG_CACHE_HOME:-${HOME}/.cache}"
DOCKER_CACHE_ROOT="${PAT3D_DOCKER_CACHE_ROOT:-}"
DASHBOARD_PORT="${PAT3D_DASHBOARD_PORT:-4173}"
REBUILD=0
SKIP_BUILD=0

load_env_file() {
  local env_path="$1"
  if [[ -f "${env_path}" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "${env_path}"
    set +a
  fi
}

load_env_file "${REPO_ROOT}/.env"
load_env_file "${HOME}/.env"
load_env_file "${REPO_ROOT}/.env.install"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      cat <<'EOF'
Usage: bash scripts/run_docker_dashboard.sh [options]

Run the PAT3D dashboard inside the supported Docker image after
scripts/setup_docker_install.sh has completed.

Options:
  --dashboard-port <n>   Host and container dashboard port (default: 4173)
  --image-tag <tag>      Docker image tag to run
  --rebuild              Rebuild the Docker image before running
  --skip-build           Assume the Docker image already exists
EOF
      exit 0
      ;;
    --dashboard-port|--port)
      DASHBOARD_PORT="$2"
      shift 2
      ;;
    --image-tag)
      IMAGE_TAG="$2"
      shift 2
      ;;
    --rebuild)
      REBUILD=1
      shift
      ;;
    --skip-build)
      SKIP_BUILD=1
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required. Install Docker Engine with the NVIDIA container runtime first." >&2
  exit 2
fi
if [[ ! -f "${REPO_ROOT}/.env.install" ]]; then
  echo "Missing ${REPO_ROOT}/.env.install. Run bash scripts/setup_docker_install.sh first." >&2
  exit 2
fi
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is required. Set it in ${REPO_ROOT}/.env or ${HOME}/.env." >&2
  exit 2
fi

if [[ "${SKIP_BUILD}" != "1" ]]; then
  if [[ "${REBUILD}" == "1" ]] || ! docker image inspect "${IMAGE_TAG}" >/dev/null 2>&1; then
    bash "${REPO_ROOT}/scripts/build_docker_image.sh" --image-tag "${IMAGE_TAG}"
  fi
fi

if [[ -n "${DOCKER_CACHE_ROOT}" ]]; then
  HUGGINGFACE_CACHE_DIR="${PAT3D_DOCKER_HF_CACHE_DIR:-${DOCKER_CACHE_ROOT}/huggingface}"
  UV_CACHE_DIR="${PAT3D_DOCKER_UV_CACHE_DIR:-${DOCKER_CACHE_ROOT}/uv}"
  UV_SHARE_DIR="${PAT3D_DOCKER_UV_SHARE_DIR:-${DOCKER_CACHE_ROOT}/uv-share}"
  NPM_CACHE_DIR="${PAT3D_DOCKER_NPM_CACHE_DIR:-${DOCKER_CACHE_ROOT}/npm}"
  PLAYWRIGHT_CACHE_DIR="${PAT3D_DOCKER_PLAYWRIGHT_CACHE_DIR:-${DOCKER_CACHE_ROOT}/playwright}"
else
  HUGGINGFACE_CACHE_DIR="${PAT3D_DOCKER_HF_CACHE_DIR:-${DEFAULT_CACHE_BASE}/huggingface}"
  UV_CACHE_DIR="${PAT3D_DOCKER_UV_CACHE_DIR:-${DEFAULT_CACHE_BASE}/uv}"
  UV_SHARE_DIR="${PAT3D_DOCKER_UV_SHARE_DIR:-${DEFAULT_CACHE_BASE}/pat3d-docker/uv-share}"
  NPM_CACHE_DIR="${PAT3D_DOCKER_NPM_CACHE_DIR:-${HOME}/.npm}"
  PLAYWRIGHT_CACHE_DIR="${PAT3D_DOCKER_PLAYWRIGHT_CACHE_DIR:-${DEFAULT_CACHE_BASE}/ms-playwright}"
fi

mkdir -p \
  "${HUGGINGFACE_CACHE_DIR}" \
  "${UV_CACHE_DIR}" \
  "${UV_SHARE_DIR}" \
  "${NPM_CACHE_DIR}" \
  "${PLAYWRIGHT_CACHE_DIR}"

gpu_args=(--gpus all)
if [[ -n "${PAT3D_DOCKER_GPU_FLAGS:-}" ]]; then
  # shellcheck disable=SC2206
  gpu_args=(${PAT3D_DOCKER_GPU_FLAGS})
fi
extra_run_args=()
if [[ -n "${PAT3D_DOCKER_EXTRA_RUN_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  extra_run_args=(${PAT3D_DOCKER_EXTRA_RUN_ARGS})
fi

echo "PAT3D Docker dashboard: http://127.0.0.1:${DASHBOARD_PORT}"

docker run --rm \
  "${gpu_args[@]}" \
  --shm-size=16g \
  "${extra_run_args[@]}" \
  -p "${DASHBOARD_PORT}:${DASHBOARD_PORT}" \
  -v "${REPO_ROOT}:${CONTAINER_REPO_ROOT}" \
  -v "${HUGGINGFACE_CACHE_DIR}:/root/.cache/huggingface" \
  -v "${UV_CACHE_DIR}:/root/.cache/uv" \
  -v "${UV_SHARE_DIR}:/root/.local/share/uv" \
  -v "${NPM_CACHE_DIR}:/root/.npm" \
  -v "${PLAYWRIGHT_CACHE_DIR}:/root/.cache/ms-playwright" \
  -w "${CONTAINER_REPO_ROOT}" \
  -e "OPENAI_API_KEY=${OPENAI_API_KEY}" \
  -e "OPENAI_BASE_URL=${OPENAI_BASE_URL:-https://api.openai.com/v1}" \
  -e "OPENAI_MODEL=${OPENAI_MODEL:-gpt-5.4}" \
  -e "OPENAI_IMAGE_MODEL=${OPENAI_IMAGE_MODEL:-gpt-image-1.5}" \
  -e "HF_TOKEN=${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}}" \
  -e "HUGGINGFACE_HUB_TOKEN=${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}}" \
  -e "HUGGING_FACE_HUB_TOKEN=${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}}" \
  -e "PAT3D_HF_TOKEN_FILE=${PAT3D_HF_TOKEN_FILE:-/root/.cache/huggingface/token}" \
  -e "NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all}" \
  -e "NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:-all}" \
  -e "PAT3D_DASHBOARD_ALLOW_PRIVATE_SUBNETS=1" \
  -e "PAT3D_DOCKER_HOST_UID=$(id -u)" \
  -e "PAT3D_DOCKER_HOST_GID=$(id -g)" \
  -e "UV_LINK_MODE=${UV_LINK_MODE:-copy}" \
  -e "PLAYWRIGHT_BROWSERS_PATH=/root/.cache/ms-playwright" \
  "${IMAGE_TAG}" \
  bash -lc "
set -euo pipefail
cleanup() {
  chown -R \"\${PAT3D_DOCKER_HOST_UID}:\${PAT3D_DOCKER_HOST_GID}\" \
    ${CONTAINER_REPO_ROOT} \
    /root/.cache/huggingface \
    /root/.cache/uv \
    /root/.local/share/uv \
    /root/.npm \
    /root/.cache/ms-playwright >/dev/null 2>&1 || true
}
trap cleanup EXIT
cd ${CONTAINER_REPO_ROOT}
bash scripts/run_dashboard.sh --host 0.0.0.0 --port $(printf '%q' "${DASHBOARD_PORT}")"
