#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTAINER_REPO_ROOT="/workspace/PAT3D_private"
IMAGE_TAG="${PAT3D_DOCKER_IMAGE_TAG:-pat3d-install:cuda13-ubuntu24.04}"
DEFAULT_CACHE_BASE="${XDG_CACHE_HOME:-${HOME}/.cache}"
DOCKER_CACHE_ROOT="${PAT3D_DOCKER_CACHE_ROOT:-}"
REBUILD=0
SKIP_BUILD=0
WITH_PLAYWRIGHT_BROWSER=0

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

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      cat <<'EOF'
Usage: bash scripts/setup_docker_install.sh [options]

Build the supported CUDA 13 PAT3D image if needed, then run the PAT3D local
installer inside that container.

Options:
  --image-tag <tag>            Docker image tag to run
  --rebuild                    Rebuild the Docker image before running
  --skip-build                 Assume the Docker image already exists
  --with-playwright-browser    Also install the local Playwright browser
EOF
      exit 0
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
    --with-playwright-browser)
      WITH_PLAYWRIGHT_BROWSER=1
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
if [[ -z "${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}}" && -z "${PAT3D_HF_TOKEN_FILE:-}" ]]; then
  echo "HF_TOKEN or PAT3D_HF_TOKEN_FILE is required." >&2
  exit 2
fi

if [[ "${SKIP_BUILD}" != "1" ]]; then
  if [[ "${REBUILD}" == "1" ]] || ! docker image inspect "${IMAGE_TAG}" >/dev/null 2>&1; then
    bash "${REPO_ROOT}/scripts/build_docker_image.sh" --image-tag "${IMAGE_TAG}"
  fi
fi

PRIVATE_WHEEL_HOST="$(python3 - <<'PY' "${PAT3D_PRIVATE_PHYSICS_WHEEL:-}" "${REPO_ROOT}"
from pathlib import Path
import sys
configured = sys.argv[1].strip()
repo_root = Path(sys.argv[2]).resolve()
if configured:
    path = Path(configured).expanduser().resolve()
else:
    candidates = sorted((repo_root / "private_wheels").glob("pyuipc-*.whl"))
    if not candidates:
        raise SystemExit(
            "PAT3D requires a Diff_GIPC wheel. Place the tracked prebuilt wheel under "
            f"{repo_root / 'private_wheels'} or set PAT3D_PRIVATE_PHYSICS_WHEEL."
        )
    path = candidates[-1]
if not path.is_file():
    raise SystemExit(f"Private physics wheel does not exist: {path}")
print(path)
PY
)"
PRIVATE_WHEEL_BASENAME="$(basename "${PRIVATE_WHEEL_HOST}")"
CONTAINER_PRIVATE_WHEEL="/opt/pat3d/private_wheels/${PRIVATE_WHEEL_BASENAME}"

HF_TOKEN_VALUE="${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}}"
if [[ -z "${HF_TOKEN_VALUE}" && -n "${PAT3D_HF_TOKEN_FILE:-}" ]]; then
  HF_TOKEN_VALUE="$(python3 - <<'PY' "${PAT3D_HF_TOKEN_FILE}"
from pathlib import Path
import sys
path = Path(sys.argv[1]).expanduser().resolve()
if not path.is_file():
    raise SystemExit(f"Hugging Face token file does not exist: {path}")
print(path.read_text(encoding='utf-8').strip())
PY
)"
fi
if [[ -z "${HF_TOKEN_VALUE}" ]]; then
  echo "Resolved Hugging Face token is empty." >&2
  exit 2
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

setup_args=(--skip-system-deps)
if [[ "${WITH_PLAYWRIGHT_BROWSER}" != "1" ]]; then
  setup_args+=(--skip-playwright-browser)
fi

container_args=(
  run --rm
  "${gpu_args[@]}"
  --shm-size=16g
  "${extra_run_args[@]}"
  -v "${REPO_ROOT}:${CONTAINER_REPO_ROOT}"
  -v "${PRIVATE_WHEEL_HOST}:${CONTAINER_PRIVATE_WHEEL}:ro"
  -v "${HUGGINGFACE_CACHE_DIR}:/root/.cache/huggingface"
  -v "${UV_CACHE_DIR}:/root/.cache/uv"
  -v "${UV_SHARE_DIR}:/root/.local/share/uv"
  -v "${NPM_CACHE_DIR}:/root/.npm"
  -v "${PLAYWRIGHT_CACHE_DIR}:/root/.cache/ms-playwright"
  -w "${CONTAINER_REPO_ROOT}"
  -e "HF_TOKEN=${HF_TOKEN_VALUE}"
  -e "HUGGINGFACE_HUB_TOKEN=${HF_TOKEN_VALUE}"
  -e "HUGGING_FACE_HUB_TOKEN=${HF_TOKEN_VALUE}"
  -e "NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all}"
  -e "NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:-all}"
  -e "PAT3D_PRIVATE_PHYSICS_WHEEL=${CONTAINER_PRIVATE_WHEEL}"
  -e "PAT3D_DOCKER_HOST_UID=$(id -u)"
  -e "PAT3D_DOCKER_HOST_GID=$(id -g)"
  -e "UV_LINK_MODE=${UV_LINK_MODE:-copy}"
  -e "PLAYWRIGHT_BROWSERS_PATH=/root/.cache/ms-playwright"
)

container_args+=(
  "${IMAGE_TAG}"
  bash -lc
  "set -euo pipefail
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
python3 scripts/setup_install.py ${setup_args[*]}"
)

echo "[docker-install] launching image ${IMAGE_TAG}"
docker "${container_args[@]}"

echo "docker image: ${IMAGE_TAG}"
echo "launch env file: ${REPO_ROOT}/.env.install"
