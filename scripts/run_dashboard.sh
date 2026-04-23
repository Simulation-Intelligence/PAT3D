#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_HF_TOKEN_FILE="${HOME}/.cache/huggingface/token"
HOST="${PAT3D_DASHBOARD_HOST:-127.0.0.1}"
PORT="${PAT3D_DASHBOARD_PORT:-4173}"
PAT3D_DASHBOARD_PYTHON="${PAT3D_DASHBOARD_PYTHON:-${REPO_ROOT}/.venv/bin/python}"
PAT3D_SAM3_PYTHON="${PAT3D_SAM3_PYTHON:-${REPO_ROOT}/.venv-sam3/bin/python}"
PAT3D_SAM3_CHECKPOINT_PATH="${PAT3D_SAM3_CHECKPOINT_PATH:-${REPO_ROOT}/checkpoints/sam3/sam3.pt}"
PAT3D_FTETWILD_BIN="${PAT3D_FTETWILD_BIN:-${REPO_ROOT}/extern/fTetWild/build-local/FloatTetwild_bin}"

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
Usage: bash scripts/run_dashboard.sh [options]

Run the PAT3D dashboard with the exact production backend profile using the
existing install created by scripts/setup_install.py.

Options:
  --host <host>         Dashboard bind host (default: 127.0.0.1)
  --port <port>         Dashboard bind port (default: 4173)
EOF
      exit 0
      ;;
    --host)
      HOST="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

resolve_hf_token() {
  local token_value
  token_value="${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}}"
  if [[ -n "${token_value}" ]]; then
    printf '%s' "${token_value}"
    return 0
  fi
  local token_file="${PAT3D_HF_TOKEN_FILE:-${DEFAULT_HF_TOKEN_FILE}}"
  if [[ -f "${token_file}" ]]; then
    python3 - <<'PY' "${token_file}"
from pathlib import Path
import sys
print(Path(sys.argv[1]).expanduser().read_text(encoding="utf-8").strip())
PY
    return 0
  fi
  return 1
}

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is required. Set it in ${REPO_ROOT}/.env or ${HOME}/.env." >&2
  exit 2
fi

HF_TOKEN_VALUE="$(resolve_hf_token || true)"
if [[ -z "${HF_TOKEN_VALUE}" ]]; then
  echo "HF_TOKEN or PAT3D_HF_TOKEN_FILE is required for the production dashboard path." >&2
  exit 2
fi

for required_path in \
  "${PAT3D_DASHBOARD_PYTHON}" \
  "${PAT3D_SAM3_PYTHON}" \
  "${PAT3D_SAM3_CHECKPOINT_PATH}" \
  "${PAT3D_FTETWILD_BIN}" \
  "${REPO_ROOT}/dashboard/node_modules"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "Missing required install artifact: ${required_path}" >&2
    echo "Run python3 scripts/setup_install.py first." >&2
    exit 2
  fi
done

echo "PAT3D dashboard: http://${HOST}:${PORT}"
echo "Using repo root: ${REPO_ROOT}"

cd "${REPO_ROOT}/dashboard"
export HOST="${HOST}"
export PORT="${PORT}"
export OPENAI_API_KEY="${OPENAI_API_KEY}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}"
export OPENAI_MODEL="${OPENAI_MODEL:-gpt-5.4}"
export OPENAI_IMAGE_MODEL="${OPENAI_IMAGE_MODEL:-gpt-image-1.5}"
export HF_TOKEN="${HF_TOKEN_VALUE}"
export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN_VALUE}"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN_VALUE}"
export PAT3D_DASHBOARD_PYTHON="${PAT3D_DASHBOARD_PYTHON}"
export PAT3D_SAM3_PYTHON="${PAT3D_SAM3_PYTHON}"
export PAT3D_SAM3_CHECKPOINT_PATH="${PAT3D_SAM3_CHECKPOINT_PATH}"
export PAT3D_SAM3_LOAD_FROM_HF="${PAT3D_SAM3_LOAD_FROM_HF:-0}"
export PAT3D_FTETWILD_BIN="${PAT3D_FTETWILD_BIN}"
export PAT3D_DASHBOARD_STAGE_BACKENDS_PROFILE="${PAT3D_DASHBOARD_STAGE_BACKENDS_PROFILE:-default}"
export PAT3D_DASHBOARD_FORCE_STAGE_BACKENDS_PROFILE="${PAT3D_DASHBOARD_FORCE_STAGE_BACKENDS_PROFILE:-0}"
export PAT3D_DASHBOARD_ALLOW_PRIVATE_SUBNETS="${PAT3D_DASHBOARD_ALLOW_PRIVATE_SUBNETS:-0}"
export PAT3D_REPO_ROOT="${PAT3D_REPO_ROOT:-${REPO_ROOT}}"
export NODE_ENV="${NODE_ENV:-development}"
exec node server/index.mjs
