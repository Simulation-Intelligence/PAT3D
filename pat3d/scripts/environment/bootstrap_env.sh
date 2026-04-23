#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
VENV_DIR="${REPO_ROOT}/.conda/pat3d"
VENV_PY="${VENV_DIR}/bin/python"
VENV_ACTIVATE="${VENV_DIR}/bin/activate"
BOOTSTRAP_PYTHON="${PAT3D_BOOTSTRAP_PYTHON:-python3.10}"

python_is_supported() {
  local candidate="${1}"
  "${candidate}" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info[:2] == (3, 10) else 1)
PY
}

is_venv_ready() {
  [ -x "${VENV_PY}" ] && [ -f "${VENV_ACTIVATE}" ]
}

if ! command -v "${BOOTSTRAP_PYTHON}" >/dev/null 2>&1; then
  echo "Expected ${BOOTSTRAP_PYTHON} for the supported PAT3D bootstrap environment." >&2
  echo "Install Python 3.10 or create the full Conda environment with: conda env create -f pat3d.yml" >&2
  exit 2
fi

if ! python_is_supported "${BOOTSTRAP_PYTHON}"; then
  echo "${BOOTSTRAP_PYTHON} is not Python 3.10, which is the supported PAT3D bootstrap target." >&2
  exit 2
fi

if is_venv_ready; then
  if python_is_supported "${VENV_PY}"; then
    echo "PAT3D environment already exists at ${VENV_DIR}"
  else
    echo "PAT3D environment at ${VENV_DIR} uses an unsupported Python version. Recreating."
    rm -rf "${VENV_DIR}"
  fi
fi

if ! is_venv_ready; then
  if [ -d "${VENV_DIR}" ]; then
    echo "PAT3D environment at ${VENV_DIR} is incomplete or invalid. Recreating."
  else
    echo "Creating PAT3D environment at ${VENV_DIR}"
  fi
  rm -rf "${VENV_DIR}"
  "${BOOTSTRAP_PYTHON}" -m venv "${VENV_DIR}"
fi

"${VENV_PY}" -m pip install --upgrade pip

if [ -f "${REPO_ROOT}/requirements-dev.txt" ]; then
  "${VENV_PY}" -m pip install --no-cache-dir -r "${REPO_ROOT}/requirements-dev.txt"
fi
