#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
DEFAULT_PYTHON="${REPO_ROOT}/.conda/pat3d/bin/python"
if [[ ! -x "${DEFAULT_PYTHON}" ]]; then
  DEFAULT_PYTHON="python3"
fi

TEST_PYTHON="${PAT3D_RELEASE_TEST_PYTHON:-${DEFAULT_PYTHON}}"
DIFF_GIPC_ROOT="${PAT3D_DIFF_GIPC_ROOT:-${REPO_ROOT}/references/mask_loss/extern/Diff_GIPC}"
OUTPUT_DIR="${PAT3D_PRIVATE_WHEEL_OUTPUT_DIR:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      TEST_PYTHON="$2"
      shift 2
      ;;
    --diff-gipc-root)
      DIFF_GIPC_ROOT="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ ! -d "${DIFF_GIPC_ROOT}" ]]; then
  echo "Diff_GIPC root does not exist: ${DIFF_GIPC_ROOT}" >&2
  exit 2
fi

if [[ -z "${OUTPUT_DIR}" ]]; then
  OUTPUT_DIR="$(mktemp -d)"
else
  mkdir -p "${OUTPUT_DIR}"
fi

ext_suffix="$("${TEST_PYTHON}" - <<'PY'
import sysconfig
print(sysconfig.get_config_var("EXT_SUFFIX") or "")
PY
)"

py_minor_tag="$("${TEST_PYTHON}" - <<'PY'
import sys
print(f"py{sys.version_info.major}{sys.version_info.minor}")
PY
)"

candidate_dirs=()
preferred_dir="${DIFF_GIPC_ROOT}/build-${py_minor_tag}/python"
if [[ -d "${preferred_dir}" ]]; then
  candidate_dirs+=("${preferred_dir}")
fi
while IFS= read -r candidate; do
  if [[ "${candidate}" != "${preferred_dir}" ]]; then
    candidate_dirs+=("${candidate}")
  fi
done < <(find "${DIFF_GIPC_ROOT}" -maxdepth 2 -type d -path '*/build*/python' | sort)

selected_dir=""
for candidate in "${candidate_dirs[@]}"; do
  if [[ -f "${candidate}/src/uipc/modules/Release/bin/pyuipc${ext_suffix}" ]]; then
    selected_dir="${candidate}"
    break
  fi
done

if [[ -z "${selected_dir}" ]]; then
  echo "Could not find a Diff_GIPC python package tree with pyuipc${ext_suffix}." >&2
  echo "Searched under: ${DIFF_GIPC_ROOT}" >&2
  find "${DIFF_GIPC_ROOT}" -path '*/pyuipc.cpython-*.so' | sort >&2 || true
  exit 1
fi

echo "[private-wheel] selected package tree: ${selected_dir}" >&2
rm -rf "${selected_dir}/build" "${selected_dir}/pyuipc.egg-info"
"${TEST_PYTHON}" -m pip wheel --no-deps --no-build-isolation "${selected_dir}" -w "${OUTPUT_DIR}" >/dev/null

wheel_path="$(find "${OUTPUT_DIR}" -maxdepth 1 -name '*.whl' | head -n 1)"
if [[ -z "${wheel_path}" ]]; then
  echo "Wheel build did not produce a wheel artifact." >&2
  exit 1
fi

bash "${REPO_ROOT}/pat3d/scripts/physics/test_private_physics_wheel.sh" \
  --python "${TEST_PYTHON}" \
  --wheel "${wheel_path}" >&2

echo "[private-wheel] built wheel: ${wheel_path}" >&2
echo "${wheel_path}"
