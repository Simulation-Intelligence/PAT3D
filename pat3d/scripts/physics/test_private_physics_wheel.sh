#!/usr/bin/env bash
set -euo pipefail

DEFAULT_PYTHON="python3"
TEST_PYTHON="${PAT3D_RELEASE_TEST_PYTHON:-${DEFAULT_PYTHON}}"
PRIVATE_PHYSICS_WHEEL="${PAT3D_PRIVATE_PHYSICS_WHEEL:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      TEST_PYTHON="$2"
      shift 2
      ;;
    --wheel)
      PRIVATE_PHYSICS_WHEEL="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "${PRIVATE_PHYSICS_WHEEL}" ]]; then
  echo "A private physics wheel is required. Set PAT3D_PRIVATE_PHYSICS_WHEEL or pass --wheel." >&2
  exit 2
fi

TMPDIR="$(mktemp -d)"
trap 'rm -rf "${TMPDIR}"' EXIT
VENV_DIR="${TMPDIR}/venv"

"${TEST_PYTHON}" -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/python" -m pip install --upgrade pip >/dev/null
"${VENV_DIR}/bin/python" -m pip install "${PRIVATE_PHYSICS_WHEEL}" >/dev/null
"${VENV_DIR}/bin/python" - <<'PY'
import importlib
import json
import sys

targets = [
    "uipc",
    "uipc.core",
    "uipc.geometry",
    "uipc.constitution",
    "uipc.gui",
    "uipc.torch",
    "pyuipc",
]
results = {}
failed = False
for name in targets:
    try:
        module = importlib.import_module(name)
        results[name] = {"ok": True, "file": getattr(module, "__file__", None)}
    except Exception as exc:
        failed = True
        results[name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

print(json.dumps(results, indent=2, sort_keys=True))
sys.exit(1 if failed else 0)
PY

echo "private physics wheel import passed"
