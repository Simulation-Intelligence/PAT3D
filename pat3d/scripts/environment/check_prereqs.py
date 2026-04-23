from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import shutil
import sys
from importlib import metadata
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SUPPORTED_TARGET = {
    "platform": "linux",
    "distro_id": "ubuntu",
    "distro_version_prefix": "24.04",
    "machine": "x86_64",
    "python_version": "3.10",
}

REQUIRED_DIRS = [
    "data/ref_img",
    "data/depth",
    "data/seg",
    "data/items",
    "data/descrip",
    "data/ref_img_obj",
    "data/raw_obj",
    "data/organized_obj",
    "data/layout",
    "data/low_poly",
    "data/size",
    "data/contain",
    "data/contain_on",
    "results/preprocess_timing",
    "results/runtime",
    "results/dashboard_jobs",
    "results/metrics",
    "results/workspaces",
    "results/rendered_images",
    "_phys_result",
]

REQUIRED_PATHS = [
    "pat3d.yml",
]

REQUIRED_EXTERNAL_PATHS = {
    "extern/Hunyuan3Dv2": "required for Hunyuan3D object asset generation",
    "extern/fTetWild": "required for low-poly and tet-ready simulation mesh prep",
    "extern/sam3": "required for SAM 3 text-prompt segmentation",
}

OPTIONAL_PATHS = {
    "extern/t2v_metrics": "optional for VQAScore metrics only",
}

OPTIONAL_BINARIES = {
    "blender": "required for visualization/export and Blender-backed rendering paths",
}


def parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value
    return values


def load_declared_env(repo_root: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for env_path in (repo_root / ".env", Path.home() / ".env", repo_root / ".env.install"):
        values.update(parse_env_file(env_path))
    values.update(os.environ)
    return values


def display_path(repo_root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def check_exists(repo_root: Path, path: Path) -> dict[str, str | bool]:
    resolved = path.resolve() if path.is_absolute() else (repo_root / path).resolve()
    return {
        "path": display_path(repo_root, resolved),
        "exists": resolved.exists(),
        "kind": "directory" if resolved.is_dir() else "file",
    }


def check_binary(name: str) -> dict[str, str | bool | None]:
    resolved = shutil.which(name)
    return {"name": name, "exists": resolved is not None, "path": resolved}


def check_api_key(api_key_path: Path, repo_root: Path, declared_env: dict[str, str]) -> dict[str, str | bool]:
    env_present = bool(str(declared_env.get("OPENAI_API_KEY", "")).strip())
    file_present = api_key_path.exists()
    file_nonempty = file_present and api_key_path.stat().st_size > 0
    return {
        "env_present": env_present,
        "file_present": file_present,
        "file_nonempty": file_nonempty,
        "path": str(api_key_path.relative_to(repo_root)),
    }


def resolve_runtime_python_path(repo_root: Path, declared_env: dict[str, str]) -> Path:
    candidates: list[Path] = []
    for env_name in ("PAT3D_DASHBOARD_PYTHON", "PAT3D_LEGACY_PHYSICS_PYTHON"):
        raw_value = str(declared_env.get(env_name, "")).strip()
        if raw_value:
            candidates.append(Path(raw_value).expanduser())
    candidates.extend(
        [
            repo_root / ".venv" / "bin" / "python",
            repo_root / ".conda" / "pat3d" / "bin" / "python",
        ]
    )

    seen: set[Path] = set()
    ordered: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve() if candidate.is_absolute() else (repo_root / candidate).resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        ordered.append(resolved)

    for candidate in ordered:
        if candidate.exists():
            return candidate
    return ordered[0] if ordered else (repo_root / ".venv" / "bin" / "python")


def _load_os_release() -> dict[str, str]:
    try:
        values = platform.freedesktop_os_release()
    except (AttributeError, OSError):
        values = {}

    if values:
        return {str(key).lower(): str(value) for key, value in values.items()}

    os_release_path = Path("/etc/os-release")
    if not os_release_path.exists():
        return {}

    payload: dict[str, str] = {}
    for raw_line in os_release_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or "=" not in line or line.startswith("#"):
            continue
        key, value = line.split("=", 1)
        payload[key.lower()] = value.strip().strip('"')
    return payload


def detect_environment() -> dict[str, object]:
    os_release = _load_os_release()
    libc_name, libc_version = platform.libc_ver()
    return {
        "platform": platform.system().lower(),
        "machine": platform.machine().lower(),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "python_executable": sys.executable,
        "libc": {
            "name": libc_name or None,
            "version": libc_version or None,
        },
        "os_release": {
            "id": os_release.get("id"),
            "version_id": os_release.get("version_id"),
            "pretty_name": os_release.get("pretty_name"),
        },
        "nvidia_smi_available": shutil.which("nvidia-smi") is not None,
    }


def detect_supported_environment(environment: dict[str, object] | None = None) -> dict[str, object]:
    environment = environment if environment is not None else detect_environment()
    os_release = environment.get("os_release", {})
    if not isinstance(os_release, dict):
        os_release = {}

    checks = {
        "platform": environment.get("platform") == SUPPORTED_TARGET["platform"],
        "machine": environment.get("machine") == SUPPORTED_TARGET["machine"],
        "python_version": str(environment.get("python_version", "")).startswith(
            f"{SUPPORTED_TARGET['python_version']}."
        ),
        "distro_id": str(os_release.get("id", "")).lower() == SUPPORTED_TARGET["distro_id"],
        "distro_version": str(os_release.get("version_id", "")).startswith(
            SUPPORTED_TARGET["distro_version_prefix"]
        ),
    }
    return {
        "target": dict(SUPPORTED_TARGET),
        "checks": checks,
        "matches": all(checks.values()),
        "note": (
            "PAT3D release preparation currently targets Ubuntu 24.04 x86_64 with Python 3.10. "
            "The private Diff_GIPC wheel should be built for this exact ABI first."
        ),
    }


def detect_private_physics_backend() -> dict[str, object]:
    payload: dict[str, object] = {
        "optional": True,
        "distribution_name": "pyuipc",
        "installed": False,
        "compatible": False,
        "package_version": None,
        "uipc_module_path": None,
        "pyuipc_module_path": None,
        "checks": {},
        "note": (
            "Optional private Diff_GIPC-backed physics backend. The public PAT3D release does not "
            "ship this source tree; install the private wheel separately for physics-enabled runs."
        ),
    }

    try:
        payload["package_version"] = metadata.version("pyuipc")
        payload["installed"] = True
    except metadata.PackageNotFoundError:
        pass

    try:
        uipc_module = importlib.import_module("uipc")
        core_module = importlib.import_module("uipc.core")
        geometry_module = importlib.import_module("uipc.geometry")
        constitution_module = importlib.import_module("uipc.constitution")
        payload["installed"] = True
        payload["uipc_module_path"] = getattr(uipc_module, "__file__", None)
    except Exception as error:  # pragma: no cover - exercised through payload inspection
        payload["import_error"] = f"{type(error).__name__}: {error}"
        return payload

    torch_module = None
    torch_module_name = None
    for candidate in ("uipc.torch", "uipc.diff_sim"):
        try:
            torch_module = importlib.import_module(candidate)
            torch_module_name = candidate
            break
        except Exception:
            continue
    payload["torch_module_name"] = torch_module_name

    try:
        pyuipc_module = importlib.import_module("pyuipc")
        payload["pyuipc_module_path"] = getattr(pyuipc_module, "__file__", None)
    except Exception:
        try:
            pyuipc_module = importlib.import_module("uipc._native.pyuipc")
            payload["pyuipc_module_path"] = getattr(pyuipc_module, "__file__", None)
        except Exception:
            payload["pyuipc_module_path"] = None

    checks = {
        "Vector3i_or_Vector3": hasattr(core_module, "Vector3i")
        or hasattr(uipc_module, "Vector3i")
        or hasattr(uipc_module, "Vector3"),
        "Matrix4x4i_or_Matrix4x4": hasattr(core_module, "Matrix4x4i")
        or hasattr(uipc_module, "Matrix4x4i")
        or hasattr(uipc_module, "Matrix4x4"),
        "geometry.SimplicialComplexIO": hasattr(geometry_module, "SimplicialComplexIO"),
        "constitution.AffineBodyConstitution": hasattr(constitution_module, "AffineBodyConstitution"),
        "DiffSimParameter_or_torch": hasattr(constitution_module, "DiffSimParameter")
        or (torch_module is not None and hasattr(torch_module, "DiffSimParameter")),
    }
    payload["checks"] = checks
    payload["compatible"] = all(checks.values())
    return payload


def collect_prereq_report(
    repo_root: Path = REPO_ROOT,
    required_dirs: list[str] | None = None,
    required_paths: list[str] | None = None,
    required_external_paths: dict[str, str] | None = None,
    optional_binaries: dict[str, str] | None = None,
    optional_paths: dict[str, str] | None = None,
    require_supported_environment: bool = False,
    require_private_physics: bool = False,
    require_api_key: bool = True,
    require_required_external_paths: bool = True,
) -> dict[str, object]:
    required_dirs = required_dirs if required_dirs is not None else REQUIRED_DIRS
    required_paths = required_paths if required_paths is not None else REQUIRED_PATHS
    required_external_paths = (
        required_external_paths
        if required_external_paths is not None
        else REQUIRED_EXTERNAL_PATHS
    )
    optional_binaries = optional_binaries if optional_binaries is not None else OPTIONAL_BINARIES
    optional_paths = optional_paths if optional_paths is not None else OPTIONAL_PATHS

    api_key_file = repo_root / "pat3d/preprocessing/gpt_utils/apikey.txt"
    declared_env = load_declared_env(repo_root)
    environment = detect_environment()
    supported_environment = detect_supported_environment(environment)
    private_physics_backend = detect_private_physics_backend()
    runtime_python_path = resolve_runtime_python_path(repo_root, declared_env)

    required_dirs_payload = [
        check_exists(repo_root, repo_root / rel_path) for rel_path in required_dirs
    ]
    required_paths_payload = [
        check_exists(repo_root, repo_root / rel_path) for rel_path in required_paths
    ]
    required_paths_payload.append(
        {
            **check_exists(repo_root, runtime_python_path),
            "note": "dashboard runtime python resolved from .env.install/.env or default install paths",
        }
    )
    required_external_paths_payload = [
        {
            **check_exists(repo_root, repo_root / rel_path),
            "note": note,
        }
        for rel_path, note in required_external_paths.items()
    ]
    optional_paths_payload = [
        {
            **check_exists(repo_root, repo_root / rel_path),
            "note": note,
        }
        for rel_path, note in optional_paths.items()
    ]
    binaries = {
        name: {
            **check_binary(name),
            "note": note,
        }
        for name, note in optional_binaries.items()
    }
    api_key = check_api_key(api_key_file, repo_root, declared_env)

    missing_required_dirs = [item["path"] for item in required_dirs_payload if not item["exists"]]
    missing_required_paths = [item["path"] for item in required_paths_payload if not item["exists"]]
    missing_required_external_paths = (
        [item["path"] for item in required_external_paths_payload if not item["exists"]]
        if require_required_external_paths
        else []
    )
    missing_api_key = require_api_key and not (api_key["env_present"] or api_key["file_nonempty"])
    missing_optional_paths = [item["path"] for item in optional_paths_payload if not item["exists"]]
    missing_supported_environment = require_supported_environment and not bool(
        supported_environment["matches"]
    )
    missing_private_physics = require_private_physics and not bool(private_physics_backend["compatible"])

    payload = {
        "repo_root": str(repo_root),
        "environment": environment,
        "supported_environment": supported_environment,
        "private_physics_backend": private_physics_backend,
        "required_dirs": required_dirs_payload,
        "required_paths": required_paths_payload,
        "required_external_paths": required_external_paths_payload,
        "optional_paths": optional_paths_payload,
        "binaries": binaries,
        "api_key": api_key,
        "status": "ok",
    }

    if (
        missing_required_dirs
        or missing_required_paths
        or missing_required_external_paths
        or missing_api_key
        or missing_supported_environment
        or missing_private_physics
    ):
        payload["status"] = "missing"
        payload["missing_required_dirs"] = missing_required_dirs
        payload["missing_required_paths"] = missing_required_paths
        if missing_required_external_paths:
            payload["missing_required_external_paths"] = missing_required_external_paths
        if missing_optional_paths:
            payload["missing_optional_paths"] = missing_optional_paths
        if missing_api_key:
            payload["missing_api_key"] = True
        if missing_supported_environment:
            payload["missing_supported_environment"] = True
        if missing_private_physics:
            payload["missing_private_physics_backend"] = True
    elif missing_optional_paths:
        payload["missing_optional_paths"] = missing_optional_paths

    return payload


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify PAT3D environment setup.")
    parser.add_argument(
        "--repo-root",
        default=str(REPO_ROOT),
        help="Repo root path to validate (default: repo root containing this script).",
    )
    parser.add_argument(
        "--require-supported-environment",
        action="store_true",
        help="Fail if the current host does not match the supported Ubuntu 24.04 x86_64 / Python 3.10 target.",
    )
    parser.add_argument(
        "--require-private-physics",
        action="store_true",
        help="Fail if the required private Diff_GIPC-backed pyuipc install is unavailable or incompatible.",
    )
    parser.add_argument(
        "--skip-api-key",
        action="store_true",
        help="Do not fail when the OpenAI API key file or environment variable is absent.",
    )
    parser.add_argument(
        "--skip-required-external-paths",
        action="store_true",
        help="Do not fail when repo-only external dependency paths are absent.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    payload = run_with_root(
        Path(args.repo_root),
        require_supported_environment=args.require_supported_environment,
        require_private_physics=args.require_private_physics,
        require_api_key=not args.skip_api_key,
        require_required_external_paths=not args.skip_required_external_paths,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "ok" else 1


def run_with_root(
    repo_root: Path,
    *,
    require_supported_environment: bool = False,
    require_private_physics: bool = False,
    require_api_key: bool = True,
    require_required_external_paths: bool = True,
) -> dict[str, object]:
    return collect_prereq_report(
        repo_root=repo_root,
        require_supported_environment=require_supported_environment,
        require_private_physics=require_private_physics,
        require_api_key=require_api_key,
        require_required_external_paths=require_required_external_paths,
    )


if __name__ == "__main__":
    raise SystemExit(main())
