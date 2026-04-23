from __future__ import annotations

import ctypes
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType


_PRELOADED_SHARED_LIBRARIES: set[Path] = set()


@dataclass(frozen=True)
class HunyuanRuntimeEnvironment:
    repo_root: Path
    hunyuan_root: Path
    torch_lib_dir: Path | None
    cuda_lib_dirs: tuple[Path, ...]
    preloaded_libraries: tuple[str, ...]


def _prepend_sys_path(path: Path) -> None:
    resolved = str(path.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


def _prepend_env_path(variable: str, path: Path) -> None:
    resolved = str(path.resolve())
    current_entries = [
        entry
        for entry in os.environ.get(variable, "").split(os.pathsep)
        if entry
    ]
    if resolved in current_entries:
        current_entries.remove(resolved)
    current_entries.insert(0, resolved)
    os.environ[variable] = os.pathsep.join(current_entries)


def _load_torch_module() -> ModuleType | None:
    torch_module = sys.modules.get("torch")
    if torch_module is not None:
        return torch_module if isinstance(torch_module, ModuleType) else None
    try:
        import torch as torch_module  # type: ignore[import-not-found]
    except Exception:
        return None
    return torch_module


def _torch_lib_dir_from_module(torch_module: ModuleType | None) -> Path | None:
    if torch_module is None:
        return None
    torch_file = getattr(torch_module, "__file__", None)
    if not torch_file:
        return None
    candidate = Path(torch_file).resolve().parent / "lib"
    return candidate if candidate.exists() else None


def _existing_cuda_lib_dirs() -> tuple[Path, ...]:
    candidates: list[Path] = []
    for variable in ("PAT3D_CUDA_LIB_DIR", "CUDA_HOME", "CUDA_PATH"):
        raw_value = os.environ.get(variable, "").strip()
        if not raw_value:
            continue
        path = Path(raw_value)
        candidate = path if variable == "PAT3D_CUDA_LIB_DIR" else path / "lib64"
        if candidate.exists():
            candidates.append(candidate.resolve())

    for candidate in (
        Path("/usr/local/cuda/lib64"),
        Path("/usr/local/cuda-13.0/targets/x86_64-linux/lib"),
        Path("/usr/local/cuda-12.8/targets/x86_64-linux/lib"),
    ):
        if candidate.exists():
            resolved = candidate.resolve()
            if resolved not in candidates:
                candidates.append(resolved)
    return tuple(candidates)


def _resolve_library_path(directory: Path, library_name: str) -> Path | None:
    direct = directory / library_name
    if direct.exists():
        return direct

    matches = sorted(directory.glob(f"{library_name}*"))
    return matches[0] if matches else None


def _preload_shared_libraries(
    directories: tuple[Path, ...],
    library_names: tuple[str, ...],
) -> tuple[str, ...]:
    loaded: list[str] = []
    rtld_global = getattr(ctypes, "RTLD_GLOBAL", 0)

    for directory in directories:
        for library_name in library_names:
            library_path = _resolve_library_path(directory, library_name)
            if library_path is None:
                continue

            resolved = library_path.resolve()
            if resolved in _PRELOADED_SHARED_LIBRARIES:
                continue

            ctypes.CDLL(str(resolved), mode=rtld_global)
            _PRELOADED_SHARED_LIBRARIES.add(resolved)
            loaded.append(str(resolved))

    return tuple(loaded)


def ensure_hunyuan_runtime(
    *,
    repo_root: Path,
    hunyuan_root: Path,
) -> HunyuanRuntimeEnvironment:
    repo_root = repo_root.resolve()
    hunyuan_root = hunyuan_root.resolve()

    _prepend_sys_path(repo_root)
    _prepend_sys_path(hunyuan_root)
    _prepend_env_path("PYTHONPATH", hunyuan_root)
    _prepend_env_path("PYTHONPATH", repo_root)

    torch_module = _load_torch_module()
    torch_lib_dir = _torch_lib_dir_from_module(torch_module)
    cuda_lib_dirs = _existing_cuda_lib_dirs()

    preload_directories: list[Path] = []
    if torch_lib_dir is not None:
        _prepend_env_path("LD_LIBRARY_PATH", torch_lib_dir)
        preload_directories.append(torch_lib_dir)
    for cuda_lib_dir in cuda_lib_dirs:
        _prepend_env_path("LD_LIBRARY_PATH", cuda_lib_dir)
        preload_directories.append(cuda_lib_dir)

    preloaded_libraries = _preload_shared_libraries(
        tuple(preload_directories),
        (
            "libc10.so",
            "libtorch_cpu.so",
            "libtorch.so",
            "libc10_cuda.so",
            "libtorch_cuda.so",
            "libtorch_python.so",
            "libcudart.so",
        ),
    )

    return HunyuanRuntimeEnvironment(
        repo_root=repo_root,
        hunyuan_root=hunyuan_root,
        torch_lib_dir=torch_lib_dir,
        cuda_lib_dirs=cuda_lib_dirs,
        preloaded_libraries=preloaded_libraries,
    )
