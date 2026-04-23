from __future__ import annotations

import importlib
import importlib.util
import os
from functools import lru_cache
from pathlib import Path
from typing import Callable


PAT3D_REPO_ROOT_ENV_VAR = "PAT3D_REPO_ROOT"
_PACKAGE_ROOT = Path(__file__).resolve().parent
_FALLBACK_CONFIG_MODULE = "pat3d.default_legacy_config"


def _looks_like_repo_root(candidate: Path) -> bool:
    return (candidate / "modules" / "preprocess_utils").is_dir() and (
        candidate / "pat3d"
    ).is_dir()


def _iter_repo_root_candidates() -> list[Path]:
    candidates: list[Path] = []
    configured = os.environ.get(PAT3D_REPO_ROOT_ENV_VAR, "").strip()
    if configured:
        candidates.append(Path(configured).expanduser())

    cwd = Path.cwd()
    candidates.extend((cwd, *cwd.parents))
    candidates.extend(_PACKAGE_ROOT.parents)
    return candidates


@lru_cache(maxsize=1)
def resolve_repo_root() -> Path:
    seen: set[Path] = set()
    for candidate in _iter_repo_root_candidates():
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if _looks_like_repo_root(resolved):
            return resolved

    raise RuntimeError(
        "Could not locate the PAT3D bundle root containing pat3d/ and modules/preprocess_utils. "
        f"Run from the extracted PAT3D bundle directory or set {PAT3D_REPO_ROOT_ENV_VAR}=/path/to/PAT3D."
    )


@lru_cache(maxsize=1)
def load_root_parse_options() -> Callable[..., object]:
    config_path = resolve_repo_root() / "config.py"
    if config_path.is_file():
        spec = importlib.util.spec_from_file_location("pat3d_root_config", config_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"could not load root config module: {config_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(_FALLBACK_CONFIG_MODULE)
    parse_options = getattr(module, "parse_options", None)
    if not callable(parse_options):
        raise RuntimeError(
            "root config module does not define parse_options(): "
            f"{config_path if config_path.is_file() else _FALLBACK_CONFIG_MODULE}"
        )
    return parse_options


def clear_legacy_config_caches() -> None:
    resolve_repo_root.cache_clear()
    load_root_parse_options.cache_clear()
