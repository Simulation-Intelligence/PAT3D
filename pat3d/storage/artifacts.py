from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
from uuid import uuid4


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slug(value: str | None, fallback: str) -> str:
    if not value:
        return fallback
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value.strip())
    safe = safe.strip("_")
    return safe or fallback


def _resolve_model(symbol_name: str) -> type[Any] | None:
    module_names = [
        "pat3d.models",
        "pat3d.models.base",
        "pat3d.models.scene",
        "pat3d.models.understanding",
        "pat3d.models.objects",
        "pat3d.models.physics",
    ]
    for module_name in module_names:
        try:
            module = __import__(module_name, fromlist=[symbol_name])
        except Exception:
            continue
        symbol = getattr(module, symbol_name, None)
        if symbol is not None:
            return symbol
    return None


def _instantiate_model(symbol_name: str, payload: dict[str, Any]) -> Any:
    model_type = _resolve_model(symbol_name)
    if model_type is None:
        return payload
    if hasattr(model_type, "model_validate"):
        return model_type.model_validate(payload)
    if is_dataclass(model_type):
        return model_type(**payload)
    try:
        return model_type(**payload)
    except TypeError:
        return payload


class RunDirectoryHelper:
    def __init__(self, root: str | Path = "artifacts") -> None:
        self.root = Path(root)

    def create_run_directory(
        self,
        scene_id: str,
        stage_name: str,
        run_id: str | None = None,
    ) -> Path:
        actual_run_id = _slug(run_id, uuid4().hex[:12])
        run_dir = self.root / _slug(scene_id, "scene") / _slug(stage_name, "stage") / actual_run_id
        self.ensure_run_subdirs(run_dir)
        return run_dir

    def ensure_run_subdirs(self, run_dir: str | Path) -> dict[str, Path]:
        run_dir = Path(run_dir)
        subdirs = {
            "run_dir": run_dir,
            "artifacts": run_dir / "artifacts",
            "metadata": run_dir / "metadata",
            "raw": run_dir / "raw",
        }
        for path in subdirs.values():
            path.mkdir(parents=True, exist_ok=True)
        return subdirs

    def artifact_path(
        self,
        run_dir: str | Path,
        artifact_type: str,
        filename: str,
    ) -> Path:
        subdirs = self.ensure_run_subdirs(run_dir)
        artifact_dir = subdirs["artifacts"] / _slug(artifact_type, "artifact")
        artifact_dir.mkdir(parents=True, exist_ok=True)
        return artifact_dir / filename

    def metadata_path(self, run_dir: str | Path, stem: str = "metadata") -> Path:
        subdirs = self.ensure_run_subdirs(run_dir)
        return subdirs["metadata"] / f"{_slug(stem, 'metadata')}.json"

    def raw_response_path(
        self,
        run_dir: str | Path,
        provider_name: str,
        suffix: str = "txt",
    ) -> Path:
        subdirs = self.ensure_run_subdirs(run_dir)
        return subdirs["raw"] / f"{_slug(provider_name, 'provider')}_response.{suffix}"

    def write_text(self, path: str | Path, content: str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def write_json(self, path: str | Path, payload: Any) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if is_dataclass(payload):
            payload = asdict(payload)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return path


class ArtifactStore(RunDirectoryHelper):
    def build_artifact_ref(
        self,
        path: str | Path,
        artifact_type: str,
        format: str | None = None,
        role: str | None = None,
        metadata_path: str | Path | None = None,
    ) -> Any:
        payload = {
            "artifact_type": artifact_type,
            "path": str(path),
            "format": format or Path(path).suffix.lstrip("."),
            "role": role,
            "metadata_path": str(metadata_path) if metadata_path else None,
        }
        return _instantiate_model("ArtifactRef", payload)

    def build_stage_metadata(
        self,
        stage_name: str,
        provider_name: str | None = None,
        run_id: str | None = None,
        status: str = "completed",
        raw_response_artifacts: list[Any] | None = None,
        notes: list[str] | None = None,
        started_at: str | None = None,
        finished_at: str | None = None,
        schema_version: str = "v1",
    ) -> Any:
        payload = {
            "run_id": run_id or uuid4().hex[:12],
            "stage_name": stage_name,
            "schema_version": schema_version,
            "provider_name": provider_name,
            "provider_config_digest": None,
            "started_at": started_at or _utc_now(),
            "finished_at": finished_at or _utc_now(),
            "status": status,
            "raw_response_artifacts": raw_response_artifacts or [],
            "notes": notes or [],
        }
        return _instantiate_model("StageRunMetadata", payload)
