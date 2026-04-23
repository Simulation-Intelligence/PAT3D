from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from enum import Enum
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

JsonPrimitive = str | int | float | bool | None
JsonValue = JsonPrimitive | list["JsonValue"] | dict[str, "JsonValue"]
MetadataMap = Mapping[str, JsonValue]


def _normalize_json_value(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _normalize_json_value(item) for key, item in asdict(value).items()}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_normalize_json_value(item) for item in value]
    if isinstance(value, list):
        return [_normalize_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _normalize_json_value(item) for key, item in value.items()}
    return value


class SerializableModel:
    """Small helper for recursive JSON-friendly serialization."""

    def to_dict(self) -> dict[str, Any]:
        return _normalize_json_value(asdict(self))

    def to_json(self, *, indent: int = 2, sort_keys: bool = True) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=sort_keys)


class StageRunStatus(str, Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ArtifactRef(SerializableModel):
    artifact_type: str
    path: str
    format: str
    role: str | None = None
    metadata_path: str | None = None

    def __post_init__(self) -> None:
        if not self.artifact_type.strip():
            raise ValueError("artifact_type must be non-empty")
        if not self.path.strip():
            raise ValueError("path must be non-empty")
        if not self.format.strip():
            raise ValueError("format must be non-empty")


@dataclass(frozen=True, slots=True)
class StageRunMetadata(SerializableModel):
    run_id: str
    stage_name: str
    schema_version: str
    status: StageRunStatus = StageRunStatus.PENDING
    provider_name: str | None = None
    provider_config_digest: str | None = None
    started_at: str = datetime.now(timezone.utc).isoformat()
    finished_at: str | None = None
    raw_response_artifacts: tuple[ArtifactRef, ...] = ()
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id must be non-empty")
        if not self.stage_name.strip():
            raise ValueError("stage_name must be non-empty")
        if not self.schema_version.strip():
            raise ValueError("schema_version must be non-empty")
        _validate_iso8601(self.started_at, "started_at")
        if self.finished_at is not None:
            _validate_iso8601(self.finished_at, "finished_at")

    @classmethod
    def completed(
        cls,
        *,
        run_id: str,
        stage_name: str,
        schema_version: str,
        provider_name: str | None = None,
        provider_config_digest: str | None = None,
        raw_response_artifacts: Sequence[ArtifactRef] = (),
        notes: Sequence[str] = (),
        started_at: str | None = None,
        finished_at: str | None = None,
    ) -> "StageRunMetadata":
        return cls(
            run_id=run_id,
            stage_name=stage_name,
            schema_version=schema_version,
            status=StageRunStatus.COMPLETED,
            provider_name=provider_name,
            provider_config_digest=provider_config_digest,
            started_at=started_at or datetime.now(timezone.utc).isoformat(),
            finished_at=finished_at or datetime.now(timezone.utc).isoformat(),
            raw_response_artifacts=tuple(raw_response_artifacts),
            notes=tuple(notes),
        )


@dataclass(frozen=True, slots=True)
class StructuredPromptRequest(SerializableModel):
    schema_name: str
    prompt_text: str
    system_prompt: str | None = None
    context_artifacts: tuple[ArtifactRef, ...] = ()
    metadata: dict[str, JsonValue] | None = None

    def __post_init__(self) -> None:
        if not self.schema_name.strip():
            raise ValueError("schema_name must be non-empty")
        if not self.prompt_text.strip():
            raise ValueError("prompt_text must be non-empty")


@dataclass(frozen=True, slots=True)
class StructuredPromptResult(SerializableModel):
    schema_name: str
    parsed_output: dict[str, JsonValue]
    metadata: StageRunMetadata
    raw_response_artifact: ArtifactRef | None = None

    def __post_init__(self) -> None:
        if not self.schema_name.strip():
            raise ValueError("schema_name must be non-empty")
        if not isinstance(self.parsed_output, dict):
            raise TypeError("parsed_output must be a dictionary")


def _validate_iso8601(value: str, field_name: str) -> None:
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:  # pragma: no cover - defensive path
        raise ValueError(f"{field_name} must be an ISO-8601 timestamp") from exc
