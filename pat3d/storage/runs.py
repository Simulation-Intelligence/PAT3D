from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence
import uuid

from pat3d.models import ArtifactRef, StageRunMetadata, StageRunStatus


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_run_id(prefix: str) -> str:
    token = uuid.uuid4().hex[:8]
    return f"{prefix}-{token}"


@dataclass(frozen=True, slots=True)
class RunPaths:
    root: Path
    run_id: str

    @classmethod
    def create(cls, root: str | Path, prefix: str = "run") -> "RunPaths":
        return cls(root=Path(root), run_id=make_run_id(prefix))

    @property
    def run_dir(self) -> Path:
        return self.root / self.run_id

    def ensure(self) -> Path:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        return self.run_dir

    def stage_dir(self, stage_name: str) -> Path:
        path = self.run_dir / stage_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def artifact_ref(
        self,
        *,
        stage_name: str,
        filename: str,
        artifact_type: str,
        format: str,
        role: str | None = None,
        metadata_path: str | None = None,
    ) -> ArtifactRef:
        path = self.stage_dir(stage_name) / filename
        return ArtifactRef(
            artifact_type=artifact_type,
            path=str(path),
            format=format,
            role=role,
            metadata_path=metadata_path,
        )


def make_stage_metadata(
    *,
    stage_name: str,
    schema_version: str = "v1",
    provider_name: str | None = None,
    provider_config_digest: str | None = None,
    run_id: str | None = None,
    status: StageRunStatus = StageRunStatus.COMPLETED,
    raw_response_artifacts: Sequence[ArtifactRef] = (),
    notes: Sequence[str] = (),
) -> StageRunMetadata:
    started_at = utc_now_iso()
    finished_at = utc_now_iso() if status != StageRunStatus.PENDING else None
    return StageRunMetadata(
        run_id=run_id or make_run_id(stage_name),
        stage_name=stage_name,
        schema_version=schema_version,
        status=status,
        provider_name=provider_name,
        provider_config_digest=provider_config_digest,
        started_at=started_at,
        finished_at=finished_at,
        raw_response_artifacts=tuple(raw_response_artifacts),
        notes=tuple(notes),
    )

