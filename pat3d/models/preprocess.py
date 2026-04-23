from __future__ import annotations

from dataclasses import dataclass

from .common import ArtifactRef, SerializableModel, StageRunMetadata


@dataclass(frozen=True, slots=True)
class LegacyPreprocessResult(SerializableModel):
    scene_id: str | None
    text_prompt: str | None
    metadata: StageRunMetadata
    requested_objects: tuple[str, ...] | None = None
    artifacts: tuple[ArtifactRef, ...] = ()

    def artifact_by_role(self, role: str) -> ArtifactRef | None:
        for artifact in self.artifacts:
            if artifact.role == role:
                return artifact
        return None
