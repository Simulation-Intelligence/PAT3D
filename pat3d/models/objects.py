from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from .common import ArtifactRef, JsonValue, SerializableModel, StageRunMetadata

if TYPE_CHECKING:
    from .layout import ObjectPose


class RelationType(str, Enum):
    ON = "on"
    IN = "in"
    CONTAINS = "contains"
    SUPPORTS = "supports"


@dataclass(frozen=True, slots=True)
class DetectedObject(SerializableModel):
    object_id: str
    canonical_name: str
    display_name: str | None
    count: int
    source_instance_ids: tuple[str, ...] = ()
    attributes: dict[str, str] | None = None

    def __post_init__(self) -> None:
        if not self.object_id.strip():
            raise ValueError("object_id must be non-empty")
        if not self.canonical_name.strip():
            raise ValueError("canonical_name must be non-empty")
        if self.count < 1:
            raise ValueError("count must be >= 1")


@dataclass(frozen=True, slots=True)
class ObjectCatalog(SerializableModel):
    scene_id: str
    objects: tuple[DetectedObject, ...]
    metadata: StageRunMetadata

    def __post_init__(self) -> None:
        if not self.scene_id.strip():
            raise ValueError("scene_id must be non-empty")
        seen: set[str] = set()
        for detected_object in self.objects:
            if detected_object.object_id in seen:
                raise ValueError(f"duplicate object_id detected: {detected_object.object_id}")
            seen.add(detected_object.object_id)


@dataclass(frozen=True, slots=True)
class ObjectDescription(SerializableModel):
    object_id: str
    canonical_name: str
    prompt_text: str
    visual_attributes: dict[str, str] | None = None
    material_attributes: dict[str, str] | None = None
    orientation_hints: dict[str, str] | None = None
    metadata: StageRunMetadata | None = None

    def __post_init__(self) -> None:
        if not self.object_id.strip():
            raise ValueError("object_id must be non-empty")
        if not self.canonical_name.strip():
            raise ValueError("canonical_name must be non-empty")
        if not self.prompt_text.strip():
            raise ValueError("prompt_text must be non-empty")


@dataclass(frozen=True, slots=True)
class SizePrior(SerializableModel):
    object_id: str
    dimensions_m: dict[str, float] | None = None
    relative_scale_to_scene: float | None = None
    source: str = "unknown"
    metadata: StageRunMetadata | None = None

    def __post_init__(self) -> None:
        if not self.object_id.strip():
            raise ValueError("object_id must be non-empty")
        if self.dimensions_m is not None:
            for axis, value in self.dimensions_m.items():
                if value <= 0:
                    raise ValueError(f"dimensions_m[{axis}] must be positive")
        if self.relative_scale_to_scene is not None and self.relative_scale_to_scene <= 0:
            raise ValueError("relative_scale_to_scene must be positive")
        if not self.source.strip():
            raise ValueError("source must be non-empty")


@dataclass(frozen=True, slots=True)
class SizeInferenceReport(SerializableModel):
    scene_description: str = ""
    anchor_object_ids: tuple[str, ...] = ()
    scene_scale_summary: str = ""
    objects: dict[str, dict[str, JsonValue]] | None = None
    metadata: StageRunMetadata | None = None


@dataclass(frozen=True, slots=True)
class SizeInferenceResult(SerializableModel):
    size_priors: tuple[SizePrior, ...]
    size_inference_report: SizeInferenceReport | None = None


@dataclass(frozen=True, slots=True)
class ContainmentRelation(SerializableModel):
    parent_object_id: str
    child_object_id: str
    relation_type: RelationType
    confidence: float | None = None
    evidence: str | None = None

    def __post_init__(self) -> None:
        if not self.parent_object_id.strip() or not self.child_object_id.strip():
            raise ValueError("parent_object_id and child_object_id must be non-empty")
        if self.parent_object_id == self.child_object_id:
            raise ValueError("parent_object_id and child_object_id must be distinct")
        if self.confidence is not None and not (0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence must be between 0.0 and 1.0")


@dataclass(frozen=True, slots=True)
class SceneRelationGraph(SerializableModel):
    scene_id: str
    relations: tuple[ContainmentRelation, ...]
    root_object_ids: tuple[str, ...] = ()
    metadata: StageRunMetadata | None = None

    def __post_init__(self) -> None:
        if not self.scene_id.strip():
            raise ValueError("scene_id must be non-empty")
        self._validate_acyclic()

    def _validate_acyclic(self) -> None:
        edges: dict[str, tuple[str, ...]] = {}
        for relation in self.relations:
            edges.setdefault(relation.parent_object_id, tuple())
            edges[relation.parent_object_id] = edges[relation.parent_object_id] + (relation.child_object_id,)

        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(node: str) -> None:
            if node in visited:
                return
            if node in visiting:
                raise ValueError("SceneRelationGraph must be acyclic")
            visiting.add(node)
            for child in edges.get(node, ()):  # pragma: no branch - tiny traversal
                visit(child)
            visiting.remove(node)
            visited.add(node)

        all_nodes = set(self.root_object_ids)
        for relation in self.relations:
            all_nodes.add(relation.parent_object_id)
            all_nodes.add(relation.child_object_id)
        for node in all_nodes:
            visit(node)


@dataclass(frozen=True, slots=True)
class GeneratedObjectAsset(SerializableModel):
    object_id: str
    mesh_obj: ArtifactRef
    mesh_mtl: ArtifactRef | None = None
    texture_image: ArtifactRef | None = None
    preview_image: ArtifactRef | None = None
    provider_asset_id: str | None = None
    asset_local_pose: ObjectPose | None = None
    metadata: StageRunMetadata | None = None

    def __post_init__(self) -> None:
        if not self.object_id.strip():
            raise ValueError("object_id must be non-empty")
        if self.mesh_obj.artifact_type not in {"mesh", "mesh_obj"}:
            raise ValueError("mesh_obj artifact_type must be 'mesh' or 'mesh_obj'")
        if (
            self.asset_local_pose is not None
            and self.asset_local_pose.object_id != self.object_id
        ):
            raise ValueError("asset_local_pose.object_id must match GeneratedObjectAsset.object_id")


@dataclass(frozen=True, slots=True)
class ObjectAssetCatalog(SerializableModel):
    scene_id: str
    assets: tuple[GeneratedObjectAsset, ...]
    metadata: StageRunMetadata | None = None

    def __post_init__(self) -> None:
        if not self.scene_id.strip():
            raise ValueError("scene_id must be non-empty")
        seen: set[str] = set()
        for asset in self.assets:
            if asset.object_id in seen:
                raise ValueError(f"duplicate asset object_id detected: {asset.object_id}")
            seen.add(asset.object_id)
