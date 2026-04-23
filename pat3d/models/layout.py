from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .common import ArtifactRef, SerializableModel, StageRunMetadata
from .objects import ObjectAssetCatalog, SceneRelationGraph


@dataclass(frozen=True, slots=True)
class ObjectPose(SerializableModel):
    object_id: str
    translation_xyz: tuple[float, float, float]
    rotation_type: str
    rotation_value: tuple[float, ...]
    scale_xyz: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        if not self.object_id.strip():
            raise ValueError("object_id must be non-empty")
        if len(self.translation_xyz) != 3:
            raise ValueError("translation_xyz must contain three values")
        if not self.rotation_type.strip():
            raise ValueError("rotation_type must be non-empty")
        if len(self.rotation_value) == 0:
            raise ValueError("rotation_value must contain at least one value")
        if self.scale_xyz is not None:
            if len(self.scale_xyz) != 3:
                raise ValueError("scale_xyz must contain three values")
            if any(value <= 0 for value in self.scale_xyz):
                raise ValueError("scale_xyz values must be positive")


@dataclass(frozen=True, slots=True)
class SceneLayout(SerializableModel):
    scene_id: str
    object_poses: tuple[ObjectPose, ...]
    layout_space: str
    support_graph: SceneRelationGraph | None = None
    artifacts: tuple[ArtifactRef, ...] = ()
    metadata: StageRunMetadata | None = None

    def __post_init__(self) -> None:
        if not self.scene_id.strip():
            raise ValueError("scene_id must be non-empty")
        if not self.layout_space.strip():
            raise ValueError("layout_space must be non-empty")
        seen: set[str] = set()
        for pose in self.object_poses:
            if pose.object_id in seen:
                raise ValueError(f"duplicate pose object_id detected: {pose.object_id}")
            seen.add(pose.object_id)


@dataclass(frozen=True, slots=True)
class PhysicsReadyScene(SerializableModel):
    scene_id: str
    layout: SceneLayout
    simulation_meshes: tuple[ArtifactRef, ...]
    object_poses: tuple[ObjectPose, ...]
    collision_settings: dict[str, float | int | bool] | None = None
    metadata: StageRunMetadata | None = None

    def __post_init__(self) -> None:
        if not self.scene_id.strip():
            raise ValueError("scene_id must be non-empty")
        layout_ids = {pose.object_id for pose in self.layout.object_poses}
        pose_ids = {pose.object_id for pose in self.object_poses}
        if layout_ids != pose_ids:
            raise ValueError("PhysicsReadyScene.object_poses must match SceneLayout.object_poses by object_id")


@dataclass(frozen=True, slots=True)
class PhysicsOptimizationResult(SerializableModel):
    scene_id: str
    initial_scene: PhysicsReadyScene
    optimized_object_poses: tuple[ObjectPose, ...]
    metrics: dict[str, float] | None = None
    artifacts: tuple[ArtifactRef, ...] = ()
    metadata: StageRunMetadata | None = None

    def __post_init__(self) -> None:
        if not self.scene_id.strip():
            raise ValueError("scene_id must be non-empty")
        initial_ids = {pose.object_id for pose in self.initial_scene.object_poses}
        optimized_ids = {pose.object_id for pose in self.optimized_object_poses}
        if initial_ids != optimized_ids:
            raise ValueError("optimized_object_poses must match initial_scene object IDs")


@dataclass(frozen=True, slots=True)
class RenderResult(SerializableModel):
    scene_id: str
    render_images: tuple[ArtifactRef, ...]
    camera_metadata: ArtifactRef | None = None
    render_config: dict[str, str | int | float | bool] | None = None
    metadata: StageRunMetadata | None = None

    def __post_init__(self) -> None:
        if not self.scene_id.strip():
            raise ValueError("scene_id must be non-empty")
        if not self.render_images:
            raise ValueError("render_images must contain at least one artifact")
