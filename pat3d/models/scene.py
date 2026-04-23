from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .common import ArtifactRef, JsonValue, SerializableModel, StageRunMetadata


@dataclass(frozen=True, slots=True)
class SceneRequest(SerializableModel):
    scene_id: str
    text_prompt: str | None = None
    requested_objects: tuple[str, ...] | None = None
    reference_image: ArtifactRef | None = None
    tags: tuple[str, ...] = ()
    constraints: dict[str, JsonValue] | None = None

    def __post_init__(self) -> None:
        if not self.scene_id.strip():
            raise ValueError("scene_id must be non-empty")
        if not self.text_prompt and self.reference_image is None:
            raise ValueError("either text_prompt or reference_image must be provided")


@dataclass(frozen=True, slots=True)
class ReferenceImageResult(SerializableModel):
    request: SceneRequest
    image: ArtifactRef
    generation_prompt: str | None
    seed: int | None
    width: int
    height: int
    metadata: StageRunMetadata

    def __post_init__(self) -> None:
        if self.image.artifact_type != "image":
            raise ValueError("image artifact_type must be 'image'")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("image dimensions must be positive")


@dataclass(frozen=True, slots=True)
class DepthResult(SerializableModel):
    image: ArtifactRef
    depth_array: ArtifactRef
    depth_visualization: ArtifactRef | None
    point_cloud: ArtifactRef | None
    focal_length_px: float | None
    metadata: StageRunMetadata

    def __post_init__(self) -> None:
        if self.focal_length_px is not None and self.focal_length_px <= 0:
            raise ValueError("focal_length_px must be positive when provided")


@dataclass(frozen=True, slots=True)
class MaskInstance(SerializableModel):
    instance_id: str
    label: str | None
    mask: ArtifactRef
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float | None = None
    area_px: int | None = None

    def __post_init__(self) -> None:
        if not self.instance_id.strip():
            raise ValueError("instance_id must be non-empty")
        if len(self.bbox_xyxy) != 4:
            raise ValueError("bbox_xyxy must contain four values")
        if self.confidence is not None and not (0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence must be between 0.0 and 1.0")
        if self.area_px is not None and self.area_px < 0:
            raise ValueError("area_px must be non-negative")


@dataclass(frozen=True, slots=True)
class SegmentationResult(SerializableModel):
    image: ArtifactRef
    instances: tuple[MaskInstance, ...]
    composite_visualization: ArtifactRef | None
    metadata: StageRunMetadata

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for instance in self.instances:
            if instance.instance_id in seen:
                raise ValueError(f"duplicate instance_id detected: {instance.instance_id}")
            seen.add(instance.instance_id)
