from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from pat3d.models import (
    ArtifactRef,
    DepthResult,
    GeneratedObjectAsset,
    ObjectAssetCatalog,
    ObjectCatalog,
    ObjectDescription,
    PhysicsOptimizationResult,
    PhysicsReadyScene,
    ReferenceImageResult,
    RenderResult,
    SceneLayout,
    SceneRelationGraph,
    SceneRequest,
    SegmentationResult,
    SizeInferenceResult,
    SizePrior,
    StructuredPromptRequest,
    StructuredPromptResult,
)


@runtime_checkable
class StructuredLLM(Protocol):
    def generate(self, request: StructuredPromptRequest) -> StructuredPromptResult:
        ...


@runtime_checkable
class TextToImageProvider(Protocol):
    def generate(self, request: SceneRequest) -> ReferenceImageResult:
        ...


@runtime_checkable
class DepthEstimator(Protocol):
    def predict(self, reference_image: ReferenceImageResult) -> DepthResult:
        ...


@runtime_checkable
class Segmenter(Protocol):
    def segment(
        self,
        reference_image: ReferenceImageResult,
        object_hints: Sequence[str] | None = None,
    ) -> SegmentationResult:
        ...


@runtime_checkable
class ObjectTextDescriber(Protocol):
    def describe_objects(
        self,
        object_catalog: ObjectCatalog,
        reference_image: ReferenceImageResult,
        segmentation: SegmentationResult,
    ) -> Sequence[ObjectDescription]:
        ...


@runtime_checkable
class RelationExtractor(Protocol):
    def extract(
        self,
        object_catalog: ObjectCatalog,
        reference_image: ReferenceImageResult,
        segmentation: SegmentationResult,
        *,
        depth_result: DepthResult | None = None,
    ) -> SceneRelationGraph:
        ...


@runtime_checkable
class SizePriorEstimator(Protocol):
    def estimate(
        self,
        object_catalog: ObjectCatalog,
        reference_image: ReferenceImageResult,
        *,
        segmentation: SegmentationResult | None = None,
        depth_result: DepthResult | None = None,
        object_descriptions: Sequence[ObjectDescription] = (),
        relation_graph: SceneRelationGraph | None = None,
    ) -> SizeInferenceResult:
        ...


@runtime_checkable
class TextTo3DProvider(Protocol):
    def generate(
        self,
        object_description: ObjectDescription,
        *,
        object_reference_image: ArtifactRef | None = None,
    ) -> GeneratedObjectAsset:
        ...


@runtime_checkable
class MeshSimplifier(Protocol):
    def simplify(
        self,
        scene_layout: SceneLayout,
        object_assets: ObjectAssetCatalog,
    ) -> PhysicsReadyScene:
        ...


@runtime_checkable
class PhysicsOptimizer(Protocol):
    def optimize(self, physics_ready_scene: PhysicsReadyScene) -> PhysicsOptimizationResult:
        ...


@runtime_checkable
class SceneRenderer(Protocol):
    def render(
        self,
        scene_state: SceneLayout | PhysicsReadyScene | PhysicsOptimizationResult,
    ) -> RenderResult:
        ...
