from __future__ import annotations

from pat3d.contracts import (
    DepthEstimator,
    MeshSimplifier,
    PhysicsOptimizer,
    SceneRenderer,
    Segmenter,
    StructuredLLM,
    TextToImageProvider,
    TextTo3DProvider,
)
from pat3d.pipelines.first_contract_slice import FirstContractSlicePipeline
from pat3d.pipelines.paper_core_pipeline import PaperCorePipeline
from pat3d.stages import (
    LayoutInitializationStage,
    ObjectAssetGenerationStage,
    ObjectRelationExtractionStage,
    PhysicsOptimizationStage,
    ReferenceImageStage,
    SceneUnderstandingStage,
    SimulationPreparationStage,
    VisualizationExportStage,
)
from pat3d.storage import make_stage_metadata
from pat3d.stages.layout_initialization import SceneLayoutBuilder


def _build_default_scene_understanding_providers() -> tuple[DepthEstimator, Segmenter]:
    from pat3d.providers import CurrentDepthEstimator, CurrentSegmenter

    return CurrentDepthEstimator(), CurrentSegmenter()


def _build_object_catalog_builder():
    from pat3d.providers import HeuristicObjectCatalogBuilder

    return HeuristicObjectCatalogBuilder()


def _build_prompt_backed_relation_providers(
    structured_llm: StructuredLLM,
    *,
    size_inference_pause_on_failure: bool = False,
):
    from pat3d.providers import (
        PromptFileObjectTextDescriber,
        PromptFileRelationExtractor,
        PromptFileSizePriorEstimator,
    )

    return (
        PromptFileObjectTextDescriber(structured_llm),
        PromptFileRelationExtractor(structured_llm),
        PromptFileSizePriorEstimator(
            structured_llm,
            pause_on_failure=size_inference_pause_on_failure,
        ),
    )


def build_prompt_backed_contract_slice_pipeline(
    *,
    structured_llm: StructuredLLM,
    text_to_image_provider: TextToImageProvider | None = None,
    depth_estimator: DepthEstimator | None = None,
    segmenter: Segmenter | None = None,
    size_inference_pause_on_failure: bool = False,
) -> FirstContractSlicePipeline:
    object_catalog_builder = _build_object_catalog_builder()
    resolved_depth_estimator = depth_estimator
    resolved_segmenter = segmenter
    if resolved_depth_estimator is None or resolved_segmenter is None:
        default_depth_estimator, default_segmenter = _build_default_scene_understanding_providers()
        resolved_depth_estimator = resolved_depth_estimator or default_depth_estimator
        resolved_segmenter = resolved_segmenter or default_segmenter
    object_describer, relation_extractor, size_inferer = _build_prompt_backed_relation_providers(
        structured_llm,
        size_inference_pause_on_failure=size_inference_pause_on_failure,
    )
    return FirstContractSlicePipeline(
        reference_image_stage=ReferenceImageStage(
            metadata_factory=make_stage_metadata,
            text_to_image_provider=text_to_image_provider,
        ),
        scene_understanding_stage=SceneUnderstandingStage(
            depth_estimator=resolved_depth_estimator,
            segmenter=resolved_segmenter,
            object_catalog_builder=object_catalog_builder,
        ),
        object_relation_stage=ObjectRelationExtractionStage(
            object_describer=object_describer,
            relation_extractor=relation_extractor,
            size_inferer=size_inferer,
            object_catalog_builder=object_catalog_builder,
        ),
    )


def build_paper_core_pipeline(
    *,
    structured_llm: StructuredLLM,
    text_to_image_provider: TextToImageProvider | None = None,
    text_to_3d_provider: TextTo3DProvider,
    layout_builder: SceneLayoutBuilder,
    mesh_simplifier: MeshSimplifier,
    depth_estimator: DepthEstimator | None = None,
    segmenter: Segmenter | None = None,
    physics_optimizer: PhysicsOptimizer | None = None,
    scene_renderer: SceneRenderer | None = None,
    size_inference_pause_on_failure: bool = False,
    workspace_root: str | None = None,
) -> PaperCorePipeline:
    _ = workspace_root
    first_contract_slice_pipeline = build_prompt_backed_contract_slice_pipeline(
        structured_llm=structured_llm,
        text_to_image_provider=text_to_image_provider,
        depth_estimator=depth_estimator,
        segmenter=segmenter,
        size_inference_pause_on_failure=size_inference_pause_on_failure,
    )
    return PaperCorePipeline(
        first_contract_slice_pipeline=first_contract_slice_pipeline,
        object_asset_stage=ObjectAssetGenerationStage(text_to_3d_provider),
        layout_initialization_stage=LayoutInitializationStage(layout_builder),
        simulation_preparation_stage=SimulationPreparationStage(mesh_simplifier),
        physics_optimization_stage=(
            PhysicsOptimizationStage(physics_optimizer)
            if physics_optimizer is not None
            else None
        ),
        visualization_stage=(
            VisualizationExportStage(scene_renderer)
            if scene_renderer is not None
            else None
        ),
    )
