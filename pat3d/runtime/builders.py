from __future__ import annotations

from typing import Any

from pat3d.pipelines.factories import (
    build_paper_core_pipeline,
    build_prompt_backed_contract_slice_pipeline,
)
from pat3d.models import ArtifactRef, DepthResult, SegmentationResult
from pat3d.runtime.config import ProviderBinding, RuntimeConfig
from pat3d.runtime.errors import coerce_runtime_failure
from pat3d.runtime.registry import PipelineRegistry, ProviderRegistry
from pat3d.storage import make_stage_metadata


def _build_openai_structured_llm(binding: ProviderBinding) -> Any:
    from pat3d.providers.openai_structured_llm import OpenAIStructuredLLM

    return OpenAIStructuredLLM(**binding.options)


def _build_openai_text_to_image(binding: ProviderBinding) -> Any:
    from pat3d.providers.openai_text_to_image import OpenAITextToImageProvider

    return OpenAITextToImageProvider(**binding.options)


def _build_smoke_structured_llm(binding: ProviderBinding) -> Any:
    from pat3d.providers.release_smoke import SmokeStructuredLLM

    return SmokeStructuredLLM(**binding.options)


def _build_smoke_text_to_image(binding: ProviderBinding) -> Any:
    from pat3d.providers.release_smoke import SmokeTextToImageProvider

    return SmokeTextToImageProvider(**binding.options)


def _build_current_depth(binding: ProviderBinding) -> Any:
    from pat3d.providers.current_depth import CurrentDepthEstimator

    return CurrentDepthEstimator(**binding.options)


def _build_smoke_depth_estimator(binding: ProviderBinding) -> Any:
    from pat3d.providers.release_smoke import SmokeDepthEstimator

    return SmokeDepthEstimator(**binding.options)


def _build_current_segmenter(binding: ProviderBinding) -> Any:
    from pat3d.providers.current_segmentation import CurrentSegmenter

    return CurrentSegmenter(**binding.options)


def _build_smoke_segmenter(binding: ProviderBinding) -> Any:
    from pat3d.providers.release_smoke import SmokeSegmenter

    return SmokeSegmenter(**binding.options)


def _build_sam3_segmenter(binding: ProviderBinding) -> Any:
    from pat3d.providers.sam3_segmentation import SAM3Segmenter

    return SAM3Segmenter(**binding.options)


def _build_current_text_to_3d(binding: ProviderBinding) -> Any:
    from pat3d.providers.current_text_to_3d import CurrentTextTo3DProvider

    return CurrentTextTo3DProvider(**binding.options)


def _build_primitive_text_to_3d(binding: ProviderBinding) -> Any:
    from pat3d.providers.release_smoke import PrimitiveTextTo3DProvider

    return PrimitiveTextTo3DProvider(**binding.options)


def _build_sam3d_image_to_3d(binding: ProviderBinding) -> Any:
    from pat3d.providers._sam3d_image_to_3d_impl import SAM3DImageTo3DProvider

    return SAM3DImageTo3DProvider(**binding.options)


def _build_sam3d_multi_object_image_to_3d(binding: ProviderBinding) -> Any:
    from pat3d.providers._sam3d_multi_object_image_to_3d_impl import (
        SAM3DMultiObjectImageTo3DProvider,
    )

    return SAM3DMultiObjectImageTo3DProvider(**binding.options)


def _build_current_mesh_simplifier(binding: ProviderBinding) -> Any:
    from pat3d.providers.current_mesh_simplifier import CurrentMeshSimplifier

    return CurrentMeshSimplifier(**binding.options)


def _build_passthrough_mesh_simplifier(binding: ProviderBinding) -> Any:
    from pat3d.providers.current_mesh_simplifier import CurrentMeshSimplifier

    return CurrentMeshSimplifier(simplifier=lambda *args: None, **binding.options)


def _build_legacy_layout(binding: ProviderBinding) -> Any:
    from pat3d.providers.legacy_layout_bridge import LegacyLayoutPlanner

    return LegacyLayoutPlanner(**binding.options)


def _build_heuristic_layout_builder(binding: ProviderBinding) -> Any:
    from pat3d.providers.release_smoke import HeuristicLayoutBuilder

    return HeuristicLayoutBuilder(**binding.options)


def _build_sam3d_multi_object_layout(binding: ProviderBinding) -> Any:
    from pat3d.providers._sam3d_multi_object_layout_impl import SAM3DMultiObjectLayoutBuilder

    return SAM3DMultiObjectLayoutBuilder(**binding.options)


def _build_legacy_physics(binding: ProviderBinding) -> Any:
    from pat3d.providers.legacy_physics_optimizer import LegacyDiffSimPhysicsOptimizer

    return LegacyDiffSimPhysicsOptimizer(**binding.options)


def _build_passthrough_physics(binding: ProviderBinding) -> Any:
    from pat3d.providers.passthrough_physics import PassthroughPhysicsOptimizer

    return PassthroughPhysicsOptimizer(**binding.options)


def _build_legacy_renderer(binding: ProviderBinding) -> Any:
    from pat3d.providers.legacy_scene_renderer import LegacySceneRenderer

    return LegacySceneRenderer(**binding.options)


def _build_geometry_renderer(binding: ProviderBinding) -> Any:
    from pat3d.providers.geometry_scene_renderer import GeometrySceneRenderer

    return GeometrySceneRenderer(**binding.options)


class _NoopDepthEstimator:
    def predict(self, reference_image) -> DepthResult:
        return DepthResult(
            image=reference_image.image,
            depth_array=ArtifactRef(
                artifact_type="depth_array",
                path=reference_image.image.path,
                format=reference_image.image.format,
                role="resume_placeholder_depth",
                metadata_path=None,
            ),
            depth_visualization=None,
            point_cloud=None,
            focal_length_px=None,
            metadata=make_stage_metadata(
                stage_name="depth",
                provider_name="noop_depth_estimator",
                notes=("resume_placeholder",),
            ),
        )


class _NoopSegmenter:
    def segment(self, reference_image, object_hints=None) -> SegmentationResult:
        _ = object_hints
        return SegmentationResult(
            image=reference_image.image,
            instances=(),
            composite_visualization=None,
            metadata=make_stage_metadata(
                stage_name="segmentation",
                provider_name="noop_segmenter",
                notes=("resume_placeholder",),
            ),
        )


def _build_noop_depth_estimator(binding: ProviderBinding) -> Any:
    _ = binding
    return _NoopDepthEstimator()


def _build_noop_segmenter(binding: ProviderBinding) -> Any:
    _ = binding
    return _NoopSegmenter()


def default_provider_registry() -> ProviderRegistry:
    return ProviderRegistry(
        builders={
            "smoke_structured_llm": _build_smoke_structured_llm,
            "openai_structured_llm": _build_openai_structured_llm,
            "smoke_text_to_image": _build_smoke_text_to_image,
            "openai_text_to_image": _build_openai_text_to_image,
            "smoke_depth_estimator": _build_smoke_depth_estimator,
            "current_depth": _build_current_depth,
            "noop_depth_estimator": _build_noop_depth_estimator,
            "smoke_segmenter": _build_smoke_segmenter,
            "current_segmenter": _build_current_segmenter,
            "sam3_segmenter": _build_sam3_segmenter,
            "noop_segmenter": _build_noop_segmenter,
            "primitive_text_to_3d": _build_primitive_text_to_3d,
            "current_text_to_3d": _build_current_text_to_3d,
            "sam3d_image_to_3d": _build_sam3d_image_to_3d,
            "sam3d_multi_object_image_to_3d": _build_sam3d_multi_object_image_to_3d,
            "sam3d_text_to_3d": _build_sam3d_image_to_3d,
            "passthrough_mesh_simplifier": _build_passthrough_mesh_simplifier,
            "current_mesh_simplifier": _build_current_mesh_simplifier,
            "heuristic_layout_builder": _build_heuristic_layout_builder,
            "legacy_layout": _build_legacy_layout,
            "sam3d_multi_object_layout": _build_sam3d_multi_object_layout,
            "legacy_physics": _build_legacy_physics,
            "passthrough_physics": _build_passthrough_physics,
            "legacy_renderer": _build_legacy_renderer,
            "geometry_scene_renderer": _build_geometry_renderer,
        }
    )


def default_pipeline_registry() -> PipelineRegistry:
    return PipelineRegistry(
        factories={
            "prompt_backed_contract_slice": build_prompt_backed_contract_slice_pipeline,
            "paper_core": build_paper_core_pipeline,
        }
    )


def build_pipeline_from_config(
    config: RuntimeConfig | dict[str, Any],
    *,
    provider_registry: ProviderRegistry | None = None,
    pipeline_registry: PipelineRegistry | None = None,
) -> Any:
    runtime_config = (
        config if isinstance(config, RuntimeConfig) else RuntimeConfig.from_dict(config)
    )
    runtime_config.validate()

    resolved_provider_registry = provider_registry or default_provider_registry()
    resolved_pipeline_registry = pipeline_registry or default_pipeline_registry()

    factory_kwargs: dict[str, Any] = dict(runtime_config.pipeline_options)
    for role, binding in runtime_config.enabled_provider_bindings().items():
        try:
            factory_kwargs[role] = resolved_provider_registry.build(binding)
        except Exception as error:
            raise coerce_runtime_failure(
                error,
                phase="provider_build",
                code="provider_build_failed",
                user_message=f"Could not initialize the configured provider for '{role}'.",
                technical_message=str(error),
                provider_role=role,
                provider_kind=binding.kind,
                retryable=False,
                details={"pipeline": runtime_config.pipeline},
            ) from error

    try:
        return resolved_pipeline_registry.build(runtime_config.pipeline, **factory_kwargs)
    except Exception as error:
        raise coerce_runtime_failure(
            error,
            phase="pipeline_build",
            code="pipeline_build_failed",
            user_message=f"Could not build pipeline '{runtime_config.pipeline}'.",
            technical_message=str(error),
            retryable=False,
            details={"pipeline": runtime_config.pipeline},
        ) from error
