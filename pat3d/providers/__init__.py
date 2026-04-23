from __future__ import annotations

from importlib import import_module

__all__ = [
    "HeuristicObjectCatalogBuilder",
    "CurrentDepthEstimator",
    "CurrentMeshSimplifier",
    "CurrentSegmenter",
    "SAM3Segmenter",
    "CurrentTextTo3DProvider",
    "LegacyDiffSimPhysicsOptimizer",
    "LegacyLayoutPlanner",
    "LegacySceneRenderer",
    "OpenAITextToImageProvider",
    "OpenAIStructuredLLM",
    "PassthroughPhysicsOptimizer",
    "SAM3DImageTo3DProvider",
    "SAM3DMultiObjectImageTo3DProvider",
    "SAM3DMultiObjectLayoutBuilder",
    "SAM3DTextTo3DProvider",
    "PromptFileObjectTextDescriber",
    "PromptFileRelationExtractor",
    "PromptFileSizePriorEstimator",
]


_PROVIDER_MODULES = {
    "HeuristicObjectCatalogBuilder": "catalog_builder",
    "CurrentDepthEstimator": "current_depth",
    "CurrentMeshSimplifier": "current_mesh_simplifier",
    "CurrentSegmenter": "current_segmentation",
    "SAM3Segmenter": "sam3_segmentation",
    "CurrentTextTo3DProvider": "current_text_to_3d",
    "LegacyDiffSimPhysicsOptimizer": "legacy_physics_optimizer",
    "LegacyLayoutPlanner": "legacy_layout_bridge",
    "LegacySceneRenderer": "legacy_scene_renderer",
    "OpenAITextToImageProvider": "openai_text_to_image",
    "OpenAIStructuredLLM": "openai_structured_llm",
    "PassthroughPhysicsOptimizer": "passthrough_physics",
    "SAM3DImageTo3DProvider": "_sam3d_image_to_3d_impl",
    "SAM3DMultiObjectImageTo3DProvider": "_sam3d_multi_object_image_to_3d_impl",
    "SAM3DMultiObjectLayoutBuilder": "_sam3d_multi_object_layout_impl",
    "SAM3DTextTo3DProvider": "_sam3d_text_to_3d_impl",
    "PromptFileObjectTextDescriber": "prompt_based_extractors",
    "PromptFileRelationExtractor": "prompt_based_extractors",
    "PromptFileSizePriorEstimator": "prompt_based_extractors",
}


def __getattr__(name: str):
    module_name = _PROVIDER_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(f".{module_name}", __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
