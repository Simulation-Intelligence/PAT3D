from pat3d.runtime.config import (
    PIPELINE_OPTIONAL_PROVIDER_ROLES,
    PIPELINE_REQUIRED_PROVIDER_ROLES,
    ProviderBinding,
    RuntimeConfig,
    make_paper_core_runtime_config,
    make_prompt_backed_runtime_config,
)
from pat3d.runtime.execution import (
    artifact_ref_from_dict,
    artifact_ref_map_from_dict,
    build_and_run_pipeline,
    execute_pipeline,
    load_json_file,
    object_catalog_from_dict,
    scene_request_from_dict,
    string_list_from_json_file,
    to_jsonable,
    write_execution_result,
)
from pat3d.runtime.registry import PipelineRegistry, ProviderRegistry, RuntimeRegistryError


def build_pipeline_from_config(*args, **kwargs):
    from pat3d.runtime.builders import build_pipeline_from_config as _impl

    return _impl(*args, **kwargs)


def default_pipeline_registry(*args, **kwargs):
    from pat3d.runtime.builders import default_pipeline_registry as _impl

    return _impl(*args, **kwargs)


def default_provider_registry(*args, **kwargs):
    from pat3d.runtime.builders import default_provider_registry as _impl

    return _impl(*args, **kwargs)


def build_legacy_preprocess_args(*args, **kwargs):
    from pat3d.runtime.legacy_preprocess import build_legacy_preprocess_args as _impl

    return _impl(*args, **kwargs)


def run_legacy_preprocess(*args, **kwargs):
    from pat3d.runtime.legacy_preprocess import run_legacy_preprocess as _impl

    return _impl(*args, **kwargs)


def run_legacy_preprocess_from_args(*args, **kwargs):
    from pat3d.runtime.legacy_preprocess import run_legacy_preprocess_from_args as _impl

    return _impl(*args, **kwargs)

__all__ = [
    "PIPELINE_OPTIONAL_PROVIDER_ROLES",
    "PIPELINE_REQUIRED_PROVIDER_ROLES",
    "PipelineRegistry",
    "ProviderBinding",
    "ProviderRegistry",
    "RuntimeConfig",
    "RuntimeRegistryError",
    "artifact_ref_from_dict",
    "artifact_ref_map_from_dict",
    "build_and_run_pipeline",
    "build_legacy_preprocess_args",
    "build_pipeline_from_config",
    "default_pipeline_registry",
    "default_provider_registry",
    "execute_pipeline",
    "load_json_file",
    "make_paper_core_runtime_config",
    "make_prompt_backed_runtime_config",
    "object_catalog_from_dict",
    "run_legacy_preprocess",
    "run_legacy_preprocess_from_args",
    "scene_request_from_dict",
    "string_list_from_json_file",
    "to_jsonable",
    "write_execution_result",
]
