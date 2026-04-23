from __future__ import annotations

import inspect
import json
from json import JSONDecodeError
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from pat3d.models.common import ArtifactRef
from pat3d.models.objects import DetectedObject, ObjectCatalog
from pat3d.models.scene import SceneRequest
from pat3d.runtime.config import RuntimeConfig
from pat3d.runtime.errors import coerce_runtime_failure
from pat3d.runtime.registry import PipelineRegistry, ProviderRegistry
from pat3d.storage.runs import make_stage_metadata


def load_json_file(path: str | Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise coerce_runtime_failure(
            error,
            phase="input_loading",
            code="input_not_found",
            user_message=f"Input file '{path}' does not exist.",
            technical_message=str(error),
            retryable=False,
            details={"path": str(path)},
        ) from error
    except JSONDecodeError as error:
        raise coerce_runtime_failure(
            error,
            phase="input_loading",
            code="invalid_json",
            user_message=f"Input file '{path}' is not valid JSON.",
            technical_message=str(error),
            retryable=False,
            details={"path": str(path)},
        ) from error
    except OSError as error:
        raise coerce_runtime_failure(
            error,
            phase="input_loading",
            code="input_read_failed",
            user_message=f"Could not read input file '{path}'.",
            technical_message=str(error),
            retryable=False,
            details={"path": str(path)},
        ) from error


def artifact_ref_from_dict(data: Mapping[str, Any]) -> ArtifactRef:
    return ArtifactRef(
        artifact_type=str(data["artifact_type"]),
        path=str(data["path"]),
        format=str(data["format"]),
        role=str(data["role"]) if data.get("role") is not None else None,
        metadata_path=(
            str(data["metadata_path"]) if data.get("metadata_path") is not None else None
        ),
    )


def scene_request_from_dict(data: Mapping[str, Any]) -> SceneRequest:
    reference_image = data.get("reference_image")
    return SceneRequest(
        scene_id=str(data["scene_id"]),
        text_prompt=str(data["text_prompt"]) if data.get("text_prompt") is not None else None,
        requested_objects=(
            tuple(str(item) for item in data["requested_objects"])
            if data.get("requested_objects") is not None
            else None
        ),
        reference_image=(
            artifact_ref_from_dict(reference_image)
            if isinstance(reference_image, Mapping)
            else None
        ),
        tags=tuple(str(item) for item in data.get("tags", ())),
        constraints=(
            dict(data["constraints"]) if data.get("constraints") is not None else None
        ),
    )


def detected_object_from_dict(data: Mapping[str, Any]) -> DetectedObject:
    return DetectedObject(
        object_id=str(data["object_id"]),
        canonical_name=str(data["canonical_name"]),
        display_name=(
            str(data["display_name"]) if data.get("display_name") is not None else None
        ),
        count=int(data["count"]),
        source_instance_ids=tuple(str(item) for item in data.get("source_instance_ids", ())),
        attributes=(
            {str(key): str(value) for key, value in dict(data["attributes"]).items()}
            if data.get("attributes") is not None
            else None
        ),
    )


def object_catalog_from_dict(data: Mapping[str, Any]) -> ObjectCatalog:
    return ObjectCatalog(
        scene_id=str(data["scene_id"]),
        objects=tuple(
            detected_object_from_dict(item) for item in data.get("objects", ())
        ),
        metadata=make_stage_metadata(
            stage_name="runtime_input_object_catalog",
            provider_name="runtime",
        ),
    )


def artifact_ref_map_from_dict(data: Mapping[str, Any]) -> dict[str, ArtifactRef]:
    return {
        str(key): artifact_ref_from_dict(value)
        for key, value in dict(data).items()
    }


def string_list_from_json_file(path: str | Path) -> list[str]:
    data = load_json_file(path)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list at '{path}'")
    return [str(item) for item in data]


def execute_pipeline(
    pipeline: Any,
    request: SceneRequest,
    *,
    object_hints: Sequence[str] | None = None,
    object_catalog: ObjectCatalog | None = None,
    object_reference_images: Mapping[str, ArtifactRef] | None = None,
) -> Any:
    run_method = getattr(pipeline, "run")
    supported = set(inspect.signature(run_method).parameters)
    kwargs: dict[str, Any] = {}
    effective_object_hints = object_hints
    if effective_object_hints is None and request.requested_objects is not None:
        effective_object_hints = list(request.requested_objects)
    if effective_object_hints is not None and "object_hints" in supported:
        kwargs["object_hints"] = list(effective_object_hints)
    if object_catalog is not None and "object_catalog" in supported:
        kwargs["object_catalog"] = object_catalog
    if object_reference_images is not None and "object_reference_images" in supported:
        kwargs["object_reference_images"] = dict(object_reference_images)
    try:
        return run_method(request, **kwargs)
    except Exception as error:
        raise coerce_runtime_failure(
            error,
            phase="pipeline_run",
            code="pipeline_run_failed",
            user_message="The pipeline failed while executing the scene request.",
            technical_message=str(error),
            retryable=False,
            details={"scene_id": request.scene_id},
        ) from error


def build_and_run_pipeline(
    runtime_config: RuntimeConfig | dict[str, Any],
    request: SceneRequest,
    *,
    object_hints: Sequence[str] | None = None,
    object_catalog: ObjectCatalog | None = None,
    object_reference_images: Mapping[str, ArtifactRef] | None = None,
    provider_registry: ProviderRegistry | None = None,
    pipeline_registry: PipelineRegistry | None = None,
) -> Any:
    try:
        from pat3d.runtime.builders import build_pipeline_from_config

        pipeline = build_pipeline_from_config(
            runtime_config,
            provider_registry=provider_registry,
            pipeline_registry=pipeline_registry,
        )
    except Exception as error:
        raise coerce_runtime_failure(
            error,
            phase="pipeline_build",
            code="pipeline_build_failed",
            user_message="Could not prepare the configured pipeline.",
            technical_message=str(error),
            retryable=False,
            details={"scene_id": request.scene_id},
        ) from error
    return execute_pipeline(
        pipeline,
        request,
        object_hints=object_hints,
        object_catalog=object_catalog,
        object_reference_images=object_reference_images,
    )


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def write_execution_result(result: Any, path: str | Path) -> None:
    try:
        Path(path).write_text(
            json.dumps(to_jsonable(result), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except OSError as error:
        raise coerce_runtime_failure(
            error,
            phase="result_write",
            code="result_write_failed",
            user_message=f"Could not write runtime output to '{path}'.",
            technical_message=str(error),
            retryable=False,
            details={"path": str(path)},
        ) from error
