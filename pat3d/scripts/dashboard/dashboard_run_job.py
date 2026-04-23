from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timezone
import gc
import importlib
import inspect
import json
import os
from pathlib import Path
import re
import shutil
import sys
import traceback
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pat3d.models import (
    ArtifactRef,
    ContainmentRelation,
    DepthResult,
    GeneratedObjectAsset,
    ObjectDescription,
    ObjectAssetCatalog,
    ObjectCatalog,
    ObjectPose,
    PhysicsOptimizationResult,
    PhysicsReadyScene,
    RelationType,
    RenderResult,
    MaskInstance,
    ReferenceImageResult,
    SceneLayout,
    SceneRelationGraph,
    SceneRequest,
    SegmentationResult,
    SizePrior,
    StageRunMetadata,
    StageRunStatus,
)
from pat3d.pipelines.paper_core_pipeline import _materialize_object_reference_images
from pat3d.runtime.builders import build_pipeline_from_config
from pat3d.runtime.config import ProviderBinding, RuntimeConfig, make_paper_core_runtime_config
from pat3d.runtime.errors import Pat3DRuntimeError, coerce_runtime_failure
from pat3d.runtime.execution import (
    artifact_ref_from_dict,
    detected_object_from_dict,
    scene_request_from_dict,
    write_execution_result,
)
from pat3d.stages.object_assets import (
    clone_generated_asset_for_object_id,
    expand_object_asset_requests,
    instance_object_ids,
)
from pat3d.providers._relation_utils import materialize_size_priors_for_instances
from pat3d.providers.prompt_based_extractors import _default_size_dimensions
from pat3d.stages import (
    LayoutInitializationOutputs,
    ObjectAssetGenerationOutputs,
    ObjectRelationExtractionOutputs,
    PhysicsOptimizationOutputs,
    SceneUnderstandingOutputs,
    SimulationPreparationOutputs,
    VisualizationExportOutputs,
)
from pat3d.storage import make_stage_metadata


STAGE_ORDER = (
    ("reference-image", "Reference image"),
    ("scene-understanding", "Scene understanding"),
    ("object-relation", "Object description and relations"),
    ("object-assets", "Object asset generation"),
    ("layout-initialization", "Layout initialization"),
    ("simulation-preparation", "Simulation preparation"),
    ("physics-optimization", "Physics simulation"),
    ("visualization", "Visualization and export"),
)

CHAT_MODEL_OPTIONS = (
    "gpt-5.4",
    "gpt-5.3",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4.1",
    "gpt-4.1-mini",
    "o4-mini",
    "o3-mini",
)

IMAGE_MODEL_OPTIONS = (
    "gpt-image-1",
    "gpt-image-1.5",
    "dall-e-3",
    "dall-e-2",
)

INFERENCE_SYSTEM_PROMPT = """You are a vision-language planning assistant.

Given a user prompt describing a target scene, identify the list of physical object
types that are expected to appear so downstream segmentation can be seeded.

Return only JSON in the exact form:

{"requested_objects": ["item1", "item2", ...]}

Use singular object nouns where possible. Keep each object concise (1-3 words),
dedupe duplicates, and output no more than 12 items.

Do not include scene/background/support-surface terms such as ground, floor,
tabletop, wall, ceiling, room, scene, background, sky, white background, or
white ground unless the prompt clearly asks for a standalone physical prop with
that exact category."""
REASONING_EFFORT_OPTIONS = ("auto", "low", "medium", "high")
DEFAULT_PHYSICS_SETTINGS: dict[str, bool | int | float] = {
    "diff_sim_enabled": False,
    "end_frame": 300,
    "ground_y_value": -1.1,
    "total_opt_epoch": 50,
    "phys_lr": 0.001,
    "contact_d_hat": 5e-4,
    "contact_eps_velocity": 1e-5,
}
SMOKE_BACKEND_KINDS = {
    "smoke_text_to_image",
    "smoke_depth_estimator",
    "smoke_segmenter",
    "smoke_structured_llm",
    "primitive_text_to_3d",
    "heuristic_layout_builder",
    "passthrough_mesh_simplifier",
}

INFERRED_SCENE_CONTEXT_OBJECTS = {
    "background",
    "ceiling",
    "floor",
    "ground",
    "ground plane",
    "room",
    "scene",
    "sky",
    "surface",
    "table surface",
    "tabletop",
    "wall",
    "white background",
    "white ground",
}
LOCAL_SMOKE_OBJECT_NAMES = {
    "apple",
    "ball",
    "basket",
    "book",
    "bookshelf",
    "bottle",
    "bowl",
    "cabinet",
    "can",
    "counter",
    "cup",
    "desk",
    "mug",
    "nightstand",
    "object",
    "orange",
    "peach",
    "pear",
    "phone",
    "plate",
    "remote",
    "shelf",
    "table",
}
LOCAL_SMOKE_OBJECT_ALIASES = {
    "apples": "apple",
    "baskets": "basket",
    "books": "book",
    "bottles": "bottle",
    "bowls": "bowl",
    "cabinets": "cabinet",
    "cans": "can",
    "counters": "counter",
    "cups": "cup",
    "desks": "desk",
    "mugs": "mug",
    "nightstands": "nightstand",
    "objects": "object",
    "oranges": "orange",
    "peaches": "peach",
    "pears": "pear",
    "phones": "phone",
    "plates": "plate",
    "remotes": "remote",
    "shelves": "shelf",
    "tables": "table",
}
SMOKE_RESULT_ROOT = "results/dashboard_release_smoke"

WORKSPACE_ROOT = Path("results/workspaces")

LEGACY_SCENE_DIRS = (
    Path("data/seg"),
    Path("data/depth"),
    Path("data/layout"),
    Path("data/low_poly"),
    Path("data/organized_obj"),
    Path("results/rendered_images"),
    Path("_phys_result"),
)

LEGACY_SCENE_FILE_PATHS = (
    Path("data/ref_img/{scene_id}.png"),
    Path("data/descrip/{scene_id}.json"),
    Path("data/contain/{scene_id}.json"),
    Path("data/contain_on/{scene_id}.json"),
    Path("data/size/{scene_id}.json"),
    Path("data/items/{scene_id}.json"),
)

LEGACY_SCENE_PREFIX_DIRS = (
    Path("data/raw_obj"),
    Path("data/ref_img_obj"),
)

MANUAL_SEGMENTATION_PROVIDER = "dashboard_manual_segmenter"
RESUMABLE_STAGE_IDS = tuple(stage_id for stage_id, _label in STAGE_ORDER)
RESUME_FROM_CHOICES = RESUMABLE_STAGE_IDS
RESUME_FROM_SCENE_UNDERSTANDING = "scene-understanding"
STAGE_INDEX_BY_ID = {stage_id: index for index, (stage_id, _label) in enumerate(STAGE_ORDER)}
DOWNSTREAM_RUNTIME_KEYS_BY_RETRY_STAGE = {
    "reference-image": (),
    "scene-understanding": (
        ("first_contract_slice", "scene_understanding"),
        ("first_contract_slice", "object_relation"),
        ("object_assets",),
        ("layout_initialization",),
        ("simulation_preparation",),
        ("physics_optimization",),
        ("visualization",),
    ),
    "object-relation": (
        ("first_contract_slice", "object_relation"),
        ("object_assets",),
        ("layout_initialization",),
        ("simulation_preparation",),
        ("physics_optimization",),
        ("visualization",),
    ),
    "object-assets": (
        ("object_assets",),
        ("layout_initialization",),
        ("simulation_preparation",),
        ("physics_optimization",),
        ("visualization",),
    ),
    "layout-initialization": (
        ("layout_initialization",),
        ("simulation_preparation",),
        ("physics_optimization",),
        ("visualization",),
    ),
    "simulation-preparation": (
        ("simulation_preparation",),
        ("physics_optimization",),
        ("visualization",),
    ),
    "physics-optimization": (
        ("physics_optimization",),
        ("visualization",),
    ),
    "visualization": (
        ("visualization",),
    ),
}
PREVIOUS_COMPLETED_STAGE_BY_RETRY_STAGE = {
    "scene-understanding": "reference-image",
    "object-relation": "scene-understanding",
    "object-assets": "object-relation",
    "layout-initialization": "object-assets",
    "simulation-preparation": "layout-initialization",
    "physics-optimization": "simulation-preparation",
    "visualization": "physics-optimization",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp_path.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _path_text(path: Path) -> str:
    return path.as_posix()


def scene_workspace_paths(scene_id: str) -> dict[str, Path]:
    root = WORKSPACE_ROOT / scene_id
    object_relation_root = root / "object_relation"
    object_assets_root = root / "object_assets"
    layout_root = root / "layout"
    physics_root = root / "physics"
    return {
        "root": root,
        "reference_root": root / "reference",
        "depth_root": root / "scene_understanding" / "depth",
        "segmentation_root": root / "scene_understanding" / "segmentation",
        "description_root": object_relation_root / "descriptions",
        "contain_root": object_relation_root / "contain",
        "contain_on_root": object_relation_root / "contain_on",
        "size_root": object_relation_root / "size",
        "items_root": object_relation_root / "items",
        "raw_asset_root": object_assets_root / "raw",
        "object_reference_root": object_assets_root / "reference_images",
        "object_reference_completed_root": object_assets_root / "reference_images_completed",
        "organized_object_root": layout_root / "organized_objects",
        "layout_root": layout_root / "scene",
        "low_poly_root": root / "simulation" / "low_poly",
        "physics_root": physics_root,
        "physics_high_layout_root": physics_root / "high_layer_layout",
        "physics_diff_layout_root": physics_root / "diff_sim_layer_layout",
        "physics_forward_layout_root": physics_root / "forward_sim_layer_layout",
        "physics_xyz_order_root": physics_root / "xyz_order",
        "physics_layer_root": physics_root / "layer",
        "bbox_put_root": physics_root / "bbox_put",
        "render_root": root / "visualization" / "rendered",
    }


def scene_workspace_folder_overrides(scene_id: str) -> dict[str, str]:
    paths = scene_workspace_paths(scene_id)
    return {
        "ref_image_folder": _path_text(paths["reference_root"]),
        "depth_folder": _path_text(paths["depth_root"]),
        "seg_folder": _path_text(paths["segmentation_root"]),
        "descrip_folder": _path_text(paths["description_root"]),
        "contain_folder": _path_text(paths["contain_root"]),
        "contain_on_folder": _path_text(paths["contain_on_root"]),
        "size_folder": _path_text(paths["size_root"]),
        "items_folder": _path_text(paths["items_root"]),
        "raw_obj_folder": _path_text(paths["raw_asset_root"]),
        "ref_image_obj_folder": _path_text(paths["object_reference_root"]),
        "organized_obj_folder": _path_text(paths["organized_object_root"]),
        "layout_folder": _path_text(paths["layout_root"]),
        "low_poly_folder": _path_text(paths["low_poly_root"]),
        "phys_result_folder": _path_text(paths["physics_root"]),
        "high_layer_layout_folder": _path_text(paths["physics_high_layout_root"]),
        "diff_sim_layer_layout_folder": _path_text(paths["physics_diff_layout_root"]),
        "forward_sim_layer_layout_folder": _path_text(paths["physics_forward_layout_root"]),
        "xyz_order_folder": _path_text(paths["physics_xyz_order_root"]),
        "layer_folder": _path_text(paths["physics_layer_root"]),
        "bbox_put_folder": _path_text(paths["bbox_put_root"]),
    }


def provider_binding_with_options(binding: ProviderBinding, **updates: Any) -> ProviderBinding:
    return ProviderBinding(
        kind=binding.kind,
        enabled=binding.enabled,
        options={
            **dict(binding.options),
            **updates,
        },
    )


def short_scene_object_name(scene_id: str, object_id: str) -> str:
    raw = str(object_id or "").strip()
    if not raw:
        return "object"
    prefix = f"{scene_id}:"
    if raw.startswith(prefix):
        return raw[len(prefix) :]
    if ":" in raw:
        return raw.split(":")[-1]
    return raw


def _relation_type_value(value: Any) -> str:
    return str(getattr(value, "value", value)).strip().lower()


def _deduped_strings(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        cleaned = str(value or "").strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        deduped.append(cleaned)
    return deduped


def materialize_object_relation_workspace(
    scene_id: str,
    object_relation: ObjectRelationExtractionOutputs,
) -> None:
    paths = scene_workspace_paths(scene_id)
    item_payload: dict[str, int] = {}
    description_payload: dict[str, str] = {}
    size_payload: dict[str, Any] = {}
    all_object_names: set[str] = set()

    for detected_object in object_relation.object_catalog.objects:
        object_name = short_scene_object_name(scene_id, detected_object.object_id)
        item_payload[object_name] = int(detected_object.count)
        all_object_names.add(object_name)

    for description in object_relation.object_descriptions:
        object_name = short_scene_object_name(scene_id, description.object_id)
        description_payload[object_name] = str(description.prompt_text)
        all_object_names.add(object_name)

    for size_prior in object_relation.size_priors:
        object_name = short_scene_object_name(scene_id, size_prior.object_id)
        payload: dict[str, Any] = {}
        if isinstance(size_prior.dimensions_m, Mapping):
            numeric_dimensions: dict[str, float] = {}
            for axis, value in dict(size_prior.dimensions_m).items():
                try:
                    numeric_dimensions[str(axis)] = float(value)
                except (TypeError, ValueError):
                    continue
            if numeric_dimensions:
                payload["dimensions_m"] = numeric_dimensions
        relative_scale = getattr(size_prior, "relative_scale_to_scene", None)
        if isinstance(relative_scale, (int, float)):
            payload["relative_scale_to_scene"] = float(relative_scale)
        if getattr(size_prior, "source", None):
            payload["source"] = str(size_prior.source)
        size_payload[object_name] = payload if payload else None
        all_object_names.add(object_name)

    contain_payload = {name: [] for name in sorted(all_object_names)}
    contain_on_payload = {name: [] for name in sorted(all_object_names)}
    relation_graph = object_relation.relation_graph
    if relation_graph is not None:
        for relation in relation_graph.relations:
            child_name = short_scene_object_name(scene_id, relation.child_object_id)
            parent_name = short_scene_object_name(scene_id, relation.parent_object_id)
            relation_type = _relation_type_value(relation.relation_type)
            all_object_names.update((child_name, parent_name))
            if relation_type in {"contains", "in"}:
                contain_payload.setdefault(child_name, []).append(parent_name)
            if relation_type in {"supports", "on", "contains", "in"}:
                contain_on_payload.setdefault(child_name, []).append(parent_name)

    for object_name in sorted(all_object_names):
        contain_payload[object_name] = _deduped_strings(contain_payload.get(object_name, ()))
        contain_on_payload[object_name] = _deduped_strings(contain_on_payload.get(object_name, ()))

    write_json(paths["description_root"] / f"{scene_id}.json", description_payload)
    write_json(paths["contain_root"] / f"{scene_id}.json", contain_payload)
    write_json(paths["contain_on_root"] / f"{scene_id}.json", contain_on_payload)
    write_json(paths["size_root"] / f"{scene_id}.json", size_payload)
    write_json(paths["items_root"] / f"{scene_id}.json", item_payload)


def _maybe_import_torch() -> Any | None:
    try:
        import torch
    except Exception:
        return None
    return torch


def cuda_memory_snapshot(torch_module: Any | None = None) -> dict[str, Any]:
    torch_module = torch_module or _maybe_import_torch()
    snapshot: dict[str, Any] = {
        "torch_available": torch_module is not None,
        "cuda_available": False,
    }
    if torch_module is None:
        return snapshot
    cuda = getattr(torch_module, "cuda", None)
    if cuda is None or not callable(getattr(cuda, "is_available", None)) or not cuda.is_available():
        return snapshot

    snapshot["cuda_available"] = True
    try:
        device_index = int(cuda.current_device())
    except Exception:
        device_index = 0
    snapshot["device_index"] = device_index

    for key, fn_name in (
        ("memory_allocated", "memory_allocated"),
        ("memory_reserved", "memory_reserved"),
    ):
        fn = getattr(cuda, fn_name, None)
        if callable(fn):
            try:
                snapshot[key] = int(fn(device_index))
            except Exception:
                snapshot[key] = None
    return snapshot


def release_cuda_memory(*, reason: str) -> dict[str, Any]:
    torch_module = _maybe_import_torch()
    before = cuda_memory_snapshot(torch_module)

    gc.collect()

    after = before
    if before.get("cuda_available"):
        cuda = torch_module.cuda
        synchronize = getattr(cuda, "synchronize", None)
        if callable(synchronize):
            try:
                synchronize()
            except Exception:
                pass
        empty_cache = getattr(cuda, "empty_cache", None)
        if callable(empty_cache):
            empty_cache()
        ipc_collect = getattr(cuda, "ipc_collect", None)
        if callable(ipc_collect):
            try:
                ipc_collect()
            except Exception:
                pass
        after = cuda_memory_snapshot(torch_module)

    print(
        "[cuda-cleanup]",
        reason,
        json.dumps(
            {
                "before": before,
                "after": after,
            },
            sort_keys=True,
        ),
    )
    return {
        "before": before,
        "after": after,
    }


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug[:48] or "scene"


def canonical_mask_label(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    cleaned = "".join(ch for ch in normalized if ch.isalnum() or ch == "_").strip("_")
    cleaned = re.sub(r"\d+$", "", cleaned).strip("_")
    return cleaned or "object"


def resolve_scene_id(raw_scene_id: str | None, prompt: str, job_id: str) -> str:
    if raw_scene_id and raw_scene_id.strip():
        return raw_scene_id.strip()
    return f"{slugify(prompt)}-{job_id[-8:]}"


def resolve_segmentation_mode(value: Any) -> str:
    return "manual" if str(value).strip().lower() == "manual" else "automatic"


def resolve_model(value: Any, allowed_values: tuple[str, ...], fallback: str) -> str:
    if isinstance(value, str):
        normalized = value.strip()
        if normalized in allowed_values:
            return normalized
    return fallback


def resolve_reasoning_effort(value: Any, fallback: str = "auto") -> str:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in REASONING_EFFORT_OPTIONS:
            return normalized
    return fallback


def normalize_inferred_objects(payload: Any) -> list[str]:
    if isinstance(payload, dict):
        if isinstance(payload.get("requested_objects"), list):
            raw_values = payload["requested_objects"]
        elif isinstance(payload.get("objects"), list):
            raw_values = payload["objects"]
        elif isinstance(payload.get("items"), list):
            raw_values = payload["items"]
        elif isinstance(payload.get("requestedObjectHints"), list):
            raw_values = payload["requestedObjectHints"]
        else:
            raw_values = []
    elif isinstance(payload, list):
        raw_values = payload
    else:
        return []

    seen: set[str] = set()
    values: list[str] = []

    for item in raw_values:
        if not isinstance(item, str):
            continue
        cleaned = re.sub(r"\s+", " ", item.strip().lower())
        if not cleaned:
            continue
        if cleaned in INFERRED_SCENE_CONTEXT_OBJECTS:
            continue
        if cleaned in seen:
            continue
        seen.add(cleaned)
        values.append(cleaned)

    return values


def parse_json_response(content: str) -> dict[str, Any]:
    stripped = content.strip()
    if not stripped:
        raise ValueError("Provider response is empty.")

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            raise ValueError(f"Provider response is not JSON: {stripped[:120]}") from None
        parsed = json.loads(match.group(0))

    if not isinstance(parsed, dict):
        raise ValueError("Provider response must be a JSON object.")
    return parsed


def infer_requested_objects_from_local_smoke_prompt(prompt: str) -> list[str]:
    if not prompt.strip():
        return []

    resolved: list[str] = []
    for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_]*", prompt):
        normalized = canonical_mask_label(token)
        normalized = LOCAL_SMOKE_OBJECT_ALIASES.get(normalized, normalized)
        if normalized not in LOCAL_SMOKE_OBJECT_NAMES:
            continue
        if normalized in resolved:
            continue
        resolved.append(normalized)
    return resolved


def _load_openai_client_types() -> tuple[type[Exception], Any]:
    try:
        from openai import BadRequestError, OpenAI
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "OpenAI-backed dashboard providers require the `openai` package. "
            "Install it explicitly or use the smoke dashboard profile."
        ) from error
    return BadRequestError, OpenAI


def _supports_non_default_temperature(model: str) -> bool:
    return not str(model).strip().lower().startswith("gpt-5")


def _supports_reasoning_effort(model: str) -> bool:
    return str(model).strip().lower().startswith("gpt-5")


def _should_send_reasoning_effort(model: str, reasoning_effort: str | None) -> bool:
    if not reasoning_effort:
        return False
    return reasoning_effort != "auto" and _supports_reasoning_effort(model)


def resolve_requested_objects_inferred(value: Any) -> bool:
    return bool(value) if isinstance(value, bool) else False


def _chat_completion_create_with_compatible_token_limit(
    client: Any,
    *,
    model: str,
    messages: list[dict[str, Any]],
    temperature: float,
    response_format: dict[str, str],
    max_tokens: int,
    reasoning_effort: str | None = None,
    bad_request_error_cls: type[Exception] = Exception,
) -> Any:
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "response_format": response_format,
    }
    if temperature is not None and _supports_non_default_temperature(model):
        kwargs["temperature"] = temperature
    if _should_send_reasoning_effort(model, reasoning_effort):
        kwargs["reasoning_effort"] = reasoning_effort
    payload_variants = (
        {"max_completion_tokens": max_tokens},
        {"max_tokens": max_tokens},
        {},
    )
    last_error: Exception | None = None
    for index, payload in enumerate(payload_variants):
        try:
            return client.chat.completions.create(**{**kwargs, **payload})
        except Exception as error:
            if not isinstance(error, bad_request_error_cls):
                raise
            last_error = error
            detail = str(error).lower()
            unsupported = "unsupported parameter" in detail
            if not unsupported:
                raise
            if "max_completion_tokens" not in detail and "max_tokens" not in detail:
                raise
            if index + 1 >= len(payload_variants):
                raise
            continue
    raise last_error


def infer_requested_objects_from_prompt(
    prompt: str,
    *,
    model: str = "gpt-5.4",
    api_key: str | None = None,
    reasoning_effort: str | None = None,
    inference_budget: int = 1280,
) -> list[str]:
    if not prompt.strip():
        return []

    bad_request_error_cls, openai_client_ctor = _load_openai_client_types()
    base_url = os.environ.get("OPENAI_BASE_URL", "").strip() or "https://api.openai.com/v1"
    client_kwargs: dict[str, str] = {"base_url": base_url}
    if api_key is not None:
        client_kwargs["api_key"] = api_key
    client = openai_client_ctor(**client_kwargs)

    response = _chat_completion_create_with_compatible_token_limit(
        client,
        model=model,
        messages=[
            {
                "role": "system",
                "content": INFERENCE_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0,
        max_tokens=max(256, int(inference_budget)),
        response_format={"type": "json_object"},
        reasoning_effort=reasoning_effort,
        bad_request_error_cls=bad_request_error_cls,
    )
    raw_text = response.choices[0].message.content or ""
    parsed = parse_json_response(raw_text)
    return normalize_inferred_objects(parsed)


def resolve_preview_angle_count(value: Any) -> int:
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return 12
    return max(1, min(24, numeric))


def resolve_structured_llm_max_attempts(value: Any) -> int:
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return 3
    return max(1, min(10, numeric))


def resolve_structured_llm_reasoning_budget(value: Any) -> int:
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return 12800
    return max(256, min(65536, numeric))


def resolve_requested_object_inference_budget(value: Any) -> int:
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return 1280
    return max(256, min(65536, numeric))


def _resolve_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    return default


def _resolve_int(value: Any, default: int, *, minimum: int, maximum: int) -> int:
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, numeric))


def _resolve_float(value: Any, default: float, *, minimum: float, maximum: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, numeric))


def resolve_physics_settings(value: Any) -> dict[str, bool | int | float]:
    raw = value if isinstance(value, Mapping) else {}
    return {
        "diff_sim_enabled": _resolve_bool(
            raw.get("diff_sim_enabled", raw.get("diffSimEnabled")),
            bool(DEFAULT_PHYSICS_SETTINGS["diff_sim_enabled"]),
        ),
        "end_frame": _resolve_int(
            raw.get("end_frame", raw.get("endFrame")),
            int(DEFAULT_PHYSICS_SETTINGS["end_frame"]),
            minimum=1,
            maximum=5000,
        ),
        "ground_y_value": _resolve_float(
            raw.get("ground_y_value", raw.get("groundYValue")),
            float(DEFAULT_PHYSICS_SETTINGS["ground_y_value"]),
            minimum=-10.0,
            maximum=10.0,
        ),
        "total_opt_epoch": _resolve_int(
            raw.get("total_opt_epoch", raw.get("totalOptEpoch")),
            int(DEFAULT_PHYSICS_SETTINGS["total_opt_epoch"]),
            minimum=1,
            maximum=1000,
        ),
        "phys_lr": _resolve_float(
            raw.get("phys_lr", raw.get("physLr")),
            float(DEFAULT_PHYSICS_SETTINGS["phys_lr"]),
            minimum=1e-6,
            maximum=1.0,
        ),
        "contact_d_hat": _resolve_float(
            raw.get("contact_d_hat", raw.get("contactDHat")),
            float(DEFAULT_PHYSICS_SETTINGS["contact_d_hat"]),
            minimum=1e-7,
            maximum=1e-1,
        ),
        "contact_eps_velocity": _resolve_float(
            raw.get("contact_eps_velocity", raw.get("contactEpsVelocity")),
            float(DEFAULT_PHYSICS_SETTINGS["contact_eps_velocity"]),
            minimum=1e-8,
            maximum=1e-1,
        ),
    }


def resolve_relation_graph_override(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    relation_graph = value.get("relation_graph")
    if not isinstance(relation_graph, Mapping):
        return None
    return {
        "relation_graph": json.loads(json.dumps(relation_graph)),
        "updated_at": str(value.get("updated_at") or now_iso()),
    }


def apply_relation_graph_override(
    object_relation: ObjectRelationExtractionOutputs,
    relation_graph_override: Mapping[str, Any] | None,
) -> ObjectRelationExtractionOutputs:
    resolved_override = resolve_relation_graph_override(relation_graph_override)
    if resolved_override is None:
        return object_relation

    override_graph = scene_relation_graph_from_dict(resolved_override["relation_graph"])
    valid_object_ids: set[str] = set()
    used_ids: set[str] = set()
    for detected in object_relation.object_catalog.objects:
        valid_object_ids.add(detected.object_id)
        expanded_ids = instance_object_ids(detected, used_ids)
        valid_object_ids.update(expanded_ids)
        used_ids.update(expanded_ids)
    invalid_object_ids = sorted(
        {
            object_id
            for relation in override_graph.relations
            for object_id in (relation.parent_object_id, relation.child_object_id)
            if object_id not in valid_object_ids
        }
    )
    if invalid_object_ids:
        raise ValueError(
            "Relation override references unknown objects: "
            + ", ".join(invalid_object_ids)
        )

    next_relation_graph = SceneRelationGraph(
        scene_id=object_relation.object_catalog.scene_id,
        relations=override_graph.relations,
        root_object_ids=tuple(
            object_id for object_id in override_graph.root_object_ids if object_id in valid_object_ids
        ),
        metadata=object_relation.relation_graph.metadata,
    )
    next_size_priors = materialize_size_priors_for_instances(
        object_relation.size_priors,
        object_catalog=object_relation.object_catalog,
    )
    return replace(
        object_relation,
        relation_graph=next_relation_graph,
        size_priors=next_size_priors,
    )


def load_api_key() -> str:
    env_value = os.environ.get("OPENAI_API_KEY", "").strip()
    if env_value:
        return env_value
    api_key_path = Path("pat3d/preprocessing/gpt_utils/apikey.txt")
    if api_key_path.exists():
        file_value = api_key_path.read_text(encoding="utf-8").strip().strip(",")
        if file_value:
            return file_value
    raise RuntimeError("No OPENAI_API_KEY is available for dashboard-run jobs.")


def resolve_stage_backend_profile(value: Any) -> str | None:
    normalized = str(value).strip().lower() if value is not None else ""
    if not normalized:
        return None
    if normalized in {"default", "host-compatible"}:
        return normalized
    raise RuntimeError(
        f"Unsupported stage backend profile '{normalized}'. "
        "Supported values: default, host-compatible"
    )


def load_stage_backend_catalog(*, profile: str | None = None) -> dict[str, Any]:
    resolved_profile = resolve_stage_backend_profile(profile) or "default"
    override_path = ""
    if resolved_profile == "default":
        override_path = (
            os.environ.get("PAT3D_DASHBOARD_STAGE_BACKENDS_CATALOG", "").strip()
            or os.environ.get("PAT3D_STAGE_BACKENDS_CATALOG", "").strip()
        )
    default_catalog_path = REPO_ROOT / "dashboard" / "src" / "stageBackends.json"
    profile_catalog_paths = {
        "default": default_catalog_path,
        "host-compatible": REPO_ROOT / "dashboard" / "src" / "stageBackends.host-compatible.json",
    }
    catalog_path = Path(override_path) if override_path else profile_catalog_paths[resolved_profile]
    return json.loads(catalog_path.read_text(encoding="utf-8"))


def stage_backends_require_openai_api_key(
    stage_backends: Mapping[str, str],
    *,
    requested_objects: Sequence[str],
    segmentation_mode: str,
    object_crop_completion_enabled: bool = True,
) -> bool:
    selected_kinds = {str(value).strip() for value in stage_backends.values() if str(value).strip()}
    smoke_local_only_kinds = set(SMOKE_BACKEND_KINDS) | {"legacy_physics", "geometry_scene_renderer", "disabled"}

    if segmentation_mode != "manual" and not requested_objects:
        return not selected_kinds or not selected_kinds.issubset(smoke_local_only_kinds)

    if {"openai_text_to_image", "openai_structured_llm"} & selected_kinds:
        return True

    if object_crop_completion_enabled and stage_backends.get("object-assets") == "current_text_to_3d":
        return True

    return False


def validate_stage_backend_prerequisites(
    stage_backends: Mapping[str, str],
    *,
    segmentation_mode: str,
) -> None:
    selected_kinds = {str(value).strip() for value in stage_backends.values() if str(value).strip()}
    issues: list[str] = []

    def record_issue(label: str, error: Exception) -> None:
        issues.append(f"{label}: {type(error).__name__}: {error}")

    def has_huggingface_auth() -> bool:
        for env_name in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
            if os.environ.get(env_name, "").strip():
                return True
        token_path = Path.home() / ".cache" / "huggingface" / "token"
        try:
            return token_path.exists() and bool(token_path.read_text(encoding="utf-8").strip())
        except OSError:
            return False

    def resolve_sam3_checkpoint_override() -> Path | None:
        raw = os.environ.get("PAT3D_SAM3_CHECKPOINT_PATH", "").strip()
        if not raw:
            return None
        checkpoint_path = Path(raw).expanduser()
        if not checkpoint_path.is_absolute():
            checkpoint_path = REPO_ROOT / checkpoint_path
        return checkpoint_path

    if "current_depth" in selected_kinds:
        for module_name in ("depth_pro", "plyfile"):
            try:
                importlib.import_module(module_name)
            except Exception as error:
                record_issue(f"current_depth requires importable `{module_name}`", error)
        try:
            from pat3d.preprocessing.depth import _resolve_depth_pro_checkpoint

            _resolve_depth_pro_checkpoint()
        except Exception as error:
            record_issue(
                "current_depth requires a local Depth Pro checkpoint",
                error,
            )

    if "sam3_segmenter" in selected_kinds and segmentation_mode != "manual":
        try:
            from pat3d.providers.sam3_segmentation import SAM3Segmenter

            SAM3Segmenter()._resolve_python_executable()
        except Exception as error:
            record_issue(
                "sam3_segmenter requires a working PAT3D_SAM3_PYTHON / .venv-sam3 / pat3d-sam3 env",
                error,
            )
        sam3_checkpoint_override = resolve_sam3_checkpoint_override()
        if sam3_checkpoint_override is not None and not sam3_checkpoint_override.exists():
            issues.append(
                "sam3_segmenter received PAT3D_SAM3_CHECKPOINT_PATH but the file does not exist: "
                f"`{sam3_checkpoint_override}`."
            )
        elif sam3_checkpoint_override is None and not has_huggingface_auth():
            issues.append(
                "sam3_segmenter requires Hugging Face auth for `facebook/sam3` "
                "(set HF_TOKEN/HUGGINGFACE_HUB_TOKEN or populate ~/.cache/huggingface/token)."
            )

    if "current_text_to_3d" in selected_kinds:
        hunyuan_root = REPO_ROOT / "extern" / "Hunyuan3Dv2"
        if not hunyuan_root.exists():
            issues.append(f"current_text_to_3d requires `{hunyuan_root}`.")
        else:
            try:
                import pat3d.preprocessing.obj_gen  # noqa: F401
            except Exception as error:
                record_issue(
                    "current_text_to_3d requires Hunyuan3Dv2 plus a working runtime environment",
                    error,
                )

    if "current_mesh_simplifier" in selected_kinds:
        try:
            from pat3d.preprocessing.low_poly import resolve_float_tetwild_path

            resolve_float_tetwild_path()
            importlib.import_module("open3d")
            importlib.import_module("pymeshlab")
        except Exception as error:
            record_issue(
                "current_mesh_simplifier requires FloatTetwild_bin, open3d, and pymeshlab",
                error,
            )

    if "legacy_physics" in selected_kinds:
        try:
            from pat3d.providers._legacy_physics_adapter import LegacySequentialPhysicsAdapter

            LegacySequentialPhysicsAdapter().assert_runtime_support()
        except Exception as error:
            record_issue(
                "legacy_physics requires a compatible Diff_GIPC/UIPC runtime",
                error,
            )

    if "geometry_scene_renderer" in selected_kinds:
        try:
            importlib.import_module("open3d")
        except Exception as error:
            record_issue("geometry_scene_renderer requires importable `open3d`", error)

    if issues:
        raise RuntimeError(
            "Selected dashboard backends are not ready for a full production run:\n- "
            + "\n- ".join(issues)
        )


def normalize_stage_backend_alias(stage_id: str, selection: str) -> str:
    normalized = selection.strip()
    if stage_id == "object-assets" and normalized == "sam3d_text_to_3d":
        return "sam3d_image_to_3d"
    return normalized


def resolve_stage_backends(raw_value: Any, *, profile: str | None = None) -> dict[str, str]:
    catalog = load_stage_backend_catalog(profile=profile)
    raw_mapping = raw_value if isinstance(raw_value, dict) else {}
    resolved: dict[str, str] = {}
    for stage_id, entry in catalog.items():
        default_value = str(entry["default"])
        allowed_values = {str(option["value"]) for option in entry.get("options", [])}
        selected_value = normalize_stage_backend_alias(
            stage_id,
            str(raw_mapping.get(stage_id, default_value)).strip() or default_value,
        )
        if selected_value != "disabled" and selected_value not in allowed_values:
            raise RuntimeError(
                f"Invalid backend '{selected_value}' for stage '{stage_id}'."
            )
        resolved[stage_id] = selected_value
    return resolved


def _remove_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
        return
    path.unlink()


def purge_scene_artifacts(scene_id: str, *, preserve_reference_image: bool = False) -> None:
    workspace_paths = scene_workspace_paths(scene_id)
    workspace_root = workspace_paths["root"]
    if preserve_reference_image:
        if workspace_root.exists():
            for candidate in workspace_root.iterdir():
                if candidate == workspace_paths["reference_root"]:
                    continue
                _remove_path(candidate)
    else:
        _remove_path(workspace_root)

    for root in LEGACY_SCENE_DIRS:
        _remove_path(root / scene_id)

    for template in LEGACY_SCENE_FILE_PATHS:
        candidate = Path(str(template).format(scene_id=scene_id))
        if preserve_reference_image and candidate == Path(f"data/ref_img/{scene_id}.png"):
            continue
        _remove_path(candidate)

    for root in LEGACY_SCENE_PREFIX_DIRS:
        if root.exists():
            for candidate in root.glob(f"{scene_id}:*"):
                _remove_path(candidate)


def _remove_runtime_output_key(outputs: dict[str, Any], key_path: tuple[str, ...]) -> None:
    target: Any = outputs
    for key in key_path[:-1]:
        if not isinstance(target, dict):
            return
        target = target.get(key)
    if isinstance(target, dict):
        target.pop(key_path[-1], None)


def purge_downstream_artifacts(scene_id: str, *, completed_stage: str) -> None:
    workspace_paths = scene_workspace_paths(scene_id)
    preserve_reference_image = STAGE_INDEX_BY_ID.get(completed_stage, -1) >= STAGE_INDEX_BY_ID["reference-image"]
    if STAGE_INDEX_BY_ID.get(completed_stage, -1) < STAGE_INDEX_BY_ID["scene-understanding"]:
        _remove_path(workspace_paths["depth_root"])
        _remove_path(workspace_paths["segmentation_root"])
        _remove_path(Path("data/depth") / scene_id)
        _remove_path(Path("data/seg") / scene_id)
    if STAGE_INDEX_BY_ID.get(completed_stage, -1) < STAGE_INDEX_BY_ID["object-relation"]:
        for path in (
            workspace_paths["description_root"] / f"{scene_id}.json",
            workspace_paths["contain_root"] / f"{scene_id}.json",
            workspace_paths["contain_on_root"] / f"{scene_id}.json",
            workspace_paths["size_root"] / f"{scene_id}.json",
            workspace_paths["items_root"] / f"{scene_id}.json",
        ):
            _remove_path(path)
        for path in (
            Path(f"data/descrip/{scene_id}.json"),
            Path(f"data/contain/{scene_id}.json"),
            Path(f"data/contain_on/{scene_id}.json"),
            Path(f"data/size/{scene_id}.json"),
            Path(f"data/items/{scene_id}.json"),
        ):
            _remove_path(path)
    if STAGE_INDEX_BY_ID.get(completed_stage, -1) < STAGE_INDEX_BY_ID["object-assets"]:
        _remove_path(workspace_paths["object_reference_root"])
        _remove_path(workspace_paths["object_reference_completed_root"])
        _remove_path(workspace_paths["raw_asset_root"])
        _remove_path(Path("data/ref_img_obj") / scene_id)
        _remove_path(Path("data/raw_obj") / scene_id)
    if STAGE_INDEX_BY_ID.get(completed_stage, -1) < STAGE_INDEX_BY_ID["layout-initialization"]:
        _remove_path(workspace_paths["organized_object_root"])
        _remove_path(workspace_paths["layout_root"])
        _remove_path(Path("data/layout") / scene_id)
    if STAGE_INDEX_BY_ID.get(completed_stage, -1) < STAGE_INDEX_BY_ID["simulation-preparation"]:
        _remove_path(workspace_paths["low_poly_root"])
        _remove_path(Path("data/low_poly") / scene_id)
    if STAGE_INDEX_BY_ID.get(completed_stage, -1) < STAGE_INDEX_BY_ID["physics-optimization"]:
        _remove_path(workspace_paths["physics_root"])
        _remove_path(Path("_phys_result") / scene_id)
    if STAGE_INDEX_BY_ID.get(completed_stage, -1) < STAGE_INDEX_BY_ID["visualization"]:
        _remove_path(workspace_paths["render_root"])
        _remove_path(Path("results/rendered_images") / scene_id)
    if not preserve_reference_image:
        _remove_path(workspace_paths["reference_root"] / f"{scene_id}.png")
        _remove_path(Path(f"data/ref_img/{scene_id}.png"))


def normalize_resume_from(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    if normalized in RESUME_FROM_CHOICES:
        return normalized
    raise RuntimeError(
        f"Unsupported resume target '{normalized}'. Supported values: "
        f"{', '.join(RESUME_FROM_CHOICES)}"
    )


def summarize_error(error: Exception) -> str:
    text = str(error).strip()
    if not text:
        text = error.__class__.__name__

    lowered = text.lower()
    if "temporary failure in name resolution" in lowered or "connection error" in lowered:
        return (
            "Could not reach the configured provider endpoint. "
            "Check network or DNS access from this machine and retry."
        )
    return text


def normalize_job_error(
    error: Exception,
    *,
    stage_id: str | None = None,
) -> dict[str, Any]:
    failure = coerce_runtime_failure(
        error,
        phase="dashboard_job",
        code="dashboard_job_failed",
        user_message="The dashboard job failed.",
        technical_message=str(error),
        retryable=False,
        details={"stage_id": stage_id} if stage_id else {},
    )
    payload = failure.to_dict() if isinstance(failure, Pat3DRuntimeError) else {
        "phase": "dashboard_job",
        "code": "dashboard_job_failed",
        "user_message": summarize_error(error),
        "technical_message": str(error),
        "retryable": False,
        "details": {"stage_id": stage_id} if stage_id else {},
    }
    if stage_id and "stage_id" not in payload["details"]:
        payload["details"]["stage_id"] = stage_id
    return payload


def make_stage_rows() -> list[dict[str, Any]]:
    return [
        {
            "id": stage_id,
            "label": label,
            "status": "pending",
            "started_at": None,
            "completed_at": None,
            "error": None,
            "error_detail": None,
            "retryable": False,
            "progress": None,
            "progress_completed": None,
            "progress_total": None,
            "last_log": None,
        }
        for stage_id, label in STAGE_ORDER
    ]


def prepare_stage_rows_for_resume(
    stage_rows: list[dict[str, Any]],
    *,
    resume_from_stage: str,
) -> list[dict[str, Any]]:
    stage_index = next(
        (index for index, row in enumerate(stage_rows) if row.get("id") == resume_from_stage),
        None,
    )
    if stage_index is None:
        raise RuntimeError(f"Could not resume from unknown stage '{resume_from_stage}'.")

    prepared_rows: list[dict[str, Any]] = []
    for index, row in enumerate(stage_rows):
        next_row = dict(row)
        next_row["error"] = None
        next_row["error_detail"] = None
        next_row["retryable"] = False
        if index < stage_index:
            next_row["status"] = "completed"
            next_row["progress"] = 100
            next_row["progress_total"] = (
                next_row["progress_total"] if next_row.get("progress_total") not in (None, 0) else 100
            )
            next_row["progress_completed"] = next_row["progress_total"]
            next_row["completed_at"] = next_row.get("completed_at") or now_iso()
            next_row["last_log"] = next_row.get("last_log") or "Completed."
        elif index == stage_index:
            next_row["status"] = "queued"
            next_row["started_at"] = None
            next_row["completed_at"] = None
            next_row["progress"] = None
            next_row["progress_total"] = None
            next_row["progress_completed"] = None
            next_row["last_log"] = f"Queued to retry stage '{resume_from_stage}'."
        else:
            next_row["status"] = "pending"
            next_row["started_at"] = None
            next_row["completed_at"] = None
            next_row["progress"] = None
            next_row["progress_total"] = None
            next_row["progress_completed"] = None
            next_row["last_log"] = None
        prepared_rows.append(next_row)
    return prepared_rows


def set_stage_status(
    stage_rows: list[dict[str, Any]],
    stage_id: str,
    status: str,
    *,
    error: dict[str, Any] | None = None,
) -> None:
    for row in stage_rows:
        if row["id"] != stage_id:
            continue
        if status == "running":
            row["progress"] = 0 if row["progress"] is None else row["progress"]
            if row["progress_total"] is None:
                row["progress_total"] = 100
            if row["progress_completed"] is None:
                row["progress_completed"] = 0
        if status == "awaiting_input":
            row["progress"] = 0
            row["progress_completed"] = 0
            row["progress_total"] = max(1, row["progress_total"] or 1)
        if status in {"running", "awaiting_input"}:
            row["started_at"] = row["started_at"] or now_iso()
        if status in {"completed", "failed"}:
            row["completed_at"] = now_iso()
            if status == "completed":
                row["progress"] = 100
                row["progress_completed"] = row["progress_total"] if row["progress_total"] not in (None, 0) else 100
            else:
                row["progress"] = 0 if row["progress"] is None else row["progress"]
                row["progress_completed"] = row["progress_completed"] if row["progress_completed"] else 0
            row["last_log"] = row["last_log"] or f"{status.title()}."
        row["status"] = status
        row["error"] = error["user_message"] if error else None
        row["error_detail"] = error["technical_message"] if error else None
        row["retryable"] = bool(error["retryable"]) if error else False
        return


def set_stage_progress(
    stage_rows: list[dict[str, Any]],
    stage_id: str,
    *,
    progress_completed: int | float | None = None,
    progress_total: int | float | None = None,
    progress: int | float | None = None,
    message: str | None = None,
) -> None:
    for row in stage_rows:
        if row["id"] != stage_id:
            continue
        if progress_total is not None:
            row["progress_total"] = max(0, float(progress_total))
        if progress_completed is not None:
            row["progress_completed"] = max(0, float(progress_completed))
        elif progress is not None and row.get("progress_total") is not None and row["progress_total"] > 0:
            row["progress_completed"] = max(0, float(progress)) * row["progress_total"] / 100
        if progress is not None:
            row["progress"] = max(0, min(100, float(progress)))
        elif row["progress_total"] not in (None, 0):
            row["progress"] = row["progress_completed"] * 100 / row["progress_total"]
        if message is not None:
            row["last_log"] = message
        if row["status"] == "running":
            row["started_at"] = row["started_at"] or now_iso()
        return


def _safe_prompt_prefix(prompt: str, max_length: int = 64) -> str:
    safe = " ".join(prompt.strip().split())
    if len(safe) <= max_length:
        return safe
    return f"{safe[:max_length].rstrip()}..."


def run_object_asset_stage(
    stage_rows: list[dict[str, Any]],
    stage_id: str,
    *,
    object_asset_stage: Any,
    object_relation,
    reference_image_result: ReferenceImageResult,
    segmentation_result: SegmentationResult,
    input_payload: dict[str, Any],
    job_id: str,
    status_path: Path,
    output_name: str,
    log_name: str,
    output_path: Path | None = None,
    outputs: dict[str, Any] | None = None,
) -> ObjectAssetGenerationOutputs:
    asset_requests = expand_object_asset_requests(
        object_relation.object_catalog,
        object_relation.object_descriptions,
    )
    total = len(asset_requests)
    if total == 0:
        raise RuntimeError("No detected objects were available for asset generation.")

    provider = getattr(object_asset_stage, "_text_to_3d_provider")
    generate_callable = getattr(provider, "generate")
    generate_signature_target = generate_callable
    side_effect = getattr(generate_callable, "side_effect", None)
    if callable(side_effect):
        generate_signature_target = side_effect
    try:
        generate_signature = inspect.signature(generate_signature_target)
    except (TypeError, ValueError):
        generate_signature = None

    def _supports_generate_keyword(keyword: str) -> bool:
        if generate_signature is None:
            return False
        return any(
            parameter.name == keyword or parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in generate_signature.parameters.values()
        )

    supports_source_object_count = _supports_generate_keyword("source_object_count")
    supports_scene_object_canonical_names = _supports_generate_keyword("scene_object_canonical_names")
    scene_object_canonical_names = tuple(
        dict.fromkeys(
            detected_object.canonical_name.strip()
            for detected_object in object_relation.object_catalog.objects
            if detected_object.canonical_name.strip()
        )
    )
    object_reference_images = _materialize_object_reference_images(
        object_catalog=object_relation.object_catalog,
        reference_image_path=reference_image_result.image.path,
        segmentation_result=segmentation_result,
    )
    object_mask_artifacts = _materialize_object_mask_artifacts(
        object_catalog=object_relation.object_catalog,
        segmentation_result=segmentation_result,
    )
    prepare_scene_assets = getattr(provider, "prepare_scene_assets", None)
    if callable(prepare_scene_assets):
        prepare_scene_assets(
            asset_requests=asset_requests,
            reference_image_result=reference_image_result,
            object_reference_images=object_reference_images,
            object_mask_artifacts=object_mask_artifacts,
        )
    set_stage_progress(
        stage_rows,
        stage_id,
        progress=0,
        progress_total=total,
        message=f"Preparing to generate assets for {total} object(s).",
    )
    generated_assets = []
    expanded_descriptions = []
    cached_assets_by_source_id: dict[str, GeneratedObjectAsset] = {}
    for index, (description, source_object_id, source_object_count) in enumerate(asset_requests, start=1):
        expanded_descriptions.append(description)
        set_stage_progress(
            stage_rows,
            stage_id,
            progress_completed=index - 1,
            progress_total=total,
            message=(
                f"Generating asset for '{description.object_id}' "
                f"({index}/{total}) with prompt: {_safe_prompt_prefix(description.prompt_text)}"
            ),
        )
        write_status(
            status_path=status_path,
            job_id=job_id,
            input_payload=input_payload,
            output_name=output_name,
            stage_rows=stage_rows,
            state="running",
            current_stage_id=stage_id,
            log_name=log_name,
        )
        cached_asset = cached_assets_by_source_id.get(source_object_id)
        if cached_asset is not None:
            generated_asset = clone_generated_asset_for_object_id(cached_asset, description.object_id)
        else:
            generation_kwargs = {
                "object_reference_image": (
                    object_reference_images.get(description.object_id)
                    or object_reference_images.get(source_object_id)
                ),
            }
            if supports_source_object_count:
                generation_kwargs["source_object_count"] = source_object_count
            if supports_scene_object_canonical_names:
                generation_kwargs["scene_object_canonical_names"] = scene_object_canonical_names
            generated_asset = generate_callable(description, **generation_kwargs)
            cached_assets_by_source_id[source_object_id] = generated_asset
        generated_assets.append(generated_asset)
        if outputs is not None and output_path is not None:
            outputs["object_assets"] = ObjectAssetGenerationOutputs(
                object_catalog=object_relation.object_catalog,
                object_descriptions=tuple(expanded_descriptions),
                object_assets=ObjectAssetCatalog(
                    scene_id=object_relation.object_catalog.scene_id,
                    assets=tuple(generated_assets),
                    metadata=object_relation.object_catalog.metadata,
                ),
            )
            write_execution_result(outputs, output_path)
        set_stage_progress(
            stage_rows,
            stage_id,
            progress_completed=index,
            progress_total=total,
            message=f"Completed asset for '{description.object_id}' ({index}/{total}).",
        )
        write_status(
            status_path=status_path,
            job_id=job_id,
            input_payload=input_payload,
            output_name=output_name,
            stage_rows=stage_rows,
            state="running",
            current_stage_id=stage_id,
            log_name=log_name,
        )

    set_stage_progress(
        stage_rows,
        stage_id,
        progress=100,
        progress_total=total,
        message=f"Completed object asset generation for {total} object(s).",
    )

    return ObjectAssetGenerationOutputs(
        object_catalog=object_relation.object_catalog,
        object_descriptions=tuple(expanded_descriptions),
        object_assets=ObjectAssetCatalog(
            scene_id=object_relation.object_catalog.scene_id,
            assets=tuple(generated_assets),
            metadata=object_relation.object_catalog.metadata,
        ),
    )


def _materialize_object_mask_artifacts(
    *,
    object_catalog: ObjectCatalog,
    segmentation_result: SegmentationResult,
) -> dict[str, MaskInstance]:
    if not object_catalog.objects or not segmentation_result.instances:
        return {}

    instances_by_id = {instance.instance_id: instance for instance in segmentation_result.instances}
    instances_by_label: dict[str, list[MaskInstance]] = defaultdict(list)
    for instance in segmentation_result.instances:
        label_key = canonical_mask_label(str(instance.label or ""))
        if label_key:
            instances_by_label[label_key].append(instance)

    generated: dict[str, MaskInstance] = {}
    used_ids: set[str] = set()
    for detected_object in object_catalog.objects:
        candidate_instances = [
            instance
            for instance_id in detected_object.source_instance_ids
            for instance in (instances_by_id.get(instance_id),)
            if instance is not None
        ]
        if not candidate_instances:
            for label in (detected_object.display_name or "", detected_object.canonical_name):
                label_key = canonical_mask_label(label)
                if not label_key:
                    continue
                matches = instances_by_label.get(label_key)
                if matches:
                    candidate_instances = list(matches)
                    break
        if not candidate_instances:
            continue

        target_ids = (
            instance_object_ids(detected_object, used_ids)
            if detected_object.count > 1
            else (detected_object.object_id,)
        )
        for index, target_id in enumerate(target_ids):
            candidate_instance = candidate_instances[min(index, len(candidate_instances) - 1)]
            generated[target_id] = candidate_instance
            used_ids.add(target_id)

    return generated


class PauseForManualSizeInput(RuntimeError):
    pass


def build_pending_manual_size_entries(object_catalog: ObjectCatalog) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for detected_object in object_catalog.objects:
        suggested = _default_size_dimensions(detected_object.canonical_name)
        entries.append(
            {
                "object_id": detected_object.object_id,
                "canonical_name": detected_object.canonical_name,
                "display_name": detected_object.display_name,
                "dimensions_m": dict(suggested) if suggested is not None else None,
            }
        )
    return entries


def apply_manual_size_priors_to_object_relation(
    object_relation: ObjectRelationExtractionOutputs,
    manual_size_payload: Mapping[str, Any] | None,
) -> ObjectRelationExtractionOutputs:
    if not isinstance(manual_size_payload, Mapping):
        return object_relation
    entries = manual_size_payload.get("entries")
    if not isinstance(entries, list) or not entries:
        return object_relation

    prior_map: dict[str, SizePrior] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        object_id = str(entry.get("object_id") or "").strip()
        dimensions = entry.get("dimensions_m")
        if not object_id or not isinstance(dimensions, Mapping):
            continue
        normalized_dimensions: dict[str, float] = {}
        for axis in ("x", "y", "z"):
            try:
                value = float(dimensions.get(axis))
            except (TypeError, ValueError):
                value = 0.0
            if value <= 0:
                normalized_dimensions = {}
                break
            normalized_dimensions[axis] = value
        if len(normalized_dimensions) != 3:
            continue
        prior_map[object_id] = SizePrior(
            object_id=object_id,
            dimensions_m=normalized_dimensions,
            source="manual_input",
            metadata=make_stage_metadata(
                stage_name="size_inference",
                provider_name="manual_input",
                notes=("manual_size_input",),
            ),
        )

    if not prior_map:
        return object_relation

    ordered_priors: list[SizePrior] = []
    for detected_object in object_relation.object_catalog.objects:
        prior = prior_map.get(detected_object.object_id)
        if prior is not None:
            ordered_priors.append(prior)

    return ObjectRelationExtractionOutputs(
        reference_image=object_relation.reference_image,
        segmentation_result=object_relation.segmentation_result,
        object_catalog=object_relation.object_catalog,
        object_descriptions=object_relation.object_descriptions,
        relation_graph=object_relation.relation_graph,
        size_priors=tuple(ordered_priors),
        size_inference_report=object_relation.size_inference_report,
        depth_result=object_relation.depth_result,
    )


def count_generated_object_assets(outputs: ObjectAssetGenerationOutputs) -> int:
    return len(outputs.object_assets.assets)


def write_status(
    *,
    status_path: Path,
    job_id: str,
    input_payload: dict[str, Any],
    output_name: str,
    stage_rows: list[dict[str, Any]],
    state: str,
    current_stage_id: str | None = None,
    error: dict[str, Any] | None = None,
    log_name: str | None = None,
) -> None:
    write_json(
        status_path,
        {
            "job_id": job_id,
            "state": state,
            "prompt": input_payload["prompt"],
            "scene_id": input_payload["scene_id"],
            "workspace_root": input_payload.get("workspace_root"),
            "requested_objects": input_payload.get("requested_objects", []),
            "stage_backends": input_payload.get("stage_backends", {}),
            "stage_backends_profile": input_payload.get("stage_backends_profile"),
            "segmentation_mode": input_payload.get("segmentation_mode", "automatic"),
            "preview_angle_count": input_payload.get("preview_angle_count", 12),
            "requested_object_inference_budget": input_payload.get("requested_object_inference_budget", 1280),
            "physics_settings": input_payload.get("physics_settings", dict(DEFAULT_PHYSICS_SETTINGS)),
            "llm_model": input_payload.get("llm_model", "gpt-5.4"),
            "image_model": input_payload.get("image_model", "gpt-image-1.5"),
            "object_crop_completion_enabled": bool(
                input_payload.get("object_crop_completion_enabled", True)
            ),
            "object_crop_completion_model": input_payload.get(
                "object_crop_completion_model",
                "gpt-image-1.5",
            ),
            "structured_llm_max_attempts": input_payload.get("structured_llm_max_attempts", 3),
            "structured_llm_reasoning_budget": input_payload.get("structured_llm_reasoning_budget", 12800),
            "reasoning_effort": input_payload.get("reasoning_effort", "high"),
            "runtime_output_name": output_name,
            "current_stage_id": current_stage_id,
            "stages": stage_rows,
            "requested_objects_inferred": input_payload.get("requested_objects_inferred", False),
            "started_at": input_payload["started_at"],
            "updated_at": now_iso(),
            "finished_at": now_iso() if state in {"completed", "failed"} else None,
            "error": error,
            "log_name": log_name,
            "manual_segmentation": input_payload.get("manual_segmentation"),
            "manual_size_priors": input_payload.get("manual_size_priors"),
            "relation_graph_override": input_payload.get("relation_graph_override"),
        },
    )


def build_runtime_config(
    scene_id: str,
    stage_backends: dict[str, str] | None = None,
    stage_backends_profile: str | None = None,
    physics_settings: Mapping[str, Any] | None = None,
    preview_angle_count: int = 12,
    llm_model: str = "gpt-5.4",
    image_model: str = "gpt-image-1.5",
    object_crop_completion_enabled: bool = True,
    object_crop_completion_model: str = "gpt-image-1.5",
    structured_llm_max_attempts: int = 3,
    structured_llm_reasoning_budget: int = 12800,
    reasoning_effort: str = "high",
    resume_from_stage: str | None = None,
) -> RuntimeConfig:
    base = make_paper_core_runtime_config(
        scene_id=scene_id,
        model=llm_model,
        image_model=image_model,
        reasoning_effort=reasoning_effort,
        include_text_to_image=True,
        include_depth=True,
        include_segmentation=True,
        include_physics=True,
        include_renderer=True,
    )
    providers = dict(base.providers)
    resolved_backends = resolve_stage_backends(stage_backends, profile=stage_backends_profile)
    resolved_physics_settings = resolve_physics_settings(physics_settings)
    workspace_paths = scene_workspace_paths(scene_id)
    workspace_overrides = scene_workspace_folder_overrides(scene_id)
    catalog = load_stage_backend_catalog(profile=stage_backends_profile)
    for stage_id, selected_kind in resolved_backends.items():
        role = str(catalog[stage_id]["role"])
        if selected_kind == "disabled":
            providers.pop(role, None)
            continue
        existing = providers.get(role)
        providers[role] = ProviderBinding(
            kind=selected_kind,
            enabled=True,
            options=dict(existing.options) if existing is not None else {},
        )
    text_to_3d_binding = providers.get("text_to_3d_provider")
    if text_to_3d_binding is not None:
        if text_to_3d_binding.kind == "current_text_to_3d":
            providers["text_to_3d_provider"] = provider_binding_with_options(
                text_to_3d_binding,
                output_root=_path_text(workspace_paths["raw_asset_root"]),
                reference_image_root=_path_text(workspace_paths["object_reference_root"]),
                crop_completion_output_root=_path_text(
                    workspace_paths["object_reference_completed_root"]
                ),
                crop_completion_enabled=bool(object_crop_completion_enabled),
                crop_completion_model=str(object_crop_completion_model or "gpt-image-1.5"),
            )
        elif text_to_3d_binding.kind in {
            "sam3d_image_to_3d",
            "sam3d_multi_object_image_to_3d",
            "sam3d_text_to_3d",
        }:
            providers["text_to_3d_provider"] = provider_binding_with_options(
                text_to_3d_binding,
                output_root=_path_text(workspace_paths["raw_asset_root"]),
            )
    structured_llm_binding = providers.get("structured_llm")
    if structured_llm_binding is not None:
        providers["structured_llm"] = provider_binding_with_options(
            structured_llm_binding,
            max_attempts=max(1, int(structured_llm_max_attempts)),
            max_completion_tokens=max(256, int(structured_llm_reasoning_budget)),
        )
    reference_image_binding = providers.get("text_to_image_provider")
    if reference_image_binding is not None and reference_image_binding.kind == "openai_text_to_image":
        providers["text_to_image_provider"] = provider_binding_with_options(
            reference_image_binding,
            output_dir=_path_text(workspace_paths["reference_root"]),
        )
    depth_binding = providers.get("depth_estimator")
    if depth_binding is not None and depth_binding.kind == "current_depth":
        providers["depth_estimator"] = provider_binding_with_options(
            depth_binding,
            output_dir=_path_text(workspace_paths["depth_root"]),
            legacy_input_dir=_path_text(workspace_paths["reference_root"]),
        )
    segmenter_binding = providers.get("segmenter")
    if segmenter_binding is not None:
        if segmenter_binding.kind == "current_segmenter":
            providers["segmenter"] = provider_binding_with_options(
                segmenter_binding,
                output_dir=_path_text(workspace_paths["segmentation_root"]),
                legacy_input_dir=_path_text(workspace_paths["reference_root"]),
            )
        elif segmenter_binding.kind == "sam3_segmenter":
            sam3_options: dict[str, Any] = {
                "output_dir": _path_text(workspace_paths["segmentation_root"]),
            }
            sam3_checkpoint_override = os.environ.get("PAT3D_SAM3_CHECKPOINT_PATH", "").strip()
            if sam3_checkpoint_override:
                sam3_options["checkpoint_path"] = sam3_checkpoint_override
            sam3_load_from_hf = os.environ.get("PAT3D_SAM3_LOAD_FROM_HF", "").strip().lower()
            if sam3_load_from_hf:
                sam3_options["load_from_hf"] = sam3_load_from_hf not in {"0", "false", "no", "off"}
            providers["segmenter"] = provider_binding_with_options(
                segmenter_binding,
                **sam3_options,
            )
    layout_binding = providers.get("layout_builder")
    if layout_binding is not None:
        if layout_binding.kind == "legacy_layout":
            existing_legacy_overrides = layout_binding.options.get("legacy_arg_overrides")
            providers["layout_builder"] = provider_binding_with_options(
                layout_binding,
                layout_root=_path_text(workspace_paths["layout_root"]),
                legacy_arg_overrides={
                    **(
                        dict(existing_legacy_overrides)
                        if isinstance(existing_legacy_overrides, Mapping)
                        else {}
                    ),
                    **workspace_overrides,
                },
            )
        elif layout_binding.kind == "sam3d_multi_object_layout":
            providers["layout_builder"] = provider_binding_with_options(
                layout_binding,
                layout_root=_path_text(workspace_paths["layout_root"]),
            )
    renderer_binding = providers.get("scene_renderer")
    if renderer_binding is not None:
        if renderer_binding.kind == "geometry_scene_renderer":
            providers["scene_renderer"] = provider_binding_with_options(
                renderer_binding,
                output_root=_path_text(workspace_paths["render_root"]),
                layout_root=_path_text(workspace_paths["layout_root"]),
                raw_asset_root=_path_text(workspace_paths["raw_asset_root"]),
                preview_count=preview_angle_count,
            )
        elif renderer_binding.kind == "legacy_renderer":
            providers["scene_renderer"] = provider_binding_with_options(
                renderer_binding,
                output_root=_path_text(workspace_paths["render_root"]),
                reference_image_root=_path_text(workspace_paths["reference_root"]),
            )
    mesh_simplifier_binding = providers.get("mesh_simplifier")
    if mesh_simplifier_binding is not None:
        existing_physics_settings = mesh_simplifier_binding.options.get("physics_settings")
        providers["mesh_simplifier"] = provider_binding_with_options(
            mesh_simplifier_binding,
            layout_folder=_path_text(workspace_paths["layout_root"]),
            low_poly_folder=_path_text(workspace_paths["low_poly_root"]),
            physics_settings={
                **(
                    dict(existing_physics_settings)
                    if isinstance(existing_physics_settings, Mapping)
                    else {}
                ),
                "ground_y_value": float(resolved_physics_settings["ground_y_value"]),
            },
        )
    physics_binding = providers.get("physics_optimizer")
    if physics_binding is not None and physics_binding.kind == "legacy_physics":
        existing_legacy_overrides = physics_binding.options.get("legacy_arg_overrides")
        legacy_arg_overrides: dict[str, str | int | float | bool] = {
            **(
                dict(existing_legacy_overrides)
                if isinstance(existing_legacy_overrides, Mapping)
                else {}
            ),
            **workspace_overrides,
            "end_frame": int(resolved_physics_settings["end_frame"]),
            "ground_y_value": float(resolved_physics_settings["ground_y_value"]),
            "contact_d_hat": float(resolved_physics_settings["contact_d_hat"]),
            "contact_eps_velocity": float(resolved_physics_settings["contact_eps_velocity"]),
        }
        if bool(resolved_physics_settings["diff_sim_enabled"]):
            legacy_arg_overrides.update(
                {
                    "total_opt_epoch": int(resolved_physics_settings["total_opt_epoch"]),
                    "phys_lr": float(resolved_physics_settings["phys_lr"]),
                }
            )
        providers["physics_optimizer"] = provider_binding_with_options(
            physics_binding,
            mode=(
                "optimize_then_forward"
                if bool(resolved_physics_settings["diff_sim_enabled"])
                else "forward_only"
            ),
            legacy_arg_overrides=legacy_arg_overrides,
        )
    if (
        resume_from_stage is not None
        and STAGE_INDEX_BY_ID.get(resume_from_stage, -1) > STAGE_INDEX_BY_ID["scene-understanding"]
    ):
        providers.pop("text_to_image_provider", None)
        providers["depth_estimator"] = ProviderBinding(kind="noop_depth_estimator")
        providers["segmenter"] = ProviderBinding(kind="noop_segmenter")
    pipeline_options = dict(base.pipeline_options or {})
    pipeline_options["size_inference_pause_on_failure"] = True
    pipeline_options["workspace_root"] = _path_text(workspace_paths["root"])
    return RuntimeConfig(
        pipeline=base.pipeline,
        providers=providers,
        pipeline_options=pipeline_options,
    )


def stage_metadata_from_dict(data: Mapping[str, Any] | None, *, default_stage_name: str) -> StageRunMetadata:
    payload = data or {}
    status_value = str(payload.get("status", StageRunStatus.COMPLETED.value))
    try:
        status = StageRunStatus(status_value)
    except ValueError:
        status = StageRunStatus.COMPLETED
    raw_response_artifacts = tuple(
        artifact_ref_from_dict(item)
        for item in payload.get("raw_response_artifacts", ())
        if isinstance(item, Mapping)
    )
    return StageRunMetadata(
        run_id=str(payload.get("run_id", f"{default_stage_name}-resume")),
        stage_name=str(payload.get("stage_name", default_stage_name)),
        schema_version=str(payload.get("schema_version", "v1")),
        status=status,
        provider_name=str(payload["provider_name"]) if payload.get("provider_name") is not None else None,
        provider_config_digest=(
            str(payload["provider_config_digest"])
            if payload.get("provider_config_digest") is not None
            else None
        ),
        started_at=str(payload.get("started_at", now_iso())),
        finished_at=str(payload["finished_at"]) if payload.get("finished_at") is not None else None,
        raw_response_artifacts=raw_response_artifacts,
        notes=tuple(str(item) for item in payload.get("notes", ())),
    )


def reference_image_result_from_dict(data: Mapping[str, Any]) -> ReferenceImageResult:
    request_payload = data.get("request")
    if not isinstance(request_payload, Mapping):
        raise RuntimeError("Reference image payload is missing the original SceneRequest.")
    return ReferenceImageResult(
        request=scene_request_from_dict(request_payload),
        image=artifact_ref_from_dict(data["image"]),
        generation_prompt=(
            str(data["generation_prompt"])
            if data.get("generation_prompt") is not None
            else None
        ),
        seed=int(data["seed"]) if data.get("seed") is not None else None,
        width=int(data["width"]),
        height=int(data["height"]),
        metadata=stage_metadata_from_dict(
            data.get("metadata"),
            default_stage_name="reference_image",
        ),
    )


def depth_result_from_dict(data: Mapping[str, Any]) -> DepthResult:
    return DepthResult(
        image=artifact_ref_from_dict(data["image"]),
        depth_array=artifact_ref_from_dict(data["depth_array"]),
        depth_visualization=(
            artifact_ref_from_dict(data["depth_visualization"])
            if isinstance(data.get("depth_visualization"), Mapping)
            else None
        ),
        point_cloud=(
            artifact_ref_from_dict(data["point_cloud"])
            if isinstance(data.get("point_cloud"), Mapping)
            else None
        ),
        focal_length_px=float(data["focal_length_px"]) if data.get("focal_length_px") is not None else None,
        metadata=stage_metadata_from_dict(data.get("metadata"), default_stage_name="depth_estimation"),
    )


def mask_instance_from_dict(data: Mapping[str, Any]) -> MaskInstance:
    return MaskInstance(
        instance_id=str(data["instance_id"]),
        label=str(data["label"]) if data.get("label") is not None else None,
        mask=artifact_ref_from_dict(data["mask"]),
        bbox_xyxy=tuple(float(value) for value in data.get("bbox_xyxy", ())),
        confidence=float(data["confidence"]) if data.get("confidence") is not None else None,
        area_px=int(data["area_px"]) if data.get("area_px") is not None else None,
    )


def segmentation_result_from_dict(data: Mapping[str, Any]) -> SegmentationResult:
    return SegmentationResult(
        image=artifact_ref_from_dict(data["image"]),
        instances=tuple(
            mask_instance_from_dict(item)
            for item in data.get("instances", ())
            if isinstance(item, Mapping)
        ),
        composite_visualization=(
            artifact_ref_from_dict(data["composite_visualization"])
            if isinstance(data.get("composite_visualization"), Mapping)
            else None
        ),
        metadata=stage_metadata_from_dict(data.get("metadata"), default_stage_name="segmentation"),
    )


def object_catalog_from_dict(data: Mapping[str, Any]) -> ObjectCatalog:
    return ObjectCatalog(
        scene_id=str(data["scene_id"]),
        objects=tuple(
            detected_object_from_dict(item)
            for item in data.get("objects", ())
            if isinstance(item, Mapping)
        ),
        metadata=stage_metadata_from_dict(data.get("metadata"), default_stage_name="object_catalog"),
    )


def object_description_from_dict(data: Mapping[str, Any]) -> ObjectDescription:
    return ObjectDescription(
        object_id=str(data["object_id"]),
        canonical_name=str(data["canonical_name"]),
        prompt_text=str(data["prompt_text"]),
        visual_attributes=(
            {str(key): str(value) for key, value in dict(data["visual_attributes"]).items()}
            if data.get("visual_attributes") is not None
            else None
        ),
        material_attributes=(
            {str(key): str(value) for key, value in dict(data["material_attributes"]).items()}
            if data.get("material_attributes") is not None
            else None
        ),
        orientation_hints=(
            {str(key): str(value) for key, value in dict(data["orientation_hints"]).items()}
            if data.get("orientation_hints") is not None
            else None
        ),
        metadata=(
            stage_metadata_from_dict(data.get("metadata"), default_stage_name="object_description")
            if isinstance(data.get("metadata"), Mapping)
            else None
        ),
    )


def size_prior_from_dict(data: Mapping[str, Any]) -> SizePrior:
    return SizePrior(
        object_id=str(data["object_id"]),
        dimensions_m=(
            {str(key): float(value) for key, value in dict(data["dimensions_m"]).items()}
            if data.get("dimensions_m") is not None
            else None
        ),
        relative_scale_to_scene=(
            float(data["relative_scale_to_scene"])
            if data.get("relative_scale_to_scene") is not None
            else None
        ),
        source=str(data.get("source", "unknown")),
        metadata=(
            stage_metadata_from_dict(data.get("metadata"), default_stage_name="size_prior")
            if isinstance(data.get("metadata"), Mapping)
            else None
        ),
    )


def containment_relation_from_dict(data: Mapping[str, Any]) -> ContainmentRelation:
    relation_value = str(data["relation_type"])
    try:
        relation_type = RelationType(relation_value)
    except ValueError:
        relation_type = RelationType.CONTAINS
    return ContainmentRelation(
        parent_object_id=str(data["parent_object_id"]),
        child_object_id=str(data["child_object_id"]),
        relation_type=relation_type,
        confidence=float(data["confidence"]) if data.get("confidence") is not None else None,
        evidence=str(data["evidence"]) if data.get("evidence") is not None else None,
    )


def scene_relation_graph_from_dict(data: Mapping[str, Any]) -> SceneRelationGraph:
    return SceneRelationGraph(
        scene_id=str(data["scene_id"]),
        relations=tuple(
            containment_relation_from_dict(item)
            for item in data.get("relations", ())
            if isinstance(item, Mapping)
        ),
        root_object_ids=tuple(str(item) for item in data.get("root_object_ids", ())),
        metadata=(
            stage_metadata_from_dict(data.get("metadata"), default_stage_name="scene_relation_graph")
            if isinstance(data.get("metadata"), Mapping)
            else None
        ),
    )


def generated_object_asset_from_dict(data: Mapping[str, Any]) -> GeneratedObjectAsset:
    return GeneratedObjectAsset(
        object_id=str(data["object_id"]),
        mesh_obj=artifact_ref_from_dict(data["mesh_obj"]),
        mesh_mtl=artifact_ref_from_dict(data["mesh_mtl"]) if isinstance(data.get("mesh_mtl"), Mapping) else None,
        texture_image=(
            artifact_ref_from_dict(data["texture_image"])
            if isinstance(data.get("texture_image"), Mapping)
            else None
        ),
        preview_image=(
            artifact_ref_from_dict(data["preview_image"])
            if isinstance(data.get("preview_image"), Mapping)
            else None
        ),
        provider_asset_id=str(data["provider_asset_id"]) if data.get("provider_asset_id") is not None else None,
        asset_local_pose=(
            object_pose_from_dict(data["asset_local_pose"])
            if isinstance(data.get("asset_local_pose"), Mapping)
            else None
        ),
        metadata=(
            stage_metadata_from_dict(data.get("metadata"), default_stage_name="generated_object_asset")
            if isinstance(data.get("metadata"), Mapping)
            else None
        ),
    )


def object_asset_catalog_from_dict(data: Mapping[str, Any]) -> ObjectAssetCatalog:
    return ObjectAssetCatalog(
        scene_id=str(data["scene_id"]),
        assets=tuple(
            generated_object_asset_from_dict(item)
            for item in data.get("assets", ())
            if isinstance(item, Mapping)
        ),
        metadata=(
            stage_metadata_from_dict(data.get("metadata"), default_stage_name="object_asset_catalog")
            if isinstance(data.get("metadata"), Mapping)
            else None
        ),
    )


def object_pose_from_dict(data: Mapping[str, Any]) -> ObjectPose:
    return ObjectPose(
        object_id=str(data["object_id"]),
        translation_xyz=tuple(float(value) for value in data.get("translation_xyz", ())),
        rotation_type=str(data["rotation_type"]),
        rotation_value=tuple(float(value) for value in data.get("rotation_value", ())),
        scale_xyz=(
            tuple(float(value) for value in data.get("scale_xyz", ()))
            if data.get("scale_xyz") is not None
            else None
        ),
    )


def scene_layout_from_dict(data: Mapping[str, Any]) -> SceneLayout:
    return SceneLayout(
        scene_id=str(data["scene_id"]),
        object_poses=tuple(
            object_pose_from_dict(item)
            for item in data.get("object_poses", ())
            if isinstance(item, Mapping)
        ),
        layout_space=str(data["layout_space"]),
        support_graph=(
            scene_relation_graph_from_dict(data["support_graph"])
            if isinstance(data.get("support_graph"), Mapping)
            else None
        ),
        artifacts=tuple(
            artifact_ref_from_dict(item)
            for item in data.get("artifacts", ())
            if isinstance(item, Mapping)
        ),
        metadata=(
            stage_metadata_from_dict(data.get("metadata"), default_stage_name="scene_layout")
            if isinstance(data.get("metadata"), Mapping)
            else None
        ),
    )


def physics_ready_scene_from_dict(data: Mapping[str, Any]) -> PhysicsReadyScene:
    return PhysicsReadyScene(
        scene_id=str(data["scene_id"]),
        layout=scene_layout_from_dict(data["layout"]),
        simulation_meshes=tuple(
            artifact_ref_from_dict(item)
            for item in data.get("simulation_meshes", ())
            if isinstance(item, Mapping)
        ),
        object_poses=tuple(
            object_pose_from_dict(item)
            for item in data.get("object_poses", ())
            if isinstance(item, Mapping)
        ),
        collision_settings=(
            dict(data["collision_settings"])
            if data.get("collision_settings") is not None
            else None
        ),
        metadata=(
            stage_metadata_from_dict(data.get("metadata"), default_stage_name="physics_ready_scene")
            if isinstance(data.get("metadata"), Mapping)
            else None
        ),
    )


def physics_optimization_result_from_dict(data: Mapping[str, Any]) -> PhysicsOptimizationResult:
    return PhysicsOptimizationResult(
        scene_id=str(data["scene_id"]),
        initial_scene=physics_ready_scene_from_dict(data["initial_scene"]),
        optimized_object_poses=tuple(
            object_pose_from_dict(item)
            for item in data.get("optimized_object_poses", ())
            if isinstance(item, Mapping)
        ),
        metrics=(
            {str(key): float(value) for key, value in dict(data["metrics"]).items()}
            if data.get("metrics") is not None
            else None
        ),
        artifacts=tuple(
            artifact_ref_from_dict(item)
            for item in data.get("artifacts", ())
            if isinstance(item, Mapping)
        ),
        metadata=(
            stage_metadata_from_dict(data.get("metadata"), default_stage_name="physics_optimization")
            if isinstance(data.get("metadata"), Mapping)
            else None
        ),
    )


def render_result_from_dict(data: Mapping[str, Any]) -> RenderResult:
    return RenderResult(
        scene_id=str(data["scene_id"]),
        render_images=tuple(
            artifact_ref_from_dict(item)
            for item in data.get("render_images", ())
            if isinstance(item, Mapping)
        ),
        camera_metadata=(
            artifact_ref_from_dict(data["camera_metadata"])
            if isinstance(data.get("camera_metadata"), Mapping)
            else None
        ),
        render_config=dict(data["render_config"]) if data.get("render_config") is not None else None,
        metadata=(
            stage_metadata_from_dict(data.get("metadata"), default_stage_name="render_result")
            if isinstance(data.get("metadata"), Mapping)
            else None
        ),
    )


def scene_understanding_outputs_from_dict(data: Mapping[str, Any]) -> SceneUnderstandingOutputs:
    return SceneUnderstandingOutputs(
        reference_image=reference_image_result_from_dict(data["reference_image"]),
        depth_result=depth_result_from_dict(data["depth_result"]),
        segmentation_result=segmentation_result_from_dict(data["segmentation_result"]),
        object_catalog=(
            object_catalog_from_dict(data["object_catalog"])
            if isinstance(data.get("object_catalog"), Mapping)
            else None
        ),
    )


def object_relation_outputs_from_dict(data: Mapping[str, Any]) -> ObjectRelationExtractionOutputs:
    return ObjectRelationExtractionOutputs(
        reference_image=reference_image_result_from_dict(data["reference_image"]),
        segmentation_result=segmentation_result_from_dict(data["segmentation_result"]),
        object_catalog=object_catalog_from_dict(data["object_catalog"]),
        object_descriptions=tuple(
            object_description_from_dict(item)
            for item in data.get("object_descriptions", ())
            if isinstance(item, Mapping)
        ),
        relation_graph=scene_relation_graph_from_dict(data["relation_graph"]),
        size_priors=tuple(
            size_prior_from_dict(item)
            for item in data.get("size_priors", ())
            if isinstance(item, Mapping)
        ),
        depth_result=(
            depth_result_from_dict(data["depth_result"])
            if isinstance(data.get("depth_result"), Mapping)
            else None
        ),
    )


def object_asset_generation_outputs_from_dict(data: Mapping[str, Any]) -> ObjectAssetGenerationOutputs:
    return ObjectAssetGenerationOutputs(
        object_catalog=object_catalog_from_dict(data["object_catalog"]),
        object_descriptions=tuple(
            object_description_from_dict(item)
            for item in data.get("object_descriptions", ())
            if isinstance(item, Mapping)
        ),
        object_assets=object_asset_catalog_from_dict(data["object_assets"]),
    )


def layout_initialization_outputs_from_dict(data: Mapping[str, Any]) -> LayoutInitializationOutputs:
    return LayoutInitializationOutputs(
        scene_layout=scene_layout_from_dict(data["scene_layout"]),
        object_assets=object_asset_catalog_from_dict(data["object_assets"]),
        object_catalog=(
            object_catalog_from_dict(data["object_catalog"])
            if isinstance(data.get("object_catalog"), Mapping)
            else None
        ),
        relation_graph=(
            scene_relation_graph_from_dict(data["relation_graph"])
            if isinstance(data.get("relation_graph"), Mapping)
            else None
        ),
        size_priors=tuple(
            size_prior_from_dict(item)
            for item in data.get("size_priors", ())
            if isinstance(item, Mapping)
        ),
        reference_image_result=(
            reference_image_result_from_dict(data["reference_image_result"])
            if isinstance(data.get("reference_image_result"), Mapping)
            else None
        ),
        segmentation_result=(
            segmentation_result_from_dict(data["segmentation_result"])
            if isinstance(data.get("segmentation_result"), Mapping)
            else None
        ),
        depth_result=(
            depth_result_from_dict(data["depth_result"])
            if isinstance(data.get("depth_result"), Mapping)
            else None
        ),
    )


def simulation_preparation_outputs_from_dict(data: Mapping[str, Any]) -> SimulationPreparationOutputs:
    return SimulationPreparationOutputs(
        scene_layout=scene_layout_from_dict(data["scene_layout"]),
        object_assets=object_asset_catalog_from_dict(data["object_assets"]),
        physics_ready_scene=physics_ready_scene_from_dict(data["physics_ready_scene"]),
    )


def physics_optimization_outputs_from_dict(data: Mapping[str, Any]) -> PhysicsOptimizationOutputs:
    return PhysicsOptimizationOutputs(
        physics_ready_scene=physics_ready_scene_from_dict(data["physics_ready_scene"]),
        optimization_result=physics_optimization_result_from_dict(data["optimization_result"]),
    )


def visualization_outputs_from_dict(data: Mapping[str, Any]) -> VisualizationExportOutputs:
    scene_state_payload = (
        data.get("scene_state")
        if isinstance(data.get("scene_state"), Mapping)
        else None
    )
    scene_state: SceneLayout | PhysicsReadyScene | PhysicsOptimizationResult
    if scene_state_payload is None:
        raise RuntimeError("Visualization payload is missing scene_state.")
    if scene_state_payload.get("optimized_object_poses") is not None:
        scene_state = physics_optimization_result_from_dict(scene_state_payload)
    elif scene_state_payload.get("simulation_meshes") is not None:
        scene_state = physics_ready_scene_from_dict(scene_state_payload)
    else:
        scene_state = scene_layout_from_dict(scene_state_payload)
    return VisualizationExportOutputs(
        scene_state=scene_state,
        render_result=render_result_from_dict(data["render_result"]),
    )


def resolve_repo_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_outputs_and_reference(output_path: Path) -> tuple[dict[str, Any], ReferenceImageResult]:
    outputs = read_json(output_path) if output_path.exists() else {}
    reference_payload = outputs.get("first_contract_slice", {}).get("reference_image_result")
    if not isinstance(reference_payload, Mapping):
        raise RuntimeError("Could not load the reference image result needed to resume the job.")
    return outputs, reference_image_result_from_dict(reference_payload)


def prune_outputs_for_resume(outputs: dict[str, Any], *, resume_from_stage: str) -> None:
    if resume_from_stage == "reference-image":
        outputs.clear()
        return
    for key_path in DOWNSTREAM_RUNTIME_KEYS_BY_RETRY_STAGE.get(resume_from_stage, ()):
        _remove_runtime_output_key(outputs, key_path)


def parse_color_to_rgb(value: Any, *, fallback_index: int) -> tuple[int, int, int]:
    if isinstance(value, str):
        text = value.strip()
        if re.fullmatch(r"#[0-9a-fA-F]{6}", text):
            return tuple(int(text[index:index + 2], 16) for index in (1, 3, 5))
    hue = (fallback_index * 137.508) % 360.0
    saturation = 0.72
    lightness = 0.54
    chroma = (1.0 - abs(2.0 * lightness - 1.0)) * saturation
    hue_prime = hue / 60.0
    secondary = chroma * (1.0 - abs(hue_prime % 2.0 - 1.0))
    red = green = blue = 0.0
    if 0.0 <= hue_prime < 1.0:
        red, green = chroma, secondary
    elif 1.0 <= hue_prime < 2.0:
        red, green = secondary, chroma
    elif 2.0 <= hue_prime < 3.0:
        green, blue = chroma, secondary
    elif 3.0 <= hue_prime < 4.0:
        green, blue = secondary, chroma
    elif 4.0 <= hue_prime < 5.0:
        red, blue = secondary, chroma
    else:
        red, blue = chroma, secondary
    match = lightness - chroma / 2.0
    return tuple(int(round((channel + match) * 255.0)) for channel in (red, green, blue))


def mask_array_from_image(mask_path: Path, expected_size: tuple[int, int]) -> np.ndarray:
    with Image.open(mask_path) as image:
        width, height = image.size
        if (width, height) != expected_size:
            raise RuntimeError(
                f"Manual mask '{mask_path}' has size {width}x{height}, expected "
                f"{expected_size[0]}x{expected_size[1]}."
            )
        rgba = np.asarray(image.convert("RGBA"))
    alpha = rgba[..., 3]
    rgb = rgba[..., :3]
    foreground = np.logical_or(alpha > 0, np.any(rgb > 0, axis=-1))
    return foreground


def cutout_from_mask(image_rgb: np.ndarray, mask: np.ndarray, bbox: tuple[int, int, int, int]) -> Image.Image:
    x0, y0, x1, y1 = bbox
    crop = image_rgb[y0 : y1 + 1, x0 : x1 + 1].copy()
    crop_mask = mask[y0 : y1 + 1, x0 : x1 + 1]
    white = np.full_like(crop, 255)
    crop = np.where(crop_mask[..., None], crop, white)
    return Image.fromarray(crop)


def mask_crop_from_mask(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> Image.Image:
    x0, y0, x1, y1 = bbox
    crop_mask = (mask[y0 : y1 + 1, x0 : x1 + 1] * 255).astype(np.uint8)
    return Image.fromarray(crop_mask, mode="L")


def build_manual_segmentation_result(
    *,
    reference_image_result: ReferenceImageResult,
    manual_segmentation: Mapping[str, Any],
) -> SegmentationResult:
    instances = manual_segmentation.get("instances", ())
    if not isinstance(instances, list) or not instances:
        raise RuntimeError("At least one manual mask is required to continue the pipeline.")

    image_path = resolve_repo_path(reference_image_result.image.path)
    image_rgb = np.asarray(Image.open(image_path).convert("RGB"))
    width, height = reference_image_result.width, reference_image_result.height
    if image_rgb.shape[1] != width or image_rgb.shape[0] != height:
        raise RuntimeError("The reference image dimensions do not match the stored metadata.")

    scene_id = reference_image_result.request.scene_id
    seg_dir = REPO_ROOT / scene_workspace_paths(scene_id)["segmentation_root"] / scene_id
    if seg_dir.exists():
        shutil.rmtree(seg_dir)
    seg_dir.mkdir(parents=True, exist_ok=True)

    overlay = image_rgb.astype(np.float32).copy()
    counts: dict[str, int] = defaultdict(int)
    segmentation_instances: list[MaskInstance] = []

    for index, entry in enumerate(instances):
        if not isinstance(entry, Mapping):
            continue
        label = canonical_mask_label(str(entry.get("label", "")).strip())
        source_mask_path = entry.get("mask_path")
        if not isinstance(source_mask_path, str) or not source_mask_path.strip():
            raise RuntimeError("Each saved manual mask must include a mask_path.")

        mask = mask_array_from_image(
            resolve_repo_path(source_mask_path),
            expected_size=(width, height),
        )
        area_px = int(np.count_nonzero(mask))
        if area_px <= 0:
            continue

        ys, xs = np.nonzero(mask)
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        counts[label] += 1
        instance_name = f"{label}{counts[label]}"
        stem = f"{scene_id}_{instance_name}_{instance_name}"

        cutout_from_mask(image_rgb, mask, bbox).save(seg_dir / f"{stem}.png")
        mask_crop_from_mask(mask, bbox).save(seg_dir / f"{stem}_ann.png")

        color = np.asarray(parse_color_to_rgb(entry.get("color"), fallback_index=index), dtype=np.float32)
        overlay[mask] = overlay[mask] * 0.35 + color * 0.65

        segmentation_instances.append(
            MaskInstance(
                instance_id=stem,
                label=label,
                mask=ArtifactRef(
                    artifact_type="mask",
                    path=_path_text(
                        scene_workspace_paths(scene_id)["segmentation_root"] / scene_id / f"{stem}_ann.png"
                    ),
                    format="png",
                    role="instance_mask",
                    metadata_path=None,
                ),
                bbox_xyxy=tuple(float(value) for value in bbox),
                confidence=None,
                area_px=area_px,
            )
        )

    if not segmentation_instances:
        raise RuntimeError("At least one manual mask with painted pixels is required.")

    composite_path = seg_dir / f"{scene_id}_segmentation.png"
    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8), mode="RGB").save(composite_path)

    return SegmentationResult(
        image=reference_image_result.image,
        instances=tuple(segmentation_instances),
        composite_visualization=ArtifactRef(
            artifact_type="segmentation_visualization",
            path=_path_text(
                scene_workspace_paths(scene_id)["segmentation_root"] / scene_id / f"{scene_id}_segmentation.png"
            ),
            format="png",
            role="segmentation_visualization",
            metadata_path=None,
        ),
        metadata=make_stage_metadata(
            stage_name="segmentation",
            provider_name=MANUAL_SEGMENTATION_PROVIDER,
            notes=("mode=manual",),
        ),
    )


def build_manual_scene_understanding(
    *,
    pipeline: Any,
    reference_image_result: ReferenceImageResult,
    manual_segmentation: Mapping[str, Any],
    preserved_depth_result: DepthResult | None = None,
) -> SceneUnderstandingOutputs:
    scene_stage = pipeline._first_contract_slice_pipeline._scene_understanding_stage
    depth_result = preserved_depth_result or scene_stage._depth_estimator.predict(reference_image_result)
    segmentation_result = build_manual_segmentation_result(
        reference_image_result=reference_image_result,
        manual_segmentation=manual_segmentation,
    )

    object_catalog = None
    if scene_stage._object_catalog_builder is not None:
        object_catalog = scene_stage._object_catalog_builder(
            reference_image_result,
            segmentation_result,
        )

    return SceneUnderstandingOutputs(
        reference_image=reference_image_result,
        depth_result=depth_result,
        segmentation_result=segmentation_result,
        object_catalog=object_catalog,
    )


def preserved_depth_result_from_manual_segmentation(
    manual_segmentation: Mapping[str, Any] | None,
) -> DepthResult | None:
    if not isinstance(manual_segmentation, Mapping):
        return None
    payload = manual_segmentation.get("preserved_depth_result")
    if isinstance(payload, Mapping):
        return depth_result_from_dict(payload)
    return None


def run_automatic_scene_understanding_with_progress(
    *,
    pipeline: Any,
    reference_image_result: ReferenceImageResult,
    requested_objects: list[str],
    stage_rows: list[dict[str, Any]],
    status_path: Path,
    job_id: str,
    input_payload: dict[str, Any],
    output_name: str,
    log_name: str,
    intro_message: str,
    completion_message: str,
) -> SceneUnderstandingOutputs:
    scene_stage = pipeline._first_contract_slice_pipeline._scene_understanding_stage

    set_stage_status(stage_rows, "scene-understanding", "running")
    set_stage_progress(stage_rows, "scene-understanding", progress=0, message=intro_message)
    write_status(
        status_path=status_path,
        job_id=job_id,
        input_payload=input_payload,
        output_name=output_name,
        stage_rows=stage_rows,
        state="running",
        current_stage_id="scene-understanding",
        log_name=log_name,
    )

    set_stage_progress(
        stage_rows,
        "scene-understanding",
        progress=15,
        message="Estimating monocular depth from the reference image.",
    )
    write_status(
        status_path=status_path,
        job_id=job_id,
        input_payload=input_payload,
        output_name=output_name,
        stage_rows=stage_rows,
        state="running",
        current_stage_id="scene-understanding",
        log_name=log_name,
    )
    depth_result = scene_stage._depth_estimator.predict(reference_image_result)

    set_stage_progress(
        stage_rows,
        "scene-understanding",
        progress=55,
        message="Segmenting requested objects from the reference image.",
    )
    write_status(
        status_path=status_path,
        job_id=job_id,
        input_payload=input_payload,
        output_name=output_name,
        stage_rows=stage_rows,
        state="running",
        current_stage_id="scene-understanding",
        log_name=log_name,
    )
    segmentation_result = scene_stage._segmenter.segment(
        reference_image_result,
        object_hints=requested_objects or None,
    )

    object_catalog = None
    if scene_stage._object_catalog_builder is not None:
        set_stage_progress(
            stage_rows,
            "scene-understanding",
            progress=85,
            message="Building the object catalog from segmentation instances.",
        )
        write_status(
            status_path=status_path,
            job_id=job_id,
            input_payload=input_payload,
            output_name=output_name,
            stage_rows=stage_rows,
            state="running",
            current_stage_id="scene-understanding",
            log_name=log_name,
        )
        object_catalog = scene_stage._object_catalog_builder(
            reference_image_result,
            segmentation_result,
        )

    scene_understanding = SceneUnderstandingOutputs(
        reference_image=reference_image_result,
        depth_result=depth_result,
        segmentation_result=segmentation_result,
        object_catalog=object_catalog,
    )
    set_stage_progress(
        stage_rows,
        "scene-understanding",
        progress=100,
        message=completion_message,
    )
    return scene_understanding


def run_manual_scene_understanding_with_progress(
    *,
    pipeline: Any,
    reference_image_result: ReferenceImageResult,
    manual_segmentation: Mapping[str, Any],
    preserved_depth_result: DepthResult | None = None,
    stage_rows: list[dict[str, Any]],
    status_path: Path,
    job_id: str,
    input_payload: dict[str, Any],
    output_name: str,
    log_name: str,
    intro_message: str,
    completion_message: str,
) -> SceneUnderstandingOutputs:
    scene_stage = pipeline._first_contract_slice_pipeline._scene_understanding_stage

    set_stage_status(stage_rows, "scene-understanding", "running")
    set_stage_progress(stage_rows, "scene-understanding", progress=0, message=intro_message)
    write_status(
        status_path=status_path,
        job_id=job_id,
        input_payload=input_payload,
        output_name=output_name,
        stage_rows=stage_rows,
        state="running",
        current_stage_id="scene-understanding",
        log_name=log_name,
    )

    set_stage_progress(
        stage_rows,
        "scene-understanding",
        progress=20,
        message=(
            "Reusing the saved depth estimate from the previous scene understanding pass."
            if preserved_depth_result is not None
            else "Estimating monocular depth from the reference image."
        ),
    )
    write_status(
        status_path=status_path,
        job_id=job_id,
        input_payload=input_payload,
        output_name=output_name,
        stage_rows=stage_rows,
        state="running",
        current_stage_id="scene-understanding",
        log_name=log_name,
    )
    depth_result = preserved_depth_result or scene_stage._depth_estimator.predict(
        reference_image_result
    )

    set_stage_progress(
        stage_rows,
        "scene-understanding",
        progress=60,
        message="Materializing user-provided masks into segmentation outputs.",
    )
    write_status(
        status_path=status_path,
        job_id=job_id,
        input_payload=input_payload,
        output_name=output_name,
        stage_rows=stage_rows,
        state="running",
        current_stage_id="scene-understanding",
        log_name=log_name,
    )
    segmentation_result = build_manual_segmentation_result(
        reference_image_result=reference_image_result,
        manual_segmentation=manual_segmentation,
    )

    object_catalog = None
    if scene_stage._object_catalog_builder is not None:
        set_stage_progress(
            stage_rows,
            "scene-understanding",
            progress=85,
            message="Building the object catalog from manual masks.",
        )
        write_status(
            status_path=status_path,
            job_id=job_id,
            input_payload=input_payload,
            output_name=output_name,
            stage_rows=stage_rows,
            state="running",
            current_stage_id="scene-understanding",
            log_name=log_name,
        )
        object_catalog = scene_stage._object_catalog_builder(
            reference_image_result,
            segmentation_result,
        )

    scene_understanding = SceneUnderstandingOutputs(
        reference_image=reference_image_result,
        depth_result=depth_result,
        segmentation_result=segmentation_result,
        object_catalog=object_catalog,
    )
    set_stage_progress(
        stage_rows,
        "scene-understanding",
        progress=100,
        message=completion_message,
    )
    return scene_understanding


def run_object_relation_stage(
    *,
    pipeline: Any,
    reference_image_result: ReferenceImageResult,
    scene_understanding: SceneUnderstandingOutputs,
    input_payload: dict[str, Any],
    job_id: str,
    stage_rows: list[dict[str, Any]],
    status_path: Path,
    output_name: str,
    output_path: Path,
    log_name: str,
    outputs: dict[str, Any],
) -> ObjectRelationExtractionOutputs:
    first_pipeline = pipeline._first_contract_slice_pipeline
    set_stage_status(stage_rows, "object-relation", "running")
    set_stage_progress(
        stage_rows,
        "object-relation",
        progress=5,
        message="Analyzing relations and size priors from understood objects.",
    )
    write_status(
        status_path=status_path,
        job_id=job_id,
        input_payload=input_payload,
        output_name=output_name,
        stage_rows=stage_rows,
        state="running",
        current_stage_id="object-relation",
        log_name=log_name,
    )
    try:
        object_relation = first_pipeline._object_relation_stage.run(
            reference_image_result=reference_image_result,
            segmentation_result=scene_understanding.segmentation_result,
            depth_result=scene_understanding.depth_result,
            object_catalog=scene_understanding.object_catalog,
        )
    except Pat3DRuntimeError as error:
        if error.payload.code != "size_inference_requires_input":
            raise
        partial_object_relation = first_pipeline._object_relation_stage.prepare_without_size_inference(
            reference_image_result=reference_image_result,
            segmentation_result=scene_understanding.segmentation_result,
            depth_result=scene_understanding.depth_result,
            object_catalog=scene_understanding.object_catalog,
        )
        partial_object_relation = apply_relation_graph_override(
            partial_object_relation,
            input_payload.get("relation_graph_override"),
        )
        materialize_object_relation_workspace(
            reference_image_result.request.scene_id,
            partial_object_relation,
        )
        outputs.setdefault("first_contract_slice", {})["object_relation"] = partial_object_relation
        write_execution_result(outputs, output_path)
        updated_at = now_iso()
        input_payload["manual_size_priors"] = {
            "state": "pending",
            "updated_at": updated_at,
            "entries": build_pending_manual_size_entries(partial_object_relation.object_catalog),
            "error": error.to_dict(),
        }
        set_stage_status(stage_rows, "object-relation", "awaiting_input", error=error.to_dict())
        set_stage_progress(
            stage_rows,
            "object-relation",
            progress=0,
            message=(
                "Structured size inference did not return a usable response. "
                "Retry stage 03 or enter dimensions manually to continue."
            ),
        )
        write_status(
            status_path=status_path,
            job_id=job_id,
            input_payload=input_payload,
            output_name=output_name,
            stage_rows=stage_rows,
            state="awaiting_size_input",
            current_stage_id="object-relation",
            error=error.to_dict(),
            log_name=log_name,
        )
        raise PauseForManualSizeInput() from error
    object_relation = apply_relation_graph_override(
        object_relation,
        input_payload.get("relation_graph_override"),
    )
    materialize_object_relation_workspace(
        reference_image_result.request.scene_id,
        object_relation,
    )
    set_stage_progress(
        stage_rows,
        "object-relation",
        progress=100,
        message="Object relations and size priors are prepared.",
    )
    outputs.setdefault("first_contract_slice", {})["object_relation"] = object_relation
    write_execution_result(outputs, output_path)
    set_stage_status(stage_rows, "object-relation", "completed")
    return object_relation

def run_layout_initialization_stage(
    *,
    pipeline: Any,
    request: SceneRequest,
    object_assets: ObjectAssetGenerationOutputs,
    object_relation: ObjectRelationExtractionOutputs,
    reference_image_result: ReferenceImageResult,
    scene_understanding: SceneUnderstandingOutputs,
    input_payload: dict[str, Any],
    job_id: str,
    stage_rows: list[dict[str, Any]],
    status_path: Path,
    output_name: str,
    output_path: Path,
    log_name: str,
    outputs: dict[str, Any],
) -> LayoutInitializationOutputs:
    set_stage_status(stage_rows, "layout-initialization", "running")
    set_stage_progress(
        stage_rows,
        "layout-initialization",
        progress=0,
        message="Initializing scene layout from relation graph and size priors.",
    )
    write_status(
        status_path=status_path,
        job_id=job_id,
        input_payload=input_payload,
        output_name=output_name,
        stage_rows=stage_rows,
        state="running",
        current_stage_id="layout-initialization",
        log_name=log_name,
    )
    layout_initialization = pipeline._layout_initialization_stage.run(
        scene_id=request.scene_id,
        object_assets=object_assets.object_assets,
        object_catalog=object_relation.object_catalog,
        reference_image_result=reference_image_result,
        segmentation_result=scene_understanding.segmentation_result,
        depth_result=scene_understanding.depth_result,
        relation_graph=object_relation.relation_graph,
        size_priors=object_relation.size_priors,
    )
    set_stage_progress(
        stage_rows,
        "layout-initialization",
        progress=100,
        message="Layout initialized successfully.",
    )
    outputs["layout_initialization"] = layout_initialization
    write_execution_result(outputs, output_path)
    set_stage_status(stage_rows, "layout-initialization", "completed")
    return layout_initialization


def run_simulation_preparation_stage(
    *,
    pipeline: Any,
    layout_initialization: LayoutInitializationOutputs,
    input_payload: dict[str, Any],
    job_id: str,
    stage_rows: list[dict[str, Any]],
    status_path: Path,
    output_name: str,
    output_path: Path,
    log_name: str,
    outputs: dict[str, Any],
) -> SimulationPreparationOutputs:
    set_stage_status(stage_rows, "simulation-preparation", "running")
    set_stage_progress(
        stage_rows,
        "simulation-preparation",
        progress=0,
        message="Converting assets for simulation and collision package.",
    )
    write_status(
        status_path=status_path,
        job_id=job_id,
        input_payload=input_payload,
        output_name=output_name,
        stage_rows=stage_rows,
        state="running",
        current_stage_id="simulation-preparation",
        log_name=log_name,
    )
    simulation_preparation = pipeline._simulation_preparation_stage.run(
        layout_initialization.scene_layout,
        layout_initialization.object_assets,
    )
    set_stage_progress(
        stage_rows,
        "simulation-preparation",
        progress=100,
        message="Simulation assets prepared.",
    )
    outputs["simulation_preparation"] = simulation_preparation
    write_execution_result(outputs, output_path)
    set_stage_status(stage_rows, "simulation-preparation", "completed")
    return simulation_preparation


def run_physics_optimization_stage(
    *,
    pipeline: Any,
    simulation_preparation: SimulationPreparationOutputs,
    input_payload: dict[str, Any],
    job_id: str,
    stage_rows: list[dict[str, Any]],
    status_path: Path,
    output_name: str,
    output_path: Path,
    log_name: str,
    outputs: dict[str, Any],
) -> PhysicsOptimizationOutputs | None:
    if pipeline._physics_optimization_stage is None:
        return None
    set_stage_status(stage_rows, "physics-optimization", "running")
    set_stage_progress(
        stage_rows,
        "physics-optimization",
        progress=0,
        message="Running forward physics simulation until the scene settles.",
    )
    write_status(
        status_path=status_path,
        job_id=job_id,
        input_payload=input_payload,
        output_name=output_name,
        stage_rows=stage_rows,
        state="running",
        current_stage_id="physics-optimization",
        log_name=log_name,
    )
    physics_optimization = pipeline._physics_optimization_stage.run(
        simulation_preparation.physics_ready_scene
    )
    set_stage_progress(
        stage_rows,
        "physics-optimization",
        progress=100,
        message="Physics simulation complete.",
    )
    outputs["physics_optimization"] = physics_optimization
    write_execution_result(outputs, output_path)
    set_stage_status(stage_rows, "physics-optimization", "completed")
    return physics_optimization

def run_visualization_stage(
    *,
    pipeline: Any,
    scene_state: SceneLayout | PhysicsReadyScene | PhysicsOptimizationResult,
    input_payload: dict[str, Any],
    job_id: str,
    stage_rows: list[dict[str, Any]],
    status_path: Path,
    output_name: str,
    output_path: Path,
    log_name: str,
    outputs: dict[str, Any],
) -> VisualizationExportOutputs | None:
    if pipeline._visualization_stage is None:
        return None
    set_stage_status(stage_rows, "visualization", "running")
    set_stage_progress(
        stage_rows,
        "visualization",
        progress=0,
        message="Rendering final previews and exporting scene outputs.",
    )
    write_status(
        status_path=status_path,
        job_id=job_id,
        input_payload=input_payload,
        output_name=output_name,
        stage_rows=stage_rows,
        state="running",
        current_stage_id="visualization",
        log_name=log_name,
    )
    visualization = pipeline._visualization_stage.run(scene_state)
    set_stage_progress(
        stage_rows,
        "visualization",
        progress=100,
        message="Visualization and render export complete.",
    )
    outputs["visualization"] = visualization
    write_execution_result(outputs, output_path)
    set_stage_status(stage_rows, "visualization", "completed")
    return visualization


def run_post_understanding_stages(
    *,
    pipeline: Any,
    request: SceneRequest,
    input_payload: dict[str, Any],
    job_id: str,
    stage_rows: list[dict[str, Any]],
    status_path: Path,
    output_name: str,
    output_path: Path,
    log_name: str,
    outputs: dict[str, Any],
    reference_image_result: ReferenceImageResult,
    scene_understanding: SceneUnderstandingOutputs,
    starting_stage: str = "object-relation",
    existing_object_relation: ObjectRelationExtractionOutputs | None = None,
    existing_object_assets: ObjectAssetGenerationOutputs | None = None,
    existing_layout_initialization: LayoutInitializationOutputs | None = None,
    existing_simulation_preparation: SimulationPreparationOutputs | None = None,
    existing_physics_optimization: PhysicsOptimizationOutputs | None = None,
) -> None:
    object_relation = existing_object_relation
    if starting_stage == "object-relation":
        object_relation = run_object_relation_stage(
            pipeline=pipeline,
            reference_image_result=reference_image_result,
            scene_understanding=scene_understanding,
            input_payload=input_payload,
            job_id=job_id,
            stage_rows=stage_rows,
            status_path=status_path,
            output_name=output_name,
            output_path=output_path,
            log_name=log_name,
            outputs=outputs,
        )
    if object_relation is None:
        raise RuntimeError("Object-relation outputs are required before continuing downstream stages.")

    object_assets = existing_object_assets
    if starting_stage in {"object-relation", "object-assets"}:
        set_stage_status(stage_rows, "object-assets", "running")
        set_stage_progress(
            stage_rows,
            "object-assets",
            progress=0,
            message="Building object descriptions into meshes and textures.",
        )
        write_status(
            status_path=status_path,
            job_id=job_id,
            input_payload=input_payload,
            output_name=output_name,
            stage_rows=stage_rows,
            state="running",
            current_stage_id="object-assets",
            log_name=log_name,
        )
        object_assets = run_object_asset_stage(
            stage_rows=stage_rows,
            stage_id="object-assets",
            object_asset_stage=pipeline._object_asset_stage,
            object_relation=object_relation,
            reference_image_result=reference_image_result,
            segmentation_result=scene_understanding.segmentation_result,
            input_payload=input_payload,
            job_id=job_id,
            status_path=status_path,
            output_name=output_name,
            log_name=log_name,
            output_path=output_path,
            outputs=outputs,
        )
        set_stage_progress(
            stage_rows,
            "object-assets",
            progress=100,
            message=f"Generated object assets for {count_generated_object_assets(object_assets)} object(s).",
        )
        write_status(
            status_path=status_path,
            job_id=job_id,
            input_payload=input_payload,
            output_name=output_name,
            stage_rows=stage_rows,
            state="running",
            current_stage_id="object-assets",
            log_name=log_name,
        )
        outputs["object_assets"] = object_assets
        write_execution_result(outputs, output_path)
        set_stage_status(stage_rows, "object-assets", "completed")
        release_cuda_memory(reason="after-object-assets")
    if object_assets is None:
        raise RuntimeError("Object asset outputs are required before layout initialization.")

    layout_initialization = existing_layout_initialization
    if starting_stage in {"object-relation", "object-assets", "layout-initialization"}:
        layout_initialization = run_layout_initialization_stage(
            pipeline=pipeline,
            request=request,
            object_assets=object_assets,
            object_relation=object_relation,
            reference_image_result=reference_image_result,
            scene_understanding=scene_understanding,
            input_payload=input_payload,
            job_id=job_id,
            stage_rows=stage_rows,
            status_path=status_path,
            output_name=output_name,
            output_path=output_path,
            log_name=log_name,
            outputs=outputs,
        )
    if layout_initialization is None:
        raise RuntimeError("Layout initialization outputs are required before simulation preparation.")

    simulation_preparation = existing_simulation_preparation
    if starting_stage in {"object-relation", "object-assets", "layout-initialization", "simulation-preparation"}:
        simulation_preparation = run_simulation_preparation_stage(
            pipeline=pipeline,
            layout_initialization=layout_initialization,
            input_payload=input_payload,
            job_id=job_id,
            stage_rows=stage_rows,
            status_path=status_path,
            output_name=output_name,
            output_path=output_path,
            log_name=log_name,
            outputs=outputs,
        )
    if simulation_preparation is None:
        raise RuntimeError("Simulation preparation outputs are required before downstream stages.")

    physics_optimization = existing_physics_optimization
    if starting_stage in {
        "object-relation",
        "object-assets",
        "layout-initialization",
        "simulation-preparation",
        "physics-optimization",
    }:
        physics_optimization = run_physics_optimization_stage(
            pipeline=pipeline,
            simulation_preparation=simulation_preparation,
            input_payload=input_payload,
            job_id=job_id,
            stage_rows=stage_rows,
            status_path=status_path,
            output_name=output_name,
            output_path=output_path,
            log_name=log_name,
            outputs=outputs,
        ) or physics_optimization

    scene_state_for_visualization: SceneLayout | PhysicsReadyScene | PhysicsOptimizationResult = (
        physics_optimization.optimization_result
        if physics_optimization is not None
        else simulation_preparation.physics_ready_scene
    )

    if starting_stage in {
        "object-relation",
        "object-assets",
        "layout-initialization",
        "simulation-preparation",
        "physics-optimization",
        "visualization",
    }:
        run_visualization_stage(
            pipeline=pipeline,
            scene_state=scene_state_for_visualization,
            input_payload=input_payload,
            job_id=job_id,
            stage_rows=stage_rows,
            status_path=status_path,
            output_name=output_name,
            output_path=output_path,
            log_name=log_name,
            outputs=outputs,
        )


def load_resume_state(output_path: Path) -> tuple[
    dict[str, Any],
    ReferenceImageResult,
    SceneUnderstandingOutputs | None,
    ObjectRelationExtractionOutputs | None,
    ObjectAssetGenerationOutputs | None,
    LayoutInitializationOutputs | None,
    SimulationPreparationOutputs | None,
    PhysicsOptimizationOutputs | None,
]:
    outputs, reference_image_result = load_outputs_and_reference(output_path)
    first_contract_slice = outputs.get("first_contract_slice", {})
    scene_understanding = (
        scene_understanding_outputs_from_dict(first_contract_slice["scene_understanding"])
        if isinstance(first_contract_slice.get("scene_understanding"), Mapping)
        else None
    )
    object_relation = (
        object_relation_outputs_from_dict(first_contract_slice["object_relation"])
        if isinstance(first_contract_slice.get("object_relation"), Mapping)
        else None
    )
    object_assets = (
        object_asset_generation_outputs_from_dict(outputs["object_assets"])
        if isinstance(outputs.get("object_assets"), Mapping)
        else None
    )
    layout_initialization = (
        layout_initialization_outputs_from_dict(outputs["layout_initialization"])
        if isinstance(outputs.get("layout_initialization"), Mapping)
        else None
    )
    simulation_preparation = (
        simulation_preparation_outputs_from_dict(outputs["simulation_preparation"])
        if isinstance(outputs.get("simulation_preparation"), Mapping)
        else None
    )
    physics_optimization = (
        physics_optimization_outputs_from_dict(outputs["physics_optimization"])
        if isinstance(outputs.get("physics_optimization"), Mapping)
        else None
    )
    return (
        outputs,
        reference_image_result,
        scene_understanding,
        object_relation,
        object_assets,
        layout_initialization,
        simulation_preparation,
        physics_optimization,
    )


def run_initial_job(
    *,
    pipeline: Any,
    request: SceneRequest,
    input_payload: dict[str, Any],
    segmentation_mode: str,
    requested_objects: list[str],
    job_id: str,
    stage_rows: list[dict[str, Any]],
    status_path: Path,
    output_name: str,
    output_path: Path,
    log_name: str,
) -> None:
    outputs: dict[str, Any] = {}
    first_pipeline = pipeline._first_contract_slice_pipeline

    set_stage_status(stage_rows, "reference-image", "running")
    set_stage_progress(
        stage_rows,
        "reference-image",
        progress=0,
        message="Generating reference image from prompt.",
    )
    write_status(
        status_path=status_path,
        job_id=job_id,
        input_payload=input_payload,
        output_name=output_name,
        stage_rows=stage_rows,
        state="running",
        current_stage_id="reference-image",
        log_name=log_name,
    )
    reference_image_result = first_pipeline._reference_image_stage.run(request)
    set_stage_progress(
        stage_rows,
        "reference-image",
        progress=100,
        message="Reference image generation complete.",
    )
    outputs["first_contract_slice"] = {"reference_image_result": reference_image_result}
    write_execution_result(outputs, output_path)
    set_stage_status(stage_rows, "reference-image", "completed")

    if segmentation_mode == "manual":
        input_payload["manual_segmentation"] = {
            "state": "pending",
            "reference_image_path": reference_image_result.image.path,
            "updated_at": now_iso(),
            "instances": [],
        }
        set_stage_status(stage_rows, "scene-understanding", "awaiting_input")
        set_stage_progress(
            stage_rows,
            "scene-understanding",
            progress=0,
            message="Reference image ready. Waiting for manual masks.",
        )
        write_status(
            status_path=status_path,
            job_id=job_id,
            input_payload=input_payload,
            output_name=output_name,
            stage_rows=stage_rows,
            state="awaiting_mask_input",
            current_stage_id="scene-understanding",
            log_name=log_name,
        )
        return

    scene_understanding = run_automatic_scene_understanding_with_progress(
        pipeline=pipeline,
        reference_image_result=reference_image_result,
        requested_objects=requested_objects,
        stage_rows=stage_rows,
        status_path=status_path,
        job_id=job_id,
        input_payload=input_payload,
        output_name=output_name,
        log_name=log_name,
        intro_message="Running automatic scene understanding and object cataloging.",
        completion_message="Automatic scene understanding complete.",
    )
    outputs["first_contract_slice"]["scene_understanding"] = scene_understanding
    write_execution_result(outputs, output_path)
    set_stage_status(stage_rows, "scene-understanding", "completed")
    release_cuda_memory(reason="after-scene-understanding")

    run_post_understanding_stages(
        pipeline=pipeline,
        request=request,
        input_payload=input_payload,
        job_id=job_id,
        stage_rows=stage_rows,
        status_path=status_path,
        output_name=output_name,
        output_path=output_path,
        log_name=log_name,
        outputs=outputs,
        reference_image_result=reference_image_result,
        scene_understanding=scene_understanding,
    )


def run_manual_resume(
    *,
    pipeline: Any,
    request: SceneRequest,
    input_payload: dict[str, Any],
    job_id: str,
    stage_rows: list[dict[str, Any]],
    status_path: Path,
    output_name: str,
    output_path: Path,
    log_name: str,
) -> None:
    if not isinstance(stage_rows, list) or not stage_rows:
        raise RuntimeError("Could not resume the manual job because the stage state is missing.")

    manual_segmentation = input_payload.get("manual_segmentation")
    if not isinstance(manual_segmentation, Mapping):
        raise RuntimeError("No saved manual segmentation payload is available for this job.")
    preserved_depth_result = preserved_depth_result_from_manual_segmentation(
        manual_segmentation
    )

    outputs, reference_image_result = load_outputs_and_reference(output_path)
    outputs.setdefault("first_contract_slice", {})

    set_stage_status(stage_rows, "scene-understanding", "running")
    set_stage_progress(
        stage_rows,
        "scene-understanding",
        progress=0,
        message="Resuming manual stage 2 with user-provided masks.",
    )
    write_status(
        status_path=status_path,
        job_id=job_id,
        input_payload=input_payload,
        output_name=output_name,
        stage_rows=stage_rows,
        state="running",
        current_stage_id="scene-understanding",
        log_name=log_name,
    )
    scene_understanding = run_manual_scene_understanding_with_progress(
        pipeline=pipeline,
        reference_image_result=reference_image_result,
        manual_segmentation=manual_segmentation,
        preserved_depth_result=preserved_depth_result,
        stage_rows=stage_rows,
        status_path=status_path,
        job_id=job_id,
        input_payload=input_payload,
        output_name=output_name,
        log_name=log_name,
        intro_message="Resuming manual stage 2 with user-provided masks.",
        completion_message="Manual scene understanding complete.",
    )
    write_status(
        status_path=status_path,
        job_id=job_id,
        input_payload=input_payload,
        output_name=output_name,
        stage_rows=stage_rows,
        state="running",
        current_stage_id="scene-understanding",
        log_name=log_name,
    )
    outputs["first_contract_slice"]["scene_understanding"] = scene_understanding
    write_execution_result(outputs, output_path)
    set_stage_status(stage_rows, "scene-understanding", "completed")
    release_cuda_memory(reason="after-scene-understanding")
    input_payload["manual_segmentation"] = {
        **manual_segmentation,
        "state": "applied",
        "reference_image_path": reference_image_result.image.path,
        "updated_at": now_iso(),
    }

    run_post_understanding_stages(
        pipeline=pipeline,
        request=request,
        input_payload=input_payload,
        job_id=job_id,
        stage_rows=stage_rows,
        status_path=status_path,
        output_name=output_name,
        output_path=output_path,
        log_name=log_name,
        outputs=outputs,
        reference_image_result=reference_image_result,
        scene_understanding=scene_understanding,
    )


def _require_resume_output(value: Any, *, stage_id: str, dependency_name: str) -> Any:
    if value is None:
        raise RuntimeError(
            f"Could not retry stage '{stage_id}' because the saved '{dependency_name}' output is missing."
        )
    return value


def run_resume_from_stage(
    *,
    pipeline: Any,
    request: SceneRequest,
    input_payload: dict[str, Any],
    segmentation_mode: str,
    requested_objects: list[str],
    resume_from_stage: str,
    job_id: str,
    stage_rows: list[dict[str, Any]],
    status_path: Path,
    output_name: str,
    output_path: Path,
    log_name: str,
) -> None:
    if resume_from_stage == "reference-image":
        purge_scene_artifacts(input_payload["scene_id"])
        run_initial_job(
            pipeline=pipeline,
            request=request,
            input_payload=input_payload,
            segmentation_mode=segmentation_mode,
            requested_objects=requested_objects,
            job_id=job_id,
            stage_rows=stage_rows,
            status_path=status_path,
            output_name=output_name,
            output_path=output_path,
            log_name=log_name,
        )
        return

    (
        outputs,
        reference_image_result,
        scene_understanding,
        object_relation,
        object_assets,
        layout_initialization,
        simulation_preparation,
        physics_optimization,
    ) = load_resume_state(output_path)
    outputs.setdefault("first_contract_slice", {})["reference_image_result"] = reference_image_result
    prune_outputs_for_resume(outputs, resume_from_stage=resume_from_stage)
    previous_completed_stage = PREVIOUS_COMPLETED_STAGE_BY_RETRY_STAGE.get(resume_from_stage)
    if previous_completed_stage is None:
        raise RuntimeError(f"Unsupported resume target '{resume_from_stage}'.")
    purge_downstream_artifacts(input_payload["scene_id"], completed_stage=previous_completed_stage)
    write_execution_result(outputs, output_path)

    if resume_from_stage == "scene-understanding":
        set_stage_status(stage_rows, resume_from_stage, "running")
        set_stage_progress(
            stage_rows,
            resume_from_stage,
            progress=0,
            message="Retrying scene understanding from the saved reference image.",
        )
        write_status(
            status_path=status_path,
            job_id=job_id,
            input_payload=input_payload,
            output_name=output_name,
            stage_rows=stage_rows,
            state="running",
            current_stage_id=resume_from_stage,
            log_name=log_name,
        )

        if segmentation_mode == "manual":
            manual_segmentation = input_payload.get("manual_segmentation")
            if not isinstance(manual_segmentation, Mapping) or not manual_segmentation.get("instances"):
                raise RuntimeError(
                    "Manual scene-understanding resume requires saved manual masks."
                )
            scene_understanding = build_manual_scene_understanding(
                pipeline=pipeline,
                reference_image_result=reference_image_result,
                manual_segmentation=manual_segmentation,
            )
            input_payload["manual_segmentation"] = {
                **manual_segmentation,
                "state": "applied",
                "reference_image_path": reference_image_result.image.path,
                "updated_at": now_iso(),
            }
        else:
            scene_understanding = run_automatic_scene_understanding_with_progress(
                pipeline=pipeline,
                reference_image_result=reference_image_result,
                requested_objects=requested_objects,
                stage_rows=stage_rows,
                status_path=status_path,
                job_id=job_id,
                input_payload=input_payload,
                output_name=output_name,
                log_name=log_name,
                intro_message="Retrying automatic scene understanding from the saved reference image.",
                completion_message="Automatic scene understanding retry complete.",
            )

        outputs["first_contract_slice"]["scene_understanding"] = scene_understanding
        write_execution_result(outputs, output_path)
        set_stage_status(stage_rows, resume_from_stage, "completed")
        release_cuda_memory(reason="after-scene-understanding")
        run_post_understanding_stages(
            pipeline=pipeline,
            request=request,
            input_payload=input_payload,
            job_id=job_id,
            stage_rows=stage_rows,
            status_path=status_path,
            output_name=output_name,
            output_path=output_path,
            log_name=log_name,
            outputs=outputs,
            reference_image_result=reference_image_result,
            scene_understanding=scene_understanding,
        )
        return

    scene_understanding = _require_resume_output(
        scene_understanding,
        stage_id=resume_from_stage,
        dependency_name="scene-understanding",
    )
    if resume_from_stage == "object-relation":
        run_post_understanding_stages(
            pipeline=pipeline,
            request=request,
            input_payload=input_payload,
            job_id=job_id,
            stage_rows=stage_rows,
            status_path=status_path,
            output_name=output_name,
            output_path=output_path,
            log_name=log_name,
            outputs=outputs,
            reference_image_result=reference_image_result,
            scene_understanding=scene_understanding,
            starting_stage="object-relation",
        )
        return

    object_relation = _require_resume_output(
        object_relation,
        stage_id=resume_from_stage,
        dependency_name="object-relation",
    )
    manual_size_priors = input_payload.get("manual_size_priors")
    manual_sizes_applied = False
    if isinstance(manual_size_priors, Mapping) and manual_size_priors.get("state") == "submitted":
        object_relation = apply_manual_size_priors_to_object_relation(object_relation, manual_size_priors)
        input_payload["manual_size_priors"] = {
            **manual_size_priors,
            "state": "applied",
            "updated_at": now_iso(),
        }
        manual_sizes_applied = True
    object_relation = apply_relation_graph_override(
        object_relation,
        input_payload.get("relation_graph_override"),
    )
    materialize_object_relation_workspace(
        reference_image_result.request.scene_id,
        object_relation,
    )
    outputs.setdefault("first_contract_slice", {})["object_relation"] = object_relation
    if manual_sizes_applied:
        write_execution_result(outputs, output_path)
    object_assets = object_assets if resume_from_stage != "object-assets" else None
    layout_initialization = layout_initialization if resume_from_stage != "layout-initialization" else None
    simulation_preparation = simulation_preparation if resume_from_stage != "simulation-preparation" else None
    if resume_from_stage == "visualization":
        simulation_preparation = _require_resume_output(
            simulation_preparation,
            stage_id=resume_from_stage,
            dependency_name="simulation-preparation",
        )

    run_post_understanding_stages(
        pipeline=pipeline,
        request=request,
        input_payload=input_payload,
        job_id=job_id,
        stage_rows=stage_rows,
        status_path=status_path,
        output_name=output_name,
        output_path=output_path,
        log_name=log_name,
        outputs=outputs,
        reference_image_result=reference_image_result,
        scene_understanding=scene_understanding,
        starting_stage=resume_from_stage,
        existing_object_relation=object_relation,
        existing_object_assets=object_assets,
        existing_layout_initialization=layout_initialization,
        existing_simulation_preparation=simulation_preparation,
        existing_physics_optimization=physics_optimization,
    )


def build_scene_request(
    *,
    scene_id: str,
    prompt: str,
    requested_objects: list[str],
) -> SceneRequest:
    return scene_request_from_dict(
        {
            "scene_id": scene_id,
            "text_prompt": prompt,
            "requested_objects": requested_objects or None,
            "tags": ["dashboard"],
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a dashboard-managed PAT3D job with stage progress.")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--runtime-config-out", required=True)
    parser.add_argument("--resume-manual", action="store_true")
    parser.add_argument("--resume-from", choices=RESUME_FROM_CHOICES)
    args = parser.parse_args()

    job_id = args.job_id
    input_path = Path(args.input)
    status_path = Path(args.status)
    output_path = Path(args.output)
    runtime_config_out = Path(args.runtime_config_out)
    log_name = f"{job_id}.log"

    input_payload = read_json(input_path)
    prompt = str(input_payload["prompt"]).strip()
    if not prompt:
        raise RuntimeError("Prompt must be non-empty.")

    input_payload["started_at"] = input_payload.get("started_at") or now_iso()
    input_payload["scene_id"] = resolve_scene_id(input_payload.get("scene_id"), prompt, job_id)
    input_payload["workspace_root"] = _path_text(
        scene_workspace_paths(input_payload["scene_id"])["root"]
    )
    requested_objects = [
        str(item).strip()
        for item in input_payload.get("requested_objects", [])
        if str(item).strip()
    ]
    requested_objects_inferred = resolve_requested_objects_inferred(input_payload.get("requested_objects_inferred"))
    if requested_objects:
        requested_objects_inferred = False
    input_payload["requested_objects_inferred"] = requested_objects_inferred
    stage_backends_profile = resolve_stage_backend_profile(
        input_payload.get("stage_backends_profile")
        or os.environ.get("PAT3D_DASHBOARD_STAGE_BACKENDS_PROFILE")
    )
    stage_backends = resolve_stage_backends(
        input_payload.get("stage_backends"),
        profile=stage_backends_profile,
    )
    segmentation_mode = resolve_segmentation_mode(input_payload.get("segmentation_mode"))
    preview_angle_count = resolve_preview_angle_count(input_payload.get("preview_angle_count"))
    requested_object_inference_budget = resolve_requested_object_inference_budget(
        input_payload.get("requested_object_inference_budget")
    )
    physics_settings = resolve_physics_settings(
        input_payload.get("physics_settings") or input_payload.get("physicsSettings")
    )
    resume_from_stage = normalize_resume_from(args.resume_from)
    llm_model = resolve_model(
        input_payload.get("llm_model") or input_payload.get("llmModel"),
        CHAT_MODEL_OPTIONS,
        "gpt-5.4",
    )
    image_model = resolve_model(
        input_payload.get("image_model") or input_payload.get("imageModel"),
        IMAGE_MODEL_OPTIONS,
        "gpt-image-1.5",
    )
    reasoning_effort = resolve_reasoning_effort(input_payload.get("reasoning_effort"), "high")
    structured_llm_max_attempts = resolve_structured_llm_max_attempts(
        input_payload.get("structured_llm_max_attempts")
    )
    structured_llm_reasoning_budget = resolve_structured_llm_reasoning_budget(
        input_payload.get("structured_llm_reasoning_budget")
    )
    input_payload["stage_backends"] = stage_backends
    input_payload["stage_backends_profile"] = stage_backends_profile
    input_payload["segmentation_mode"] = segmentation_mode
    input_payload["preview_angle_count"] = preview_angle_count
    input_payload["requested_object_inference_budget"] = requested_object_inference_budget
    input_payload["physics_settings"] = physics_settings
    input_payload["llm_model"] = llm_model
    input_payload["image_model"] = image_model
    input_payload["structured_llm_max_attempts"] = structured_llm_max_attempts
    input_payload["structured_llm_reasoning_budget"] = structured_llm_reasoning_budget
    input_payload["reasoning_effort"] = reasoning_effort

    if args.resume_manual and resume_from_stage is not None:
        raise RuntimeError("Choose either --resume-manual or --resume-from, not both.")

    if not args.resume_manual and resume_from_stage is None:
        stage_rows = make_stage_rows()
        write_status(
            status_path=status_path,
            job_id=job_id,
            input_payload=input_payload,
            output_name=output_path.name,
            stage_rows=stage_rows,
            state="running",
            log_name=log_name,
        )
    else:
        status_payload = read_json(status_path)
        stage_rows = status_payload.get("stages") or make_stage_rows()
        if resume_from_stage is not None:
            stage_rows = prepare_stage_rows_for_resume(
                stage_rows,
                resume_from_stage=resume_from_stage,
            )

    try:
        validate_stage_backend_prerequisites(
            stage_backends,
            segmentation_mode=segmentation_mode,
        )

        if stage_backends_require_openai_api_key(
            stage_backends,
            requested_objects=requested_objects,
            segmentation_mode=segmentation_mode,
            object_crop_completion_enabled=bool(input_payload.get("object_crop_completion_enabled", True)),
        ):
            os.environ.setdefault("OPENAI_API_KEY", load_api_key())

        if not args.resume_manual and segmentation_mode != "manual" and not requested_objects:
            if stage_backends_require_openai_api_key(
                stage_backends,
                requested_objects=requested_objects,
                segmentation_mode=segmentation_mode,
                object_crop_completion_enabled=bool(input_payload.get("object_crop_completion_enabled", True)),
            ):
                requested_objects = infer_requested_objects_from_prompt(
                    prompt,
                    api_key=os.environ.get("OPENAI_API_KEY"),
                    model=llm_model,
                    reasoning_effort=reasoning_effort,
                    inference_budget=requested_object_inference_budget,
                )
            else:
                requested_objects = infer_requested_objects_from_local_smoke_prompt(prompt)
            input_payload["requested_objects"] = requested_objects
            requested_objects_inferred = bool(requested_objects)
            input_payload["requested_objects_inferred"] = requested_objects_inferred

            if not requested_objects:
                raise RuntimeError(
                    "No requested objects could be inferred from the prompt. "
                    "Please provide requested objects explicitly."
                )

        if not args.resume_manual and resume_from_stage is None:
            purge_scene_artifacts(input_payload["scene_id"])

        runtime_config = build_runtime_config(
            input_payload["scene_id"],
            stage_backends,
            stage_backends_profile=stage_backends_profile,
            physics_settings=physics_settings,
            preview_angle_count=preview_angle_count,
            llm_model=llm_model,
            image_model=image_model,
            object_crop_completion_enabled=bool(input_payload.get("object_crop_completion_enabled", True)),
            object_crop_completion_model=str(
                input_payload.get("object_crop_completion_model", "gpt-image-1.5")
            ),
            structured_llm_max_attempts=structured_llm_max_attempts,
            structured_llm_reasoning_budget=structured_llm_reasoning_budget,
            reasoning_effort=reasoning_effort,
            resume_from_stage=resume_from_stage,
        )
        runtime_config.write_json(runtime_config_out)
        pipeline = build_pipeline_from_config(runtime_config)
        request = build_scene_request(
            scene_id=input_payload["scene_id"],
            prompt=prompt,
            requested_objects=requested_objects,
        )

        if args.resume_manual:
            if segmentation_mode != "manual":
                raise RuntimeError("Manual resume was requested for a non-manual dashboard job.")
            run_manual_resume(
                pipeline=pipeline,
                request=request,
                input_payload=input_payload,
                job_id=job_id,
                stage_rows=stage_rows,
                status_path=status_path,
                output_name=output_path.name,
                output_path=output_path,
                log_name=log_name,
            )
        elif resume_from_stage is not None:
            run_resume_from_stage(
                pipeline=pipeline,
                request=request,
                input_payload=input_payload,
                segmentation_mode=segmentation_mode,
                requested_objects=requested_objects,
                resume_from_stage=resume_from_stage,
                job_id=job_id,
                stage_rows=stage_rows,
                status_path=status_path,
                output_name=output_path.name,
                output_path=output_path,
                log_name=log_name,
            )
        else:
            run_initial_job(
                pipeline=pipeline,
                request=request,
                input_payload=input_payload,
                segmentation_mode=segmentation_mode,
                requested_objects=requested_objects,
                job_id=job_id,
                stage_rows=stage_rows,
                status_path=status_path,
                output_name=output_path.name,
                output_path=output_path,
                log_name=log_name,
            )

        if segmentation_mode == "manual" and not args.resume_manual and input_payload.get("manual_segmentation"):
            write_status(
                status_path=status_path,
                job_id=job_id,
                input_payload=input_payload,
                output_name=output_path.name,
                stage_rows=stage_rows,
                state="completed",
                log_name=log_name,
            )
            return 0

        write_status(
            status_path=status_path,
            job_id=job_id,
            input_payload=input_payload,
            output_name=output_path.name,
            stage_rows=stage_rows,
            state="completed",
            log_name=log_name,
        )
        return 0
    except PauseForManualSizeInput:
        return 0
    except Exception as error:
        active_stage = next((row["id"] for row in stage_rows if row["status"] == "running"), None)
        normalized_error = normalize_job_error(error, stage_id=active_stage)
        if active_stage is not None:
            set_stage_status(stage_rows, active_stage, "failed", error=normalized_error)
        write_status(
            status_path=status_path,
            job_id=job_id,
            input_payload=input_payload,
            output_name=output_path.name,
            stage_rows=stage_rows,
            state="failed",
            current_stage_id=active_stage,
            error=normalized_error,
            log_name=log_name,
        )
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
