from __future__ import annotations

from dataclasses import is_dataclass
import json
from pathlib import Path
import re
from typing import Any, Iterable


MODEL_MODULES = [
    "pat3d.models",
    "pat3d.models.base",
    "pat3d.models.scene",
    "pat3d.models.understanding",
    "pat3d.models.objects",
    "pat3d.models.physics",
]

CONTRACT_MODULES = [
    "pat3d.contracts",
    "pat3d.contracts.base",
    "pat3d.contracts.providers",
]


def resolve_symbol(symbol_name: str, module_names: Iterable[str]) -> Any | None:
    for module_name in module_names:
        try:
            module = __import__(module_name, fromlist=[symbol_name])
        except Exception:
            continue
        symbol = getattr(module, symbol_name, None)
        if symbol is not None:
            return symbol
    return None


def instantiate_model(symbol_name: str, payload: dict[str, Any]) -> Any:
    model_type = resolve_symbol(symbol_name, MODEL_MODULES)
    if model_type is None:
        return payload
    if hasattr(model_type, "model_validate"):
        return model_type.model_validate(payload)
    if is_dataclass(model_type):
        return model_type(**payload)
    try:
        return model_type(**payload)
    except TypeError:
        return payload


def validate_schema(response_schema: Any, payload: Any) -> Any:
    if response_schema is None:
        return payload
    if hasattr(response_schema, "model_validate"):
        return response_schema.model_validate(payload)
    if is_dataclass(response_schema):
        return response_schema(**payload)
    try:
        return response_schema(**payload)
    except Exception:
        return payload


def extract_json_payload(text: str) -> Any:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    matches = list(re.finditer(r"({.*}|\[.*\])", text, flags=re.DOTALL))
    for match in matches:
        snippet = match.group(0)
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            continue
    raise ValueError("Unable to parse JSON payload from provider response")


def strip_numeric_suffix(value: str) -> str:
    return re.sub(r"\d+$", "", value)


def scene_id_from_reference(reference_image_result: Any, default: str = "scene") -> str:
    if isinstance(reference_image_result, dict):
        request = reference_image_result.get("request")
        if isinstance(request, dict) and request.get("scene_id"):
            return str(request["scene_id"])
        if reference_image_result.get("scene_id"):
            return str(reference_image_result["scene_id"])
    request = getattr(reference_image_result, "request", None)
    if request is not None:
        request_scene_id = getattr(request, "scene_id", None)
        if request_scene_id:
            return str(request_scene_id)
    direct_scene_id = getattr(reference_image_result, "scene_id", None)
    if direct_scene_id:
        return str(direct_scene_id)
    return default


def artifact_path_from_ref(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return value.get("path")
    return getattr(value, "path", None)


def image_path_from_reference(reference_image_result: Any) -> str:
    if isinstance(reference_image_result, dict):
        image = reference_image_result.get("image")
        path = artifact_path_from_ref(image)
        if path:
            return path
        if reference_image_result.get("reference_image"):
            return str(reference_image_result["reference_image"])
    image = getattr(reference_image_result, "image", None)
    path = artifact_path_from_ref(image)
    if path:
        return path
    raise ValueError("Unable to resolve reference image path from result")


def coerce_mask_instances(instances: list[dict[str, Any]]) -> list[Any]:
    return [instantiate_model("MaskInstance", item) for item in instances]


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))
