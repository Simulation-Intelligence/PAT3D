from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import re
import shutil
import sys
import traceback
from typing import Any, Callable, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
SEMANTICS_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SEMANTICS_ROOT) not in sys.path:
    sys.path.insert(0, str(SEMANTICS_ROOT))

from clip.compute_clip_score import compute_clip_score
from pat3d.models import ObjectPose
from pat3d.models.pose_utils import pose_to_matrix
from phys_plausibility.compute_physical_plausibility_score import compute_physical_plausibility_score
from render.compute_render import render_scene_views
from vqa.compute_vqa_score import compute_vqa_score


DEFAULT_RESULTS_ROOT = REPO_ROOT / "results" / "metrics"
DEFAULT_METRIC_SCENE_ROOT = REPO_ROOT / "data" / "metrics" / "scene_inputs"
METRIC_RENDER_SCHEMA_VERSION = 2


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _nested_get(data: dict[str, Any], path: str) -> Any:
    current: Any = data
    for part in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def extract_scene_id(runtime: dict[str, Any], runtime_path: Path | None = None) -> str:
    for path in (
        "layout_initialization.scene_layout.scene_id",
        "simulation_preparation.physics_ready_scene.scene_id",
        "physics_optimization.optimization_result.scene_id",
        "visualization.render_result.scene_id",
        "object_assets.object_assets.scene_id",
        "first_contract_slice.object_relation.object_catalog.scene_id",
    ):
        value = _nested_get(runtime, path)
        if isinstance(value, str) and value.strip():
            return value.strip()
    if runtime_path is not None:
        return runtime_path.name.replace(".paper_core.json", "").replace(".json", "")
    return "unknown"


def metric_case_file_name(scene_id: str) -> str:
    cleaned = str(scene_id or "unknown").replace("/", "_").replace("\\", "_").strip() or "unknown"
    return f"{cleaned}.json"


def extract_prompt(runtime: dict[str, Any]) -> str:
    for path in (
        "input_payload.prompt",
        "first_contract_slice.reference_image_result.generation_prompt",
        "reference_image_result.generation_prompt",
        "prompt",
    ):
        value = _nested_get(runtime, path)
        if isinstance(value, str) and value.strip():
            return value.strip()
    catalog = _nested_get(runtime, "first_contract_slice.object_relation.object_catalog")
    objects = _as_list(catalog.get("objects") if isinstance(catalog, dict) else None)
    labels = [
        str(obj.get("display_name") or obj.get("canonical_name") or obj.get("object_id"))
        for obj in objects
        if isinstance(obj, dict) and (obj.get("display_name") or obj.get("canonical_name") or obj.get("object_id"))
    ]
    return ", ".join(labels)


def resolve_layout_scene_dir(runtime: dict[str, Any], scene_id: str) -> Path:
    workspace_scene_dir = (
        REPO_ROOT
        / "results"
        / "workspaces"
        / scene_id
        / "layout"
        / "scene"
        / scene_id
    )
    if workspace_scene_dir.exists():
        return workspace_scene_dir

    direct_scene_dir = REPO_ROOT / "data" / "layout" / scene_id
    if direct_scene_dir.exists():
        return direct_scene_dir

    artifacts = _as_list(_nested_get(runtime, "layout_initialization.scene_layout.artifacts"))
    layout_paths = [
        Path(artifact["path"])
        for artifact in artifacts
        if isinstance(artifact, dict)
        and artifact.get("artifact_type") == "layout_mesh"
        and isinstance(artifact.get("path"), str)
    ]
    existing_parents = []
    for path in layout_paths:
        candidate = path if path.is_absolute() else REPO_ROOT / path
        if candidate.exists():
            existing_parents.append(candidate.parent)
    unique_parents = sorted({parent.resolve() for parent in existing_parents})
    if len(unique_parents) == 1:
        return unique_parents[0]
    raise FileNotFoundError(
        "Could not resolve layout mesh directory for scene "
        f"'{scene_id}'. Expected {workspace_scene_dir} or {direct_scene_dir}"
    )


def _extract_layout(runtime: dict[str, Any]) -> dict[str, Any] | None:
    value = _nested_get(runtime, "layout_initialization.scene_layout")
    return value if isinstance(value, dict) else None


def _extract_simulation(runtime: dict[str, Any]) -> dict[str, Any] | None:
    value = _nested_get(runtime, "simulation_preparation.physics_ready_scene")
    return value if isinstance(value, dict) else None


def _extract_physics(runtime: dict[str, Any]) -> dict[str, Any] | None:
    value = _nested_get(runtime, "physics_optimization.optimization_result")
    return value if isinstance(value, dict) else None


def _extract_object_assets(runtime: dict[str, Any]) -> dict[str, Any] | None:
    for path in (
        "layout_initialization.object_assets",
        "simulation_preparation.object_assets",
        "object_assets.object_assets",
    ):
        value = _nested_get(runtime, path)
        if isinstance(value, dict):
            return value
    return None


def _extract_final_object_poses(runtime: dict[str, Any]) -> list[dict[str, Any]]:
    physics = _extract_physics(runtime)
    if isinstance(physics, dict):
        optimized = _as_list(physics.get("optimized_object_poses"))
        if optimized:
            return [pose for pose in optimized if isinstance(pose, dict)]
    simulation = _extract_simulation(runtime)
    if isinstance(simulation, dict):
        object_poses = _as_list(simulation.get("object_poses"))
        if object_poses:
            return [pose for pose in object_poses if isinstance(pose, dict)]
    layout = _extract_layout(runtime)
    return [pose for pose in _as_list(layout.get("object_poses") if isinstance(layout, dict) else None) if isinstance(pose, dict)]


def _safe_scene_name(value: str, fallback: str = "object") -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip())
    return normalized.strip("._-") or fallback


def _resolve_artifact_path(path_value: str, *, runtime_path: Path | None = None) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    repo_path = REPO_ROOT / path
    if repo_path.exists():
        return repo_path
    if runtime_path is not None:
        runtime_candidate = runtime_path.resolve().parent / path
        if runtime_candidate.exists():
            return runtime_candidate
    return repo_path


def _normalize_object_id_candidates(value: object) -> list[str]:
    raw = str(value or "").strip()
    if not raw:
        return []
    candidates = {raw.lower()}
    if ":" in raw:
        last_segment = raw.split(":")[-1].strip()
        if last_segment:
            candidates.add(last_segment.lower())
    current = raw
    while True:
        stripped = re.sub(r"(?:(?:::)|[_-]|\b)(\d+)$", "", current).rstrip(":_-")
        if not stripped or stripped == current:
            break
        current = stripped
        candidates.add(current.lower())
        if ":" in current:
            last_segment = current.split(":")[-1].strip()
            if last_segment:
                candidates.add(last_segment.lower())
    return list(candidates)


def _record_id_candidates(record: dict[str, Any], extra_ids: Sequence[object] = ()) -> list[str]:
    candidates = _normalize_object_id_candidates(record.get("object_id"))
    for extra_id in extra_ids:
        candidates.extend(_normalize_object_id_candidates(extra_id))
    return candidates


def _resolve_record_by_object_id(
    records: Sequence[dict[str, Any]],
    object_id: object,
    *,
    extra_ids_selector: Callable[[dict[str, Any]], Sequence[object]] | None = None,
) -> dict[str, Any] | None:
    exact_id = str(object_id or "").strip()
    if not exact_id:
        return None
    for record in records:
        if str(record.get("object_id") or "").strip() == exact_id:
            return record
    object_candidates = _normalize_object_id_candidates(exact_id)
    matches: list[dict[str, Any]] = []
    for record in records:
        extra_ids = extra_ids_selector(record) if extra_ids_selector is not None else ()
        record_candidates = _record_id_candidates(record, extra_ids)
        if any(candidate in record_candidates for candidate in object_candidates):
            matches.append(record)
    return matches[0] if len(matches) == 1 else None


def _pose_transform_for_bundle(pose_payload: dict[str, Any] | None) -> dict[str, object] | None:
    if not isinstance(pose_payload, dict) or not pose_payload.get("object_id"):
        return None
    translation = _as_list(pose_payload.get("translation_xyz") or pose_payload.get("position"))[:3]
    rotation = _as_list(pose_payload.get("rotation_value") or pose_payload.get("rotation") or [1.0, 0.0, 0.0, 0.0])
    scale = _as_list(pose_payload.get("scale_xyz") or pose_payload.get("scale") or [1.0, 1.0, 1.0])[:3]
    return {
        "translation_xyz": [float(value) for value in translation],
        "rotation_type": str(pose_payload.get("rotation_type") or "quaternion"),
        "rotation_value": [float(value) for value in rotation],
        "scale_xyz": [float(value) for value in scale],
    }


def _asset_mesh_already_in_scene_space(asset: dict[str, Any]) -> bool:
    metadata = asset.get("metadata")
    if not isinstance(metadata, dict):
        return False
    notes = [note for note in _as_list(metadata.get("notes")) if isinstance(note, str)]
    return "mesh_pose_space=scene" in notes


def _build_original_textured_bundle(
    runtime: dict[str, Any],
    *,
    scene_id: str,
) -> dict[str, Any] | None:
    object_assets = _extract_object_assets(runtime)
    if not isinstance(object_assets, dict):
        return None
    scene_assets = [
        asset
        for asset in _as_list(object_assets.get("assets"))
        if isinstance(asset, dict)
        and asset.get("object_id")
        and isinstance(_nested_get(asset, "mesh_obj.path"), str)
    ]
    if not scene_assets:
        return None

    poses = _extract_final_object_poses(runtime)
    objects: list[dict[str, object]] = []
    for asset in scene_assets:
        provider_asset_ids = _as_list(asset.get("provider_asset_id"))
        pose = _resolve_record_by_object_id(
            poses,
            asset.get("object_id"),
            extra_ids_selector=lambda _record: provider_asset_ids,
        )
        transform = None if _asset_mesh_already_in_scene_space(asset) else _pose_transform_for_bundle(pose)
        objects.append(
            {
                "object_id": str(asset["object_id"]),
                "mesh_obj_path": str(_nested_get(asset, "mesh_obj.path")),
                "mesh_mtl_path": _nested_get(asset, "mesh_mtl.path"),
                "texture_image_path": _nested_get(asset, "texture_image.path"),
                "already_transformed": bool(_asset_mesh_already_in_scene_space(asset) or transform is None),
                "transform": transform,
            }
        )
    if not objects:
        return None
    return {
        "scene_id": scene_id,
        "source_scene_type": "runtime_object_assets",
        "geometry_source_type": "object_asset_meshes",
        "geometry_variant": "original_textured",
        "objects": objects,
    }


def _bundle_transform_matrix(item: dict[str, Any]) -> np.ndarray | None:
    if item.get("already_transformed", True):
        return None
    transform = item.get("transform")
    if isinstance(transform, dict):
        translation = tuple(float(value) for value in _as_list(transform.get("translation_xyz"))[:3])
        rotation_value = tuple(float(value) for value in _as_list(transform.get("rotation_value"))[:4])
        scale_raw = transform.get("scale_xyz")
        scale_xyz = (
            tuple(float(value) for value in _as_list(scale_raw)[:3])
            if scale_raw is not None
            else None
        )
        pose = ObjectPose(
            object_id=str(item.get("object_id") or "object"),
            translation_xyz=translation,
            rotation_type=str(transform.get("rotation_type") or "quaternion"),
            rotation_value=rotation_value,
            scale_xyz=scale_xyz,
        )
        return pose_to_matrix(pose)
    if isinstance(transform, list):
        matrix = np.asarray(transform, dtype=np.float64)
        if matrix.shape == (4, 4):
            return matrix
    return None


def _format_float(value: float) -> str:
    return f"{value:.10g}"


def _transform_obj_with_matrix(source_path: Path, output_path: Path, matrix: np.ndarray) -> None:
    linear = np.asarray(matrix[:3, :3], dtype=np.float64)
    translation = np.asarray(matrix[:3, 3], dtype=np.float64)
    try:
        normal_matrix = np.linalg.inv(linear).T
    except np.linalg.LinAlgError:
        normal_matrix = linear
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with source_path.open("r", encoding="utf-8", errors="ignore") as source, output_path.open(
        "w",
        encoding="utf-8",
    ) as target:
        for line in source:
            if line.startswith("v "):
                parts = line.rstrip("\n").split()
                if len(parts) >= 4:
                    xyz = np.asarray([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float64)
                    transformed = linear @ xyz + translation
                    target.write(
                        " ".join(
                            [
                                parts[0],
                                _format_float(float(transformed[0])),
                                _format_float(float(transformed[1])),
                                _format_float(float(transformed[2])),
                                *parts[4:],
                            ]
                        )
                        + "\n"
                    )
                    continue
            if line.startswith("vn "):
                parts = line.rstrip("\n").split()
                if len(parts) >= 4:
                    normal = np.asarray([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float64)
                    transformed = normal_matrix @ normal
                    norm = np.linalg.norm(transformed)
                    if norm > 0:
                        transformed = transformed / norm
                    target.write(
                        " ".join(
                            [
                                parts[0],
                                _format_float(float(transformed[0])),
                                _format_float(float(transformed[1])),
                                _format_float(float(transformed[2])),
                                *parts[4:],
                            ]
                        )
                        + "\n"
                    )
                    continue
            target.write(line)


def _read_obj_mtllibs(obj_path: Path) -> tuple[str, ...]:
    try:
        lines = obj_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return ()
    mtllibs: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("mtllib "):
            mtllibs.extend(part for part in stripped.split()[1:] if part)
    return tuple(mtllibs)


def _read_mtl_texture_refs(mtl_path: Path) -> tuple[str, ...]:
    try:
        lines = mtl_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return ()
    refs: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith(("map_Kd ", "map_Ka ", "map_d ", "bump ", "map_Bump ")):
            parts = stripped.split()
            if len(parts) >= 2:
                refs.append(parts[-1])
    return tuple(refs)


def _copy_if_exists(source: Path, target: Path) -> Path | None:
    if not source.exists():
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != target.resolve():
        shutil.copy2(source, target)
    return target


def _rewrite_obj_mtllibs(
    obj_path: Path,
    name_map: dict[str, str],
    *,
    default_names: Sequence[str] = (),
) -> None:
    try:
        lines = obj_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return
    updated: list[str] = []
    saw_mtllib = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("mtllib "):
            saw_mtllib = True
            refs = [
                name_map.get(ref) or name_map.get(Path(ref).name) or ref
                for ref in stripped.split()[1:]
            ]
            updated.append(f"mtllib {' '.join(refs)}")
            continue
        updated.append(line)
    if not saw_mtllib and default_names:
        updated.insert(0, f"mtllib {' '.join(default_names)}")
    obj_path.write_text("\n".join(updated) + "\n", encoding="utf-8")


def _rewrite_mtl_texture_refs(mtl_path: Path, name_map: dict[str, str]) -> None:
    try:
        lines = mtl_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return
    updated: list[str] = []
    ref_prefixes = ("map_Kd ", "map_Ka ", "map_d ", "bump ", "map_Bump ")
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(ref_prefixes):
            parts = stripped.split()
            if len(parts) >= 2:
                ref = parts[-1]
                rewritten = name_map.get(ref) or name_map.get(Path(ref).name)
                if rewritten:
                    parts[-1] = rewritten
                    updated.append(" ".join(parts))
                    continue
        updated.append(line)
    mtl_path.write_text("\n".join(updated) + "\n", encoding="utf-8")


def _copy_material_sidecars(
    source_obj: Path,
    output_obj: Path,
    *,
    source_mtl_path: Path | None = None,
    source_texture_path: Path | None = None,
) -> tuple[Path | None, Path | None]:
    obj_mtllibs = _read_obj_mtllibs(source_obj)
    mtl_candidates: list[Path] = []
    if source_mtl_path is not None:
        mtl_candidates.append(source_mtl_path)
    mtl_candidates.extend(source_obj.parent / mtllib for mtllib in obj_mtllibs)
    mtl_candidates.append(source_obj.with_suffix(".mtl"))

    unique_mtls: list[Path] = []
    seen_mtls: set[str] = set()
    for candidate in mtl_candidates:
        if not candidate.exists():
            continue
        resolved = str(candidate.resolve())
        if resolved in seen_mtls:
            continue
        seen_mtls.add(resolved)
        unique_mtls.append(candidate)

    copied_mtl: Path | None = None
    copied_texture: Path | None = None
    rewritten_mtl_names: list[str] = []
    obj_mtl_name_map: dict[str, str] = {}
    for source_mtl in unique_mtls:
        target_mtl_name = f"{output_obj.stem}__{source_mtl.name}"
        target_mtl = output_obj.parent / target_mtl_name
        copied = _copy_if_exists(source_mtl, target_mtl)
        if copied is None:
            continue
        if copied_mtl is None:
            copied_mtl = copied
        rewritten_mtl_names.append(target_mtl_name)
        obj_mtl_name_map[source_mtl.name] = target_mtl_name
        for mtllib in obj_mtllibs:
            if (source_obj.parent / mtllib).resolve() == source_mtl.resolve():
                obj_mtl_name_map[mtllib] = target_mtl_name

        texture_name_map: dict[str, str] = {}
        texture_refs = list(_read_mtl_texture_refs(source_mtl))
        if source_texture_path is not None and source_texture_path.exists():
            explicit_name = source_texture_path.name
            if explicit_name not in texture_refs:
                texture_refs.append(explicit_name)
        for texture_ref in texture_refs:
            source_texture = (
                source_texture_path
                if source_texture_path is not None
                and source_texture_path.exists()
                and texture_ref == source_texture_path.name
                else source_mtl.parent / texture_ref
            )
            if not source_texture.exists():
                continue
            target_texture_name = f"{output_obj.stem}__{source_texture.name}"
            copied_texture_path = _copy_if_exists(source_texture, target_mtl.parent / target_texture_name)
            if copied_texture_path is None:
                continue
            if copied_texture is None:
                copied_texture = copied_texture_path
            texture_name_map[texture_ref] = target_texture_name
            texture_name_map[source_texture.name] = target_texture_name
        if texture_name_map:
            _rewrite_mtl_texture_refs(target_mtl, texture_name_map)

    _rewrite_obj_mtllibs(output_obj, obj_mtl_name_map, default_names=tuple(rewritten_mtl_names))

    sibling_png = source_obj.with_suffix(".png")
    if sibling_png.exists():
        target_png = output_obj.parent / f"{output_obj.stem}__{sibling_png.name}"
        copied_texture = _copy_if_exists(sibling_png, target_png) or copied_texture
    return copied_mtl, copied_texture


def _materialize_bundle_scene_dir(
    bundle: dict[str, Any],
    *,
    scene_id: str,
    runtime_path: Path | None = None,
) -> Path:
    scene_dir = DEFAULT_METRIC_SCENE_ROOT / scene_id / str(bundle.get("geometry_variant") or "bundle")
    scene_dir.mkdir(parents=True, exist_ok=True)
    for stale_path in scene_dir.iterdir():
        if stale_path.is_file():
            stale_path.unlink()
        elif stale_path.is_dir():
            shutil.rmtree(stale_path)

    written = 0
    for index, item in enumerate(_as_list(bundle.get("objects")), start=1):
        if not isinstance(item, dict):
            continue
        mesh_path_value = item.get("mesh_obj_path")
        if not isinstance(mesh_path_value, str) or not mesh_path_value.strip():
            continue
        source_obj = _resolve_artifact_path(mesh_path_value, runtime_path=runtime_path)
        if not source_obj.exists():
            continue
        object_name = _safe_scene_name(str(item.get("object_id") or f"object_{index:02d}"), f"object_{index:02d}")
        output_obj = scene_dir / f"{object_name}.obj"
        matrix = _bundle_transform_matrix(item)
        if matrix is None:
            shutil.copy2(source_obj, output_obj)
        else:
            _transform_obj_with_matrix(source_obj, output_obj, matrix)
        source_mtl_path = None
        mesh_mtl_path = item.get("mesh_mtl_path")
        if isinstance(mesh_mtl_path, str) and mesh_mtl_path.strip():
            source_mtl_path = _resolve_artifact_path(mesh_mtl_path, runtime_path=runtime_path)
        source_texture_path = None
        texture_image_path = item.get("texture_image_path")
        if isinstance(texture_image_path, str) and texture_image_path.strip():
            source_texture_path = _resolve_artifact_path(texture_image_path, runtime_path=runtime_path)
        _copy_material_sidecars(
            source_obj,
            output_obj,
            source_mtl_path=source_mtl_path,
            source_texture_path=source_texture_path,
        )
        written += 1

    if written == 0:
        raise FileNotFoundError(f"could not materialize any renderable meshes for scene '{scene_id}'")
    return scene_dir


def resolve_metric_scene_dir(runtime: dict[str, Any], runtime_path: Path, scene_id: str) -> tuple[Path, str, str]:
    original_bundle = _build_original_textured_bundle(runtime, scene_id=scene_id)
    if isinstance(original_bundle, dict) and _as_list(original_bundle.get("objects")):
        return (
            _materialize_bundle_scene_dir(original_bundle, scene_id=scene_id, runtime_path=runtime_path),
            "final_textured_scene",
            "original_textured",
        )

    scene_dir = resolve_layout_scene_dir(runtime, scene_id)
    return scene_dir, "layout_scene_dir", "layout_mesh"


def metric_error_payload(error: BaseException) -> dict[str, Any]:
    return {
        "status": "failed",
        "score": None,
        "error": {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(limit=8),
        },
    }


def _safe_compute(
    name: str,
    fn: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output):
            result = fn()
        result = dict(result)
        result.setdefault("status", "completed")
        captured_stdout = output.getvalue().strip()
        if captured_stdout:
            result["stdout"] = captured_stdout
        return result
    except Exception as error:
        payload = metric_error_payload(error)
        captured_stdout = output.getvalue().strip()
        if captured_stdout:
            payload["stdout"] = captured_stdout
        payload["metric"] = name
        return payload


def representative_metric_image_paths(render_payload: dict[str, Any]) -> list[str]:
    representative_image = render_payload.get("representative_image")
    if isinstance(representative_image, str) and representative_image.strip():
        return [representative_image]

    image_paths = [
        str(path)
        for path in _as_list(render_payload.get("image_paths"))
        if str(path).strip()
    ]
    if not image_paths:
        return []

    representative_index = render_payload.get("representative_index", 0)
    try:
        representative_index = int(representative_index)
    except (TypeError, ValueError):
        representative_index = 0
    representative_index = max(0, min(representative_index, len(image_paths) - 1))
    return [image_paths[representative_index]]


def completed_metric(metric: Any) -> bool:
    return isinstance(metric, dict) and metric.get("status") == "completed"


def physical_metric_uses_representative(payload: dict[str, Any]) -> bool:
    render_payload = payload.get("render")
    metric_payload = _nested_get(payload, "metrics.physical_plausibility_score")
    if not isinstance(render_payload, dict) or not completed_metric(metric_payload):
        return False

    expected_paths = representative_metric_image_paths(render_payload)
    actual_paths = [
        str(path)
        for path in _as_list(metric_payload.get("image_paths"))
        if str(path).strip()
    ]
    return bool(expected_paths) and actual_paths == expected_paths


def vqa_metric_uses_t2v_metrics(payload: dict[str, Any]) -> bool:
    metric_payload = _nested_get(payload, "metrics.vqa_score")
    return completed_metric(metric_payload) and metric_payload.get("backend") == "t2v_metrics_vqascore"


def render_payload_uses_final_scene(payload: dict[str, Any]) -> bool:
    render_payload = payload.get("render")
    if not isinstance(render_payload, dict):
        return False
    return (
        int(render_payload.get("render_schema_version") or 0) >= METRIC_RENDER_SCHEMA_VERSION
        and
        str(render_payload.get("scene_source_type") or "") == "final_textured_scene"
        and str(render_payload.get("geometry_variant") or "") == "original_textured"
    )


def compute_dashboard_metrics(
    runtime_path: str | Path,
    *,
    output_root: str | Path = DEFAULT_RESULTS_ROOT,
    force_render: bool = False,
    force_metrics: bool = False,
    skip_clip: bool = False,
    skip_vqa: bool = False,
    skip_physical: bool = False,
) -> dict[str, Any]:
    runtime_path = Path(runtime_path).expanduser()
    if not runtime_path.is_absolute():
        runtime_path = (Path.cwd() / runtime_path).resolve()
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    scene_id = extract_scene_id(runtime, runtime_path)
    prompt = extract_prompt(runtime)
    output_root = Path(output_root).expanduser()
    if not output_root.is_absolute():
        output_root = (Path.cwd() / output_root).resolve()
    output_path = output_root / metric_case_file_name(scene_id)

    cached_payload: dict[str, Any] | None = None
    if output_path.exists() and not force_render and not force_metrics:
        cached_payload = json.loads(output_path.read_text(encoding="utf-8"))
        if (
            render_payload_uses_final_scene(cached_payload)
            and physical_metric_uses_representative(cached_payload)
            and vqa_metric_uses_t2v_metrics(cached_payload)
        ):
            cached_payload.setdefault("metrics_result_path", str(output_path))
            return cached_payload

    payload: dict[str, Any] = {
        "case_id": scene_id,
        "runtime_path": str(runtime_path),
        "prompt": prompt,
        "generated_at": _now_iso(),
        "render": None,
        "metrics": {},
    }

    try:
        scene_dir, scene_source_type, geometry_variant = resolve_metric_scene_dir(runtime, runtime_path, scene_id)
        render_payload = render_scene_views(
            scene_dir,
            scene_id,
            force=force_render,
        )
        render_payload["scene_source_type"] = scene_source_type
        render_payload["geometry_variant"] = geometry_variant
        render_payload["render_schema_version"] = METRIC_RENDER_SCHEMA_VERSION
        payload["render"] = render_payload
        image_paths = render_payload["image_paths"]
        physical_image_paths = representative_metric_image_paths(render_payload)
    except Exception as error:
        payload["render"] = metric_error_payload(error)
        image_paths = []
        physical_image_paths = []

    if image_paths and prompt:
        cached_metrics = (
            cached_payload.get("metrics", {})
            if cached_payload is not None and not force_render and not force_metrics
            else {}
        )
        if not skip_clip:
            cached_clip_score = cached_metrics.get("clip_score")
            payload["metrics"]["clip_score"] = (
                cached_clip_score
                if completed_metric(cached_clip_score)
                else _safe_compute(
                    "clip_score",
                    lambda: compute_clip_score(image_paths, prompt),
                )
            )
        if not skip_vqa:
            cached_vqa_score = cached_metrics.get("vqa_score")
            payload["metrics"]["vqa_score"] = (
                cached_vqa_score
                if completed_metric(cached_vqa_score) and cached_vqa_score.get("backend") == "t2v_metrics_vqascore"
                else _safe_compute(
                    "vqa_score",
                    lambda: compute_vqa_score(image_paths, prompt),
                )
            )
        if not skip_physical:
            payload["metrics"]["physical_plausibility_score"] = _safe_compute(
                "physical_plausibility_score",
                lambda: compute_physical_plausibility_score(physical_image_paths, prompt),
            )
    else:
        reason = "rendered images unavailable" if not image_paths else "prompt unavailable"
        for key in ("clip_score", "vqa_score", "physical_plausibility_score"):
            payload["metrics"][key] = {
                "status": "skipped",
                "score": None,
                "error": {"type": "SkippedMetric", "message": reason},
            }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    payload["metrics_result_path"] = str(output_path)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render and compute dashboard metrics for a PAT3D runtime.")
    parser.add_argument("--runtime", required=True, help="Runtime JSON path.")
    parser.add_argument("--output-root", default=str(DEFAULT_RESULTS_ROOT), help="Metrics JSON output root.")
    parser.add_argument("--force-render", action="store_true", help="Re-render metric images.")
    parser.add_argument("--force-metrics", action="store_true", help="Recompute metrics even when cached JSON exists.")
    parser.add_argument("--skip-clip", action="store_true", help="Skip CLIP score.")
    parser.add_argument("--skip-vqa", action="store_true", help="Skip VQAScore.")
    parser.add_argument("--skip-physical", action="store_true", help="Skip physical plausibility score.")
    args = parser.parse_args(argv)

    payload = compute_dashboard_metrics(
        args.runtime,
        output_root=args.output_root,
        force_render=args.force_render,
        force_metrics=args.force_metrics,
        skip_clip=args.skip_clip,
        skip_vqa=args.skip_vqa,
        skip_physical=args.skip_physical,
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
