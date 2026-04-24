from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import trimesh

from pat3d.models import ObjectPose
from pat3d.models.pose_utils import pose_to_matrix

REPO_ROOT = Path(__file__).resolve().parents[2]
NORMALIZED_SCENE_DIAGONAL = 2.0


def load_runtime_payload(runtime_path: str | Path) -> dict[str, object]:
    return json.loads(Path(runtime_path).read_text(encoding="utf-8"))


def load_legacy_diff_sim_metrics(
    *,
    runtime_payload: Mapping[str, Any] | None = None,
    runtime_path: str | Path | None = None,
    scene_id: str | None = None,
    phys_result_root: str | Path = "_phys_result",
) -> dict[str, object]:
    final_metrics = _build_final_metrics(
        runtime_payload=runtime_payload,
        runtime_path=runtime_path,
    )
    resolved_scene_id = scene_id or _scene_id_from_runtime(runtime_payload or {})
    if not resolved_scene_id:
        return {
            "available": False,
            "reason": "scene_id_unavailable",
            "final_metrics": final_metrics,
        }

    summary_path = _resolve_loss_history_summary_path(
        runtime_payload,
        runtime_path=runtime_path,
        scene_id=resolved_scene_id,
        phys_result_root=phys_result_root,
    )
    if summary_path is not None and summary_path.exists():
        payload = _load_json_dict(summary_path)
        if payload is not None:
            return _normalize_metrics_payload(
                payload,
                scene_id=resolved_scene_id,
                final_metrics=final_metrics,
            )

    pass_paths = _resolve_loss_history_pass_paths(
        runtime_payload,
        runtime_path=runtime_path,
        scene_id=resolved_scene_id,
        phys_result_root=phys_result_root,
    )
    if not pass_paths:
        return {
            "available": False,
            "scene_id": resolved_scene_id,
            "mode": "optimize_then_forward",
            "reason": "loss_history_unavailable",
            "final_metrics": final_metrics,
        }

    series: list[dict[str, object]] = []
    best_losses: list[float] = []
    for path in pass_paths:
        payload = _load_json_dict(path)
        if payload is None:
            continue
        normalized = _normalize_series_payload(payload, fallback_pass_index=len(series))
        if normalized is None:
            continue
        series.append(normalized)
        best_loss = payload.get("best_loss")
        if isinstance(best_loss, (int, float)):
            best_losses.append(float(best_loss))

    if not series:
        return {
            "available": False,
            "scene_id": resolved_scene_id,
            "mode": "optimize_then_forward",
            "reason": "loss_history_unavailable",
            "final_metrics": final_metrics,
        }

    return {
        "available": True,
        "scene_id": resolved_scene_id,
        "mode": "optimize_then_forward",
        "best_loss": min(best_losses) if best_losses else None,
        "series": sorted(series, key=lambda item: int(item.get("pass_index", 0))),
        "final_metrics": final_metrics,
    }


def _build_final_metrics(
    *,
    runtime_payload: Mapping[str, Any] | None,
    runtime_path: str | Path | None,
) -> dict[str, object]:
    penetration_metric = {
        "available": False,
        "value": None,
        "source": "paper_ratio_of_penetrating_triangle_pairs",
        "note": "Final scene bundle is unavailable.",
    }
    displacement_metric = {
        "available": False,
        "value": None,
        "source": "pending_gipc",
        "note": "Placeholder until GIPC displacement is wired.",
    }
    if not isinstance(runtime_payload, Mapping):
        penetration_metric["note"] = "Runtime payload is unavailable."
        return {
            "penetration_metric": penetration_metric,
            "displacement_metric": displacement_metric,
        }

    scene_bundle_path = _resolve_scene_bundle_path(runtime_payload, runtime_path=runtime_path)
    if scene_bundle_path is None:
        penetration_metric["note"] = "Final scene bundle is unavailable."
        return {
            "penetration_metric": penetration_metric,
            "displacement_metric": displacement_metric,
        }
    if not scene_bundle_path.exists():
        penetration_metric["note"] = f"Scene bundle does not exist: {scene_bundle_path}"
        return {
            "penetration_metric": penetration_metric,
            "displacement_metric": displacement_metric,
        }

    try:
        value = _compute_penetration_metric_from_scene_bundle(scene_bundle_path)
    except Exception as exc:
        penetration_metric["note"] = f"{type(exc).__name__}: {exc}"
    else:
        penetration_metric = {
            "available": True,
            "value": value,
            "source": "simplified_scene_ratio_of_penetrating_triangle_pairs",
            "note": (
                "Computed directly from the final simplified scene bundle after normalizing the "
                "scene diagonal to 2. No additional remeshing is applied during metric evaluation."
            ),
        }

    return {
        "penetration_metric": penetration_metric,
        "displacement_metric": displacement_metric,
    }


def _compute_penetration_metric_from_scene_bundle(scene_bundle_path: Path) -> float:
    bundle_payload = _load_json_dict(scene_bundle_path)
    if bundle_payload is None:
        raise ValueError("scene bundle JSON is invalid")
    object_payloads = bundle_payload.get("objects")
    if not isinstance(object_payloads, list) or not object_payloads:
        raise ValueError("scene bundle does not contain any objects")

    meshes: list[trimesh.Trimesh] = []
    for item in object_payloads:
        if not isinstance(item, Mapping):
            continue
        mesh_path_value = item.get("mesh_obj_path")
        if not isinstance(mesh_path_value, str) or not mesh_path_value.strip():
            continue
        mesh_path = _resolve_artifact_path(mesh_path_value)
        if not mesh_path.exists():
            continue
        mesh = _load_trimesh(mesh_path)
        _apply_scene_bundle_transform(mesh, item)
        if mesh.vertices.size == 0 or mesh.faces.size == 0:
            continue
        meshes.append(mesh)

    if not meshes:
        raise ValueError("scene bundle does not contain any loadable meshes")

    scene_diagonal = _scene_diagonal(meshes)
    if not np.isfinite(scene_diagonal) or scene_diagonal <= 0.0:
        raise ValueError("scene diagonal must be positive")

    normalized_meshes = []
    scale = NORMALIZED_SCENE_DIAGONAL / scene_diagonal
    for mesh in meshes:
        normalized = mesh.copy()
        normalized.apply_scale(scale)
        normalized_meshes.append(normalized)

    object_pair_counts: list[int] = []
    object_edge_lengths: list[np.ndarray] = []
    for mesh in normalized_meshes:
        object_pair_counts.append(_count_triangle_pair_intersections(mesh))
        edge_lengths = np.asarray(mesh.edges_unique_length, dtype=np.float64)
        if edge_lengths.size:
            object_edge_lengths.append(edge_lengths)

    merged_mesh = trimesh.util.concatenate(normalized_meshes)
    total_pair_count = _count_triangle_pair_intersections(merged_mesh)
    penetrating_pairs = max(total_pair_count - sum(object_pair_counts), 0)
    if object_edge_lengths:
        average_edge_length = float(np.concatenate(object_edge_lengths).mean())
    else:
        average_edge_length = 0.0

    return float((penetrating_pairs * average_edge_length) / NORMALIZED_SCENE_DIAGONAL)


def _load_trimesh(mesh_path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(mesh_path, force="scene", process=False)
    if isinstance(loaded, trimesh.Scene):
        geometries = [
            geometry.copy()
            for geometry in loaded.geometry.values()
            if isinstance(geometry, trimesh.Trimesh) and geometry.vertices.size and geometry.faces.size
        ]
        if not geometries:
            raise ValueError(f"mesh contains no triangle geometry: {mesh_path}")
        mesh = trimesh.util.concatenate(geometries)
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded.copy()
    else:
        raise ValueError(f"unsupported mesh payload: {type(loaded).__name__}")

    mesh.update_faces(mesh.unique_faces())
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    return mesh


def _apply_scene_bundle_transform(mesh: trimesh.Trimesh, item: Mapping[str, object]) -> None:
    if item.get("already_transformed", True):
        return
    transform = item.get("transform")
    if not isinstance(transform, Mapping):
        return
    translation = tuple(transform.get("translation_xyz") or (0.0, 0.0, 0.0))
    rotation_value = tuple(transform.get("rotation_value") or (1.0, 0.0, 0.0, 0.0))
    scale_value = transform.get("scale_xyz")
    scale_xyz = tuple(scale_value[:3]) if isinstance(scale_value, (list, tuple)) else None
    pose = ObjectPose(
        object_id=str(item.get("object_id") or "object"),
        translation_xyz=tuple(float(value) for value in translation[:3]),
        rotation_type=str(transform.get("rotation_type") or "quaternion"),
        rotation_value=tuple(float(value) for value in rotation_value),
        scale_xyz=(
            tuple(float(value) for value in scale_xyz)
            if scale_xyz is not None
            else None
        ),
    )
    mesh.apply_transform(pose_to_matrix(pose))


def _scene_diagonal(meshes: list[trimesh.Trimesh]) -> float:
    stacked_vertices = np.concatenate([np.asarray(mesh.vertices, dtype=np.float64) for mesh in meshes], axis=0)
    bounds_min = stacked_vertices.min(axis=0)
    bounds_max = stacked_vertices.max(axis=0)
    return float(np.linalg.norm(bounds_max - bounds_min))


def _count_triangle_pair_intersections(mesh: trimesh.Trimesh) -> int:
    import open3d as o3d

    open3d_mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=np.float64)),
        o3d.utility.Vector3iVector(np.asarray(mesh.faces, dtype=np.int32)),
    )
    open3d_mesh.remove_duplicated_vertices()
    open3d_mesh.remove_duplicated_triangles()
    open3d_mesh.remove_degenerate_triangles()
    open3d_mesh.remove_unreferenced_vertices()
    pairs = np.asarray(open3d_mesh.get_self_intersecting_triangles())
    return int(len(pairs))


def _resolve_scene_bundle_path(
    runtime_payload: Mapping[str, Any],
    *,
    runtime_path: str | Path | None,
) -> Path | None:
    render_result = runtime_payload.get("visualization", {}).get("render_result", {})
    if isinstance(render_result, Mapping):
        camera_metadata = render_result.get("camera_metadata")
        if isinstance(camera_metadata, Mapping):
            path_value = camera_metadata.get("path")
            role = camera_metadata.get("role")
            artifact_type = camera_metadata.get("artifact_type")
            if (
                isinstance(path_value, str)
                and path_value.strip()
                and artifact_type == "scene_bundle"
                and role == "scene_bundle"
            ):
                return _resolve_artifact_path(path_value, runtime_path=runtime_path)

    scene_id = _scene_id_from_runtime(runtime_payload)
    if scene_id:
        fallback_path = REPO_ROOT / "results" / "rendered_images" / scene_id / "scene_bundle.json"
        if fallback_path.exists():
            return fallback_path
    return None


def _resolve_loss_history_summary_path(
    runtime_payload: Mapping[str, Any] | None,
    *,
    runtime_path: str | Path | None,
    scene_id: str,
    phys_result_root: str | Path,
) -> Path | None:
    if isinstance(runtime_payload, Mapping):
        for path in _find_runtime_artifact_paths(
            runtime_payload,
            runtime_path=runtime_path,
            role="loss_history",
            suffix="loss_history.json",
        ):
            if path.exists():
                return path
        for path in _loss_history_candidate_dirs(runtime_payload, runtime_path=runtime_path):
            candidate = path / "loss_history.json"
            if candidate.exists():
                return candidate

    for fallback_dir in _scene_loss_history_candidate_dirs(
        scene_id=scene_id,
        phys_result_root=phys_result_root,
    ):
        fallback = fallback_dir / "loss_history.json"
        if fallback.exists():
            return fallback
    return None


def _resolve_loss_history_pass_paths(
    runtime_payload: Mapping[str, Any] | None,
    *,
    runtime_path: str | Path | None,
    scene_id: str,
    phys_result_root: str | Path,
) -> list[Path]:
    candidate_dirs: list[Path] = []
    if isinstance(runtime_payload, Mapping):
        candidate_dirs.extend(_loss_history_candidate_dirs(runtime_payload, runtime_path=runtime_path))
    candidate_dirs.extend(
        _scene_loss_history_candidate_dirs(
            scene_id=scene_id,
            phys_result_root=phys_result_root,
        )
    )

    unique_dirs = _unique_paths(candidate_dirs)
    pass_paths: list[Path] = []
    for directory in unique_dirs:
        pass_paths.extend(sorted(directory.glob("loss_history_pass_*.json"), key=_snapshot_sort_key))
    return _unique_paths(pass_paths)


def _scene_loss_history_candidate_dirs(
    *,
    scene_id: str,
    phys_result_root: str | Path,
) -> list[Path]:
    return [
        Path(phys_result_root) / scene_id,
        REPO_ROOT / "results" / "workspaces" / scene_id / "physics" / scene_id,
    ]


def _loss_history_candidate_dirs(
    runtime_payload: Mapping[str, Any],
    *,
    runtime_path: str | Path | None,
) -> list[Path]:
    candidate_paths = []
    candidate_paths.extend(
        _find_runtime_artifact_paths(
            runtime_payload,
            runtime_path=runtime_path,
            role="loss_history",
        )
    )
    candidate_paths.extend(
        _find_runtime_artifact_paths(
            runtime_payload,
            runtime_path=runtime_path,
            role="diff_sim_report",
        )
    )
    candidate_paths.extend(
        _find_runtime_artifact_paths(
            runtime_payload,
            runtime_path=runtime_path,
            artifact_type="physics_debug_report",
        )
    )
    return _unique_paths(path.parent for path in candidate_paths)


def _find_runtime_artifact_paths(
    runtime_payload: Mapping[str, Any],
    *,
    runtime_path: str | Path | None,
    role: str | None = None,
    artifact_type: str | None = None,
    suffix: str | None = None,
) -> list[Path]:
    matches: list[Path] = []
    for artifact in _iter_runtime_artifacts(runtime_payload):
        role_value = artifact.get("role")
        if role is not None and role_value != role:
            continue
        artifact_type_value = artifact.get("artifact_type")
        if artifact_type is not None and artifact_type_value != artifact_type:
            continue
        path_value = artifact.get("path")
        if not isinstance(path_value, str) or not path_value.strip():
            continue
        if suffix is not None and not path_value.endswith(suffix):
            continue
        matches.append(_resolve_artifact_path(path_value, runtime_path=runtime_path))
    return _unique_paths(matches)


def _iter_runtime_artifacts(node: Any):
    if isinstance(node, Mapping):
        path_value = node.get("path")
        if isinstance(path_value, str) and path_value.strip():
            yield node
        for value in node.values():
            yield from _iter_runtime_artifacts(value)
        return
    if isinstance(node, list):
        for item in node:
            yield from _iter_runtime_artifacts(item)


def _unique_paths(paths) -> list[Path]:
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = Path(path)
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


def _resolve_artifact_path(path_value: str, *, runtime_path: str | Path | None = None) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    repo_path = REPO_ROOT / path
    if repo_path.exists():
        return repo_path
    if runtime_path is not None:
        runtime_candidate = Path(runtime_path).resolve().parent / path
        if runtime_candidate.exists():
            return runtime_candidate
    return repo_path


def _load_json_dict(path: Path) -> dict[str, object] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _normalize_metrics_payload(
    payload: Mapping[str, Any],
    *,
    scene_id: str,
    final_metrics: Mapping[str, object],
) -> dict[str, object]:
    if not bool(payload.get("available")):
        return {
            "available": False,
            "scene_id": scene_id,
            "mode": str(payload.get("mode", "optimize_then_forward")),
            "reason": str(payload.get("reason", "loss_history_unavailable")),
            "final_metrics": dict(final_metrics),
        }
    series = []
    for index, item in enumerate(payload.get("series", ())):
        if not isinstance(item, Mapping):
            continue
        normalized = _normalize_series_payload(item, fallback_pass_index=index)
        if normalized is not None:
            series.append(normalized)
    return {
        "available": True,
        "scene_id": scene_id,
        "mode": str(payload.get("mode", "optimize_then_forward")),
        "best_loss": (
            float(payload["best_loss"])
            if isinstance(payload.get("best_loss"), (int, float))
            else None
        ),
        "series": sorted(series, key=lambda item: int(item.get("pass_index", 0))),
        "final_metrics": dict(final_metrics),
    }


def _normalize_series_payload(
    payload: Mapping[str, Any],
    *,
    fallback_pass_index: int,
) -> dict[str, object] | None:
    points = payload.get("points")
    if not isinstance(points, list):
        return None
    try:
        pass_index = int(payload.get("pass_index", fallback_pass_index))
    except (TypeError, ValueError):
        pass_index = fallback_pass_index
    normalized_points: list[dict[str, object]] = []
    for point in points:
        if not isinstance(point, Mapping):
            continue
        try:
            normalized_points.append(
                {
                    "step": int(point.get("step", 0)),
                    "epoch": int(point.get("epoch", point.get("step", 0))),
                    "loss": float(point["loss"]),
                }
            )
        except (KeyError, TypeError, ValueError):
            continue
    return {
        "pass_index": pass_index,
        "label": str(payload.get("label") or f"Pass {pass_index + 1}"),
        "points": normalized_points,
    }


def _scene_id_from_runtime(runtime_payload: Mapping[str, Any]) -> str | None:
    for candidate in (
        runtime_payload.get("simulation_preparation", {}).get("physics_ready_scene", {}).get("scene_id"),
        runtime_payload.get("layout_initialization", {}).get("scene_layout", {}).get("scene_id"),
        runtime_payload.get("first_contract_slice", {}).get("reference_image_result", {}).get("request", {}).get(
            "scene_id"
        ),
    ):
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return None


def _snapshot_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"(\d+)$", path.stem)
    if match is None:
        return (0, path.name)
    return (int(match.group(1)), path.name)
