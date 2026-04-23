from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from pat3d.models import ObjectPose
from pat3d.models.pose_utils import matrix_to_pose, pose_to_matrix


def load_legacy_diff_sim_trajectory(
    runtime_payload: Mapping[str, Any],
    *,
    phys_result_root: str | Path = "_phys_result",
) -> dict[str, object]:
    scene_id = _scene_id_from_runtime(runtime_payload)
    if scene_id is None:
        return {"available": False, "reason": "scene_id_unavailable"}

    initial_poses = _initial_object_poses(runtime_payload)
    if not initial_poses:
        return {"available": False, "scene_id": scene_id, "reason": "initial_object_poses_unavailable"}

    transform_dir = _preferred_transform_dir(
        runtime_payload,
        scene_id=scene_id,
        phys_result_root=phys_result_root,
    )
    snapshot_dir = transform_dir
    snapshot_paths = sorted(transform_dir.glob("frame_*.npz"), key=_snapshot_sort_key)
    snapshot_source = "forward_simulation_frames"
    if not snapshot_paths:
        param_dir = Path(phys_result_root) / scene_id / "param"
        snapshot_dir = param_dir
        snapshot_paths = sorted(param_dir.glob("optim_*.npz"), key=_snapshot_sort_key)
        snapshot_source = "optimization_parameters"
    if not snapshot_paths:
        return {
            "available": False,
            "scene_id": scene_id,
            "reason": "trajectory_snapshots_unavailable",
            "param_dir": str(snapshot_dir),
        }

    tracks: list[dict[str, object]] = []
    for object_id, initial_pose in initial_poses.items():
        points = []
        for frame_index, snapshot_path in enumerate(snapshot_paths):
            transforms = np.load(snapshot_path, allow_pickle=True)
            delta_matrix = _delta_matrix_for_object(transforms, object_id)
            final_pose = matrix_to_pose(object_id, delta_matrix @ pose_to_matrix(initial_pose))
            points.append(
                {
                    "frame_index": frame_index,
                    "frame_label": snapshot_path.stem,
                    "translation_xyz": [
                        float(final_pose.translation_xyz[0]),
                        float(final_pose.translation_xyz[1]),
                        float(final_pose.translation_xyz[2]),
                    ],
                }
            )
        tracks.append(
            {
                "object_id": object_id,
                "points": points,
            }
        )

    return {
        "available": True,
        "scene_id": scene_id,
        "param_dir": str(snapshot_dir),
        "snapshot_source": snapshot_source,
        "snapshot_paths": [str(path) for path in snapshot_paths],
        "snapshot_count": len(snapshot_paths),
        "tracks": tracks,
    }


def load_runtime_payload(runtime_path: str | Path) -> dict[str, object]:
    return json.loads(Path(runtime_path).read_text(encoding="utf-8"))


def _scene_id_from_runtime(runtime_payload: Mapping[str, Any]) -> str | None:
    for candidate in (
        runtime_payload.get("simulation_preparation", {}).get("physics_ready_scene", {}).get("scene_id"),
        runtime_payload.get("layout_initialization", {}).get("scene_layout", {}).get("scene_id"),
        runtime_payload.get("first_contract_slice", {}).get("reference_image_result", {}).get("request", {}).get("scene_id"),
    ):
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return None


def _preferred_transform_dir(
    runtime_payload: Mapping[str, Any],
    *,
    scene_id: str,
    phys_result_root: str | Path,
) -> Path:
    report_path = _diff_init_report_path(runtime_payload, scene_id=scene_id, phys_result_root=phys_result_root)
    if report_path is not None:
        try:
            report_payload = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception:
            report_payload = None
        if isinstance(report_payload, Mapping):
            raw_passes = report_payload.get("passes")
            if isinstance(raw_passes, list):
                for pass_payload in reversed(raw_passes):
                    if not isinstance(pass_payload, Mapping):
                        continue
                    raw_dir = pass_payload.get("trajectory_dir")
                    if not isinstance(raw_dir, str) or not raw_dir.strip():
                        continue
                    candidate = Path(raw_dir)
                    if candidate.exists() and any(candidate.glob("frame_*.npz")):
                        return candidate
    return Path(phys_result_root) / scene_id / "transform"


def _diff_init_report_path(
    runtime_payload: Mapping[str, Any],
    *,
    scene_id: str,
    phys_result_root: str | Path,
) -> Path | None:
    artifact_groups = (
        runtime_payload.get("physics_optimization", {}).get("optimization_result", {}).get("artifacts", ()),
        runtime_payload.get("physics_optimization", {}).get("artifacts", ()),
    )
    for artifact_group in artifact_groups:
        if not isinstance(artifact_group, (list, tuple)):
            continue
        for artifact in artifact_group:
            if not isinstance(artifact, Mapping):
                continue
            if artifact.get("artifact_type") != "physics_debug_report":
                continue
            raw_path = artifact.get("path")
            if not isinstance(raw_path, str) or not raw_path.strip():
                continue
            candidate = Path(raw_path)
            if candidate.exists():
                return candidate
    fallback = Path(phys_result_root) / scene_id / "diff_init_report.json"
    if fallback.exists():
        return fallback
    return None


def _initial_object_poses(runtime_payload: Mapping[str, Any]) -> dict[str, ObjectPose]:
    raw_poses = (
        runtime_payload.get("simulation_preparation", {}).get("physics_ready_scene", {}).get("object_poses")
        or runtime_payload.get("layout_initialization", {}).get("scene_layout", {}).get("object_poses")
        or ()
    )
    poses: dict[str, ObjectPose] = {}
    for item in raw_poses:
        if not isinstance(item, Mapping):
            continue
        pose = _object_pose_from_payload(item)
        poses[pose.object_id] = pose
    return poses


def _object_pose_from_payload(data: Mapping[str, Any]) -> ObjectPose:
    scale_xyz = data.get("scale_xyz")
    return ObjectPose(
        object_id=str(data["object_id"]),
        translation_xyz=tuple(float(value) for value in data.get("translation_xyz", (0.0, 0.0, 0.0))),
        rotation_type=str(data.get("rotation_type", "quaternion")),
        rotation_value=tuple(float(value) for value in data.get("rotation_value", (1.0, 0.0, 0.0, 0.0))),
        scale_xyz=(
            tuple(float(value) for value in scale_xyz)
            if isinstance(scale_xyz, (list, tuple))
            else None
        ),
    )


def _delta_matrix_for_object(transforms: Mapping[str, Any], object_id: str) -> np.ndarray:
    if object_id in transforms:
        return np.asarray(transforms[object_id], dtype=np.float64)
    return np.eye(4, dtype=np.float64)


def _snapshot_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"(\d+)$", path.stem)
    if match is None:
        return (0, path.name)
    return (int(match.group(1)), path.name)
