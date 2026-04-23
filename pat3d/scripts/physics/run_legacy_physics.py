from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from pat3d.models import (
    ArtifactRef,
    ContainmentRelation,
    ObjectPose,
    PhysicsReadyScene,
    RelationType,
    SceneLayout,
    SceneRelationGraph,
)
from pat3d.providers._legacy_physics_adapter import LegacySequentialPhysicsAdapter


def _artifact_ref_from_dict(data: Mapping[str, Any]) -> ArtifactRef:
    return ArtifactRef(
        artifact_type=str(data["artifact_type"]),
        path=str(data["path"]),
        format=str(data["format"]),
        role=str(data["role"]) if data.get("role") is not None else None,
        metadata_path=(
            str(data["metadata_path"])
            if data.get("metadata_path") is not None
            else None
        ),
    )


def _object_pose_from_dict(data: Mapping[str, Any]) -> ObjectPose:
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


def _scene_relation_graph_from_dict(data: Mapping[str, Any]) -> SceneRelationGraph:
    return SceneRelationGraph(
        scene_id=str(data["scene_id"]),
        relations=tuple(
            ContainmentRelation(
                parent_object_id=str(item["parent_object_id"]),
                child_object_id=str(item["child_object_id"]),
                relation_type=RelationType(str(item["relation_type"])),
                confidence=(
                    float(item["confidence"])
                    if item.get("confidence") is not None
                    else None
                ),
                evidence=str(item["evidence"]) if item.get("evidence") is not None else None,
            )
            for item in data.get("relations", ())
            if isinstance(item, Mapping)
        ),
        root_object_ids=tuple(str(item) for item in data.get("root_object_ids", ())),
    )


def _scene_layout_from_dict(data: Mapping[str, Any]) -> SceneLayout:
    return SceneLayout(
        scene_id=str(data["scene_id"]),
        object_poses=tuple(
            _object_pose_from_dict(item)
            for item in data.get("object_poses", ())
            if isinstance(item, Mapping)
        ),
        layout_space=str(data["layout_space"]),
        support_graph=(
            _scene_relation_graph_from_dict(data["support_graph"])
            if isinstance(data.get("support_graph"), Mapping)
            else None
        ),
        artifacts=tuple(
            _artifact_ref_from_dict(item)
            for item in data.get("artifacts", ())
            if isinstance(item, Mapping)
        ),
    )


def _physics_ready_scene_from_dict(data: Mapping[str, Any]) -> PhysicsReadyScene:
    return PhysicsReadyScene(
        scene_id=str(data["scene_id"]),
        layout=_scene_layout_from_dict(data["layout"]),
        simulation_meshes=tuple(
            _artifact_ref_from_dict(item)
            for item in data.get("simulation_meshes", ())
            if isinstance(item, Mapping)
        ),
        object_poses=tuple(
            _object_pose_from_dict(item)
            for item in data.get("object_poses", ())
            if isinstance(item, Mapping)
        ),
        collision_settings=(
            dict(data["collision_settings"])
            if data.get("collision_settings") is not None
            else None
        ),
    )


def _normalize_result(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "optimized_object_poses": [
            pose.to_dict() if hasattr(pose, "to_dict") else pose
            for pose in result.get("optimized_object_poses", ())
        ],
        "artifacts": [
            artifact.to_dict() if hasattr(artifact, "to_dict") else artifact
            for artifact in result.get("artifacts", ())
        ],
        "metrics": {
            str(key): float(value)
            for key, value in dict(result.get("metrics", {})).items()
        },
    }


def run(payload: Mapping[str, Any]) -> dict[str, Any]:
    mode = str(payload.get("mode") or "forward_only")
    physics_ready_scene = _physics_ready_scene_from_dict(payload["physics_ready_scene"])
    legacy_arg_overrides = payload.get("legacy_arg_overrides")
    adapter = LegacySequentialPhysicsAdapter(
        legacy_arg_overrides=(
            dict(legacy_arg_overrides)
            if isinstance(legacy_arg_overrides, Mapping)
            else None
        )
    )
    if mode == "forward_only":
        result = adapter.simulate_to_static(physics_ready_scene)
    elif mode == "optimize_then_forward":
        result = adapter.optimize(physics_ready_scene)
    else:
        raise ValueError(
            f"Unsupported legacy physics mode '{mode}'. "
            "Expected 'forward_only' or 'optimize_then_forward'."
        )
    return _normalize_result(result)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run PAT3D legacy physics in a subprocess.")
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    try:
        payload = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
        result = run(payload)
        Path(args.output_json).write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return 0
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
