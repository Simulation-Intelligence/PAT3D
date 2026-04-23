from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any
import zipfile

import trimesh

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pat3d.models import ObjectPose
from pat3d.models.pose_utils import pose_to_matrix


def _repo_root() -> Path:
    return REPO_ROOT


def _safe_name(value: str, fallback: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return normalized.strip("._-") or fallback


def _resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (_repo_root() / path).resolve()


def _load_mesh(mesh_path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(mesh_path, force="mesh")
    if isinstance(loaded, trimesh.Scene):
        return loaded.dump(concatenate=True)
    return loaded.copy()


def _transform_matrix(item: dict[str, Any]) -> list[list[float]] | None:
    if item.get("already_transformed", True):
        return None
    transform = item.get("transform")
    if not isinstance(transform, dict):
        return None
    translation = tuple(float(value) for value in (transform.get("translation_xyz") or (0.0, 0.0, 0.0))[:3])
    rotation = tuple(float(value) for value in (transform.get("rotation_value") or (1.0, 0.0, 0.0, 0.0)))
    scale_raw = transform.get("scale_xyz")
    scale = (
        tuple(float(value) for value in scale_raw[:3])
        if isinstance(scale_raw, (list, tuple))
        else None
    )
    pose = ObjectPose(
        object_id=str(item.get("object_id") or "object"),
        translation_xyz=translation,
        rotation_type=str(transform.get("rotation_type") or "quaternion"),
        rotation_value=rotation,
        scale_xyz=scale,
    )
    return pose_to_matrix(pose).tolist()


def _transformed_mesh(item: dict[str, Any]) -> trimesh.Trimesh:
    mesh_path = _resolve_path(str(item["mesh_obj_path"]))
    mesh = _load_mesh(mesh_path)
    transform_matrix = _transform_matrix(item)
    if transform_matrix is not None:
        mesh.apply_transform(transform_matrix)
    return mesh


def export_bundle(bundle: dict[str, Any], *, mode: str, output_path: Path) -> dict[str, str]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scene_id = str(bundle.get("scene_id") or "scene")
    export_root = output_path.parent / _safe_name(scene_id, "scene")
    export_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "scene_id": scene_id,
        "mode": mode,
        "object_count": 0,
        "requested_ground_plane_y": bundle.get("requested_ground_plane_y"),
        "applied_ground_plane_y": bundle.get("applied_ground_plane_y"),
        "ground_plane_source": bundle.get("ground_plane_source"),
        "objects": [],
    }

    objects = [
        item
        for item in bundle.get("objects", [])
        if isinstance(item, dict) and isinstance(item.get("mesh_obj_path"), str)
    ]
    if not objects:
        raise RuntimeError("scene bundle did not contain any exportable objects")

    if mode == "merged":
        merged_meshes = []
        for item in objects:
            merged_meshes.append(_transformed_mesh(item))
            manifest["objects"].append(
                {
                    "object_id": item.get("object_id"),
                    "mesh_obj_path": item.get("mesh_obj_path"),
                }
            )
        merged_mesh = trimesh.util.concatenate(merged_meshes)
        merged_obj_name = f"{_safe_name(scene_id, 'scene')}.obj"
        merged_obj_path = export_root / merged_obj_name
        merged_mesh.export(merged_obj_path)
        manifest["object_count"] = len(objects)
        manifest["merged_obj"] = merged_obj_name
        download_name = f"{_safe_name(scene_id, 'scene')}_merged_scene.zip"
    elif mode == "separate":
        objects_dir = export_root / "objects"
        objects_dir.mkdir(parents=True, exist_ok=True)
        for index, item in enumerate(objects, start=1):
            object_id = str(item.get("object_id") or f"object_{index:02d}")
            object_obj_name = f"{_safe_name(object_id, f'object_{index:02d}')}.obj"
            object_obj_path = objects_dir / object_obj_name
            mesh = _transformed_mesh(item)
            mesh.export(object_obj_path)
            manifest["objects"].append(
                {
                    "object_id": object_id,
                    "mesh_obj_path": item.get("mesh_obj_path"),
                    "export_obj": str(Path("objects") / object_obj_name),
                }
            )
        manifest["object_count"] = len(objects)
        download_name = f"{_safe_name(scene_id, 'scene')}_separate_objects.zip"
    else:
        raise RuntimeError(f"unsupported export mode: {mode}")

    bundle_path = export_root / "scene_bundle.json"
    bundle_path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_path = export_root / "export_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in sorted(export_root.rglob("*")):
            if not file_path.is_file():
                continue
            archive.write(file_path, arcname=str(file_path.relative_to(export_root)))

    return {
        "zip_path": str(output_path),
        "download_name": download_name,
    }


def _cli() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--mode", choices=("merged", "separate"), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    bundle = json.loads(Path(args.bundle).read_text(encoding="utf-8"))
    result = export_bundle(bundle, mode=args.mode, output_path=Path(args.output))
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
