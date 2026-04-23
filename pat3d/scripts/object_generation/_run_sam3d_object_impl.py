from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image
import torch
import trimesh

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pat3d.models.pose_utils import matrix_to_pose, pose_transform_dict


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-json", required=True)
    return parser


def _resolve_reference_image(path: str) -> tuple[np.ndarray, np.ndarray]:
    image = Image.open(path).convert("RGBA")
    image_rgba = np.asarray(image)
    rgb = image_rgba[..., :3].copy()
    alpha = image_rgba[..., 3] > 0
    if np.any(alpha):
        return rgb, alpha
    inferred = np.any(rgb < 250, axis=-1)
    if np.any(inferred):
        return rgb, inferred
    return rgb, np.ones(rgb.shape[:2], dtype=bool)


def _as_trimesh_mesh(mesh_or_scene: Any) -> trimesh.Trimesh:
    if isinstance(mesh_or_scene, trimesh.Trimesh):
        return mesh_or_scene.copy()
    if isinstance(mesh_or_scene, trimesh.Scene):
        collapsed = mesh_or_scene.dump(concatenate=True)
        if isinstance(collapsed, trimesh.Trimesh):
            return collapsed
    raise RuntimeError(
        f"SAM 3D returned an unsupported mesh container: {type(mesh_or_scene)!r}"
    )


def _to_cpu_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _normalize_quaternion_wxyz(rotation: Any) -> np.ndarray:
    array = _to_cpu_numpy(rotation).astype(np.float32, copy=False).reshape(-1)
    if array.size != 4:
        raise RuntimeError(
            f"SAM 3D returned an unexpected rotation shape: {array.shape!r}"
        )
    norm = float(np.linalg.norm(array))
    if norm <= 1e-8:
        raise RuntimeError("SAM 3D returned a zero-length quaternion.")
    return array / norm


def _coerce_xyz(name: str, value: Any) -> np.ndarray:
    array = _to_cpu_numpy(value).astype(np.float32, copy=False).reshape(-1)
    if array.size == 1:
        return np.repeat(array, 3)
    if array.size != 3:
        raise RuntimeError(f"SAM 3D returned an unexpected {name} shape: {array.shape!r}")
    return array


def _canonical_vertices_like_generate_sam3d(glb: Any) -> np.ndarray:
    mesh = _as_trimesh_mesh(glb)
    vertices = np.asarray(mesh.vertices, dtype=np.float32).copy()
    converted = np.zeros_like(vertices)
    converted[:, 0] = vertices[:, 0]
    converted[:, 1] = -vertices[:, 2]
    converted[:, 2] = vertices[:, 1]
    return converted


def _export_canonical_mesh_ply(glb: Any, output_path: Path) -> None:
    mesh = _as_trimesh_mesh(glb)
    mesh.vertices = _canonical_vertices_like_generate_sam3d(mesh)
    mesh.export(output_path)


def _posed_mesh_like_generate_sam3d(
    glb: Any,
    *,
    rotation: Any,
    translation: Any,
    scale: Any,
) -> trimesh.Trimesh:
    mesh = _as_trimesh_mesh(glb)
    canonical_vertices = _canonical_vertices_like_generate_sam3d(mesh)
    pose_device = (
        rotation.device
        if isinstance(rotation, torch.Tensor)
        else translation.device
        if isinstance(translation, torch.Tensor)
        else scale.device
        if isinstance(scale, torch.Tensor)
        else torch.device("cpu")
    )
    local_points = torch.from_numpy(canonical_vertices).to(
        device=pose_device, dtype=torch.float32
    ).unsqueeze(0)
    scale_tensor = _to_pose_tensor("scale", scale, device=pose_device)
    rotation_tensor = _to_pose_tensor("rotation", rotation, device=pose_device)
    translation_tensor = _to_pose_tensor("translation", translation, device=pose_device)
    posed_vertices = _transform_points_like_generate_sam3d(
        local_points=local_points,
        scale=scale_tensor,
        rotation_wxyz=rotation_tensor,
        translation=translation_tensor,
    ).squeeze(0).detach().cpu().numpy()
    mesh.vertices = posed_vertices
    return mesh


def _quaternion_wxyz_to_rotation_matrix_torch(rotation_wxyz: torch.Tensor) -> torch.Tensor:
    rotation_wxyz = rotation_wxyz / torch.linalg.norm(rotation_wxyz, dim=-1, keepdim=True).clamp_min(1e-8)
    w = rotation_wxyz[..., 0]
    x = rotation_wxyz[..., 1]
    y = rotation_wxyz[..., 2]
    z = rotation_wxyz[..., 3]
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    row0 = torch.stack((1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)), dim=-1)
    row1 = torch.stack((2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)), dim=-1)
    row2 = torch.stack((2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)), dim=-1)
    return torch.stack((row0, row1, row2), dim=-2)


def _transform_points_like_generate_sam3d(
    *,
    local_points: torch.Tensor,
    scale: torch.Tensor,
    rotation_wxyz: torch.Tensor,
    translation: torch.Tensor,
) -> torch.Tensor:
    scaled = local_points * scale.unsqueeze(-2)
    rotation_matrix = _quaternion_wxyz_to_rotation_matrix_torch(rotation_wxyz)
    rotated = torch.matmul(scaled, rotation_matrix.transpose(-1, -2))
    return rotated + translation.unsqueeze(-2)


def _to_pose_tensor(name: str, value: Any, *, device: torch.device) -> torch.Tensor:
    array = _to_cpu_numpy(value).astype(np.float32, copy=False).reshape(-1)
    if name == "rotation":
        if array.size != 4:
            raise RuntimeError(
                f"SAM 3D returned an unexpected {name} shape: {array.shape!r}"
            )
        array = _normalize_quaternion_wxyz(array)
    elif array.size == 1:
        array = np.repeat(array, 3)
    elif array.size != 3:
        raise RuntimeError(f"SAM 3D returned an unexpected {name} shape: {array.shape!r}")
    return torch.from_numpy(array).to(device=device, dtype=torch.float32).unsqueeze(0)


def _export_posed_mesh_ply(
    glb: Any,
    *,
    rotation: Any,
    translation: Any,
    scale: Any,
    output_path: Path,
) -> None:
    posed_mesh = _posed_mesh_like_generate_sam3d(
        glb,
        rotation=rotation,
        translation=translation,
        scale=scale,
    )
    posed_mesh.export(output_path)


def _serializable_pose_value(value: Any) -> list[float] | None:
    if value is None:
        return None
    array = _to_cpu_numpy(value).astype(np.float32, copy=False).reshape(-1)
    return [float(item) for item in array]


def _pose_dict_from_output(*, object_id: str, output: Mapping[str, Any]) -> dict[str, object] | None:
    explicit_pose = output.get("asset_local_pose")
    if isinstance(explicit_pose, Mapping):
        rotation_type = str(explicit_pose.get("rotation_type") or "quaternion")
        default_rotation_value = {
            "quaternion": (1.0, 0.0, 0.0, 0.0),
            "euler_xyz": (0.0, 0.0, 0.0),
            "matrix4x4": (
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0,
            ),
        }.get(rotation_type, ())
        return {
            "object_id": object_id,
            "translation_xyz": tuple(
                float(value) for value in explicit_pose.get("translation_xyz", (0.0, 0.0, 0.0))
            ),
            "rotation_type": rotation_type,
            "rotation_value": tuple(
                float(value)
                for value in explicit_pose.get("rotation_value", default_rotation_value)
            ),
            "scale_xyz": (
                tuple(float(value) for value in explicit_pose.get("scale_xyz", ()))
                if explicit_pose.get("scale_xyz") is not None
                else None
            ),
        }

    for key in ("asset_local_transform_matrix", "asset_transform_matrix"):
        matrix_values = output.get(key)
        if not isinstance(matrix_values, Sequence) or isinstance(matrix_values, (str, bytes)):
            continue
        flat_values = tuple(float(value) for value in matrix_values)
        if len(flat_values) != 16:
            continue
        pose = matrix_to_pose(
            object_id,
            np.asarray(flat_values, dtype=np.float64).reshape(4, 4),
        )
        return {"object_id": object_id, **pose_transform_dict(pose)}
    return None


def _configure_cuda_linalg_backend(payload: Mapping[str, Any]) -> str | None:
    backend = payload.get("cuda_linalg_backend")
    if not isinstance(backend, str) or not backend.strip():
        return None
    normalized = backend.strip().lower()
    torch.backends.cuda.preferred_linalg_library(normalized)
    return normalized


def run(payload: dict[str, Any]) -> dict[str, Any]:
    sam3d_root = Path(str(payload["sam3d_root"])).expanduser()
    if not sam3d_root.is_absolute():
        sam3d_root = REPO_ROOT / sam3d_root
    if not sam3d_root.exists():
        raise RuntimeError(f"SAM 3D root does not exist: {sam3d_root}")
    reference_image_path = Path(str(payload["reference_image_path"])).expanduser()
    if not reference_image_path.is_absolute():
        reference_image_path = REPO_ROOT / reference_image_path
    if not reference_image_path.exists():
        raise RuntimeError(f"SAM 3D reference image does not exist: {reference_image_path}")

    notebook_dir = sam3d_root / "notebook"
    if str(notebook_dir) not in sys.path:
        sys.path.insert(0, str(notebook_dir))
    if str(sam3d_root) not in sys.path:
        sys.path.insert(0, str(sam3d_root))

    current_cwd = Path.cwd()
    try:
        os.chdir(sam3d_root)
        from inference import Inference

        config_path = Path(str(payload.get("config_path") or "checkpoints/hf/pipeline.yaml"))
        if not config_path.is_absolute():
            config_path = sam3d_root / config_path
        if not config_path.exists():
            raise RuntimeError(f"SAM 3D config does not exist: {config_path}")

        object_id = str(payload["object_id"])
        output_dir = Path(str(payload["output_dir"])).expanduser()
        if not output_dir.is_absolute():
            output_dir = REPO_ROOT / output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        cuda_linalg_backend = _configure_cuda_linalg_backend(payload)
        image_rgb, mask = _resolve_reference_image(str(reference_image_path))
        inference = Inference(str(config_path), compile=bool(payload.get("compile", False)))
        rgba_image = inference.merge_mask_to_rgba(image_rgb, mask)
        output = inference._pipeline.run(
            rgba_image,
            None,
            seed=int(payload.get("seed", 42)),
            stage1_only=False,
            with_mesh_postprocess=False,
            with_texture_baking=False,
            with_layout_postprocess=True,
            use_vertex_color=True,
            stage1_inference_steps=None,
            pointmap=None,
        )

        mesh_path = output_dir / f"{object_id}.obj"
        canonical_mesh_path = output_dir / f"{object_id}.canonical.obj"
        mesh_ply_path = output_dir / f"{object_id}.mesh.ply"
        posed_mesh_ply_path = output_dir / f"{object_id}.posed.ply"
        gaussian_path = output_dir / f"{object_id}.ply"

        glb = output.get("glb")
        if glb is None:
            raise RuntimeError("SAM 3D did not return a mesh output.")
        glb.export(canonical_mesh_path)
        _export_canonical_mesh_ply(copy.deepcopy(glb), mesh_ply_path)

        rotation = output.get("rotation")
        translation = output.get("translation")
        scale = output.get("scale")
        if rotation is not None and translation is not None and scale is not None:
            posed_mesh = _posed_mesh_like_generate_sam3d(
                copy.deepcopy(glb),
                rotation=rotation,
                translation=translation,
                scale=scale,
            )
            posed_mesh.export(mesh_path)
            _export_posed_mesh_ply(
                copy.deepcopy(glb),
                rotation=rotation,
                translation=translation,
                scale=scale,
                output_path=posed_mesh_ply_path,
            )
        else:
            glb.export(mesh_path)
            posed_mesh_ply_path = None

        gaussian = output.get("gs")
        if gaussian is not None:
            gaussian.save_ply(str(gaussian_path))
        else:
            gaussian_path = None

        return {
            "provider_asset_id": object_id,
            "mesh_path": str(mesh_path),
            "canonical_mesh_path": str(canonical_mesh_path),
            "mesh_ply_path": str(mesh_ply_path),
            "posed_mesh_ply_path": str(posed_mesh_ply_path) if posed_mesh_ply_path is not None else None,
            "gaussian_path": str(gaussian_path) if gaussian_path is not None else None,
            "rotation_wxyz": _serializable_pose_value(rotation),
            "translation_xyz": _serializable_pose_value(translation),
            "scale_xyz": _serializable_pose_value(scale),
            "device": str(payload.get("device") or ""),
            "cuda_linalg_backend": cuda_linalg_backend,
            "asset_local_pose": _pose_dict_from_output(object_id=object_id, output=output),
        }
    finally:
        os.chdir(current_cwd)


def main() -> int:
    args = build_parser().parse_args()
    input_path = Path(args.input_json)
    output_path = Path(args.output_json)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    result = run(payload)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
