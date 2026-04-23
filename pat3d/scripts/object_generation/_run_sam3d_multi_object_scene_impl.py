from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image
import torch
import trimesh

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-json", required=True)
    return parser


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


def _quaternion_wxyz_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    quaternions = quaternions.to(dtype=torch.float32)
    norm = quaternions.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    r, i, j, k = torch.unbind(quaternions / norm, -1)
    two_s = 2.0
    return torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    ).reshape(quaternions.shape[:-1] + (3, 3))


def _load_rgb_image(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    return np.asarray(image, dtype=np.uint8)


def _load_mask(path: Path) -> np.ndarray:
    mask = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    return mask > 0


def _restore_scene_mask(
    mask: np.ndarray,
    *,
    bbox_xyxy: list[float] | tuple[float, ...] | None,
    scene_shape: tuple[int, int],
) -> np.ndarray:
    if mask.shape == scene_shape:
        return mask
    if not bbox_xyxy or len(bbox_xyxy) != 4:
        raise RuntimeError(
            f"Mask shape {mask.shape!r} does not match scene shape {scene_shape!r}, and no bbox was provided."
        )

    x0, y0, x1, y1 = [int(round(float(value))) for value in bbox_xyxy]
    scene_mask = np.zeros(scene_shape, dtype=bool)
    target_h = max(0, y1 - y0)
    target_w = max(0, x1 - x0)
    paste_h = min(mask.shape[0], target_h, scene_shape[0] - max(y0, 0))
    paste_w = min(mask.shape[1], target_w, scene_shape[1] - max(x0, 0))
    if paste_h <= 0 or paste_w <= 0:
        raise RuntimeError(
            f"Could not paste mask with bbox {bbox_xyxy!r} into scene shape {scene_shape!r}."
        )
    scene_mask[max(y0, 0) : max(y0, 0) + paste_h, max(x0, 0) : max(x0, 0) + paste_w] = mask[
        :paste_h,
        :paste_w,
    ]
    return scene_mask


def _build_raw_mesh(output: dict[str, Any]) -> trimesh.Trimesh:
    mesh_output = output.get("mesh")
    if mesh_output is None:
        raise RuntimeError("SAM 3D did not return a raw mesh output.")

    mesh_data = mesh_output[0]
    vertices = mesh_data.vertices.detach().float().cpu().numpy()
    faces = mesh_data.faces.detach().cpu().numpy()
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    vertex_attrs = getattr(mesh_data, "vertex_attrs", None)
    if vertex_attrs is not None:
        colors = (
            vertex_attrs[:, :3]
            .detach()
            .float()
            .clamp(0.0, 1.0)
            .cpu()
            .numpy()
        )
        mesh.visual.vertex_colors = (colors * 255).astype(np.uint8)

    return mesh


def _build_posed_mesh(output: dict[str, Any]) -> trimesh.Trimesh:
    from sam3d_objects.data.dataset.tdfy.transforms_3d import compose_transform

    mesh = _build_raw_mesh(output)
    vertices = torch.from_numpy(np.asarray(mesh.vertices, dtype=np.float32))
    transform = compose_transform(
        scale=_to_pose_tensor(output["scale"], name="scale", device=vertices.device),
        rotation=_quaternion_wxyz_to_matrix(
            _to_pose_tensor(output["rotation"], name="rotation", device=vertices.device)
        ),
        translation=_to_pose_tensor(
            output["translation"], name="translation", device=vertices.device
        ),
    )
    posed_vertices = (
        transform.transform_points(vertices.unsqueeze(0))
        .squeeze(0)
        .detach()
        .cpu()
        .numpy()
    )
    mesh.vertices = posed_vertices
    return mesh


def _to_pose_tensor(value: Any, *, name: str, device: torch.device) -> torch.Tensor:
    if name == "rotation":
        array = _normalize_quaternion_wxyz(value)
    else:
        array = _coerce_xyz(name, value)
    return torch.from_numpy(array).to(device=device, dtype=torch.float32).unsqueeze(0)


def _serializable_pose_value(value: Any) -> list[float] | None:
    if value is None:
        return None
    array = _to_cpu_numpy(value).astype(np.float32, copy=False).reshape(-1)
    return [float(item) for item in array]


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
        raise RuntimeError(f"Reference image does not exist: {reference_image_path}")

    scene_output_dir = Path(str(payload["scene_output_dir"])).expanduser()
    if not scene_output_dir.is_absolute():
        scene_output_dir = REPO_ROOT / scene_output_dir
    scene_output_dir.mkdir(parents=True, exist_ok=True)

    notebook_dir = sam3d_root / "notebook"
    if str(notebook_dir) not in sys.path:
        sys.path.insert(0, str(notebook_dir))
    if str(sam3d_root) not in sys.path:
        sys.path.insert(0, str(sam3d_root))

    current_cwd = Path.cwd()
    try:
        os.chdir(sam3d_root)
        from inference import Inference, make_scene

        config_path = Path(str(payload.get("config_path") or "checkpoints/hf/pipeline.yaml"))
        if not config_path.is_absolute():
            config_path = sam3d_root / config_path
        if not config_path.exists():
            raise RuntimeError(f"SAM 3D config does not exist: {config_path}")

        image_rgb = _load_rgb_image(reference_image_path)
        inference = Inference(str(config_path), compile=bool(payload.get("compile", False)))
        seed = int(payload.get("seed", 42))

        object_payloads = payload.get("objects", ())
        if not isinstance(object_payloads, list) or not object_payloads:
            raise RuntimeError("Expected a non-empty 'objects' list in the scene payload.")

        object_results: list[dict[str, Any]] = []
        outputs: list[dict[str, Any]] = []
        for item in object_payloads:
            if not isinstance(item, dict):
                continue
            object_id = str(item["object_id"])
            mask_path = Path(str(item["mask_path"])).expanduser()
            if not mask_path.is_absolute():
                mask_path = REPO_ROOT / mask_path
            if not mask_path.exists():
                raise RuntimeError(f"Mask does not exist for '{object_id}': {mask_path}")

            output_dir = scene_output_dir / object_id
            output_dir.mkdir(parents=True, exist_ok=True)

            mask = _load_mask(mask_path)
            mask = _restore_scene_mask(
                mask,
                bbox_xyxy=item.get("bbox_xyxy"),
                scene_shape=image_rgb.shape[:2],
            )
            output = inference(image_rgb, mask, seed=seed)
            outputs.append(output)

            canonical_mesh_path = output_dir / f"{object_id}.canonical.obj"
            posed_mesh_path = output_dir / f"{object_id}.posed.obj"
            mesh_ply_path = output_dir / f"{object_id}.mesh.ply"
            posed_mesh_ply_path = output_dir / f"{object_id}.posed.ply"
            gaussian_path = output_dir / f"{object_id}.ply"

            raw_mesh = _build_raw_mesh(output)
            raw_mesh.export(canonical_mesh_path)
            raw_mesh.export(mesh_ply_path)

            posed_mesh = _build_posed_mesh(output)
            posed_mesh.export(posed_mesh_path)
            posed_mesh.export(posed_mesh_ply_path)

            gaussian = output.get("gs") or output.get("gaussian", [None])[0]
            if gaussian is not None:
                gaussian.save_ply(str(gaussian_path))
            else:
                gaussian_path = None

            object_results.append(
                {
                    "object_id": object_id,
                    "provider_asset_id": object_id,
                    "mesh_path": str(posed_mesh_path),
                    "canonical_mesh_path": str(canonical_mesh_path),
                    "mesh_ply_path": str(mesh_ply_path),
                    "posed_mesh_ply_path": str(posed_mesh_ply_path),
                    "gaussian_path": str(gaussian_path) if gaussian_path is not None else None,
                    "rotation_wxyz": _serializable_pose_value(output.get("rotation")),
                    "translation_xyz": _serializable_pose_value(output.get("translation")),
                    "scale_xyz": _serializable_pose_value(output.get("scale")),
                    "prompt": str(item.get("prompt") or ""),
                    "mask_path": str(mask_path),
                    "reference_image_path": str(item.get("reference_image_path") or ""),
                }
            )

        scene_manifest_path = scene_output_dir / "sam3d_multi_object_scene.json"
        scene_manifest: dict[str, Any] = {
            "scene_id": str(payload.get("scene_id") or ""),
            "reference_image_path": str(reference_image_path),
            "objects": object_results,
        }

        if outputs:
            scene_gs = make_scene(*outputs)
            scene_gaussian_path = scene_output_dir / "scene_posed.ply"
            scene_gs.save_ply(str(scene_gaussian_path))
            scene_manifest["scene_gaussian_path"] = str(scene_gaussian_path)

        scene_manifest_path.write_text(
            json.dumps(scene_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return {
            "scene_manifest_path": str(scene_manifest_path),
            "scene_gaussian_path": scene_manifest.get("scene_gaussian_path"),
            "objects": object_results,
            "device": str(payload.get("device") or ""),
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
