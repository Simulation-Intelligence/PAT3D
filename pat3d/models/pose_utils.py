from __future__ import annotations

import numpy as np
import trimesh

from .layout import ObjectPose


def pose_to_matrix(pose: ObjectPose) -> np.ndarray:
    matrix = _rotation_matrix(pose)
    if pose.scale_xyz is not None:
        scale_matrix = np.eye(4, dtype=np.float64)
        scale_matrix[0, 0] = float(pose.scale_xyz[0])
        scale_matrix[1, 1] = float(pose.scale_xyz[1])
        scale_matrix[2, 2] = float(pose.scale_xyz[2])
        matrix = matrix @ scale_matrix
    matrix = matrix.copy()
    matrix[0, 3] += float(pose.translation_xyz[0])
    matrix[1, 3] += float(pose.translation_xyz[1])
    matrix[2, 3] += float(pose.translation_xyz[2])
    return matrix


def matrix_to_pose(object_id: str, matrix: np.ndarray | list[list[float]]) -> ObjectPose:
    normalized_matrix = np.asarray(matrix, dtype=np.float64)
    if normalized_matrix.shape != (4, 4):
        raise ValueError("matrix must be 4x4")

    linear = normalized_matrix[:3, :3]
    scale = np.linalg.norm(linear, axis=0)
    safe_scale = np.where(scale > 1e-12, scale, 1.0)
    rotation = linear / safe_scale
    u, _, vh = np.linalg.svd(rotation)
    rotation = u @ vh
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vh

    rotation_matrix = np.eye(4, dtype=np.float64)
    rotation_matrix[:3, :3] = rotation
    quaternion = trimesh.transformations.quaternion_from_matrix(rotation_matrix)

    return ObjectPose(
        object_id=object_id,
        translation_xyz=(
            float(normalized_matrix[0, 3]),
            float(normalized_matrix[1, 3]),
            float(normalized_matrix[2, 3]),
        ),
        rotation_type="quaternion",
        rotation_value=tuple(float(value) for value in quaternion),
        scale_xyz=tuple(float(abs(value)) for value in scale),
    )


def pose_transform_dict(pose: ObjectPose) -> dict[str, object]:
    return {
        "translation_xyz": tuple(float(value) for value in pose.translation_xyz),
        "rotation_type": pose.rotation_type,
        "rotation_value": tuple(float(value) for value in pose.rotation_value),
        "scale_xyz": (
            tuple(float(value) for value in pose.scale_xyz)
            if pose.scale_xyz is not None
            else None
        ),
    }


def delta_transform_dict(initial_pose: ObjectPose, final_pose: ObjectPose) -> dict[str, object]:
    initial_matrix = pose_to_matrix(initial_pose)
    final_matrix = pose_to_matrix(final_pose)
    delta_matrix = final_matrix @ np.linalg.inv(initial_matrix)
    return pose_transform_dict(matrix_to_pose(final_pose.object_id, delta_matrix))


def apply_pose_to_mesh(mesh: trimesh.Trimesh, pose: ObjectPose | None) -> trimesh.Trimesh:
    if pose is None:
        return mesh
    mesh.apply_transform(pose_to_matrix(pose))
    return mesh


def _rotation_matrix(pose: ObjectPose) -> np.ndarray:
    if pose.rotation_type == "quaternion":
        if len(pose.rotation_value) != 4:
            raise ValueError("quaternion rotation_value must contain four values")
        return np.asarray(
            trimesh.transformations.quaternion_matrix(pose.rotation_value),
            dtype=np.float64,
        )
    if pose.rotation_type == "euler_xyz":
        if len(pose.rotation_value) != 3:
            raise ValueError("euler_xyz rotation_value must contain three values")
        return np.asarray(
            trimesh.transformations.euler_matrix(*pose.rotation_value, axes="sxyz"),
            dtype=np.float64,
        )
    if pose.rotation_type == "matrix4x4":
        if len(pose.rotation_value) != 16:
            raise ValueError("matrix4x4 rotation_value must contain sixteen values")
        return np.asarray(pose.rotation_value, dtype=np.float64).reshape(4, 4)
    raise ValueError(f"unsupported rotation_type: {pose.rotation_type}")
