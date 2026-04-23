from __future__ import annotations

from pathlib import Path
import shutil
from typing import Callable

import trimesh

from pat3d.contracts import MeshSimplifier
from pat3d.models import ArtifactRef, ObjectAssetCatalog, ObjectPose, PhysicsReadyScene, SceneLayout
from pat3d.models.pose_utils import apply_pose_to_mesh
from pat3d.storage import make_stage_metadata


def _load_default_simplifier() -> Callable[[str, str, str, int, str], None]:
    from pat3d.preprocessing.low_poly import get_low_poly_new as current_get_low_poly_new

    return current_get_low_poly_new


def _noop_simplifier(
    scene_name: str,
    layout_folder: str,
    low_poly_folder: str,
    face_num: int,
    manifold_path: str,
) -> None:
    _ = scene_name, layout_folder, low_poly_folder, face_num, manifold_path


class CurrentMeshSimplifier(MeshSimplifier):
    def __init__(
        self,
        *,
        layout_folder: str = "data/layout",
        low_poly_folder: str = "data/low_poly",
        target_face_num: int = 2000,
        physics_settings: dict[str, float | int | bool] | None = None,
        manifold_code_path: str = "extern/Manifold/build",
        simplifier: Callable[[str, str, str, int, str], None] | None = None,
        metadata_factory: Callable[..., object] = make_stage_metadata,
    ) -> None:
        self._layout_folder = layout_folder
        self._low_poly_folder = low_poly_folder
        self._target_face_num = target_face_num
        self._physics_settings = dict(physics_settings or {})
        self._manifold_code_path = manifold_code_path
        self._default_simplifier_note: str | None = None
        if simplifier is not None:
            self._simplifier = simplifier
        else:
            try:
                self._simplifier = _load_default_simplifier()
            except Exception as error:
                self._simplifier = _noop_simplifier
                self._default_simplifier_note = (
                    f"default_simplifier_unavailable={error.__class__.__name__}: {error}"
                )
        self._metadata_factory = metadata_factory

    def simplify(
        self,
        scene_layout: SceneLayout,
        object_assets: ObjectAssetCatalog,
    ) -> PhysicsReadyScene:
        layout_scene_dir = self._materialize_layout_meshes(scene_layout, object_assets)
        scene_dir = Path(self._low_poly_folder) / scene_layout.scene_id
        fallback_notes: list[str] = []
        try:
            self._simplifier(
                scene_layout.scene_id,
                self._layout_folder,
                self._low_poly_folder,
                self._target_face_num,
                self._manifold_code_path,
            )
        except Exception as exc:
            fallback_notes = [
                "legacy_low_poly_fallback",
                f"fallback_reason={type(exc).__name__}: {exc}",
            ]
            if scene_dir.exists():
                shutil.rmtree(scene_dir)

        self._ensure_simulation_mesh_outputs(
            scene_layout,
            object_assets,
            layout_scene_dir,
            scene_dir,
            overwrite_existing=bool(fallback_notes),
        )
        simulation_meshes = []
        for asset in object_assets.assets:
            candidate = scene_dir / f"{asset.object_id}.obj"
            simulation_meshes.append(
                ArtifactRef(
                    artifact_type="simulation_mesh",
                    path=str(candidate),
                    format="obj",
                    role=asset.object_id,
                    metadata_path=None,
                )
            )

        return PhysicsReadyScene(
            scene_id=scene_layout.scene_id,
            layout=scene_layout,
            simulation_meshes=tuple(simulation_meshes),
            object_poses=scene_layout.object_poses,
            collision_settings={
                "target_face_num": self._target_face_num,
                **self._physics_settings,
            },
            metadata=self._metadata_factory(
                stage_name="simulation_preparation",
                provider_name="current_low_poly",
                notes=tuple(
                    note
                    for note in (
                        self._default_simplifier_note,
                        *fallback_notes,
                    )
                    if note
                ),
            ),
        )

    def _materialize_layout_meshes(
        self,
        scene_layout: SceneLayout,
        object_assets: ObjectAssetCatalog,
    ) -> Path:
        scene_dir = Path(self._layout_folder) / scene_layout.scene_id
        scene_dir.mkdir(parents=True, exist_ok=True)
        pose_by_object_id = {pose.object_id: pose for pose in scene_layout.object_poses}
        artifact_by_object_id = self._layout_artifact_meshes(scene_layout)
        for asset in object_assets.assets:
            target_path = scene_dir / f"{asset.object_id}.obj"
            pose = pose_by_object_id.get(asset.object_id)
            if pose is None:
                continue
            try:
                mesh = self._load_mesh(Path(asset.mesh_obj.path))
            except Exception:
                artifact_path = artifact_by_object_id.get(asset.object_id)
                if artifact_path is not None and artifact_path.exists():
                    if artifact_path.resolve() != target_path.resolve():
                        shutil.copyfile(artifact_path, target_path)
                    continue
                raise

            if not self._asset_mesh_is_scene_space(asset):
                apply_pose_to_mesh(mesh, asset.asset_local_pose)
                self._apply_pose(mesh, pose)
            else:
                apply_pose_to_mesh(mesh, asset.asset_local_pose)
            mesh.export(target_path)
        return scene_dir

    def _layout_artifact_meshes(self, scene_layout: SceneLayout) -> dict[str, Path]:
        artifact_map: dict[str, Path] = {}
        for artifact in scene_layout.artifacts:
            if artifact.artifact_type != "layout_mesh" or artifact.role is None:
                continue
            artifact_map[artifact.role] = Path(artifact.path)
        return artifact_map

    def _ensure_simulation_mesh_outputs(
        self,
        scene_layout: SceneLayout,
        object_assets: ObjectAssetCatalog,
        layout_scene_dir: Path,
        simulation_scene_dir: Path,
        *,
        overwrite_existing: bool = False,
    ) -> None:
        simulation_scene_dir.mkdir(parents=True, exist_ok=True)
        missing_outputs = []
        for asset in object_assets.assets:
            candidate = simulation_scene_dir / f"{asset.object_id}.obj"
            if overwrite_existing or not candidate.exists():
                missing_outputs.append((layout_scene_dir / f"{asset.object_id}.obj", candidate))
        if not missing_outputs:
            return
        for source, candidate in missing_outputs:
            if source.exists():
                shutil.copyfile(source, candidate)

    def _load_mesh(self, mesh_path: Path) -> trimesh.Trimesh:
        loaded = trimesh.load(mesh_path, force="mesh")
        if isinstance(loaded, trimesh.Scene):
            return loaded.dump(concatenate=True)
        return loaded.copy()

    def _asset_mesh_is_scene_space(self, asset) -> bool:
        metadata = getattr(asset, "metadata", None)
        notes = getattr(metadata, "notes", ()) if metadata is not None else ()
        return any(note == "mesh_pose_space=scene" for note in notes)

    def _apply_pose(self, mesh: trimesh.Trimesh, pose: ObjectPose) -> None:
        apply_pose_to_mesh(mesh, pose)
