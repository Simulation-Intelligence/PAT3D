from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import trimesh

from pat3d.models import (
    ArtifactRef,
    DepthResult,
    ObjectAssetCatalog,
    ObjectCatalog,
    ObjectPose,
    ReferenceImageResult,
    SceneLayout,
    SceneRelationGraph,
    SegmentationResult,
    SizePrior,
)
from pat3d.providers._projection_layout_refiner import ProjectionAwareLayoutRefiner
from pat3d.providers._sam3d_pose_metadata_impl import (
    POSE_NOTE_PREFIX,
    SCENE_GAUSSIAN_NOTE_PREFIX,
    SCENE_MANIFEST_NOTE_PREFIX,
    parse_json_note,
)
from pat3d.storage import make_stage_metadata


class SAM3DMultiObjectLayoutBuilder:
    def __init__(
        self,
        *,
        layout_root: str = "data/layout",
        mesh_loader: Callable[..., trimesh.Trimesh | trimesh.Scene] = trimesh.load,
        layout_refiner: ProjectionAwareLayoutRefiner | None = None,
        metadata_factory: Callable[..., object] = make_stage_metadata,
    ) -> None:
        self._layout_root = Path(layout_root)
        self._mesh_loader = mesh_loader
        self._layout_refiner = layout_refiner or ProjectionAwareLayoutRefiner(
            layout_root=layout_root
        )
        self._metadata_factory = metadata_factory

    def build_layout(
        self,
        *,
        scene_id: str,
        object_assets: ObjectAssetCatalog,
        object_catalog: ObjectCatalog | None = None,
        reference_image_result: ReferenceImageResult | None = None,
        segmentation_result: SegmentationResult | None = None,
        depth_result: DepthResult | None = None,
        relation_graph: SceneRelationGraph | None = None,
        size_priors: Sequence[SizePrior] | None = None,
    ) -> SceneLayout:
        _ = object_catalog, reference_image_result, segmentation_result, depth_result, size_priors

        scene_dir = self._layout_root / scene_id
        scene_dir.mkdir(parents=True, exist_ok=True)

        object_poses: list[ObjectPose] = []
        artifacts: list[ArtifactRef] = []
        shared_manifest_path: str | None = None
        shared_gaussian_path: str | None = None

        for asset in object_assets.assets:
            layout_mesh_path = scene_dir / f"{asset.object_id}.obj"
            mesh = self._load_mesh(Path(asset.mesh_obj.path))
            layout_mesh_path.write_text(mesh.export(file_type="obj"), encoding="utf-8")

            pose = self._pose_from_asset(asset, mesh)
            object_poses.append(pose)
            artifacts.append(
                ArtifactRef(
                    artifact_type="layout_mesh",
                    path=str(layout_mesh_path),
                    format="obj",
                    role=asset.object_id,
                    metadata_path=None,
                )
            )

            metadata = getattr(asset, "metadata", None)
            notes = getattr(metadata, "notes", ()) if metadata is not None else ()
            if shared_manifest_path is None:
                manifest_payload = parse_json_note(notes, SCENE_MANIFEST_NOTE_PREFIX)
                if manifest_payload and manifest_payload.get("path"):
                    shared_manifest_path = str(manifest_payload["path"])
            if shared_gaussian_path is None:
                gaussian_payload = parse_json_note(notes, SCENE_GAUSSIAN_NOTE_PREFIX)
                if gaussian_payload and gaussian_payload.get("path"):
                    shared_gaussian_path = str(gaussian_payload["path"])

        if shared_manifest_path:
            artifacts.append(
                ArtifactRef(
                    artifact_type="sam3d_scene_manifest",
                    path=shared_manifest_path,
                    format=Path(shared_manifest_path).suffix.lstrip(".") or "json",
                    role="sam3d_scene_manifest",
                    metadata_path=None,
                )
            )
        if shared_gaussian_path:
            artifacts.append(
                ArtifactRef(
                    artifact_type="sam3d_scene_gaussian",
                    path=shared_gaussian_path,
                    format=Path(shared_gaussian_path).suffix.lstrip(".") or "ply",
                    role="sam3d_scene_gaussian",
                    metadata_path=None,
                )
            )

        payload = {
            "object_poses": tuple(object_poses),
            "artifacts": tuple(artifacts),
            "layout_space": "world",
        }
        refined = self._layout_refiner.refine(
            scene_id=scene_id,
            payload=payload,
            relation_graph=relation_graph,
        )
        refined_poses = tuple(refined.get("object_poses", ()))
        refined_artifacts = tuple(refined.get("artifacts", ()))
        return SceneLayout(
            scene_id=scene_id,
            object_poses=refined_poses,
            layout_space=str(refined.get("layout_space", "world")),
            support_graph=relation_graph,
            artifacts=refined_artifacts,
            metadata=self._metadata_factory(
                stage_name="layout_initialization",
                provider_name="sam3d_multi_object_layout",
                notes=("initial_layout=sam3d_multi_object_pose", "refinement=projection_xz"),
            ),
        )

    def _pose_from_asset(
        self,
        asset,
        mesh: trimesh.Trimesh,
    ) -> ObjectPose:
        metadata = getattr(asset, "metadata", None)
        notes = getattr(metadata, "notes", ()) if metadata is not None else ()
        pose_payload = parse_json_note(notes, POSE_NOTE_PREFIX) or {}
        rotation_wxyz = tuple(
            float(value) for value in pose_payload.get("rotation_wxyz", (1.0, 0.0, 0.0, 0.0))
        )
        scale_xyz = pose_payload.get("scale_xyz")
        center = mesh.bounding_box.centroid
        return ObjectPose(
            object_id=asset.object_id,
            translation_xyz=(float(center[0]), float(center[1]), float(center[2])),
            rotation_type="quaternion",
            rotation_value=rotation_wxyz,
            scale_xyz=(
                tuple(float(value) for value in scale_xyz)
                if isinstance(scale_xyz, (list, tuple)) and len(scale_xyz) == 3
                else (1.0, 1.0, 1.0)
            ),
        )

    def _load_mesh(self, mesh_path: Path) -> trimesh.Trimesh:
        try:
            loaded = self._mesh_loader(mesh_path, force="mesh")
        except TypeError:
            loaded = self._mesh_loader(mesh_path)
        if isinstance(loaded, trimesh.Scene):
            return loaded.dump(concatenate=True)
        return loaded.copy()
