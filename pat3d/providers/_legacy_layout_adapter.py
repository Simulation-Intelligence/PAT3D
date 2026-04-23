from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import itertools
import json
import multiprocessing as mp
import numpy as np
from pathlib import Path
import re
import shutil
import tempfile
import traceback
from typing import Callable, Sequence

from PIL import Image
import trimesh

from pat3d.legacy_config import load_root_parse_options
from pat3d.providers._projection_layout_refiner import ProjectionAwareLayoutRefiner
from pat3d.models import (
    ArtifactRef,
    GeneratedObjectAsset,
    ObjectAssetCatalog,
    ObjectCatalog,
    ObjectPose,
    ReferenceImageResult,
    SceneRelationGraph,
    SizePrior,
)
from pat3d.models.pose_utils import apply_pose_to_mesh
from pat3d.providers._relation_utils import match_object_id


def _default_args_factory() -> object:
    args, _ = load_root_parse_options()(argv=[])
    return args


def _run_layout_worker(
    *,
    layout_initializer: Callable[[str, str, str, str, str, int, dict[str, list[float]]], None],
    ground_resetter: Callable[[str, str, str, str], None],
    organized_obj_folder: str,
    depth_folder: str,
    seg_folder: str,
    scene_id: str,
    layout_folder: str,
    layout_front_num: int,
    size_ratio: dict[str, list[float]],
    log_path: str,
    status_queue,
) -> None:
    with open(log_path, "w", encoding="utf-8") as log_handle:
        with redirect_stdout(log_handle), redirect_stderr(log_handle):
            try:
                log_handle.write("layout_initializer:start\n")
                log_handle.flush()
                layout_initializer(
                    organized_obj_folder,
                    depth_folder,
                    seg_folder,
                    scene_id,
                    layout_folder,
                    layout_front_num,
                    size_ratio,
                )
                log_handle.write("layout_initializer:done\n")
                log_handle.write("ground_resetter:start\n")
                log_handle.flush()
                ground_resetter(layout_folder, depth_folder, scene_id, layout_folder)
                log_handle.write("ground_resetter:done\n")
                log_handle.flush()
            except Exception as exc:  # pragma: no cover - exercised through parent process
                traceback.print_exc(file=log_handle)
                status_queue.put(("error", str(exc)))
                return
    status_queue.put(("ok", None))


class LegacyInitialLayoutAdapter:
    def __init__(
        self,
        *,
        args_factory: Callable[[], object] = _default_args_factory,
        layout_initializer: Callable[[str, str, str, str, str, int, dict[str, list[float]]], None]
        | None = None,
        ground_resetter: Callable[[str, str, str, str], None] | None = None,
        mesh_loader: Callable[..., trimesh.Trimesh | trimesh.Scene] = trimesh.load,
        layout_refiner: ProjectionAwareLayoutRefiner | None = None,
        timeout_seconds: float = 90.0,
        worker_log_root: str | Path | None = None,
        legacy_arg_overrides: dict[str, str | int | float | bool] | None = None,
    ) -> None:
        resolved_legacy_arg_overrides = dict(legacy_arg_overrides or {})
        self._args_factory = args_factory
        if layout_initializer is None:
            from pat3d.preprocessing.layout import get_initial_layout as layout_initializer
        if ground_resetter is None:
            from pat3d.preprocessing.reset_y import reset_ground as ground_resetter

        self._layout_initializer = layout_initializer
        self._ground_resetter = ground_resetter
        self._mesh_loader = mesh_loader
        resolved_layout_root = resolved_legacy_arg_overrides.get("layout_folder")
        self._layout_refiner = layout_refiner or ProjectionAwareLayoutRefiner(
            layout_root=(
                str(resolved_layout_root)
                if isinstance(resolved_layout_root, str) and resolved_layout_root.strip()
                else "data/layout"
            )
        )
        self._timeout_seconds = timeout_seconds
        self._worker_log_root = Path(worker_log_root) if worker_log_root is not None else (
            Path(tempfile.gettempdir()) / "pat3d_legacy_layout_logs"
        )
        self._legacy_arg_overrides = resolved_legacy_arg_overrides

    def build(
        self,
        *,
        scene_id: str,
        object_assets: ObjectAssetCatalog,
        object_catalog: ObjectCatalog | None = None,
        reference_image_result: ReferenceImageResult | None = None,
        relation_graph: SceneRelationGraph | None = None,
        size_priors: Sequence[SizePrior] = (),
        depth_result: object | None = None,
        segmentation_result: object | None = None,
    ) -> dict[str, object]:
        _ = object_catalog
        args = self._build_args(scene_id, object_assets, reference_image_result)
        prior_by_id = {prior.object_id: prior for prior in size_priors}
        alias_map = self._materialize_input_meshes(
            args,
            scene_id,
            object_assets,
            prior_by_id=prior_by_id,
        )
        size_ratio = self._build_size_ratio(object_assets, size_priors, alias_map)
        log_path = self._run_bounded_layout(args, scene_id, size_ratio)
        payload = self._build_layout_payload(args, scene_id, object_assets, alias_map, relation_graph)
        refined_payload = self._layout_refiner.refine(
            scene_id=scene_id,
            payload=payload,
            relation_graph=relation_graph,
        )
        return self._cleanup_layout_meshes(
            scene_id=scene_id,
            payload=refined_payload,
            object_assets=object_assets,
            object_catalog=object_catalog,
            segmentation_result=segmentation_result,
            depth_result=depth_result,
            relation_graph=relation_graph,
        )

    def _build_args(
        self,
        scene_id: str,
        object_assets: ObjectAssetCatalog,
        reference_image_result: ReferenceImageResult | None,
    ) -> object:
        args = self._args_factory()
        for key, value in self._legacy_arg_overrides.items():
            if isinstance(value, (str, int, float, bool)):
                setattr(args, key, value)
        args.scene_name = scene_id
        args.exp_name = scene_id
        args.text_prompt = (
            reference_image_result.request.text_prompt if reference_image_result is not None else None
        )
        args.items = [self._legacy_name(scene_id, asset.object_id) for asset in object_assets.assets]
        return args

    def _materialize_input_meshes(
        self,
        args: object,
        scene_id: str,
        object_assets: ObjectAssetCatalog,
        *,
        prior_by_id: dict[str, SizePrior] | None = None,
    ) -> dict[str, str]:
        scene_dir = Path(args.organized_obj_folder) / scene_id
        if scene_dir.exists():
            shutil.rmtree(scene_dir)
        scene_dir.mkdir(parents=True, exist_ok=True)

        alias_map: dict[str, str] = {}
        for asset in object_assets.assets:
            legacy_name = self._legacy_name(scene_id, asset.object_id)
            alias_map[asset.object_id] = legacy_name
            target_obj = scene_dir / f"{legacy_name}.obj"
            source_mesh = self._load_asset_mesh(asset)
            target_extents = self._resolve_target_extents_for_source_axes(
                source_mesh,
                size_prior=(prior_by_id or {}).get(asset.object_id),
            )
            proxy_mesh = self._build_proxy_mesh(source_mesh, target_extents=target_extents)
            proxy_mesh.export(target_obj)
            self._write_placeholder_materials(scene_dir, legacy_name)

        return alias_map

    def _build_size_ratio(
        self,
        object_assets: ObjectAssetCatalog,
        size_priors: Sequence[SizePrior],
        alias_map: dict[str, str],
    ) -> dict[str, list[float]]:
        prior_by_id = {prior.object_id: prior for prior in size_priors}
        size_ratio: dict[str, list[float]] = {}
        absolute_size_categories: set[str] = set()
        for asset in object_assets.assets:
            legacy_name = alias_map[asset.object_id]
            category = self._category_name(legacy_name)
            prior_values = self._normalize_prior_values(prior_by_id.get(asset.object_id))
            values = prior_values or self._mesh_extent_values_for_asset(asset)
            size_ratio[category] = values
            if prior_values:
                absolute_size_categories.add(category)
        if absolute_size_categories:
            size_ratio["__absolute_size_categories__"] = sorted(absolute_size_categories)
        return size_ratio

    def _build_layout_payload(
        self,
        args: object,
        scene_id: str,
        object_assets: ObjectAssetCatalog,
        alias_map: dict[str, str],
        relation_graph: SceneRelationGraph | None,
    ) -> dict[str, object]:
        output_scene_dir = Path(args.layout_folder) / scene_id
        output_scene_dir.mkdir(parents=True, exist_ok=True)

        alias_map_path = output_scene_dir / "alias_map.json"
        alias_map_path.write_text(json.dumps(alias_map, indent=2, sort_keys=True), encoding="utf-8")

        artifacts: list[ArtifactRef] = [
            ArtifactRef(
                artifact_type="layout_alias_map",
                path=str(alias_map_path),
                format="json",
                role="alias_map",
                metadata_path=None,
            )
        ]
        log_path = output_scene_dir / "legacy_layout.log"
        if log_path.exists():
            artifacts.append(
                ArtifactRef(
                    artifact_type="legacy_layout_log",
                    path=str(log_path),
                    format="log",
                    role="layout_log",
                    metadata_path=None,
                )
            )
        ground_y_path = output_scene_dir / "ground_y_value.txt"
        if ground_y_path.exists():
            artifacts.append(
                ArtifactRef(
                    artifact_type="ground_plane",
                    path=str(ground_y_path),
                    format="txt",
                    role="ground_y",
                    metadata_path=None,
                )
            )

        object_poses = []
        for asset in object_assets.assets:
            legacy_name = alias_map[asset.object_id]
            layout_mesh_path = output_scene_dir / f"{legacy_name}.obj"
            pose = self._materialize_layout_mesh(asset, layout_mesh_path)
            artifacts.append(
                ArtifactRef(
                    artifact_type="layout_mesh",
                    path=str(layout_mesh_path),
                    format="obj",
                    role=asset.object_id,
                    metadata_path=None,
                )
            )
            object_poses.append(pose)

        return {
            "object_poses": tuple(object_poses),
            "artifacts": tuple(artifacts),
            "layout_space": "world",
            "relation_graph": relation_graph,
        }

    def _cleanup_layout_meshes(
        self,
        *,
        scene_id: str,
        payload: dict[str, object],
        object_assets: ObjectAssetCatalog,
        object_catalog: ObjectCatalog | None,
        segmentation_result: object | None,
        depth_result: object | None,
        relation_graph: SceneRelationGraph | None,
    ) -> dict[str, object]:
        if relation_graph is None or not relation_graph.relations:
            return payload

        evidence_by_object_id = self._build_observed_evidence(
            scene_id=scene_id,
            object_assets=object_assets,
            object_catalog=object_catalog,
            segmentation_result=segmentation_result,
            depth_result=depth_result,
        )
        if not evidence_by_object_id:
            return payload

        artifacts = list(payload.get("artifacts", ()))
        mesh_path_by_id = {
            artifact.role: Path(artifact.path)
            for artifact in artifacts
            if artifact.artifact_type == "layout_mesh" and artifact.role
        }
        asset_by_id = {asset.object_id: asset for asset in object_assets.assets}
        report: dict[str, object] = {
            "scene_id": scene_id,
            "cleaned_objects": [],
            "relations": [],
        }
        changed_object_ids: set[str] = set()

        for relation in relation_graph.relations:
            parent_id = relation.parent_object_id
            child_id = relation.child_object_id
            relation_type = str(getattr(relation.relation_type, "value", relation.relation_type)).strip().lower()
            if relation_type in {"contains", "in"}:
                continue
            parent_path = mesh_path_by_id.get(parent_id)
            parent_evidence = evidence_by_object_id.get(parent_id)
            child_evidence = evidence_by_object_id.get(child_id)
            if parent_path is None or parent_evidence is None or child_evidence is None or not parent_path.exists():
                continue

            parent_mesh = self._load_mesh(parent_path)
            components = list(parent_mesh.split(only_watertight=False))
            if len(components) <= 1:
                continue

            anchor_index = max(
                range(len(components)),
                key=lambda index: (
                    float(abs(getattr(components[index], "volume", 0.0))),
                    len(components[index].faces),
                    -self._component_distance_score(components[index], parent_evidence["points"]),
                ),
            )
            removed_indices: list[int] = []
            relation_moves: list[dict[str, object]] = []
            total_faces = max(sum(len(component.faces) for component in components), 1)
            kept_components: list[trimesh.Trimesh] = []

            for index, component in enumerate(components):
                if index == anchor_index:
                    kept_components.append(component)
                    continue

                parent_score = self._component_distance_score(component, parent_evidence["points"])
                child_score = self._component_distance_score(component, child_evidence["points"])
                component_center = np.asarray(component.bounding_box.centroid, dtype=float)
                child_bounds = child_evidence["bounds"]
                if relation_type in {"contains", "in"}:
                    within_child_region = self._point_within_horizontal_bounds(
                        component_center[[0, 2]],
                        child_bounds[:, [0, 2]],
                        padding=max(0.02, float(np.max(child_bounds[1, [0, 2]] - child_bounds[0, [0, 2]])) * 0.2),
                    )
                else:
                    within_child_region = self._point_within_bounds(
                        component_center,
                        child_bounds,
                        padding=max(0.05, float(np.max(child_bounds[1] - child_bounds[0])) * 0.25),
                    )
                face_ratio = float(len(component.faces)) / float(total_faces)

                if within_child_region and child_score + 0.02 < parent_score and face_ratio < 0.75:
                    removed_indices.append(index)
                    relation_moves.append(
                        {
                            "component_index": index,
                            "face_count": len(component.faces),
                            "parent_score": float(parent_score),
                            "child_score": float(child_score),
                        }
                    )
                    continue

                kept_components.append(component)

            if not removed_indices:
                continue

            cleaned_mesh = trimesh.util.concatenate(kept_components)
            parent_path.write_text(cleaned_mesh.export(file_type="obj"), encoding="utf-8")
            changed_object_ids.add(parent_id)
            report["relations"].append(
                {
                    "parent_object_id": parent_id,
                    "child_object_id": child_id,
                    "relation_type": str(getattr(relation.relation_type, "value", relation.relation_type)),
                    "removed_components": relation_moves,
                    "original_component_count": len(components),
                    "cleaned_component_count": len(kept_components),
                }
            )
            report["cleaned_objects"].append(parent_id)

        if not changed_object_ids:
            return payload

        updated_poses = []
        for pose in payload.get("object_poses", ()):
            asset = asset_by_id.get(pose.object_id)
            layout_mesh_path = mesh_path_by_id.get(pose.object_id)
            if asset is None or layout_mesh_path is None or not layout_mesh_path.exists():
                updated_poses.append(pose)
                continue
            if pose.object_id not in changed_object_ids:
                updated_poses.append(pose)
                continue
            source_mesh = self._load_asset_mesh(asset)
            layout_mesh = self._load_mesh(layout_mesh_path)
            updated_poses.append(self._pose_from_loaded_meshes(pose.object_id, source_mesh, layout_mesh))

        report_dir = next(iter(mesh_path_by_id.values())).parent if mesh_path_by_id else Path(self._layout_refiner._layout_root) / scene_id
        report_path = report_dir / "mesh_cleanup.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        artifacts.append(
            ArtifactRef(
                artifact_type="layout_mesh_cleanup_report",
                path=str(report_path),
                format="json",
                role="mesh_cleanup",
                metadata_path=None,
            )
        )

        next_payload = dict(payload)
        next_payload["object_poses"] = tuple(updated_poses)
        next_payload["artifacts"] = tuple(artifacts)
        return next_payload

    def _build_observed_evidence(
        self,
        *,
        scene_id: str,
        object_assets: ObjectAssetCatalog,
        object_catalog: ObjectCatalog | None,
        segmentation_result: object | None,
        depth_result: object | None,
    ) -> dict[str, dict[str, np.ndarray]]:
        point_cloud_artifact = getattr(depth_result, "point_cloud", None)
        instances = getattr(segmentation_result, "instances", ())
        if point_cloud_artifact is None or not Path(point_cloud_artifact.path).exists() or not instances:
            return {}

        point_cloud_payload = np.load(point_cloud_artifact.path)
        point_cloud = np.asarray(point_cloud_payload["point_cloud"], dtype=float)
        if point_cloud.ndim != 3 or point_cloud.shape[-1] != 3:
            return {}

        asset_ids = {asset.object_id for asset in object_assets.assets}
        evidence_points: dict[str, list[np.ndarray]] = {}
        for instance in instances:
            object_id = self._resolve_segmentation_object_id(
                scene_id=scene_id,
                asset_ids=asset_ids,
                object_catalog=object_catalog,
                instance_id=getattr(instance, "instance_id", ""),
                label=getattr(instance, "label", None),
            )
            if object_id is None:
                continue
            mask_path = Path(instance.mask.path)
            if not mask_path.exists():
                continue
            mask = self._load_mask(mask_path, target_shape=point_cloud.shape[:2])
            instance_points = point_cloud[mask > 0]
            if instance_points.size == 0:
                continue
            finite_points = instance_points[np.isfinite(instance_points).all(axis=1)]
            if len(finite_points) == 0:
                continue
            evidence_points.setdefault(object_id, []).append(finite_points)

        evidence_by_object_id: dict[str, dict[str, np.ndarray]] = {}
        for object_id, point_sets in evidence_points.items():
            merged_points = np.concatenate(point_sets, axis=0)
            sampled_points = self._sample_points(merged_points, max_points=512)
            evidence_by_object_id[object_id] = {
                "points": sampled_points,
                "bounds": np.asarray(
                    [
                        np.min(merged_points, axis=0),
                        np.max(merged_points, axis=0),
                    ],
                    dtype=float,
                ),
            }
        return evidence_by_object_id

    def _resolve_segmentation_object_id(
        self,
        *,
        scene_id: str,
        asset_ids: set[str],
        object_catalog: ObjectCatalog | None,
        instance_id: str,
        label: str | None,
    ) -> str | None:
        for candidate in (instance_id, label or ""):
            if not candidate:
                continue
            if candidate in asset_ids:
                return candidate
            prefixed = f"{scene_id}:{candidate}"
            if prefixed in asset_ids:
                return prefixed

        if object_catalog is not None:
            for candidate in (instance_id, label or ""):
                matched = match_object_id(object_catalog, candidate)
                if matched is None:
                    continue
                if matched in asset_ids:
                    return matched
                matched_category = self._normalize_token(self._legacy_name(scene_id, matched))
                category_matches = [
                    asset_id
                    for asset_id in asset_ids
                    if self._normalize_token(self._category_name(self._legacy_name(scene_id, asset_id))) == matched_category
                ]
                if len(category_matches) == 1:
                    return category_matches[0]

        normalized_candidates = [self._normalize_token(value) for value in (instance_id, label or "") if value]
        for candidate in normalized_candidates:
            category_matches = [
                asset_id
                for asset_id in asset_ids
                if self._normalize_token(self._category_name(self._legacy_name(scene_id, asset_id))) == candidate
            ]
            if len(category_matches) == 1:
                return category_matches[0]
        return None

    def _load_mask(self, mask_path: Path, *, target_shape: tuple[int, int]) -> np.ndarray:
        mask_image = Image.open(mask_path).convert("L")
        if mask_image.size != (target_shape[1], target_shape[0]):
            mask_image = mask_image.resize((target_shape[1], target_shape[0]), Image.NEAREST)
        return np.asarray(mask_image, dtype=np.uint8)

    def _sample_points(self, points: np.ndarray, *, max_points: int) -> np.ndarray:
        if len(points) <= max_points:
            return np.asarray(points, dtype=float)
        indices = np.linspace(0, len(points) - 1, num=max_points, dtype=int)
        return np.asarray(points[indices], dtype=float)

    def _component_distance_score(self, component: trimesh.Trimesh, evidence_points: np.ndarray) -> float:
        component_points = self._sample_points(np.asarray(component.vertices, dtype=float), max_points=256)
        if component_points.size == 0 or evidence_points.size == 0:
            return float("inf")
        deltas = component_points[:, None, :] - evidence_points[None, :, :]
        distances = np.linalg.norm(deltas, axis=2)
        return float(np.mean(np.min(distances, axis=1)))

    def _point_within_bounds(
        self,
        point: np.ndarray,
        bounds: np.ndarray,
        *,
        padding: float,
    ) -> bool:
        min_bounds = np.asarray(bounds[0], dtype=float) - float(padding)
        max_bounds = np.asarray(bounds[1], dtype=float) + float(padding)
        return bool(np.all(point >= min_bounds) and np.all(point <= max_bounds))

    def _point_within_horizontal_bounds(
        self,
        point_xz: np.ndarray,
        bounds_xz: np.ndarray,
        *,
        padding: float,
    ) -> bool:
        min_bounds = np.asarray(bounds_xz[0], dtype=float) - float(padding)
        max_bounds = np.asarray(bounds_xz[1], dtype=float) + float(padding)
        return bool(np.all(point_xz >= min_bounds) and np.all(point_xz <= max_bounds))

    def _normalize_token(self, value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(value).lower())

    def _materialize_layout_mesh(
        self,
        asset: GeneratedObjectAsset,
        layout_mesh_path: Path,
    ) -> ObjectPose:
        source_mesh = self._load_asset_mesh(asset)
        target_proxy_mesh = self._load_mesh(layout_mesh_path)
        transformed_mesh = self._transform_mesh_with_legacy_layout(
            source_mesh,
            target_proxy_mesh,
        )
        layout_mesh_path.write_text(transformed_mesh.export(file_type="obj"), encoding="utf-8")
        self._copy_layout_materials(asset, layout_mesh_path)
        return self._pose_from_loaded_meshes(asset.object_id, source_mesh, transformed_mesh)

    def _pose_from_meshes(
        self,
        object_id: str,
        source_mesh_path: Path,
        layout_mesh_path: Path,
    ) -> ObjectPose:
        source_mesh = self._load_mesh(source_mesh_path)
        layout_mesh = self._load_mesh(layout_mesh_path)
        return self._pose_from_loaded_meshes(object_id, source_mesh, layout_mesh)

    def _pose_from_loaded_meshes(
        self,
        object_id: str,
        source_mesh: trimesh.Trimesh,
        layout_mesh: trimesh.Trimesh,
    ) -> ObjectPose:
        source_extents = source_mesh.bounding_box.extents
        layout_extents = layout_mesh.bounding_box.extents
        scale_xyz = []
        for source_extent, layout_extent in zip(source_extents, layout_extents):
            if abs(float(source_extent)) <= 1e-8:
                scale_xyz.append(1.0)
            else:
                scale_xyz.append(float(layout_extent) / float(source_extent))

        center = layout_mesh.bounding_box.centroid
        return ObjectPose(
            object_id=object_id,
            translation_xyz=(float(center[0]), float(center[1]), float(center[2])),
            rotation_type="quaternion",
            rotation_value=(1.0, 0.0, 0.0, 0.0),
            scale_xyz=(float(scale_xyz[0]), float(scale_xyz[1]), float(scale_xyz[2])),
        )

    def _transform_mesh_to_target_bounds(
        self,
        source_mesh: trimesh.Trimesh,
        target_mesh: trimesh.Trimesh,
    ) -> trimesh.Trimesh:
        transformed_mesh = source_mesh.copy()
        source_extents = np.asarray(source_mesh.bounding_box.extents, dtype=float)
        target_extents = np.asarray(target_mesh.bounding_box.extents, dtype=float)

        scale_xyz = np.ones(3, dtype=float)
        for index, (source_extent, target_extent) in enumerate(zip(source_extents, target_extents)):
            if float(source_extent) > 1e-8 and float(target_extent) > 0.0:
                scale_xyz[index] = float(target_extent) / float(source_extent)
        scale_matrix = np.eye(4, dtype=float)
        scale_matrix[0, 0] = float(scale_xyz[0])
        scale_matrix[1, 1] = float(scale_xyz[1])
        scale_matrix[2, 2] = float(scale_xyz[2])
        transformed_mesh.apply_transform(scale_matrix)

        transformed_center = transformed_mesh.bounding_box.centroid
        target_bounds = target_mesh.bounds
        target_center = target_mesh.bounding_box.centroid
        transformed_bounds = transformed_mesh.bounds
        translation = np.asarray(
            (
                float(target_center[0] - transformed_center[0]),
                float(target_bounds[0][1] - transformed_bounds[0][1]),
                float(target_center[2] - transformed_center[2]),
            ),
            dtype=float,
        )
        transformed_mesh.apply_translation(translation)
        return transformed_mesh

    def _transform_mesh_with_legacy_layout(
        self,
        source_mesh: trimesh.Trimesh,
        target_proxy_mesh: trimesh.Trimesh,
    ) -> trimesh.Trimesh:
        return self._transform_mesh_to_target_bounds(source_mesh, target_proxy_mesh)

    def _build_proxy_mesh(
        self,
        source_mesh: trimesh.Trimesh,
        *,
        target_extents: list[float] | None = None,
    ) -> trimesh.Trimesh:
        extents = np.asarray(source_mesh.bounding_box.extents, dtype=float)
        if target_extents:
            positive_extents = [float(value) for value in target_extents if float(value) > 0.0]
            if len(positive_extents) >= 3:
                extents = np.asarray(positive_extents[:3], dtype=float)
            elif positive_extents:
                target_longest = max(positive_extents)
                source_longest = max(float(np.max(extents)), 1e-8)
                extents = extents * (target_longest / source_longest)
        center = source_mesh.bounding_box.centroid
        proxy_mesh = trimesh.creation.box(extents=extents.tolist())
        proxy_mesh.apply_translation(center)
        return proxy_mesh

    def _resolve_target_extents_for_source_axes(
        self,
        source_mesh: trimesh.Trimesh,
        *,
        size_prior: SizePrior | None,
    ) -> list[float] | None:
        semantic_extents = self._semantic_prior_extents(size_prior)
        if semantic_extents is None:
            return None

        source_extents = np.asarray(source_mesh.bounding_box.extents, dtype=float)
        if source_extents.shape != (3,):
            return semantic_extents.tolist()

        best_full_assignment = min(
            itertools.permutations((0, 1, 2)),
            key=lambda assignment: self._assignment_score(
                source_extents=source_extents,
                source_indices=(0, 1, 2),
                semantic_extents=semantic_extents,
                semantic_indices=assignment,
            ),
        )
        return [float(semantic_extents[semantic_index]) for semantic_index in best_full_assignment]

    def _semantic_prior_extents(self, size_prior: SizePrior | None) -> np.ndarray | None:
        if size_prior is None or size_prior.dimensions_m is None:
            return None
        dimensions = size_prior.dimensions_m
        xyz_values = []
        for axis in ("x", "y", "z"):
            value = dimensions.get(axis)
            if not isinstance(value, (int, float)) or float(value) <= 0.0:
                xyz_values = []
                break
            xyz_values.append(float(value))
        if len(xyz_values) == 3:
            return np.asarray(xyz_values, dtype=float)

        fallback_values = []
        for value in dimensions.values():
            if isinstance(value, (int, float)) and float(value) > 0.0:
                fallback_values.append(float(value))
            if len(fallback_values) == 3:
                break
        if len(fallback_values) != 3:
            return None
        return np.asarray(fallback_values, dtype=float)

    def _assignment_score(
        self,
        *,
        source_extents: np.ndarray,
        source_indices: Sequence[int],
        semantic_extents: np.ndarray,
        semantic_indices: Sequence[int],
    ) -> tuple[float, float]:
        epsilon = 1e-8
        total_score = 0.0
        worst_axis_score = 0.0
        for source_index, semantic_index in zip(source_indices, semantic_indices):
            source_extent = max(float(source_extents[source_index]), epsilon)
            semantic_extent = max(float(semantic_extents[semantic_index]), epsilon)
            axis_score = abs(float(np.log(semantic_extent / source_extent)))
            total_score += axis_score
            worst_axis_score = max(worst_axis_score, axis_score)
        return (worst_axis_score, total_score)

    def _copy_layout_materials(
        self,
        asset: GeneratedObjectAsset,
        layout_mesh_path: Path,
    ) -> None:
        scene_dir = layout_mesh_path.parent
        legacy_name = layout_mesh_path.stem
        fallback_texture_path = (
            Path(asset.texture_image.path)
            if asset.texture_image is not None and Path(asset.texture_image.path).exists()
            else None
        )
        mtl_filename = f"{legacy_name}.mtl"
        if asset.mesh_mtl is not None and Path(asset.mesh_mtl.path).exists():
            rewritten_mtl = self._rewrite_material_library(
                source_mtl_path=Path(asset.mesh_mtl.path),
                scene_dir=scene_dir,
                legacy_name=legacy_name,
                fallback_texture_path=fallback_texture_path,
            )
            (scene_dir / mtl_filename).write_text(rewritten_mtl, encoding="utf-8")
        else:
            self._write_placeholder_materials(scene_dir, legacy_name)
            if fallback_texture_path is not None:
                shutil.copyfile(
                    fallback_texture_path,
                    scene_dir / f"{legacy_name}{fallback_texture_path.suffix or '.png'}",
                )

        obj_text = layout_mesh_path.read_text(encoding="utf-8", errors="ignore")
        layout_mesh_path.write_text(
            self._rewrite_obj_material_library(obj_text, mtl_filename),
            encoding="utf-8",
        )

    def _rewrite_obj_material_library(self, obj_text: str, mtl_filename: str) -> str:
        rewritten_lines: list[str] = []
        replaced = False
        for line in obj_text.splitlines():
            if line.strip().startswith("mtllib "):
                rewritten_lines.append(f"mtllib {mtl_filename}")
                replaced = True
                continue
            rewritten_lines.append(line)
        if not replaced:
            rewritten_lines.insert(0, f"mtllib {mtl_filename}")
        return "\n".join(rewritten_lines).rstrip() + "\n"

    def _rewrite_material_library(
        self,
        *,
        source_mtl_path: Path,
        scene_dir: Path,
        legacy_name: str,
        fallback_texture_path: Path | None,
    ) -> str:
        source_text = source_mtl_path.read_text(encoding="utf-8", errors="ignore")
        fallback_texture_name: str | None = None
        used_texture_mapping = False
        rewritten_lines: list[str] = []

        def ensure_fallback_texture() -> str | None:
            nonlocal fallback_texture_name
            if fallback_texture_path is None:
                return None
            if fallback_texture_name is None:
                fallback_texture_name = f"{legacy_name}{fallback_texture_path.suffix or '.png'}"
                shutil.copyfile(fallback_texture_path, scene_dir / fallback_texture_name)
            return fallback_texture_name

        for line in source_text.splitlines():
            stripped = line.strip()
            if stripped.startswith(("map_Kd ", "map_Ka ", "map_Ks ", "map_d ")):
                parts = line.split()
                source_texture_name = parts[-1] if len(parts) >= 2 else None
                if source_texture_name is not None:
                    source_texture_path = source_mtl_path.parent / source_texture_name
                    if source_texture_path.exists():
                        target_texture_name = f"{legacy_name}__{source_texture_path.name}"
                        shutil.copyfile(source_texture_path, scene_dir / target_texture_name)
                        parts[-1] = target_texture_name
                        rewritten_lines.append(" ".join(parts))
                        used_texture_mapping = True
                        continue

                    fallback_name = ensure_fallback_texture()
                    if fallback_name is not None:
                        parts[-1] = fallback_name
                        rewritten_lines.append(" ".join(parts))
                        used_texture_mapping = True
                        continue

            rewritten_lines.append(line)

        if not used_texture_mapping:
            fallback_name = ensure_fallback_texture()
            if fallback_name is not None:
                rewritten_lines.append(f"map_Kd {fallback_name}")

        return "\n".join(rewritten_lines).rstrip() + "\n"

    def _load_mesh(self, mesh_path: Path) -> trimesh.Trimesh:
        try:
            loaded = self._mesh_loader(mesh_path, force="mesh")
        except TypeError:
            loaded = self._mesh_loader(mesh_path)
        if isinstance(loaded, trimesh.Scene):
            return loaded.dump(concatenate=True)
        return loaded.copy()

    def _load_asset_mesh(self, asset: GeneratedObjectAsset) -> trimesh.Trimesh:
        mesh = self._load_mesh(Path(asset.mesh_obj.path))
        apply_pose_to_mesh(mesh, asset.asset_local_pose)
        return mesh

    def _mesh_extent_values(self, mesh_path: Path) -> list[float]:
        mesh = self._load_mesh(mesh_path)
        extents = mesh.bounding_box.extents.tolist()
        values = [float(abs(value)) for value in extents if float(abs(value)) > 0.0]
        return values or [1.0, 1.0, 1.0]

    def _mesh_extent_values_for_asset(self, asset: GeneratedObjectAsset) -> list[float]:
        mesh = self._load_asset_mesh(asset)
        extents = mesh.bounding_box.extents.tolist()
        values = [float(abs(value)) for value in extents if float(abs(value)) > 0.0]
        return values or [1.0, 1.0, 1.0]

    def _normalize_prior_values(self, size_prior: SizePrior | None) -> list[float] | None:
        if size_prior is None or size_prior.dimensions_m is None:
            return None
        values = []
        for key in ("x", "y", "z", "width", "height", "depth", "length"):
            value = size_prior.dimensions_m.get(key)
            if isinstance(value, (int, float)) and value > 0:
                values.append(float(value))
        return values or None

    def _write_placeholder_materials(self, scene_dir: Path, legacy_name: str) -> None:
        mtl_path = scene_dir / f"{legacy_name}.mtl"
        png_path = scene_dir / f"{legacy_name}.png"
        if not mtl_path.exists():
            mtl_path.write_text(
                "\n".join(
                    [
                        f"newmtl {legacy_name}",
                        "Ka 1.000000 1.000000 1.000000",
                        "Kd 1.000000 1.000000 1.000000",
                        "Ks 0.000000 0.000000 0.000000",
                        f"map_Kd {legacy_name}.png",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
        if not png_path.exists():
            Image.new("RGB", (1, 1), color=(255, 255, 255)).save(png_path)

    def _legacy_name(self, scene_id: str, object_id: str) -> str:
        prefix = f"{scene_id}:"
        if object_id.startswith(prefix):
            return object_id[len(prefix) :]
        return object_id.split(":")[-1]

    def _category_name(self, legacy_name: str) -> str:
        return legacy_name.rstrip("0123456789").strip("_") or legacy_name

    def _run_bounded_layout(
        self,
        args: object,
        scene_id: str,
        size_ratio: dict[str, list[float]],
    ) -> Path:
        scene_dir = Path(args.layout_folder) / scene_id
        log_path = scene_dir / "legacy_layout.log"
        worker_log_dir = self._worker_log_root / scene_id
        if worker_log_dir.exists():
            shutil.rmtree(worker_log_dir, ignore_errors=True)
        worker_log_dir.mkdir(parents=True, exist_ok=True)
        worker_log_path = worker_log_dir / "legacy_layout.log"
        context = mp.get_context("fork")
        status_queue = context.Queue()
        process = context.Process(
            target=_run_layout_worker,
            kwargs={
                "layout_initializer": self._layout_initializer,
                "ground_resetter": self._ground_resetter,
                "organized_obj_folder": args.organized_obj_folder,
                "depth_folder": args.depth_folder,
                "seg_folder": args.seg_folder,
                "scene_id": scene_id,
                "layout_folder": args.layout_folder,
                "layout_front_num": args.layout_front_num,
                "size_ratio": size_ratio,
                "log_path": str(worker_log_path),
                "status_queue": status_queue,
            },
        )
        process.start()
        process.join(self._timeout_seconds)
        if process.is_alive():
            process.terminate()
            process.join()
            with open(worker_log_path, "a", encoding="utf-8") as log_handle:
                log_handle.write(
                    f"parent_timeout: legacy layout adapter exceeded {self._timeout_seconds} seconds\n"
                )
            self._publish_worker_log(worker_log_path, log_path)
            raise TimeoutError(
                f"legacy layout adapter timed out after {self._timeout_seconds} seconds; see {log_path}"
            )

        if status_queue.empty():
            with open(worker_log_path, "a", encoding="utf-8") as log_handle:
                log_handle.write("parent_error: legacy layout adapter exited without status\n")
            self._publish_worker_log(worker_log_path, log_path)
            raise RuntimeError(f"legacy layout adapter exited without status; see {log_path}")
        status, message = status_queue.get()
        if status != "ok":
            with open(worker_log_path, "a", encoding="utf-8") as log_handle:
                log_handle.write(f"parent_error: {message}\n")
            self._publish_worker_log(worker_log_path, log_path)
            raise RuntimeError(f"legacy layout adapter failed: {message}; see {log_path}")
        self._publish_worker_log(worker_log_path, log_path)
        return log_path

    def _publish_worker_log(self, worker_log_path: Path, final_log_path: Path) -> None:
        if not worker_log_path.exists():
            return
        final_log_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(worker_log_path, final_log_path)
