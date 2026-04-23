from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import trimesh

from pat3d.models import (
    ArtifactRef,
    DepthResult,
    GeneratedObjectAsset,
    ObjectAssetCatalog,
    ObjectCatalog,
    ObjectPose,
    ReferenceImageResult,
    SceneLayout,
    SceneRelationGraph,
    SegmentationResult,
    SizePrior,
)
from pat3d.models.pose_utils import apply_pose_to_mesh
from pat3d.providers._layout_circle_packing import pack_circle_centers_in_bounds
from pat3d.storage import make_stage_metadata


class LegacyLayoutPlanner:
    def __init__(
        self,
        *,
        planner: Callable[..., SceneLayout | dict | Sequence[ObjectPose]] | None = None,
        default_planner: Callable[..., SceneLayout | dict | Sequence[ObjectPose]] | None = None,
        layout_space: str = "world",
        metadata_factory: Callable[..., object] = make_stage_metadata,
        legacy_root: str = "pat3d/legacy_vendor/mask_loss",
        layout_root: str = "data/layout",
        legacy_arg_overrides: dict[str, str | int | float | bool] | None = None,
    ) -> None:
        self._planner = planner
        self._default_planner = default_planner
        self._layout_space = layout_space
        self._metadata_factory = metadata_factory
        self._legacy_root = legacy_root
        self._layout_root = layout_root
        self._legacy_arg_overrides = dict(legacy_arg_overrides or {})

    def build_layout(
        self,
        *,
        scene_id: str,
        object_assets: ObjectAssetCatalog,
        object_catalog: ObjectCatalog | None = None,
        reference_image_result: ReferenceImageResult | None = None,
        relation_graph: SceneRelationGraph | None = None,
        size_priors: Sequence[SizePrior] = (),
        depth_result: DepthResult | None = None,
        segmentation_result: SegmentationResult | None = None,
    ) -> SceneLayout:
        try:
            planner = self._planner or self._default_planner
            if planner is None:
                planner = self._make_default_planner()
            result = planner(
                scene_id=scene_id,
                object_assets=object_assets,
                object_catalog=object_catalog,
                reference_image_result=reference_image_result,
                relation_graph=relation_graph,
                size_priors=size_priors,
                depth_result=depth_result,
                segmentation_result=segmentation_result,
            )
        except Exception as exc:
            return self._fallback_layout(
                scene_id,
                object_assets,
                relation_graph,
                size_priors=size_priors,
                error=exc,
            )
        if isinstance(result, SceneLayout):
            return result
        if isinstance(result, dict):
            object_poses = tuple(result.get("object_poses", ()))
            artifacts = tuple(result.get("artifacts", ()))
            layout_space = result.get("layout_space", self._layout_space)
            return SceneLayout(
                scene_id=scene_id,
                object_poses=object_poses,
                layout_space=layout_space,
                support_graph=relation_graph,
                artifacts=artifacts,
                metadata=self._metadata_factory(
                    stage_name="layout_initialization",
                    provider_name="legacy_bbox_layout",
                    notes=("bridged_from_legacy",),
                ),
            )
        return SceneLayout(
            scene_id=scene_id,
            object_poses=tuple(result),
            layout_space=self._layout_space,
            support_graph=relation_graph,
            artifacts=(),
            metadata=self._metadata_factory(
                stage_name="layout_initialization",
                provider_name="legacy_bbox_layout",
                notes=("bridged_from_legacy",),
            ),
        )

    def _fallback_layout(
        self,
        scene_id: str,
        object_assets: ObjectAssetCatalog,
        relation_graph: SceneRelationGraph | None,
        *,
        size_priors: Sequence[SizePrior] = (),
        error: Exception | None = None,
    ) -> SceneLayout:
        poses = self._fallback_poses(
            object_assets=object_assets,
            relation_graph=relation_graph,
            size_priors=size_priors,
        )
        legacy_artifact = ArtifactRef(
            artifact_type="legacy_layout_reference",
            path=str(Path(self._legacy_root) / "modules" / "bbox_put_utils" / "put_obj.py"),
            format="py",
            role="legacy_source",
            metadata_path=None,
        )
        artifacts = [legacy_artifact]
        layout_log_path = Path(self._layout_root) / scene_id / "legacy_layout.log"
        if layout_log_path.exists():
            artifacts.append(
                ArtifactRef(
                    artifact_type="legacy_layout_log",
                    path=str(layout_log_path),
                    format="log",
                    role="layout_log",
                    metadata_path=None,
                )
            )
        notes = ["fallback_support_layout" if relation_graph and relation_graph.relations else "fallback_linear_layout"]
        if error is not None:
            notes.append(f"fallback_reason={type(error).__name__}: {error}")
        return SceneLayout(
            scene_id=scene_id,
            object_poses=tuple(poses),
            layout_space=self._layout_space,
            support_graph=relation_graph,
            artifacts=tuple(artifacts),
            metadata=self._metadata_factory(
                stage_name="layout_initialization",
                provider_name="legacy_bbox_layout_fallback",
                notes=tuple(notes),
            ),
        )

    def _fallback_poses(
        self,
        *,
        object_assets: ObjectAssetCatalog,
        relation_graph: SceneRelationGraph | None,
        size_priors: Sequence[SizePrior],
    ) -> tuple[ObjectPose, ...]:
        specs = self._fallback_specs(object_assets, size_priors)
        children_by_parent = self._support_children_by_parent(relation_graph)
        child_ids = {child_id for child_ids in children_by_parent.values() for child_id in child_ids}
        ordered_asset_ids = [asset.object_id for asset in object_assets.assets]
        root_ids = [
            object_id
            for object_id in (
                *(relation_graph.root_object_ids if relation_graph is not None else ()),
                *ordered_asset_ids,
            )
            if object_id in specs and object_id not in child_ids
        ]
        seen_roots: set[str] = set()
        deduped_root_ids: list[str] = []
        for object_id in root_ids:
            if object_id in seen_roots:
                continue
            seen_roots.add(object_id)
            deduped_root_ids.append(object_id)

        placements: dict[str, np.ndarray] = {}
        root_spacing = 0.12
        layer_gap = 0.003
        cursor_x = 0.0
        pending_roots = deduped_root_ids or ordered_asset_ids
        for root_index, root_id in enumerate(pending_roots):
            if root_id in placements or root_id not in specs:
                continue
            extents = specs[root_id]["extents"]
            if len(pending_roots) == 1:
                center_x = 0.0
            else:
                center_x = cursor_x + (float(extents[0]) * 0.5)
            center = np.array([center_x, float(extents[1]) * 0.5, 0.0], dtype=float)
            placements[root_id] = center
            cursor_x = center_x + (float(extents[0]) * 0.5) + root_spacing
            self._place_supported_descendants(
                parent_id=root_id,
                placements=placements,
                specs=specs,
                children_by_parent=children_by_parent,
                layer_gap=layer_gap,
            )

        for object_id in ordered_asset_ids:
            if object_id in placements or object_id not in specs:
                continue
            extents = specs[object_id]["extents"]
            center_x = cursor_x + (float(extents[0]) * 0.5)
            placements[object_id] = np.array([center_x, float(extents[1]) * 0.5, 0.0], dtype=float)
            cursor_x = center_x + (float(extents[0]) * 0.5) + root_spacing

        poses: list[ObjectPose] = []
        for asset in object_assets.assets:
            spec = specs[asset.object_id]
            center = placements[asset.object_id]
            uniform_scale = float(spec["uniform_scale"])
            poses.append(
                ObjectPose(
                    object_id=asset.object_id,
                    translation_xyz=(float(center[0]), float(center[1]), float(center[2])),
                    rotation_type="quaternion",
                    rotation_value=(1.0, 0.0, 0.0, 0.0),
                    scale_xyz=(uniform_scale, uniform_scale, uniform_scale),
                )
            )
        return tuple(poses)

    def _place_supported_descendants(
        self,
        *,
        parent_id: str,
        placements: dict[str, np.ndarray],
        specs: dict[str, dict[str, np.ndarray | float]],
        children_by_parent: dict[str, list[str]],
        layer_gap: float,
    ) -> None:
        parent_center = placements.get(parent_id)
        if parent_center is None:
            return
        child_ids = [child_id for child_id in children_by_parent.get(parent_id, []) if child_id in specs]
        if not child_ids:
            return

        parent_extents = np.asarray(specs[parent_id]["extents"], dtype=float)
        preferred_centers = []
        radii = []
        for child_index, child_id in enumerate(child_ids):
            child_extents = np.asarray(specs[child_id]["extents"], dtype=float)
            theta = (2.0 * np.pi * float(child_index)) / max(len(child_ids), 1)
            preferred_centers.append(
                np.array(
                    [
                        float(parent_center[0] + np.cos(theta) * max(parent_extents[0] * 0.12, 0.0)),
                        float(parent_center[2] + np.sin(theta) * max(parent_extents[2] * 0.12, 0.0)),
                    ],
                    dtype=float,
                )
            )
            radii.append(float(np.linalg.norm(child_extents[[0, 2]]) * 0.5) * 1.05)

        container_bounds = np.array(
            [
                [
                    float(parent_center[0] - (parent_extents[0] * 0.5)),
                    float(parent_center[2] - (parent_extents[2] * 0.5)),
                ],
                [
                    float(parent_center[0] + (parent_extents[0] * 0.5)),
                    float(parent_center[2] + (parent_extents[2] * 0.5)),
                ],
            ],
            dtype=float,
        )
        packed_centers = pack_circle_centers_in_bounds(
            np.asarray(preferred_centers, dtype=float),
            np.asarray(radii, dtype=float),
            bounds_xz=container_bounds,
            gap=layer_gap,
        )
        if not self._packed_children_satisfy_separation(packed_centers, radii, gap=layer_gap):
            packed_centers = self._pack_children_with_overflow(
                parent_center=parent_center,
                parent_extents=parent_extents,
                radii=radii,
                gap=layer_gap,
            )

        for child_id, packed_center in zip(child_ids, packed_centers, strict=False):
            child_extents = np.asarray(specs[child_id]["extents"], dtype=float)
            child_center = np.array(
                [
                    float(packed_center[0]),
                    float(parent_center[1] + (parent_extents[1] * 0.5) + (child_extents[1] * 0.5) + layer_gap),
                    float(packed_center[1]),
                ],
                dtype=float,
            )
            placements[child_id] = child_center
            self._place_supported_descendants(
                parent_id=child_id,
                placements=placements,
                specs=specs,
                children_by_parent=children_by_parent,
                layer_gap=layer_gap,
            )

    def _packed_children_satisfy_separation(
        self,
        centers_xz: np.ndarray,
        radii: Sequence[float],
        *,
        gap: float,
    ) -> bool:
        centers = np.asarray(centers_xz, dtype=float)
        if len(centers) <= 1:
            return True
        normalized_radii = [max(float(radius), 0.0) for radius in radii]
        safe_gap = max(float(gap), 0.0)
        for left_index in range(len(centers)):
            for right_index in range(left_index + 1, len(centers)):
                required_distance = normalized_radii[left_index] + normalized_radii[right_index] + safe_gap
                actual_distance = float(np.linalg.norm(centers[left_index] - centers[right_index]))
                if actual_distance < required_distance - 1e-8:
                    return False
        return True

    def _pack_children_with_overflow(
        self,
        *,
        parent_center: np.ndarray,
        parent_extents: np.ndarray,
        radii: Sequence[float],
        gap: float,
    ) -> np.ndarray:
        normalized_radii = [max(float(radius), 0.0) for radius in radii]
        if not normalized_radii:
            return np.empty((0, 2), dtype=float)

        dominant_axis = 0 if float(parent_extents[0]) >= float(parent_extents[2]) else 1
        orthogonal_value = float(parent_center[2] if dominant_axis == 0 else parent_center[0])
        safe_gap = max(float(gap), 0.0)
        total_span = sum(radius * 2.0 for radius in normalized_radii) + safe_gap * max(len(normalized_radii) - 1, 0)
        cursor = float(parent_center[0] if dominant_axis == 0 else parent_center[2]) - (total_span * 0.5)

        overflow_centers: list[np.ndarray] = []
        for radius in normalized_radii:
            axis_value = cursor + radius
            if dominant_axis == 0:
                overflow_centers.append(np.array([axis_value, orthogonal_value], dtype=float))
            else:
                overflow_centers.append(np.array([orthogonal_value, axis_value], dtype=float))
            cursor = axis_value + radius + safe_gap
        return np.asarray(overflow_centers, dtype=float)

    def _support_children_by_parent(
        self,
        relation_graph: SceneRelationGraph | None,
    ) -> dict[str, list[str]]:
        mapping: dict[str, list[str]] = defaultdict(list)
        if relation_graph is None:
            return mapping
        for relation in relation_graph.relations:
            relation_type = str(getattr(relation.relation_type, "value", relation.relation_type)).strip().lower()
            # Keep fallback placement behavior aligned with the main layout pass:
            # containment is treated as a parent-child stacking relation here too.
            if relation_type not in {"supports", "on", "contains", "in"}:
                continue
            mapping[relation.parent_object_id].append(relation.child_object_id)
        return mapping

    def _fallback_specs(
        self,
        object_assets: ObjectAssetCatalog,
        size_priors: Sequence[SizePrior],
    ) -> dict[str, dict[str, np.ndarray | float]]:
        prior_by_id = {prior.object_id: prior for prior in size_priors}
        specs: dict[str, dict[str, np.ndarray | float]] = {}
        for asset in object_assets.assets:
            mesh = self._load_asset_mesh(asset)
            source_extents = np.asarray(mesh.bounding_box.extents, dtype=float)
            source_longest = max(float(np.max(source_extents)), 1e-8)
            target_longest = self._target_longest_extent(prior_by_id.get(asset.object_id)) or source_longest
            uniform_scale = max(target_longest / source_longest, 1e-6)
            specs[asset.object_id] = {
                "extents": source_extents * uniform_scale,
                "uniform_scale": uniform_scale,
            }
        return specs

    def _target_longest_extent(self, size_prior: SizePrior | None) -> float | None:
        if size_prior is None or size_prior.dimensions_m is None:
            return None
        values: list[float] = []
        for key in ("x", "y", "z", "width", "height", "depth", "length"):
            value = size_prior.dimensions_m.get(key)
            if isinstance(value, (int, float)) and float(value) > 0.0:
                values.append(float(value))
        if not values:
            return None
        return max(values)

    def _load_asset_mesh(self, asset: GeneratedObjectAsset) -> trimesh.Trimesh:
        mesh_path = Path(asset.mesh_obj.path)
        if not mesh_path.exists():
            mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
            apply_pose_to_mesh(mesh, asset.asset_local_pose)
            return mesh
        loaded = trimesh.load(mesh_path, force="mesh")
        if isinstance(loaded, trimesh.Scene):
            mesh = loaded.dump(concatenate=True)
        else:
            mesh = loaded.copy()
        apply_pose_to_mesh(mesh, asset.asset_local_pose)
        return mesh

    def _make_default_planner(self) -> Callable[..., SceneLayout | dict | Sequence[ObjectPose]]:
        from pat3d.providers._legacy_layout_adapter import LegacyInitialLayoutAdapter

        return LegacyInitialLayoutAdapter(
            legacy_arg_overrides=self._legacy_arg_overrides,
        ).build
