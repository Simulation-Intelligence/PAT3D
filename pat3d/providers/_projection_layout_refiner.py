from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import json
from pathlib import Path
from typing import Callable

import numpy as np
import trimesh

from pat3d.models import ArtifactRef, ObjectPose, SceneRelationGraph
from pat3d.providers._layout_circle_packing import pack_circle_centers_in_bounds
from pat3d.providers._relation_utils import is_parent_child_relation_type


@dataclass
class _LayoutMeshState:
    object_id: str
    mesh_path: Path
    mesh: trimesh.Trimesh
    pose: ObjectPose

    @property
    def bounds(self) -> np.ndarray:
        return np.asarray(self.mesh.bounds, dtype=float)

    @property
    def center(self) -> np.ndarray:
        bounds = self.bounds
        return (bounds[0] + bounds[1]) * 0.5

    @property
    def horizontal_min(self) -> np.ndarray:
        bounds = self.bounds
        return bounds[0, [0, 2]]

    @property
    def horizontal_max(self) -> np.ndarray:
        bounds = self.bounds
        return bounds[1, [0, 2]]

    @property
    def vertical_min(self) -> float:
        return float(self.bounds[0, 1])

    @property
    def vertical_max(self) -> float:
        return float(self.bounds[1, 1])

    def translate(self, delta: np.ndarray) -> None:
        if float(np.linalg.norm(delta)) <= 1e-10:
            return
        self.mesh.apply_translation(delta)

    @property
    def horizontal_center(self) -> np.ndarray:
        center = self.center
        return np.array([float(center[0]), float(center[2])], dtype=float)

    @property
    def horizontal_radius(self) -> float:
        vertices = np.asarray(self.mesh.vertices, dtype=float)
        if vertices.size == 0:
            return 0.0
        center = self.horizontal_center
        distances = np.linalg.norm(vertices[:, [0, 2]] - center, axis=1)
        return float(np.max(distances)) * 1.05


class ProjectionAwareLayoutRefiner:
    def __init__(
        self,
        *,
        layout_root: str = "data/layout",
        gap_xy: float = 0.08,
        layer_gap: float = 0.003,
        max_iterations: int = 24,
        containment_fit_tolerance: float = 0.05,
        mesh_loader: Callable[..., trimesh.Trimesh | trimesh.Scene] = trimesh.load,
    ) -> None:
        self._layout_root = Path(layout_root)
        self._gap_xy = float(gap_xy)
        self._layer_gap = float(layer_gap)
        self._max_iterations = int(max_iterations)
        self._containment_fit_tolerance = float(containment_fit_tolerance)
        self._mesh_loader = mesh_loader

    def refine(
        self,
        *,
        scene_id: str,
        payload: dict[str, object],
        relation_graph: SceneRelationGraph | None = None,
    ) -> dict[str, object]:
        object_poses = tuple(payload.get("object_poses", ()))
        artifacts = list(payload.get("artifacts", ()))
        if not object_poses:
            return payload

        mesh_states = self._load_mesh_states(object_poses, artifacts)
        if not mesh_states:
            return payload

        groups = self._build_groups(tuple(mesh_states.keys()), relation_graph)
        managed_parent_ids = self._managed_relation_parent_ids(relation_graph)
        report: dict[str, object] = {
            "scene_id": scene_id,
            "groups": [],
            "moves": [],
        }

        for parent_id, object_ids in groups:
            if parent_id in managed_parent_ids:
                continue
            if len(object_ids) <= 1:
                continue
            container_bounds = self._container_bounds(parent_id, mesh_states)
            moves = self._resolve_group(
                [mesh_states[object_id] for object_id in object_ids if object_id in mesh_states],
                container_bounds=container_bounds,
            )
            report["groups"].append(
                {
                    "parent_object_id": parent_id,
                    "object_ids": list(object_ids),
                    "container_bounds_xz": list(container_bounds.reshape(-1)) if container_bounds is not None else None,
                    "move_count": len(moves),
                }
            )
            report["moves"].extend(moves)

        report["moves"].extend(
            self._resolve_global_overlaps(
                list(mesh_states.values()),
                relation_graph=relation_graph,
                ignored_pairs=self._relation_overlap_exempt_pairs(relation_graph),
            )
        )
        report["moves"].extend(
            self._project_relation_groups(
                mesh_states,
                relation_graph=relation_graph,
                report=report,
            )
        )

        for state in mesh_states.values():
            state.mesh_path.write_text(state.mesh.export(file_type="obj"), encoding="utf-8")

        report_path = self._layout_root / scene_id / "projection_refinement.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        artifacts.append(
            ArtifactRef(
                artifact_type="layout_refinement_report",
                path=str(report_path),
                format="json",
                role="projection_refinement",
                metadata_path=None,
            )
        )

        updated_poses = tuple(self._updated_pose(mesh_states[pose.object_id], pose) for pose in object_poses if pose.object_id in mesh_states)
        next_payload = dict(payload)
        next_payload["object_poses"] = updated_poses
        next_payload["artifacts"] = tuple(artifacts)
        return next_payload

    def _load_mesh_states(
        self,
        object_poses: tuple[ObjectPose, ...],
        artifacts: list[ArtifactRef],
    ) -> dict[str, _LayoutMeshState]:
        mesh_path_by_id: dict[str, Path] = {}
        for artifact in artifacts:
            if artifact.artifact_type != "layout_mesh" or not artifact.role:
                continue
            mesh_path_by_id[artifact.role] = Path(artifact.path)

        states: dict[str, _LayoutMeshState] = {}
        for pose in object_poses:
            mesh_path = mesh_path_by_id.get(pose.object_id)
            if mesh_path is None or not mesh_path.exists():
                continue
            mesh = self._load_mesh(mesh_path)
            states[pose.object_id] = _LayoutMeshState(
                object_id=pose.object_id,
                mesh_path=mesh_path,
                mesh=mesh,
                pose=pose,
            )
        return states

    def _build_groups(
        self,
        object_ids: tuple[str, ...],
        relation_graph: SceneRelationGraph | None,
    ) -> list[tuple[str | None, tuple[str, ...]]]:
        if relation_graph is None or not relation_graph.relations:
            return [(None, object_ids)]

        children: set[str] = set()
        children_by_parent: dict[str, list[str]] = {}
        for relation in relation_graph.relations:
            children.add(relation.child_object_id)
            children_by_parent.setdefault(relation.parent_object_id, []).append(relation.child_object_id)

        groups: list[tuple[str | None, tuple[str, ...]]] = []
        roots = tuple(object_id for object_id in object_ids if object_id not in children)
        if roots:
            groups.append((None, roots))
        for parent_id, child_ids in children_by_parent.items():
            filtered = tuple(object_id for object_id in child_ids if object_id in object_ids)
            if filtered:
                groups.append((parent_id, filtered))
        return groups

    def _container_bounds(
        self,
        parent_id: str | None,
        mesh_states: dict[str, _LayoutMeshState],
    ) -> np.ndarray | None:
        if parent_id is None or parent_id not in mesh_states:
            return None
        bounds = mesh_states[parent_id].bounds
        return bounds[:, [0, 2]].copy()

    def _managed_relation_parent_ids(
        self,
        relation_graph: SceneRelationGraph | None,
    ) -> set[str]:
        if relation_graph is None:
            return set()
        return {
            relation.parent_object_id
            for relation in relation_graph.relations
            if is_parent_child_relation_type(relation.relation_type)
        }

    def _resolve_group(
        self,
        states: list[_LayoutMeshState],
        *,
        container_bounds: np.ndarray | None,
    ) -> list[dict[str, object]]:
        if len(states) <= 1:
            return []

        moves: list[dict[str, object]] = []
        if container_bounds is not None:
            moves.extend(self._pack_group_with_outer_spheres(states, container_bounds))
        for _ in range(self._max_iterations):
            changed = False
            states.sort(key=lambda state: (float(state.center[0]), float(state.center[2]), state.object_id))
            for left_index in range(len(states)):
                for right_index in range(left_index + 1, len(states)):
                    left_state = states[left_index]
                    right_state = states[right_index]
                    overlap_xz = self._horizontal_overlap(left_state, right_state)
                    if overlap_xz is None:
                        continue
                    if not self._vertical_overlap(left_state, right_state):
                        continue
                    delta = self._separation_delta(left_state, right_state, overlap_xz)
                    right_state.translate(delta)
                    self._clamp_to_container(right_state, container_bounds)
                    moves.append(
                        {
                            "object_id": right_state.object_id,
                            "kind": "horizontal_separation",
                            "delta_xyz": [float(delta[0]), float(delta[1]), float(delta[2])],
                        }
                    )
                    changed = True
            if not changed:
                break

            states.sort(key=lambda state: (float(state.center[0]), float(state.center[2]), state.object_id))
        for left_index in range(len(states)):
            for right_index in range(left_index + 1, len(states)):
                left_state = states[left_index]
                right_state = states[right_index]
                overlap_xz = self._horizontal_overlap(left_state, right_state)
                if overlap_xz is None:
                    continue
                if not self._vertical_overlap(left_state, right_state):
                    continue
                lift = np.array([0.0, left_state.vertical_max - right_state.vertical_min + self._layer_gap, 0.0], dtype=float)
                right_state.translate(lift)
                moves.append(
                    {
                        "object_id": right_state.object_id,
                        "kind": "vertical_layer_lift",
                        "delta_xyz": [0.0, float(lift[1]), 0.0],
                    }
                )

        return moves

    def _pack_group_with_outer_spheres(
        self,
        states: list[_LayoutMeshState],
        container_bounds: np.ndarray,
    ) -> list[dict[str, object]]:
        if len(states) <= 1:
            return []

        preferred_centers = np.asarray([state.horizontal_center for state in states], dtype=float)
        radii = np.asarray([state.horizontal_radius for state in states], dtype=float)
        packed_centers = pack_circle_centers_in_bounds(
            preferred_centers,
            radii,
            bounds_xz=container_bounds,
            gap=self._gap_xy,
        )

        moves: list[dict[str, object]] = []
        for state, packed_center in zip(states, packed_centers, strict=False):
            delta_xz = packed_center - state.horizontal_center
            delta = np.array([float(delta_xz[0]), 0.0, float(delta_xz[1])], dtype=float)
            if float(np.linalg.norm(delta)) <= 1e-10:
                continue
            state.translate(delta)
            moves.append(
                {
                    "object_id": state.object_id,
                    "kind": "sphere_packing_reposition",
                    "delta_xyz": [float(delta[0]), 0.0, float(delta[2])],
                }
            )
        return moves

    def _resolve_global_overlaps(
        self,
        states: list[_LayoutMeshState],
        *,
        relation_graph: SceneRelationGraph | None,
        ignored_pairs: set[frozenset[str]] | None = None,
    ) -> list[dict[str, object]]:
        if len(states) <= 1:
            return []

        exempt_pairs = ignored_pairs or set()
        moves: list[dict[str, object]] = []
        for _ in range(self._max_iterations):
            changed = False
            states.sort(key=lambda state: (float(state.center[0]), float(state.center[2]), state.object_id))
            for left_index in range(len(states)):
                for right_index in range(left_index + 1, len(states)):
                    left_state = states[left_index]
                    right_state = states[right_index]
                    if frozenset((left_state.object_id, right_state.object_id)) in exempt_pairs:
                        continue
                    overlap_xz = self._horizontal_overlap(left_state, right_state)
                    if overlap_xz is None:
                        continue
                    if not self._vertical_overlap(left_state, right_state):
                        continue
                    delta = self._separation_delta(left_state, right_state, overlap_xz)
                    right_state.translate(delta)
                    moves.append(
                        {
                            "object_id": right_state.object_id,
                            "kind": "global_horizontal_separation",
                            "delta_xyz": [float(delta[0]), float(delta[1]), float(delta[2])],
                        }
                    )
                    changed = True
            if not changed:
                break

        for left_index in range(len(states)):
            for right_index in range(left_index + 1, len(states)):
                left_state = states[left_index]
                right_state = states[right_index]
                if frozenset((left_state.object_id, right_state.object_id)) in exempt_pairs:
                    continue
                overlap_xz = self._horizontal_overlap(left_state, right_state)
                if overlap_xz is None:
                    continue
                if not self._vertical_overlap(left_state, right_state):
                    continue
                lift = np.array(
                    [0.0, left_state.vertical_max - right_state.vertical_min + self._layer_gap, 0.0],
                    dtype=float,
                )
                right_state.translate(lift)
                moves.append(
                    {
                        "object_id": right_state.object_id,
                        "kind": "global_vertical_layer_lift",
                        "delta_xyz": [0.0, float(lift[1]), 0.0],
                    }
                )

        return moves

    def _project_relation_groups(
        self,
        mesh_states: dict[str, _LayoutMeshState],
        *,
        relation_graph: SceneRelationGraph | None,
        report: dict[str, object] | None = None,
    ) -> list[dict[str, object]]:
        relation_groups = self._relation_groups(relation_graph, mesh_states)
        if not relation_groups:
            return []

        moves: list[dict[str, object]] = []
        for relation_group in relation_groups:
            parent_id = relation_group["parent_id"]
            child_ids = relation_group["child_ids"]
            move_kind = relation_group["move_kind"]
            parent_state = mesh_states.get(parent_id)
            if parent_state is None:
                continue
            child_states = [mesh_states[child_id] for child_id in child_ids if child_id in mesh_states]
            if not child_states:
                continue

            preferred_centers = {
                state.object_id: state.horizontal_center.copy()
                for state in child_states
            }
            moves.extend(
                self._resolve_group(
                    child_states,
                    container_bounds=self._container_bounds(parent_id, mesh_states),
                )
            )

            layer_indices = self._compute_sublayer_indices(child_states)
            current_layer_top = float(parent_state.vertical_max)
            for layer_index in sorted(set(layer_indices.values())):
                layer_members = [state for state in child_states if layer_indices[state.object_id] == layer_index]
                layer_bottom = current_layer_top + self._layer_gap
                layer_top = layer_bottom
                for child_state in layer_members:
                    delta_y = float(layer_bottom - child_state.vertical_min)
                    if abs(delta_y) > 1e-10:
                        delta = np.array([0.0, delta_y, 0.0], dtype=float)
                        child_state.translate(delta)
                        moves.append(
                            {
                                "object_id": child_state.object_id,
                                "kind": move_kind,
                                "delta_xyz": [0.0, float(delta_y), 0.0],
                            }
                        )
                    layer_top = max(layer_top, child_state.vertical_max)
                current_layer_top = layer_top

            container_bounds = self._container_bounds(parent_id, mesh_states)
            if container_bounds is not None:
                for child_state in child_states:
                    before = child_state.center.copy()
                    self._clamp_to_container(child_state, container_bounds)
                    delta = child_state.center - before
                    if float(np.linalg.norm(delta)) <= 1e-10:
                        continue
                    moves.append(
                        {
                            "object_id": child_state.object_id,
                            "kind": f"{move_kind}_clamp",
                            "delta_xyz": [float(delta[0]), float(delta[1]), float(delta[2])],
                        }
                    )

            if report is not None:
                report["groups"].append(
                    {
                        "parent_object_id": parent_id,
                        "object_ids": list(child_ids),
                        "relation_group_type": move_kind,
                        "preferred_child_centers_xz": {
                            child_id: [float(value) for value in preferred_centers[child_id]]
                            for child_id in preferred_centers
                        },
                        "final_child_centers_xz": {
                            state.object_id: [float(value) for value in state.horizontal_center]
                            for state in child_states
                        },
                        "sublayer_index_by_child": layer_indices,
                    }
                )
        return moves

    def _relation_groups(
        self,
        relation_graph: SceneRelationGraph | None,
        mesh_states: dict[str, _LayoutMeshState],
    ) -> list[dict[str, object]]:
        if relation_graph is None or not relation_graph.relations:
            return []

        child_ids_by_parent: dict[str, list[str]] = {}
        for relation in relation_graph.relations:
            if not is_parent_child_relation_type(relation.relation_type):
                continue
            child_ids_by_parent.setdefault(relation.parent_object_id, []).append(relation.child_object_id)

        parent_ids_by_child: dict[str, set[str]] = {}
        for parent_id, child_ids in child_ids_by_parent.items():
            for child_id in child_ids:
                parent_ids_by_child.setdefault(child_id, set()).add(parent_id)

        depth_cache: dict[str, int] = {}

        def parent_depth(parent_id: str) -> int:
            cached = depth_cache.get(parent_id)
            if cached is not None:
                return cached
            direct_parents = parent_ids_by_child.get(parent_id, set())
            if not direct_parents:
                depth_cache[parent_id] = 0
                return 0
            depth_value = 1 + max(parent_depth(ancestor_id) for ancestor_id in direct_parents)
            depth_cache[parent_id] = depth_value
            return depth_value

        groups: list[dict[str, object]] = []
        for parent_id, child_ids in child_ids_by_parent.items():
            if parent_id not in mesh_states:
                continue
            filtered_child_ids = tuple(child_id for child_id in child_ids if child_id in mesh_states)
            if not filtered_child_ids:
                continue
            groups.append(
                {
                    "parent_id": parent_id,
                    "child_ids": filtered_child_ids,
                    "move_kind": "parent_child_projection",
                    "parent_depth": parent_depth(parent_id),
                }
            )
        groups.sort(key=lambda group: (int(group["parent_depth"]), str(group["parent_id"])))
        return groups

    def _relation_overlap_exempt_pairs(
        self,
        relation_graph: SceneRelationGraph | None,
    ) -> set[frozenset[str]]:
        exempt_pairs: set[frozenset[str]] = set()
        if relation_graph is None or not relation_graph.relations:
            return exempt_pairs

        managed_children_by_parent: dict[str, list[str]] = {}
        for relation in relation_graph.relations:
            if not is_parent_child_relation_type(relation.relation_type):
                continue
            managed_children_by_parent.setdefault(relation.parent_object_id, []).append(relation.child_object_id)
            exempt_pairs.add(frozenset((relation.parent_object_id, relation.child_object_id)))

        for child_ids in managed_children_by_parent.values():
            for left_id, right_id in combinations(child_ids, 2):
                exempt_pairs.add(frozenset((left_id, right_id)))
        return exempt_pairs

    def _compute_sublayer_indices(
        self,
        child_states: list[_LayoutMeshState],
    ) -> dict[str, int]:
        sorted_states = sorted(
            child_states,
            key=lambda state: (float(state.center[0]), float(state.center[2]), state.object_id),
        )
        layers: list[list[_LayoutMeshState]] = []
        indices: dict[str, int] = {}
        for state in sorted_states:
            assigned_index: int | None = None
            for index, layer_states in enumerate(layers):
                if all(self._horizontal_overlap(state, layer_state) is None for layer_state in layer_states):
                    assigned_index = index
                    break
            if assigned_index is None:
                assigned_index = len(layers)
                layers.append([])
            layers[assigned_index].append(state)
            indices[state.object_id] = assigned_index
        return indices

    def _fits_inside_container(
        self,
        *,
        container_state: _LayoutMeshState,
        child_state: _LayoutMeshState,
    ) -> bool:
        container_extents = np.asarray(container_state.mesh.bounding_box.extents, dtype=float)
        child_extents = np.asarray(child_state.mesh.bounding_box.extents, dtype=float)
        return bool(
            np.all(
                child_extents <= (container_extents + self._containment_fit_tolerance)
            )
        )

    def _horizontal_overlap(
        self,
        left_state: _LayoutMeshState,
        right_state: _LayoutMeshState,
    ) -> tuple[float, float] | None:
        left_min = left_state.horizontal_min
        left_max = left_state.horizontal_max
        right_min = right_state.horizontal_min
        right_max = right_state.horizontal_max
        overlap_x = min(float(left_max[0]), float(right_max[0])) - max(float(left_min[0]), float(right_min[0]))
        overlap_y = min(float(left_max[1]), float(right_max[1])) - max(float(left_min[1]), float(right_min[1]))
        if overlap_x <= 0.0 or overlap_y <= 0.0:
            return None
        return overlap_x, overlap_y

    def _vertical_overlap(
        self,
        left_state: _LayoutMeshState,
        right_state: _LayoutMeshState,
    ) -> bool:
        return min(left_state.vertical_max, right_state.vertical_max) > max(left_state.vertical_min, right_state.vertical_min)

    def _separation_delta(
        self,
        left_state: _LayoutMeshState,
        right_state: _LayoutMeshState,
        overlap_xz: tuple[float, float],
    ) -> np.ndarray:
        overlap_x, overlap_z = overlap_xz
        left_center = left_state.center
        right_center = right_state.center
        if overlap_x <= overlap_z:
            direction = 1.0 if float(right_center[0]) >= float(left_center[0]) else -1.0
            return np.array([direction * (overlap_x + self._gap_xy), 0.0, 0.0], dtype=float)
        direction = 1.0 if float(right_center[2]) >= float(left_center[2]) else -1.0
        return np.array([0.0, 0.0, direction * (overlap_z + self._gap_xy)], dtype=float)

    def _clamp_to_container(
        self,
        state: _LayoutMeshState,
        container_bounds: np.ndarray | None,
    ) -> None:
        if container_bounds is None:
            return
        bounds = state.bounds
        delta = np.zeros(3, dtype=float)
        if bounds[0, 0] < container_bounds[0, 0]:
            delta[0] = float(container_bounds[0, 0] - bounds[0, 0])
        elif bounds[1, 0] > container_bounds[1, 0]:
            delta[0] = float(container_bounds[1, 0] - bounds[1, 0])
        if bounds[0, 2] < container_bounds[0, 1]:
            delta[2] = float(container_bounds[0, 1] - bounds[0, 2])
        elif bounds[1, 2] > container_bounds[1, 1]:
            delta[2] = float(container_bounds[1, 1] - bounds[1, 2])
        state.translate(delta)

    def _updated_pose(self, state: _LayoutMeshState, pose: ObjectPose) -> ObjectPose:
        center = state.center
        return ObjectPose(
            object_id=pose.object_id,
            translation_xyz=(float(center[0]), float(center[1]), float(center[2])),
            rotation_type=pose.rotation_type,
            rotation_value=pose.rotation_value,
            scale_xyz=pose.scale_xyz,
        )

    def _load_mesh(self, mesh_path: Path) -> trimesh.Trimesh:
        try:
            loaded = self._mesh_loader(mesh_path, force="mesh")
        except TypeError:
            loaded = self._mesh_loader(mesh_path)
        if isinstance(loaded, trimesh.Scene):
            return loaded.dump(concatenate=True)
        return loaded.copy()
