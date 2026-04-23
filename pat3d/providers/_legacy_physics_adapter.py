from __future__ import annotations

import copy
from contextlib import contextmanager
import importlib
import json
import numpy as np
from pathlib import Path
import sys
import types
from typing import Callable

import trimesh

from pat3d.legacy_config import load_root_parse_options, resolve_repo_root
from pat3d.models import ArtifactRef, ObjectPose, PhysicsReadyScene, RelationType
from pat3d.models.pose_utils import matrix_to_pose, pose_to_matrix

def _default_args_factory() -> object:
    args, _ = load_root_parse_options()(argv=[])
    return args


LEGACY_SHARED_DEFAULTS: dict[str, float | int] = {
    "end_frame": 300,
    "tol_rate": 1e-4,
    "ground_y_value": -1.1,
    "time_step": 0.03,
    "contact_d_hat": 5e-4,
    "contact_eps_velocity": 1e-5,
}

LEGACY_OPTIMIZE_DEFAULTS: dict[str, float | int | bool | str] = {
    **LEGACY_SHARED_DEFAULTS,
    "adaptive_end_frame_enabled": True,
    "adaptive_controller": "sampled_criteria",
    "total_opt_epoch": 50,
    "phys_lr": 0.001,
    "weight_decay": 0.01,
    "optimizer_name": "adamw",
    "optimizer_amsgrad": False,
    "grad_clip_norm": 0.0,
    "offset_frame": 0,
    "optim_frame_interval": 10,
    "adaptive_end_frame_cap": 300,
    "adaptive_end_frame_min": 200,
    "adaptive_warmup_epochs": 10,
    "adaptive_required_consecutive_epochs": 3,
    "adaptive_required_stable_epochs": 5,
    "adaptive_acceleration_residual_threshold": 0.1,
    "adaptive_force_residual_threshold": 0.4,
    "adaptive_velocity_residual_threshold": 0.003,
    "adaptive_forward_validation_min_frame": 50,
    "adaptive_forward_validation_static_window": 100,
    "adaptive_forward_validation_refresh_epochs": 5,
    "adaptive_forward_validation_end_frame": 300,
    "adaptive_forward_validation_cushion_frames": 50,
    "adaptive_forward_validation_align_multiple": 10,
    "adaptive_min_stop_frame": 200,
    "adaptive_support_order": "",
    "adaptive_overlap_threshold": 0.95,
    "adaptive_height_ratio_threshold": 0.90,
    "adaptive_require_min_stop_frame": False,
    "adaptive_require_order": False,
    "adaptive_require_overlap": False,
    "adaptive_require_height_ratio": False,
    "adaptive_require_velocity": True,
    "adaptive_require_force": True,
}

DIFF_INIT_GROUND_CLEARANCE = 1e-3


class LegacySequentialPhysicsAdapter:
    def __init__(
        self,
        *,
        args_factory: Callable[[], object] = _default_args_factory,
        module_path: str = "pat3d/legacy_vendor/mask_loss/diff_sim/sequential_phys_optim.py",
        legacy_arg_overrides: dict[str, str | int | float | bool] | None = None,
        mesh_loader: Callable[..., trimesh.Trimesh | trimesh.Scene] = trimesh.load,
        static_translation_tol: float = 1e-4,
        static_rotation_tol: float = 1e-4,
        static_consecutive_frames: int = 10,
        min_static_frames: int = 20,
    ) -> None:
        self._args_factory = args_factory
        resolved_module_path = Path(module_path)
        if not resolved_module_path.is_absolute():
            resolved_module_path = resolve_repo_root() / resolved_module_path
        self._module_path = resolved_module_path.resolve()
        self._legacy_arg_overrides = dict(legacy_arg_overrides or {})
        self._mesh_loader = mesh_loader
        self._static_translation_tol = float(static_translation_tol)
        self._static_rotation_tol = float(static_rotation_tol)
        self._static_consecutive_frames = int(static_consecutive_frames)
        self._min_static_frames = int(min_static_frames)

    def optimize(self, physics_ready_scene: PhysicsReadyScene) -> dict[str, object]:
        with self._legacy_import_context():
            backend = self._load_backend()
            if len(backend) == 3:
                optimizer_cls, simplicial_complex_io_cls, transform_cls = backend
                simulator_cls = None
                view_fn = None
                gui_info_cls = None
            else:
                optimizer_cls, simulator_cls, simplicial_complex_io_cls, transform_cls, view_fn, gui_info_cls = backend
            args = self._build_args(physics_ready_scene, profile="optimize_then_forward")
            requested_ground_y = self._ground_y_value(args)
            mesh_dict, trimesh_dict = self._load_meshes(
                physics_ready_scene,
                simplicial_complex_io_cls,
                transform_cls,
            )
            report = {
                "scene_id": physics_ready_scene.scene_id,
                "mode": "optimize_then_forward",
                "hierarchical_tree_used": False,
                "progressive_pass_count": 0,
                "passes": [],
            }
            try:
                final_transformations, optimizer_metrics, report = self._run_progressive_diff_init(
                    args,
                    physics_ready_scene,
                    optimizer_cls,
                    simulator_cls,
                    simplicial_complex_io_cls,
                    transform_cls,
                    view_fn=view_fn,
                    gui_info_cls=gui_info_cls,
                    trimesh_dict=trimesh_dict,
                )
            except Exception as exc:
                report["fallback_reason"] = f"{type(exc).__name__}: {exc}"
                if simulator_cls is None:
                    raise
                simulator_mesh_dict, _ = self._load_meshes(
                    physics_ready_scene,
                    simplicial_complex_io_cls,
                    transform_cls,
                )
                final_transformations, simulator_metrics = self._simulate_forward_transforms(
                    args,
                    simulator_cls,
                    simulator_mesh_dict,
                    tuple(mesh_dict.keys()),
                    gui_info_cls=gui_info_cls,
                )
                optimizer_metrics = {
                    "diff_sim_initialization_failed": 1.0,
                    "diff_sim_initialization_used": 0.0,
                    "diff_sim_progressive_pass_count": 0.0,
                    "diff_sim_hierarchical_tree_used": 0.0,
                    "forward_only_simulation_used": 1.0,
                    "forward_diff_simulator_used": 1.0,
                    "forward_diff_simulator_failed": 0.0,
                    **simulator_metrics,
                }
                if requested_ground_y is not None:
                    optimizer_metrics["requested_ground_y_value"] = requested_ground_y
                    optimizer_metrics["applied_ground_y_value"] = requested_ground_y
                    report["requested_ground_y_value"] = requested_ground_y
                    report["applied_ground_y_value"] = requested_ground_y
                report["used_forward_only_fallback"] = True
            else:
                report["used_forward_only_fallback"] = False

        optimized_object_poses = tuple(
            self._apply_transform_delta(pose, final_transformations.get(pose.object_id))
            for pose in physics_ready_scene.object_poses
        )
        optimized_object_poses, projected_support_count = self._project_support_contacts(
            physics_ready_scene,
            optimized_object_poses,
            trimesh_dict,
            final_transformations,
        )
        artifacts = list(self._collect_artifacts(args, report=report))
        loss_history_artifact = self._write_loss_history_summary(args)
        if loss_history_artifact is not None:
            artifacts.append(loss_history_artifact)
        report_artifact = self._write_diff_init_report(args, report)
        if report_artifact is not None:
            artifacts.append(report_artifact)

        return {
            "optimized_object_poses": optimized_object_poses,
            "artifacts": tuple(artifacts),
            "metrics": {
                "legacy_diff_sim_adapter_used": 1.0,
                "optimized_object_count": float(len(optimized_object_poses)),
                "support_contact_projection_count": float(projected_support_count),
                **optimizer_metrics,
            },
        }

    def simulate_to_static(self, physics_ready_scene: PhysicsReadyScene) -> dict[str, object]:
        with self._legacy_import_context():
            backend = self._load_backend()
            if len(backend) == 3:
                _optimizer_cls, simplicial_complex_io_cls, transform_cls = backend
                simulator_cls = None
                gui_info_cls = None
            else:
                _optimizer_cls, simulator_cls, simplicial_complex_io_cls, transform_cls, _view_fn, gui_info_cls = backend
            if simulator_cls is None:
                raise RuntimeError("legacy physics backend did not expose a forward simulator")
            args = self._build_args(physics_ready_scene)
            requested_ground_y = self._ground_y_value(args)
            mesh_dict, trimesh_dict = self._load_meshes(
                physics_ready_scene,
                simplicial_complex_io_cls,
                transform_cls,
            )
            final_transformations, simulator_metrics = self._simulate_forward_transforms(
                args,
                simulator_cls,
                mesh_dict,
                tuple(mesh_dict.keys()),
                gui_info_cls=gui_info_cls,
            )

        optimized_object_poses = tuple(
            self._apply_transform_delta(pose, final_transformations.get(pose.object_id))
            for pose in physics_ready_scene.object_poses
        )
        optimized_object_poses, projected_support_count = self._project_support_contacts(
            physics_ready_scene,
            optimized_object_poses,
            trimesh_dict,
            final_transformations,
        )

        return {
            "optimized_object_poses": optimized_object_poses,
            "artifacts": self._collect_artifacts(args),
            "metrics": {
                "legacy_diff_sim_adapter_used": 1.0,
                "optimized_object_count": float(len(optimized_object_poses)),
                "forward_only_simulation_used": 1.0,
                "forward_diff_simulator_used": 1.0,
                "forward_diff_simulator_failed": 0.0,
                "support_contact_projection_count": float(projected_support_count),
                **(
                    {
                        "requested_ground_y_value": requested_ground_y,
                        "applied_ground_y_value": requested_ground_y,
                    }
                    if requested_ground_y is not None
                    else {}
                ),
                **simulator_metrics,
            },
        }

    def _ground_y_value(self, args) -> float | None:
        if not hasattr(args, "ground_y_value"):
            return None
        try:
            value = float(getattr(args, "ground_y_value"))
        except (TypeError, ValueError):
            return None
        if not np.isfinite(value):
            return None
        return value

    def _apply_transformations_to_meshes(
        self,
        mesh_dict,
        transformations: dict[str, list[list[float]]],
        view_fn,
    ) -> None:
        for object_id, mesh in mesh_dict.items():
            transform_matrix = transformations.get(object_id)
            if transform_matrix is None:
                continue
            view_fn(mesh.transforms())[:] = np.asarray(transform_matrix, dtype=np.float64)

    def _simulate_forward_transforms(
        self,
        args,
        simulator_cls,
        simulator_mesh_dict,
        object_ids: tuple[str, ...],
        *,
        gui_info_cls=None,
        transform_root: Path | None = None,
    ) -> tuple[dict[str, list[list[float]]], dict[str, float]]:
        self._prepare_headless_simulator(simulator_cls, gui_info_cls)
        simulator = self._instantiate_forward_simulator(
            args,
            simulator_cls,
            simulator_mesh_dict,
            transform_root=transform_root,
        )
        raw_transformations = simulator.get_transform_parameters()
        final_frame = 0.0
        if getattr(simulator, "final_frame", None) is not None:
            try:
                final_frame = float(simulator.final_frame)
            except Exception:
                final_frame = 0.0
        else:
            world = getattr(simulator, "world", None)
            if world is not None and callable(getattr(world, "frame", None)):
                try:
                    final_frame = float(world.frame())
                except Exception:
                    final_frame = 0.0
            else:
                final_frame = 0.0
        return (
            self._normalize_transformations(raw_transformations, object_ids),
            {
                "forward_simulator_final_frame": final_frame,
                "forward_simulator_stopped_static": (
                    1.0 if bool(getattr(simulator, "stopped_because_static", False)) else 0.0
                ),
            },
        )

    def _prepare_headless_gui(self, gui_info_cls=None) -> None:
        if gui_info_cls is not None and hasattr(gui_info_cls, "enabled"):
            gui_info_cls.enabled = staticmethod(lambda: False)

    def _prepare_headless_simulator(self, simulator_cls, gui_info_cls=None) -> None:
        self._prepare_headless_gui(gui_info_cls)

    def _instantiate_forward_simulator(
        self,
        args,
        simulator_cls,
        simulator_mesh_dict,
        *,
        transform_root: Path | None = None,
    ):
        active_transform_root = (
            transform_root if transform_root is not None else self._default_transform_root(args)
        )
        self._prepare_transform_root(active_transform_root)
        try:
            return simulator_cls(
                args,
                simulator_mesh_dict,
                int(getattr(args, "end_frame", 1000)),
                True,
                str(active_transform_root),
            )
        except TypeError:
            return simulator_cls(
                args,
                simulator_mesh_dict,
                int(getattr(args, "end_frame", 1000)),
                False,
            )

    def _collect_artifacts(
        self,
        args,
        report: dict[str, object] | None = None,
    ) -> tuple[ArtifactRef, ...]:
        artifacts = [
            ArtifactRef(
                artifact_type="legacy_physics_reference",
                path=str(self._module_path),
                format="py",
                role="legacy_source",
                metadata_path=None,
            )
        ]
        param_dir = Path(args.phys_result_folder) / args.exp_name / "param"
        param_files = sorted(param_dir.glob("optim_*.npz"))
        if param_files:
            artifacts.append(
                ArtifactRef(
                    artifact_type="physics_parameters",
                    path=str(param_files[-1]),
                    format="npz",
                    role="optimization_parameters",
                    metadata_path=None,
                )
            )
        transform_dir = self._preferred_transform_root(args, report=report)
        frame_files = self._sorted_frame_paths(transform_dir)
        if frame_files:
            artifacts.append(
                ArtifactRef(
                    artifact_type="physics_trajectory",
                    path=str(frame_files[-1]),
                    format="npz",
                    role="forward_simulation_frame",
                    metadata_path=None,
                )
            )
        return tuple(artifacts)

    def _scene_root(self, args) -> Path:
        return Path(args.phys_result_folder) / args.exp_name

    def _default_transform_root(self, args) -> Path:
        return self._scene_root(args) / "transform"

    def _progressive_pass_transform_root(self, args, pass_index: int) -> Path:
        return self._scene_root(args) / f"progressive_pass_{int(pass_index):03d}" / "transform"

    def _prepare_transform_root(self, transform_root: Path) -> None:
        transform_root.mkdir(parents=True, exist_ok=True)
        for stale_frame in transform_root.glob("frame_*.npz"):
            stale_frame.unlink()

    def _sorted_frame_paths(self, transform_root: Path) -> list[Path]:
        return sorted(
            transform_root.glob("frame_*.npz"),
            key=lambda path: int(path.stem.rsplit("_", 1)[-1]),
        )

    def _preferred_transform_root(
        self,
        args,
        *,
        report: dict[str, object] | None = None,
    ) -> Path:
        if isinstance(report, dict):
            raw_passes = report.get("passes")
            if isinstance(raw_passes, list):
                for pass_payload in reversed(raw_passes):
                    if not isinstance(pass_payload, dict):
                        continue
                    raw_dir = pass_payload.get("trajectory_dir")
                    if not isinstance(raw_dir, str) or not raw_dir.strip():
                        continue
                    candidate = Path(raw_dir)
                    if self._sorted_frame_paths(candidate):
                        return candidate
        return self._default_transform_root(args)

    def _loss_history_path(self, args) -> Path:
        return Path(args.phys_result_folder) / args.exp_name / "loss_history.json"

    def _loss_history_pass_paths(self, args) -> list[Path]:
        scene_root = Path(args.phys_result_folder) / args.exp_name
        return sorted(scene_root.glob("loss_history_pass_*.json"))

    def _load_loss_history_summary(self, args) -> dict[str, object]:
        summary_path = self._loss_history_path(args)
        if summary_path.exists():
            try:
                payload = json.loads(summary_path.read_text(encoding="utf-8"))
            except Exception:
                payload = None
            if isinstance(payload, dict):
                return payload

        series: list[dict[str, object]] = []
        best_losses: list[float] = []
        for pass_path in self._loss_history_pass_paths(args):
            try:
                payload = json.loads(pass_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            points = payload.get("points")
            if not isinstance(points, list):
                continue
            try:
                pass_index = int(payload.get("pass_index", len(series)))
            except (TypeError, ValueError):
                pass_index = len(series)
            normalized_points: list[dict[str, object]] = []
            for point in points:
                if not isinstance(point, dict):
                    continue
                try:
                    normalized_points.append(
                        {
                            "step": int(point.get("step", 0)),
                            "epoch": int(point.get("epoch", point.get("step", 0))),
                            "loss": float(point["loss"]),
                        }
                    )
                except (KeyError, TypeError, ValueError):
                    continue
            best_loss = payload.get("best_loss")
            if isinstance(best_loss, (int, float)) and np.isfinite(float(best_loss)):
                best_losses.append(float(best_loss))
            series.append(
                {
                    "pass_index": pass_index,
                    "label": str(payload.get("label") or f"Pass {pass_index + 1}"),
                    "points": normalized_points,
                }
            )

        if not series:
            return {
                "available": False,
                "scene_id": args.exp_name,
                "mode": "optimize_then_forward",
                "reason": "loss_history_unavailable",
            }

        return {
            "available": True,
            "scene_id": args.exp_name,
            "mode": "optimize_then_forward",
            "best_loss": min(best_losses) if best_losses else None,
            "series": sorted(series, key=lambda item: int(item.get("pass_index", 0))),
        }

    def _write_loss_history_summary(self, args) -> ArtifactRef | None:
        summary = self._load_loss_history_summary(args)
        if not bool(summary.get("available")):
            return None
        summary_path = self._loss_history_path(args)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return ArtifactRef(
            artifact_type="physics_metrics",
            path=str(summary_path),
            format="json",
            role="loss_history",
            metadata_path=None,
        )

    def _write_diff_init_report(self, args, report: dict[str, object]) -> ArtifactRef | None:
        if not report:
            return None
        report_path = Path(args.phys_result_folder) / args.exp_name / "diff_init_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return ArtifactRef(
            artifact_type="physics_debug_report",
            path=str(report_path),
            format="json",
            role="diff_sim_report",
            metadata_path=None,
        )

    def _project_support_contacts(
        self,
        physics_ready_scene: PhysicsReadyScene,
        optimized_object_poses: tuple[ObjectPose, ...],
        trimesh_dict: dict[str, trimesh.Trimesh],
        transformations: dict[str, list[list[float]]],
    ) -> tuple[tuple[ObjectPose, ...], int]:
        support_graph = physics_ready_scene.layout.support_graph
        if support_graph is None or not support_graph.relations:
            return optimized_object_poses, 0

        pose_by_id = {pose.object_id: pose for pose in optimized_object_poses}
        mesh_by_id: dict[str, trimesh.Trimesh] = {}
        for object_id, source_mesh in trimesh_dict.items():
            mesh = source_mesh.copy()
            transform_matrix = transformations.get(object_id)
            if transform_matrix is not None:
                mesh.apply_transform(np.asarray(transform_matrix, dtype=np.float64))
            mesh_by_id[object_id] = mesh

        root_object_ids = tuple(getattr(support_graph, "root_object_ids", ()) or ())
        ground_y = self._support_ground_y(mesh_by_id, root_object_ids)
        projected_children: set[str] = set()
        support_relations = [
            relation
            for relation in support_graph.relations
            if getattr(relation.relation_type, "value", relation.relation_type)
            in (RelationType.SUPPORTS.value, RelationType.ON.value)
        ]

        for _ in range(len(support_relations) + 1):
            changed = False
            for relation in support_relations:
                parent_mesh = mesh_by_id.get(relation.parent_object_id)
                child_mesh = mesh_by_id.get(relation.child_object_id)
                child_pose = pose_by_id.get(relation.child_object_id)
                if parent_mesh is None or child_mesh is None or child_pose is None:
                    continue
                support_surfaces = self._support_surfaces_for_child(
                    relation.parent_object_id,
                    mesh_by_id,
                    root_object_ids=root_object_ids,
                    ground_y=ground_y,
                )
                delta_y = self._support_surface_delta_y(child_mesh, support_surfaces)
                if abs(delta_y) <= 1e-6:
                    continue
                child_mesh.apply_translation((0.0, -delta_y, 0.0))
                pose_by_id[relation.child_object_id] = ObjectPose(
                    object_id=child_pose.object_id,
                    translation_xyz=(
                        float(child_pose.translation_xyz[0]),
                        float(child_pose.translation_xyz[1]) - delta_y,
                        float(child_pose.translation_xyz[2]),
                    ),
                    rotation_type=child_pose.rotation_type,
                    rotation_value=child_pose.rotation_value,
                    scale_xyz=child_pose.scale_xyz,
                )
                projected_children.add(relation.child_object_id)
                changed = True
            if not changed:
                break

        return (
            tuple(pose_by_id[pose.object_id] for pose in optimized_object_poses),
            len(projected_children),
        )

    def _support_ground_y(
        self,
        mesh_by_id: dict[str, trimesh.Trimesh],
        root_object_ids: tuple[str, ...],
    ) -> float:
        candidate_ids = [object_id for object_id in root_object_ids if object_id in mesh_by_id]
        if not candidate_ids:
            candidate_ids = list(mesh_by_id.keys())
        if not candidate_ids:
            return 0.0
        return min(float(mesh_by_id[object_id].bounds[0][1]) for object_id in candidate_ids)

    def _support_surfaces_for_child(
        self,
        parent_object_id: str,
        mesh_by_id: dict[str, trimesh.Trimesh],
        *,
        root_object_ids: tuple[str, ...],
        ground_y: float,
    ) -> tuple[dict[str, object], ...]:
        surfaces: list[dict[str, object]] = [{"y": ground_y, "footprint": None}]
        # Use the immediate parent only. Root objects are already accounted for in the
        # ground reference and can otherwise pull the child down into an unintended lower
        # support band when the support graph is a chain.
        mesh = mesh_by_id.get(parent_object_id)
        if mesh is not None:
            surface = self._top_surface(mesh)
            if surface is not None:
                surfaces.append(surface)
        return tuple(surfaces)

    def _top_surface(self, mesh: trimesh.Trimesh) -> dict[str, object] | None:
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        if vertices.size == 0:
            return None
        top_y = float(mesh.bounds[1][1])
        extent_y = float(mesh.extents[1]) if len(mesh.extents) > 1 else 0.0
        band = max(1e-4, min(2.5e-3, extent_y * 0.25 if extent_y > 0.0 else 2.5e-3))
        near_top = np.abs(vertices[:, 1] - top_y) <= band
        top_vertices = vertices[near_top] if np.any(near_top) else vertices
        return {
            "y": top_y,
            "footprint": (
                float(top_vertices[:, 0].min()),
                float(top_vertices[:, 0].max()),
                float(top_vertices[:, 2].min()),
                float(top_vertices[:, 2].max()),
            ),
        }

    def _support_surface_delta_y(
        self,
        child_mesh: trimesh.Trimesh,
        surfaces: tuple[dict[str, object], ...],
    ) -> float:
        vertices = np.asarray(child_mesh.vertices, dtype=np.float64)
        if vertices.size == 0 or not surfaces:
            return 0.0
        support_y = np.full(vertices.shape[0], float(surfaces[0]["y"]), dtype=np.float64)
        for surface in surfaces[1:]:
            footprint = surface.get("footprint")
            if footprint is None:
                support_y = np.maximum(support_y, float(surface["y"]))
                continue
            min_x, max_x, min_z, max_z = footprint
            in_footprint = (
                (vertices[:, 0] >= float(min_x))
                & (vertices[:, 0] <= float(max_x))
                & (vertices[:, 2] >= float(min_z))
                & (vertices[:, 2] <= float(max_z))
            )
            if np.any(in_footprint):
                support_y[in_footprint] = np.maximum(
                    support_y[in_footprint],
                    float(surface["y"]),
                )
        return float(np.min(vertices[:, 1] - support_y))

    def _extract_transformations(
        self,
        optimizer,
        object_ids: tuple[str, ...],
    ) -> dict[str, list[list[float]]]:
        raw_transformations = getattr(optimizer, "transformation_parameter", None)
        transformations = self._normalize_transformations(raw_transformations, object_ids)

        param = getattr(optimizer, "param", None)
        param_map_back = getattr(optimizer, "param_map_back", None)
        if param is None or not isinstance(param_map_back, dict) or not param_map_back:
            if not isinstance(raw_transformations, dict):
                raise RuntimeError("legacy sequential optimizer did not expose transformation_parameter")
            return transformations

        try:
            values = param.U()
        except Exception:
            if not isinstance(raw_transformations, dict):
                raise RuntimeError("legacy sequential optimizer did not expose transformation_parameter")
            return transformations

        for index, location in param_map_back.items():
            if not isinstance(location, (list, tuple)) or len(location) != 3:
                continue
            object_id, row, col = location
            if not isinstance(object_id, str):
                continue
            if object_id not in transformations:
                transformations[object_id] = self._identity_transform()
            transformations[object_id][int(row)][int(col)] = self._coerce_scalar(values[index])
        return transformations

    def _normalize_transformations(
        self,
        raw_transformations,
        object_ids: tuple[str, ...],
    ) -> dict[str, list[list[float]]]:
        transformations: dict[str, list[list[float]]] = {}
        for object_id in object_ids:
            matrix = raw_transformations.get(object_id) if isinstance(raw_transformations, dict) else None
            if matrix is None:
                transformations[object_id] = self._identity_transform()
                continue
            rows: list[list[float]] = []
            for row_index in range(4):
                row: list[float] = []
                for col_index in range(4):
                    row.append(self._coerce_scalar(matrix[row_index][col_index]))
                rows.append(row)
            transformations[object_id] = rows
        return transformations

    def _identity_transform(self) -> list[list[float]]:
        return [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]

    def _coerce_scalar(self, value) -> float:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        return float(value)

    @contextmanager
    def _legacy_import_context(self):
        legacy_repo_root = self._module_path.parent.parent
        inserted_paths: list[str] = []
        for candidate in (str(legacy_repo_root), str(self._module_path.parent)):
            if candidate not in sys.path:
                sys.path.insert(0, candidate)
                inserted_paths.append(candidate)

        shadowed_modules = {
            name: module
            for name, module in list(sys.modules.items())
            if name == "modules" or name.startswith("modules.")
        }
        for name in shadowed_modules:
            sys.modules.pop(name, None)

        try:
            self._install_uipc_compat()
            self._assert_uipc_compatibility()
            yield
        finally:
            for name in list(sys.modules):
                if name == "modules" or name.startswith("modules."):
                    sys.modules.pop(name, None)
            sys.modules.update(shadowed_modules)
            for candidate in reversed(inserted_paths):
                try:
                    sys.path.remove(candidate)
                except ValueError:
                    pass

    def _load_backend(self):
        spec = importlib.util.spec_from_file_location(
            "pat3d_legacy_sequential_phys_optim",
            self._module_path,
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"could not load legacy physics module: {self._module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return (
            module.PhysOptim,
            getattr(module, "PhysSimulator", None),
            module.SimplicialComplexIO,
            module.Transform,
            getattr(module, "view", None),
            getattr(module, "GUIInfo", None),
        )

    def assert_runtime_support(self) -> None:
        with self._legacy_import_context():
            self._load_backend()

    def _install_uipc_compat(self) -> None:
        try:
            import uipc  # type: ignore
        except ImportError:
            return

        core_module = None
        constitution_module = None
        try:
            core_module = importlib.import_module("uipc.core")
        except ImportError:
            pass
        try:
            constitution_module = importlib.import_module("uipc.constitution")
        except ImportError:
            pass

        if not hasattr(uipc, "Vector3i") and hasattr(uipc, "Vector3"):
            setattr(uipc, "Vector3i", uipc.Vector3)
        if not hasattr(uipc, "Matrix4x4i") and hasattr(uipc, "Matrix4x4"):
            setattr(uipc, "Matrix4x4i", uipc.Matrix4x4)

        if core_module is not None:
            if not hasattr(core_module, "Vector3i") and hasattr(uipc, "Vector3i"):
                setattr(core_module, "Vector3i", uipc.Vector3i)
            if not hasattr(core_module, "Matrix4x4i") and hasattr(uipc, "Matrix4x4i"):
                setattr(core_module, "Matrix4x4i", uipc.Matrix4x4i)

        torch_module = None
        for module_name in ("uipc.torch", "uipc.diff_sim"):
            try:
                torch_module = importlib.import_module(module_name)
                if module_name != "uipc.torch":
                    sys.modules.setdefault("uipc.torch", torch_module)
                break
            except ImportError:
                continue
        if torch_module is None:
            torch_module = sys.modules.setdefault("uipc.torch", types.ModuleType("uipc.torch"))

        if (
            constitution_module is not None
            and not hasattr(constitution_module, "DiffSimParameter")
            and hasattr(torch_module, "DiffSimParameter")
        ):
            setattr(constitution_module, "DiffSimParameter", torch_module.DiffSimParameter)

    def _assert_uipc_compatibility(self) -> None:
        required_symbols = {
            "uipc.core": ("Vector3i", "Matrix4x4i"),
            "uipc.geometry": ("SimplicialComplexIO",),
            "uipc.constitution": ("AffineBodyConstitution", "DiffSimParameter"),
        }
        missing_symbols: list[str] = []
        for module_name, symbol_names in required_symbols.items():
            module = importlib.import_module(module_name)
            for symbol_name in symbol_names:
                if not hasattr(module, symbol_name):
                    missing_symbols.append(f"{module_name}.{symbol_name}")
        if missing_symbols:
            raise RuntimeError(
                "legacy physics backend requires the custom Diff_GIPC/UIPC Python build; "
                f"missing symbols: {', '.join(missing_symbols)}"
            )

    def _build_args(
        self,
        physics_ready_scene: PhysicsReadyScene,
        *,
        profile: str = "forward_only",
    ) -> object:
        args = self._args_factory()
        args.exp_name = physics_ready_scene.scene_id
        args.static_translation_tol = getattr(args, "static_translation_tol", self._static_translation_tol)
        args.static_rotation_tol = getattr(args, "static_rotation_tol", self._static_rotation_tol)
        args.static_consecutive_frames = getattr(
            args, "static_consecutive_frames", self._static_consecutive_frames
        )
        args.min_static_start_frame = getattr(args, "min_static_start_frame", self._min_static_frames)
        defaults = LEGACY_OPTIMIZE_DEFAULTS if profile == "optimize_then_forward" else LEGACY_SHARED_DEFAULTS
        for key, value in defaults.items():
            if hasattr(args, key):
                setattr(args, key, value)
        for key, value in self._legacy_arg_overrides.items():
            if hasattr(args, key) and isinstance(value, (str, int, float, bool)):
                setattr(args, key, value)
        settings = physics_ready_scene.collision_settings or {}
        for key, value in settings.items():
            if hasattr(args, key) and isinstance(value, (int, float, bool)):
                setattr(args, key, value)
        return args

    def _load_meshes(
        self,
        physics_ready_scene: PhysicsReadyScene,
        simplicial_complex_io_cls,
        transform_cls,
        *,
        object_ids: tuple[str, ...] | None = None,
    ):
        pre_transform = transform_cls.Identity()
        pre_transform.scale(1)
        io = simplicial_complex_io_cls(pre_transform)

        mesh_dict = {}
        trimesh_dict = {}
        selected_ids = set(object_ids or ())
        for artifact in physics_ready_scene.simulation_meshes:
            object_id = artifact.role or Path(artifact.path).stem
            if selected_ids and object_id not in selected_ids:
                continue
            mesh_dict[object_id] = io.read(artifact.path)
            trimesh_dict[object_id] = self._load_trimesh(Path(artifact.path))
        return mesh_dict, trimesh_dict

    def _select_container_name(
        self,
        physics_ready_scene: PhysicsReadyScene,
        trimesh_dict: dict[str, trimesh.Trimesh],
    ) -> str:
        support_graph = physics_ready_scene.layout.support_graph
        if support_graph is not None and support_graph.relations:
            counts: dict[str, int] = {}
            for relation in support_graph.relations:
                counts[relation.parent_object_id] = counts.get(relation.parent_object_id, 0) + 1
            for object_id, _count in sorted(counts.items(), key=lambda item: item[1], reverse=True):
                if object_id in trimesh_dict:
                    return object_id

        best_object_id = None
        best_volume = -1.0
        for object_id, mesh in trimesh_dict.items():
            extents = mesh.bounding_box.extents
            volume = float(extents[0] * extents[1] * extents[2])
            if volume > best_volume:
                best_volume = volume
                best_object_id = object_id
        if best_object_id is None:
            raise RuntimeError("legacy physics adapter found no simulation meshes")
        return best_object_id

    def _apply_transform_delta(
        self,
        pose: ObjectPose,
        transform_matrix,
    ) -> ObjectPose:
        if transform_matrix is None:
            return pose
        delta_matrix = np.asarray(transform_matrix, dtype=np.float64)
        final_matrix = delta_matrix @ pose_to_matrix(pose)
        return matrix_to_pose(pose.object_id, final_matrix)

    def _load_trimesh(self, mesh_path: Path) -> trimesh.Trimesh:
        try:
            loaded = self._mesh_loader(mesh_path, force="mesh")
        except TypeError:
            loaded = self._mesh_loader(mesh_path)
        if isinstance(loaded, trimesh.Scene):
            return loaded.dump(concatenate=True)
        return loaded.copy()

    def _run_progressive_diff_init(
        self,
        args,
        physics_ready_scene: PhysicsReadyScene,
        optimizer_cls,
        simulator_cls,
        simplicial_complex_io_cls,
        transform_cls,
        *,
        view_fn,
        gui_info_cls,
        trimesh_dict: dict[str, trimesh.Trimesh],
    ) -> tuple[dict[str, list[list[float]]], dict[str, float], dict[str, object]]:
        object_ids = tuple(
            artifact.role or Path(artifact.path).stem
            for artifact in physics_ready_scene.simulation_meshes
        )
        passes, hierarchical_tree_used, fallback_reason = self._build_diff_init_plan(
            physics_ready_scene,
            object_ids,
            trimesh_dict,
        )
        report: dict[str, object] = {
            "scene_id": physics_ready_scene.scene_id,
            "mode": "optimize_then_forward",
            "hierarchical_tree_used": hierarchical_tree_used,
            "progressive_pass_count": len(passes),
            "completed_pass_count": 0,
            "active_pass_index": 0 if passes else None,
            "plan_fallback_reason": fallback_reason,
            "passes": [
                {
                    "pass_index": pass_index,
                    "container_name": str(pass_config["container_name"]),
                    "optimized_object_ids": [
                        str(value) for value in pass_config["optimizer_object_ids"]
                    ],
                    "dynamic_object_ids": [
                        str(value) for value in pass_config["dynamic_object_ids"]
                    ],
                    "frozen_object_ids": [
                        str(value) for value in pass_config.get("frozen_object_ids", ())
                    ],
                    "strategy": str(pass_config.get("strategy", "heuristic")),
                    "status": "pending",
                }
                for pass_index, pass_config in enumerate(passes)
            ],
        }
        support_tree_children_map = None
        if hierarchical_tree_used:
            support_tree_children_map, _parent_map, _roots, _tree_fallback_reason = (
                self._build_filtered_support_tree(physics_ready_scene, object_ids)
            )
        requested_ground_y = self._ground_y_value(args)
        last_applied_ground_y = requested_ground_y
        working_transforms = {
            object_id: self._identity_transform()
            for object_id in object_ids
        }
        finished_dynamic_ids: list[str] = []
        pass_losses: list[float] = []
        last_simulator_metrics: dict[str, float] = {}
        all_passes_static = True
        any_forward_simulation = False

        self._prepare_headless_gui(gui_info_cls)
        self._write_diff_init_report(args, report)

        for pass_index, pass_config in enumerate(passes):
            container_name = str(pass_config["container_name"])
            optimizer_object_ids = tuple(str(value) for value in pass_config["optimizer_object_ids"])
            dynamic_object_ids = tuple(str(value) for value in pass_config["dynamic_object_ids"])
            frozen_object_ids = tuple(str(value) for value in pass_config.get("frozen_object_ids", ()))
            strategy = str(pass_config.get("strategy", "heuristic"))
            pass_report = report["passes"][pass_index]
            pass_report["status"] = "running"
            report["active_pass_index"] = pass_index
            self._write_diff_init_report(args, report)
            pre_pass_transforms = {
                object_id: copy.deepcopy(working_transforms[object_id])
                for object_id in object_ids
            }
            optimizer_mesh_dict = {}
            optimizer_trimesh_dict = {}
            optimizer_args = args
            ground_adjustment = None
            if optimizer_object_ids:
                optimizer_mesh_dict, optimizer_trimesh_dict = self._load_meshes(
                    physics_ready_scene,
                    simplicial_complex_io_cls,
                    transform_cls,
                    object_ids=optimizer_object_ids,
                )
                if view_fn is not None:
                    self._apply_transformations_to_meshes(
                        optimizer_mesh_dict,
                        {object_id: working_transforms[object_id] for object_id in optimizer_object_ids},
                        view_fn,
                    )
                self._apply_transformations_to_trimeshes(
                    optimizer_trimesh_dict,
                    {object_id: working_transforms[object_id] for object_id in optimizer_object_ids},
                )
                optimizer_args, ground_adjustment = self._clone_args_with_safe_ground(
                    args,
                    optimizer_trimesh_dict,
                )
            setattr(optimizer_args, "diff_sim_pass_index", pass_index)
            setattr(optimizer_args, "diff_sim_pass_label", f"Pass {pass_index + 1}")
            setattr(optimizer_args, "diff_sim_frozen_object_names", frozen_object_ids)
            optimizer = None
            if strategy != "ground_drop" and optimizer_object_ids:
                optimizer = optimizer_cls(
                    optimizer_args,
                    container_name,
                    optimizer_mesh_dict,
                    optimizer_trimesh_dict,
                )
                optimized_xz = self._extract_transformations(optimizer, optimizer_object_ids)
                for object_id in optimizer_object_ids:
                    working_transforms[object_id] = self._merge_xz_transform(
                        working_transforms[object_id],
                        optimized_xz.get(object_id),
                    )

            pass_report.update(
                {
                    "pass_index": pass_index,
                    "container_name": container_name,
                    "optimized_object_ids": list(optimizer_object_ids),
                    "dynamic_object_ids": list(dynamic_object_ids),
                    "frozen_object_ids": list(frozen_object_ids),
                    "strategy": strategy,
                    "adaptive_controller": str(
                        getattr(optimizer_args, "adaptive_controller", "sampled_criteria")
                    ),
                    "weight_decay": float(getattr(optimizer_args, "weight_decay", 0.01)),
                    "optimizer_name": str(getattr(optimizer_args, "optimizer_name", "adamw")),
                    "optimizer_amsgrad": bool(getattr(optimizer_args, "optimizer_amsgrad", False)),
                    "grad_clip_norm": float(getattr(optimizer_args, "grad_clip_norm", 0.0)),
                    "adaptive_forward_validation_cushion_frames": int(
                        getattr(optimizer_args, "adaptive_forward_validation_cushion_frames", 50)
                    ),
                    "adaptive_forward_validation_align_multiple": int(
                        getattr(optimizer_args, "adaptive_forward_validation_align_multiple", 10)
                    ),
                    "adaptive_required_criteria": (
                        ["acceleration"]
                        if str(getattr(optimizer_args, "adaptive_controller", "sampled_criteria")) == "sampled_criteria"
                        else list(getattr(optimizer, "adaptive_required_criteria", ()))
                        or ["acceleration"]
                    ),
                }
            )
            if ground_adjustment is not None:
                pass_report.update(ground_adjustment)
            applied_ground_y = self._ground_y_value(optimizer_args)
            if applied_ground_y is not None:
                pass_report["ground_y_applied"] = applied_ground_y
                last_applied_ground_y = applied_ground_y
            adaptive_selected_end_frame = None
            if optimizer is not None:
                best_loss = getattr(optimizer, "best_loss", None)
                if isinstance(best_loss, (int, float)) and np.isfinite(float(best_loss)):
                    best_loss_value = float(best_loss)
                    pass_losses.append(best_loss_value)
                    pass_report["best_loss"] = best_loss_value
                adaptive_selected_end_frame = getattr(optimizer, "best_selected_end_frame", None)
                if adaptive_selected_end_frame is None:
                    adaptive_selected_end_frame = getattr(optimizer, "selected_end_frame", None)
                if isinstance(adaptive_selected_end_frame, (int, float)) and np.isfinite(float(adaptive_selected_end_frame)):
                    pass_report["adaptive_selected_end_frame"] = int(adaptive_selected_end_frame)
                adaptive_qualifying_frame = getattr(optimizer, "best_qualifying_frame", None)
                if isinstance(adaptive_qualifying_frame, (int, float)) and np.isfinite(float(adaptive_qualifying_frame)):
                    pass_report["adaptive_qualifying_frame"] = int(adaptive_qualifying_frame)
                pass_report["adaptive_end_frame_enabled"] = bool(
                    getattr(optimizer, "adaptive_end_frame_enabled", False)
                )
                pass_report["adaptive_early_stop_triggered"] = bool(
                    getattr(optimizer, "adaptive_early_stop_triggered", False)
                )
                pass_report["adaptive_completed_opt_epoch"] = int(
                    getattr(optimizer, "completed_opt_epoch", 0) or 0
                )
                adaptive_stop_reason = getattr(optimizer, "adaptive_stop_reason", None)
                if isinstance(adaptive_stop_reason, str) and adaptive_stop_reason.strip():
                    pass_report["adaptive_stop_reason"] = adaptive_stop_reason
                best_epoch_metrics = getattr(optimizer, "best_epoch_metrics", None)
                if isinstance(best_epoch_metrics, dict):
                    pass_report["adaptive_metrics"] = best_epoch_metrics
                latest_forward_validation = getattr(optimizer, "latest_forward_validation", None)
                if isinstance(latest_forward_validation, dict):
                    pass_report["latest_forward_validation"] = copy.deepcopy(latest_forward_validation)
                forward_validation_history = getattr(optimizer, "forward_validation_history", None)
                if isinstance(forward_validation_history, list):
                    pass_report["forward_validation_history"] = copy.deepcopy(
                        forward_validation_history
                    )
            else:
                pass_report["adaptive_end_frame_enabled"] = False
                pass_report["adaptive_early_stop_triggered"] = False
                pass_report["adaptive_completed_opt_epoch"] = 0

            if simulator_cls is not None and dynamic_object_ids:
                any_forward_simulation = True
                simulator_object_ids = self._ordered_unique_ids((*finished_dynamic_ids, *dynamic_object_ids))
                simulator_mesh_dict, simulator_trimesh_dict = self._load_meshes(
                    physics_ready_scene,
                    simplicial_complex_io_cls,
                    transform_cls,
                    object_ids=simulator_object_ids,
                )
                if view_fn is not None:
                    self._apply_transformations_to_meshes(
                        simulator_mesh_dict,
                        {object_id: working_transforms[object_id] for object_id in simulator_object_ids},
                        view_fn,
                    )
                self._apply_transformations_to_trimeshes(
                    simulator_trimesh_dict,
                    {object_id: working_transforms[object_id] for object_id in simulator_object_ids},
                )
                simulator_args = copy.copy(optimizer_args)
                simulator_args, simulator_ground_adjustment = self._clone_args_with_safe_ground(
                    simulator_args,
                    simulator_trimesh_dict,
                )
                if simulator_ground_adjustment is not None:
                    pass_report["ground_y_adjusted_from"] = float(
                        pass_report.get(
                            "ground_y_adjusted_from",
                            simulator_ground_adjustment["ground_y_adjusted_from"],
                        )
                    )
                    pass_report["ground_y_adjusted_to"] = float(
                        simulator_ground_adjustment["ground_y_adjusted_to"]
                    )
                    pass_report["lowest_mesh_y"] = min(
                        float(pass_report.get("lowest_mesh_y", simulator_ground_adjustment["lowest_mesh_y"])),
                        float(simulator_ground_adjustment["lowest_mesh_y"]),
                    )
                if isinstance(adaptive_selected_end_frame, (int, float)) and np.isfinite(float(adaptive_selected_end_frame)):
                    simulator_args.end_frame = int(adaptive_selected_end_frame)
                applied_ground_y = self._ground_y_value(simulator_args)
                if applied_ground_y is not None:
                    pass_report["ground_y_applied"] = applied_ground_y
                    last_applied_ground_y = applied_ground_y
                pass_report["forward_simulation_end_frame"] = int(
                    getattr(simulator_args, "end_frame", getattr(optimizer_args, "end_frame", 0))
                )
                pass_transform_root = self._progressive_pass_transform_root(args, pass_index)
                pass_report["trajectory_dir"] = str(pass_transform_root)
                simulated_transformations, simulator_metrics = self._simulate_forward_transforms(
                    simulator_args,
                    simulator_cls,
                    simulator_mesh_dict,
                    simulator_object_ids,
                    gui_info_cls=gui_info_cls,
                    transform_root=pass_transform_root,
                )
                frame_paths = self._sorted_frame_paths(pass_transform_root)
                pass_report["trajectory_snapshot_count"] = len(frame_paths)
                if frame_paths:
                    pass_report["trajectory_last_frame"] = str(frame_paths[-1])
                for object_id, transform_matrix in simulated_transformations.items():
                    working_transforms[object_id] = transform_matrix
                last_simulator_metrics = simulator_metrics
                all_passes_static = all_passes_static and (
                    simulator_metrics.get("forward_simulator_stopped_static", 0.0) > 0.0
                )
                finished_dynamic_ids = self._ordered_unique_ids(
                    (*finished_dynamic_ids, *dynamic_object_ids)
                )
                pass_report["simulated_object_ids"] = list(simulator_object_ids)
                if support_tree_children_map is not None:
                    pass_report["subtree_xz_propagation"] = self._propagate_pending_subtree_xz(
                        working_transforms,
                        pre_pass_transforms=pre_pass_transforms,
                        source_object_ids=simulator_object_ids,
                        finished_dynamic_ids=tuple(finished_dynamic_ids),
                        children_map=support_tree_children_map,
                    )

            pass_report["status"] = "completed"
            report["completed_pass_count"] = pass_index + 1
            report["active_pass_index"] = (
                pass_index + 1 if pass_index + 1 < len(report["passes"]) else None
            )
            self._write_diff_init_report(args, report)

        metrics: dict[str, float] = {
            "diff_sim_initialization_used": 1.0,
            "diff_sim_progressive_pass_count": float(len(passes)),
            "diff_sim_hierarchical_tree_used": 1.0 if hierarchical_tree_used else 0.0,
            "forward_diff_simulator_used": 1.0 if any_forward_simulation else 0.0,
            "forward_diff_simulator_failed": 0.0,
            "forward_simulation_pass_count": float(len([p for p in report["passes"] if p.get("dynamic_object_ids")])),
        }
        if pass_losses:
            metrics["diff_sim_best_loss"] = float(min(pass_losses))
        if last_simulator_metrics:
            metrics.update(last_simulator_metrics)
            if any_forward_simulation:
                metrics["forward_simulator_stopped_static"] = 1.0 if all_passes_static else 0.0
        adaptive_selected_frames = [
            int(pass_report["adaptive_selected_end_frame"])
            for pass_report in report["passes"]
            if isinstance(pass_report.get("adaptive_selected_end_frame"), int)
        ]
        if adaptive_selected_frames:
            metrics["diff_sim_adaptive_end_frame_selected"] = float(adaptive_selected_frames[-1])
        if any(bool(pass_report.get("adaptive_end_frame_enabled")) for pass_report in report["passes"]):
            metrics["diff_sim_adaptive_end_frame_enabled"] = 1.0
        if any(bool(pass_report.get("adaptive_early_stop_triggered")) for pass_report in report["passes"]):
            metrics["diff_sim_adaptive_early_stop_triggered"] = 1.0
        if requested_ground_y is not None:
            metrics["requested_ground_y_value"] = requested_ground_y
            metrics["applied_ground_y_value"] = (
                last_applied_ground_y if last_applied_ground_y is not None else requested_ground_y
            )
            report["requested_ground_y_value"] = requested_ground_y
            report["applied_ground_y_value"] = (
                last_applied_ground_y if last_applied_ground_y is not None else requested_ground_y
            )
        report["progressive_pass_count"] = len(report["passes"])
        return working_transforms, metrics, report

    def _clone_args_with_safe_ground(
        self,
        args,
        trimesh_dict: dict[str, trimesh.Trimesh],
    ) -> tuple[object, dict[str, float] | None]:
        cloned_args = copy.copy(args)
        if not hasattr(cloned_args, "ground_y_value") or not trimesh_dict:
            return cloned_args, None

        lowest_mesh_y: float | None = None
        for mesh in trimesh_dict.values():
            vertices = np.asarray(mesh.vertices, dtype=np.float64)
            if vertices.size == 0:
                continue
            mesh_min_y = float(mesh.bounds[0][1])
            if lowest_mesh_y is None or mesh_min_y < lowest_mesh_y:
                lowest_mesh_y = mesh_min_y

        if lowest_mesh_y is None:
            return cloned_args, None

        current_ground_y = float(getattr(cloned_args, "ground_y_value"))
        adjusted_ground_y = min(
            current_ground_y,
            lowest_mesh_y - DIFF_INIT_GROUND_CLEARANCE,
        )
        if adjusted_ground_y >= current_ground_y:
            return cloned_args, None
        setattr(cloned_args, "ground_y_value", adjusted_ground_y)
        return cloned_args, {
            "ground_y_adjusted_from": current_ground_y,
            "ground_y_adjusted_to": adjusted_ground_y,
            "lowest_mesh_y": lowest_mesh_y,
        }

    def _build_diff_init_plan(
        self,
        physics_ready_scene: PhysicsReadyScene,
        object_ids: tuple[str, ...],
        trimesh_dict: dict[str, trimesh.Trimesh],
    ) -> tuple[tuple[dict[str, object], ...], bool, str | None]:
        heuristic_pass = (
            {
                "container_name": self._select_container_name(physics_ready_scene, trimesh_dict),
                "optimizer_object_ids": object_ids,
                "dynamic_object_ids": object_ids,
                "frozen_object_ids": tuple(),
                "strategy": "heuristic",
            },
        )
        filtered_tree = self._build_filtered_support_tree(physics_ready_scene, object_ids)
        children_map, _parent_map, deduped_roots, fallback_reason = filtered_tree
        if fallback_reason is not None or children_map is None or deduped_roots is None:
            return heuristic_pass, False, fallback_reason
        passes: list[dict[str, object]] = []
        passes.append(
            {
                "container_name": "ground",
                "optimizer_object_ids": tuple(),
                "dynamic_object_ids": deduped_roots,
                "frozen_object_ids": tuple(),
                "strategy": "ground_drop",
            }
        )
        for root_object_id in deduped_roots:
            descendant_ids = tuple(self._descendant_object_ids(children_map, root_object_id))
            if not descendant_ids:
                continue
            optimizer_object_ids = (root_object_id, *descendant_ids)
            frozen_object_ids = tuple()
            passes.append(
                {
                    "container_name": root_object_id,
                    "optimizer_object_ids": optimizer_object_ids,
                    "dynamic_object_ids": descendant_ids,
                    "frozen_object_ids": frozen_object_ids,
                    "strategy": "subtree",
                }
            )

        return tuple(passes), True, None

    def _build_filtered_support_tree(
        self,
        physics_ready_scene: PhysicsReadyScene,
        object_ids: tuple[str, ...],
    ) -> tuple[
        dict[str, list[str]] | None,
        dict[str, str] | None,
        tuple[str, ...] | None,
        str | None,
    ]:
        support_graph = physics_ready_scene.layout.support_graph
        object_id_set = set(object_ids)
        parent_map: dict[str, str] = {}
        children_map: dict[str, list[str]] = {object_id: [] for object_id in object_ids}
        if support_graph is None or not support_graph.relations:
            return children_map, parent_map, tuple(self._ordered_unique_ids(object_ids)), None

        allowed_relation_types = {
            RelationType.SUPPORTS.value,
            RelationType.ON.value,
            RelationType.CONTAINS.value,
            RelationType.IN.value,
        }
        relation_nodes: set[str] = set()
        seen_edges: set[tuple[str, str]] = set()

        for relation in support_graph.relations:
            relation_type = self._normalize_relation_type(relation.relation_type)
            if relation_type not in allowed_relation_types:
                continue
            parent_id = relation.parent_object_id
            child_id = relation.child_object_id
            if parent_id not in object_id_set or child_id not in object_id_set:
                continue
            relation_nodes.add(parent_id)
            relation_nodes.add(child_id)
            edge = (parent_id, child_id)
            if edge in seen_edges:
                continue
            seen_edges.add(edge)
            existing_parent = parent_map.get(child_id)
            if existing_parent is not None and existing_parent != parent_id:
                return None, None, None, f"multi_parent:{child_id}"
            parent_map[child_id] = parent_id
            children_map.setdefault(parent_id, []).append(child_id)
            children_map.setdefault(child_id, [])

        if not relation_nodes:
            return children_map, parent_map, tuple(self._ordered_unique_ids(object_ids)), None

        roots: list[str] = []
        for object_id in getattr(support_graph, "root_object_ids", ()) or ():
            if object_id in object_id_set and object_id not in parent_map:
                roots.append(object_id)
        for object_id in object_ids:
            if object_id not in parent_map and object_id not in roots:
                roots.append(object_id)
        if not roots:
            return None, None, None, "missing_roots"

        return children_map, parent_map, tuple(self._ordered_unique_ids(tuple(roots))), None

    def _normalize_relation_type(self, relation_type: RelationType | str) -> str:
        if isinstance(relation_type, RelationType):
            return relation_type.value
        return str(relation_type).strip().lower()

    def _ordered_unique_ids(self, object_ids: tuple[str, ...] | list[str]) -> list[str]:
        ordered: list[str] = []
        seen: set[str] = set()
        for object_id in object_ids:
            if object_id in seen:
                continue
            ordered.append(object_id)
            seen.add(object_id)
        return ordered

    def _apply_transformations_to_trimeshes(
        self,
        trimesh_dict: dict[str, trimesh.Trimesh],
        transformations: dict[str, list[list[float]]],
    ) -> None:
        for object_id, mesh in trimesh_dict.items():
            transform_matrix = transformations.get(object_id)
            if transform_matrix is None:
                continue
            mesh.apply_transform(np.asarray(transform_matrix, dtype=np.float64))

    def _merge_xz_transform(
        self,
        base_transform,
        optimized_transform,
    ) -> list[list[float]]:
        merged = np.asarray(
            base_transform if base_transform is not None else self._identity_transform(),
            dtype=np.float64,
        ).copy()
        if optimized_transform is None:
            return merged.tolist()
        optimized_matrix = np.asarray(optimized_transform, dtype=np.float64)
        merged[0, 3] = float(optimized_matrix[0, 3])
        merged[2, 3] = float(optimized_matrix[2, 3])
        return merged.tolist()

    def _propagate_pending_subtree_xz(
        self,
        working_transforms: dict[str, list[list[float]]],
        *,
        pre_pass_transforms: dict[str, list[list[float]]],
        source_object_ids: tuple[str, ...],
        finished_dynamic_ids: tuple[str, ...],
        children_map: dict[str, list[str]],
    ) -> dict[str, object]:
        finished_dynamic_id_set = set(finished_dynamic_ids)
        pending_ids = {
            object_id
            for object_id in working_transforms
            if object_id not in finished_dynamic_id_set
        }
        moved_source_object_ids: list[str] = []
        propagated_target_object_ids: list[str] = []
        source_reports: list[dict[str, object]] = []
        for source_object_id in source_object_ids:
            all_descendants = self._descendant_object_ids(children_map, source_object_id)
            target_object_ids = [
                object_id for object_id in all_descendants
                if object_id in pending_ids
            ]
            if not target_object_ids:
                continue
            delta_x, delta_z = self._xz_translation_delta(
                pre_pass_transforms.get(source_object_id),
                working_transforms.get(source_object_id),
            )
            if np.isclose(delta_x, 0.0) and np.isclose(delta_z, 0.0):
                continue
            moved_source_object_ids.append(source_object_id)
            propagated_target_object_ids.extend(target_object_ids)
            source_reports.append(
                {
                    "source_object_id": source_object_id,
                    "delta_x": float(delta_x),
                    "delta_z": float(delta_z),
                    "target_object_ids": list(target_object_ids),
                }
            )
            for target_object_id in target_object_ids:
                working_transforms[target_object_id] = self._translate_xz_transform(
                    working_transforms[target_object_id],
                    delta_x=delta_x,
                    delta_z=delta_z,
                )
        return {
            "moved_source_object_ids": moved_source_object_ids,
            "propagated_target_object_ids": self._ordered_unique_ids(propagated_target_object_ids),
            "sources": source_reports,
        }

    def _descendant_object_ids(
        self,
        children_map: dict[str, list[str]],
        source_object_id: str,
    ) -> list[str]:
        descendants: list[str] = []
        queue = list(children_map.get(source_object_id, ()))
        while queue:
            child_object_id = queue.pop(0)
            descendants.append(child_object_id)
            queue.extend(children_map.get(child_object_id, ()))
        return descendants

    def _xz_translation_delta(
        self,
        before_transform,
        after_transform,
    ) -> tuple[float, float]:
        before_matrix = np.asarray(
            before_transform if before_transform is not None else self._identity_transform(),
            dtype=np.float64,
        )
        after_matrix = np.asarray(
            after_transform if after_transform is not None else self._identity_transform(),
            dtype=np.float64,
        )
        return (
            float(after_matrix[0, 3] - before_matrix[0, 3]),
            float(after_matrix[2, 3] - before_matrix[2, 3]),
        )

    def _translate_xz_transform(
        self,
        transform_matrix,
        *,
        delta_x: float,
        delta_z: float,
    ) -> list[list[float]]:
        translated = np.asarray(
            transform_matrix if transform_matrix is not None else self._identity_transform(),
            dtype=np.float64,
        ).copy()
        translated[0, 3] += float(delta_x)
        translated[2, 3] += float(delta_z)
        return translated.tolist()
