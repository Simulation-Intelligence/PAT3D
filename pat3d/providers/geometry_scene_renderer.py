from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Callable

import numpy as np
from PIL import Image, ImageDraw
import trimesh

from pat3d.contracts import SceneRenderer
from pat3d.models import (
    ArtifactRef,
    ObjectPose,
    PhysicsOptimizationResult,
    PhysicsReadyScene,
    RenderResult,
    SceneLayout,
)
from pat3d.models.pose_utils import delta_transform_dict, pose_to_matrix
from pat3d.storage import make_stage_metadata


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class GeometrySceneRenderer(SceneRenderer):
    def __init__(
        self,
        *,
        output_root: str = "results/rendered_images",
        layout_root: str = "data/layout",
        raw_asset_root: str = "data/raw_obj",
        preview_size: tuple[int, int] = (1400, 960),
        preview_count: int = 12,
        preview_timeout_seconds: float = 90.0,
        metadata_factory: Callable[..., object] = make_stage_metadata,
    ) -> None:
        self._output_root = Path(output_root)
        self._layout_root = Path(layout_root)
        self._raw_asset_root = Path(raw_asset_root)
        self._preview_size = preview_size
        self._preview_count = max(1, min(24, int(preview_count)))
        self._preview_timeout_seconds = max(5.0, float(preview_timeout_seconds))
        self._metadata_factory = metadata_factory

    def render(
        self,
        scene_state: SceneLayout | PhysicsReadyScene | PhysicsOptimizationResult,
    ) -> RenderResult:
        scene_id = scene_state.scene_id
        output_dir = self._output_root / scene_id
        output_dir.mkdir(parents=True, exist_ok=True)

        bundle = self._build_scene_bundle(scene_state, use_simulation_meshes=True)
        bundle_path = output_dir / "scene_bundle.json"
        bundle_path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        preview_bundle = self._build_scene_bundle(
            scene_state,
            use_simulation_meshes=False,
            prefer_layout_meshes=True,
        )
        preview_bundle_path = output_dir / "scene_bundle_rendered.json"
        preview_bundle_path.write_text(json.dumps(preview_bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        render_images, render_mode, render_notes = self._render_geometry_previews(
            preview_bundle,
            preview_bundle_path,
            output_dir,
            output_prefix="geometry_preview",
            role_prefix="geometry_preview",
        )

        return RenderResult(
            scene_id=scene_id,
            render_images=tuple(render_images),
            camera_metadata=ArtifactRef(
                artifact_type="scene_bundle",
                path=str(bundle_path),
                format="json",
                role="scene_bundle",
                metadata_path=None,
            ),
            render_config={
                "mode": render_mode,
                "viewer": "threejs",
                "projection": "perspective_mesh_render",
                "preview_count": len(render_images),
                "variants": ("full_geometry", "simplified_geometry"),
            },
            metadata=self._metadata_factory(
                stage_name="visualization",
                provider_name="geometry_scene_renderer",
                notes=render_notes,
            ),
        )

    def _build_scene_bundle(
        self,
        scene_state: SceneLayout | PhysicsReadyScene | PhysicsOptimizationResult,
        *,
        use_simulation_meshes: bool = False,
        prefer_layout_meshes: bool = False,
    ) -> dict[str, object]:
        layout = self._resolve_layout(scene_state)
        initial_pose_map = self._initial_pose_map(scene_state)
        optimized_pose_map = self._optimized_pose_map(scene_state)
        mesh_path_by_object_id = self._preferred_mesh_paths(
            scene_state,
            layout,
            use_simulation_meshes=use_simulation_meshes,
            prefer_layout_meshes=prefer_layout_meshes,
        )
        requested_ground_y, applied_ground_y, ground_source = self._resolve_ground_plane(scene_state)
        objects: list[dict[str, object]] = []

        for pose in layout.object_poses:
            object_id = pose.object_id
            mesh_path = mesh_path_by_object_id.get(object_id)
            if mesh_path is None:
                continue
            mtl_path, texture_path = self._resolve_mesh_artifacts(
                mesh_path,
                scene_id=scene_state.scene_id,
                object_id=object_id,
            )
            transform = self._bundle_transform_for_object(
                object_id,
                initial_pose_map=initial_pose_map,
                optimized_pose_map=optimized_pose_map,
            )
            objects.append(
                {
                    "object_id": object_id,
                    "mesh_obj_path": str(mesh_path),
                    "mesh_mtl_path": str(mtl_path) if mtl_path is not None else None,
                    "texture_image_path": str(texture_path) if texture_path is not None else None,
                    "already_transformed": transform is None,
                    "transform": transform,
                }
            )

        return {
            "scene_id": scene_state.scene_id,
            "generated_at": _now_iso(),
            "source_scene_type": type(scene_state).__name__,
            "geometry_source_type": (
                "simulation_mesh_artifacts"
                if (use_simulation_meshes or self._scene_mesh_artifacts(scene_state))
                else "layout_mesh_artifacts"
            ),
            "geometry_variant": "simplified" if (use_simulation_meshes or self._scene_mesh_artifacts(scene_state)) else "full",
            "requested_ground_plane_y": requested_ground_y,
            "applied_ground_plane_y": applied_ground_y,
            "ground_plane_source": ground_source,
            "objects": objects,
        }

    def _resolve_ground_plane(
        self,
        scene_state: SceneLayout | PhysicsReadyScene | PhysicsOptimizationResult,
    ) -> tuple[float | None, float | None, str | None]:
        if isinstance(scene_state, SceneLayout):
            return None, None, None

        if isinstance(scene_state, PhysicsReadyScene):
            collision_ground_y = self._collision_ground_y(scene_state.collision_settings)
            if collision_ground_y is not None:
                return collision_ground_y, collision_ground_y, "physics_collision_settings"
            return -1.1, -1.1, "physics_legacy_default"

        metrics = scene_state.metrics or {}
        requested_ground_y = self._numeric_value(metrics.get("requested_ground_y_value"))
        applied_ground_y = self._numeric_value(metrics.get("applied_ground_y_value"))
        if requested_ground_y is not None or applied_ground_y is not None:
            collision_ground_y = self._collision_ground_y(scene_state.initial_scene.collision_settings)
            resolved_requested = (
                requested_ground_y
                if requested_ground_y is not None
                else (collision_ground_y if collision_ground_y is not None else -1.1)
            )
            resolved_applied = (
                applied_ground_y
                if applied_ground_y is not None
                else resolved_requested
            )
            return resolved_requested, resolved_applied, "physics_metrics"

        report = self._read_diff_init_report(scene_state)
        report_applied_ground_y = self._ground_y_from_report(report, key="applied_ground_y_value")
        if report_applied_ground_y is None:
            report_applied_ground_y = self._ground_y_from_report(report, key="ground_y_applied")
        if report_applied_ground_y is None:
            report_applied_ground_y = self._ground_y_from_report(report, key="ground_y_adjusted_to")
        if report_applied_ground_y is not None:
            report_requested_ground_y = self._ground_y_from_report(report, key="requested_ground_y_value")
            if report_requested_ground_y is None:
                report_requested_ground_y = self._ground_y_from_report(report, key="ground_y_adjusted_from")
            if report_requested_ground_y is None:
                collision_ground_y = self._collision_ground_y(scene_state.initial_scene.collision_settings)
                report_requested_ground_y = collision_ground_y if collision_ground_y is not None else -1.1
            return report_requested_ground_y, report_applied_ground_y, "physics_diff_init_adjusted"

        collision_ground_y = self._collision_ground_y(scene_state.initial_scene.collision_settings)
        if collision_ground_y is not None:
            return collision_ground_y, collision_ground_y, "physics_collision_settings"
        return -1.1, -1.1, "physics_legacy_default"

    def _collision_ground_y(self, collision_settings: dict[str, float | int | bool] | None) -> float | None:
        if not isinstance(collision_settings, dict):
            return None
        return self._numeric_value(collision_settings.get("ground_y_value"))

    def _numeric_value(self, value: object) -> float | None:
        if isinstance(value, bool):
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(numeric):
            return None
        return numeric

    def _read_diff_init_report(self, scene_state: PhysicsOptimizationResult) -> dict[str, object] | None:
        for artifact in scene_state.artifacts:
            if artifact.role != "diff_sim_report":
                continue
            try:
                return json.loads(Path(artifact.path).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return None
        return None

    def _ground_y_from_report(self, report: dict[str, object] | None, *, key: str) -> float | None:
        if not isinstance(report, dict):
            return None
        direct_value = self._numeric_value(report.get(key))
        if direct_value is not None:
            return direct_value
        passes = report.get("passes")
        if not isinstance(passes, list):
            return None
        for pass_report in reversed(passes):
            if not isinstance(pass_report, dict):
                continue
            pass_value = self._numeric_value(pass_report.get(key))
            if pass_value is not None:
                return pass_value
        return None

    def _resolve_layout(
        self,
        scene_state: SceneLayout | PhysicsReadyScene | PhysicsOptimizationResult,
    ) -> SceneLayout:
        if isinstance(scene_state, SceneLayout):
            return scene_state
        if isinstance(scene_state, PhysicsReadyScene):
            return scene_state.layout
        return scene_state.initial_scene.layout

    def _initial_pose_map(
        self,
        scene_state: SceneLayout | PhysicsReadyScene | PhysicsOptimizationResult,
    ) -> dict[str, ObjectPose]:
        if isinstance(scene_state, PhysicsOptimizationResult):
            return {
                pose.object_id: pose
                for pose in scene_state.initial_scene.object_poses
            }
        return {}

    def _optimized_pose_map(
        self,
        scene_state: SceneLayout | PhysicsReadyScene | PhysicsOptimizationResult,
    ) -> dict[str, ObjectPose]:
        if isinstance(scene_state, PhysicsOptimizationResult):
            return {
                pose.object_id: pose
                for pose in scene_state.optimized_object_poses
            }
        return {}

    def _preferred_mesh_paths(
        self,
        scene_state: SceneLayout | PhysicsReadyScene | PhysicsOptimizationResult,
        layout: SceneLayout,
        *,
        use_simulation_meshes: bool = False,
        prefer_layout_meshes: bool = False,
    ) -> dict[str, Path]:
        path_by_object_id: dict[str, Path] = {}

        if use_simulation_meshes:
            for artifact in self._scene_mesh_artifacts(scene_state):
                if artifact.role is None:
                    continue
                path_by_object_id.setdefault(artifact.role, Path(artifact.path))
        else:
            for pose in layout.object_poses:
                if not prefer_layout_meshes:
                    raw_mesh_path = self._raw_asset_mesh_path(scene_state.scene_id, pose.object_id)
                    if raw_mesh_path is not None:
                        path_by_object_id.setdefault(pose.object_id, raw_mesh_path)
                        continue

                # Fall back to the materialized layout mesh when no raw asset mesh exists,
                # or prefer it outright when preview rendering needs scene-space geometry.
                for candidate_name in self._mesh_path_candidates(pose.object_id):
                    layout_mesh_path = self._layout_root / scene_state.scene_id / f"{candidate_name}.obj"
                    if layout_mesh_path.exists():
                        path_by_object_id.setdefault(pose.object_id, layout_mesh_path)
                        break

                if prefer_layout_meshes and pose.object_id not in path_by_object_id:
                    raw_mesh_path = self._raw_asset_mesh_path(scene_state.scene_id, pose.object_id)
                    if raw_mesh_path is not None:
                        path_by_object_id.setdefault(pose.object_id, raw_mesh_path)

            for artifact in self._scene_mesh_artifacts(scene_state):
                if artifact.role is None:
                    continue
                path_by_object_id.setdefault(artifact.role, Path(artifact.path))

        for artifact in layout.artifacts:
            if artifact.artifact_type != "layout_mesh" or artifact.role is None:
                continue
            path_by_object_id.setdefault(artifact.role, Path(artifact.path))

        return path_by_object_id

    def _raw_asset_directory_candidates(self, scene_id: str, object_id: str) -> tuple[Path, ...]:
        normalized_scene_id = scene_id.replace(":", "_")
        normalized_object_id = object_id.replace(":", "_")
        return (
            self._raw_asset_root / object_id,
            self._raw_asset_root / scene_id / object_id,
            self._raw_asset_root / normalized_scene_id / object_id,
            self._raw_asset_root / normalized_scene_id / normalized_object_id,
            self._raw_asset_root / normalized_object_id,
        )

    def _raw_asset_mesh_path(self, scene_id: str, object_id: str) -> Path | None:
        for candidate_directory in self._raw_asset_directory_candidates(scene_id, object_id):
            if not candidate_directory.is_dir():
                continue
            obj_candidates = sorted(
                candidate_directory.glob("*.obj"),
                key=self._raw_asset_mesh_sort_key,
            )
            if obj_candidates:
                return obj_candidates[0]
        return None

    def _raw_asset_mesh_sort_key(self, mesh_path: Path) -> tuple[int, int, str]:
        lower_name = mesh_path.name.lower()
        if "texture" in lower_name:
            priority = 0
        elif "mesh" in lower_name:
            priority = 1
        else:
            priority = 2
        return priority, len(lower_name), lower_name

    def _scene_mesh_artifacts(
        self,
        scene_state: SceneLayout | PhysicsReadyScene | PhysicsOptimizationResult,
    ) -> tuple[ArtifactRef, ...]:
        if isinstance(scene_state, PhysicsOptimizationResult):
            return tuple(scene_state.initial_scene.simulation_meshes)
        if isinstance(scene_state, PhysicsReadyScene):
            return tuple(scene_state.simulation_meshes)
        return ()

    def _mesh_path_candidates(self, object_id: str) -> tuple[str, ...]:
        if ":" in object_id:
            return (object_id.split(":", 1)[1], object_id)
        return (object_id,)

    def _bundle_transform_for_object(
        self,
        object_id: str,
        *,
        initial_pose_map: dict[str, ObjectPose],
        optimized_pose_map: dict[str, ObjectPose],
    ) -> dict[str, object] | None:
        initial_pose = initial_pose_map.get(object_id)
        optimized_pose = optimized_pose_map.get(object_id)
        if initial_pose is None or optimized_pose is None:
            return None
        return delta_transform_dict(initial_pose, optimized_pose)

    def _render_geometry_preview(self, bundle: dict[str, object], output_path: Path) -> None:
        width, height = self._preview_size
        objects: list[tuple[object, tuple[int, int, int], dict[str, object]]] = []
        min_corner: np.ndarray | None = None
        max_corner: np.ndarray | None = None

        for index, item in enumerate(bundle.get("objects", ())):
            if not isinstance(item, dict):
                continue
            mesh_path = item.get("mesh_obj_path")
            if not isinstance(mesh_path, str):
                continue
            mesh = self._load_preview_mesh(mesh_path)
            if mesh is None:
                continue
            self._apply_bundle_transform(mesh, item)

            bounds = np.asarray(mesh.get_axis_aligned_bounding_box().get_box_points(), dtype=float)
            mesh_min = bounds.min(axis=0)
            mesh_max = bounds.max(axis=0)
            min_corner = mesh_min if min_corner is None else np.minimum(min_corner, mesh_min)
            max_corner = mesh_max if max_corner is None else np.maximum(max_corner, mesh_max)

            color = self._sample_object_color(
                texture_path=item.get("texture_image_path"),
                fallback_index=index,
            )
            objects.append((mesh, color, item))

        if not objects or min_corner is None or max_corner is None:
            self._write_empty_preview(bundle, output_path, width, height)
            return

        import open3d as o3d

        center = (min_corner + max_corner) / 2.0
        extent = max(float(np.max(max_corner - min_corner)), 1.0)
        eye = center + np.array([extent * 1.25, extent * 0.95, -extent * 1.7], dtype=float)
        up = np.array([0.0, 1.0, 0.0], dtype=float)
        ground_plane_y = self._numeric_value(bundle.get("applied_ground_plane_y"))

        renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
        try:
            scene = renderer.scene
            scene.set_background(np.array([246 / 255, 241 / 255, 231 / 255, 1.0], dtype=np.float32))
            try:
                scene.set_lighting(
                    o3d.visualization.rendering.Open3DScene.LightingProfile.SOFT_SHADOWS,
                    (0.577, -0.577, -0.577),
                )
            except Exception:
                pass

            for index, (mesh, color, item) in enumerate(objects):
                material = o3d.visualization.rendering.MaterialRecord()
                material.shader = "defaultLit"
                texture_path = item.get("texture_image_path")
                if isinstance(texture_path, str) and Path(texture_path).exists():
                    try:
                        material.albedo_img = o3d.io.read_image(texture_path)
                        material.base_color = (1.0, 1.0, 1.0, 1.0)
                    except Exception:
                        material.base_color = (
                            float(color[0]) / 255.0,
                            float(color[1]) / 255.0,
                            float(color[2]) / 255.0,
                            1.0,
                        )
                else:
                    material.base_color = (
                        float(color[0]) / 255.0,
                        float(color[1]) / 255.0,
                        float(color[2]) / 255.0,
                        1.0,
                    )
                material.base_roughness = 0.82
                material.base_reflectance = 0.08
                scene.add_geometry(f"object_{index}", mesh, material)

            if ground_plane_y is not None:
                self._add_preview_floor(
                    scene,
                    o3d,
                    center=center,
                    extent=extent,
                    ground_plane_y=ground_plane_y,
                )

            renderer.setup_camera(45.0, center, eye, up)
            image = renderer.render_to_image()
            o3d.io.write_image(str(output_path), image, 9)
        finally:
            renderer.scene.clear_geometry()

    def _render_geometry_previews(
        self,
        bundle: dict[str, object],
        bundle_path: Path,
        output_dir: Path,
        *,
        output_prefix: str,
        role_prefix: str,
    ) -> tuple[list[ArtifactRef], str, tuple[str, ...]]:
        for preview_path in self._preview_paths_for_prefix(output_dir, output_prefix):
            preview_path.unlink(missing_ok=True)

        render_result = self._run_preview_renderer_subprocess(bundle_path, output_dir, output_prefix=output_prefix)
        preview_paths = self._collect_preview_paths(output_dir, output_prefix=output_prefix)
        if preview_paths:
            return self._artifact_refs_from_preview_paths(preview_paths, role_prefix=role_prefix), role_prefix, ()

        fallback_reason = (
            render_result.stderr.strip()
            or render_result.stdout.strip()
            or (f"preview subprocess exited with code {render_result.returncode}" if render_result.returncode else "")
        )
        fallback_paths = self._write_fallback_previews(bundle, output_dir, output_prefix=output_prefix)
        notes = ()
        if fallback_reason:
            notes = (f"{role_prefix}_fallback={fallback_reason}",)
        return (
            self._artifact_refs_from_preview_paths(fallback_paths, role_prefix=role_prefix),
            f"{role_prefix}_fallback",
            notes,
        )

    def _preview_paths_for_prefix(self, output_dir: Path, output_prefix: str) -> list[Path]:
        preview_paths = [output_dir / f"{output_prefix}.png"]
        preview_paths.extend(sorted(output_dir.glob(f"{output_prefix}_*.png")))
        return [preview_path for preview_path in preview_paths if preview_path.exists()]

    def _collect_preview_paths(self, output_dir: Path, *, output_prefix: str) -> list[Path]:
        preview_paths = self._preview_paths_for_prefix(output_dir, output_prefix)
        if not preview_paths:
            return []

        unique_preview_paths: list[Path] = []
        seen_paths: set[Path] = set()
        for preview_path in preview_paths:
            if preview_path in seen_paths:
                continue
            seen_paths.add(preview_path)
            unique_preview_paths.append(preview_path)
        return unique_preview_paths

    def _artifact_refs_from_preview_paths(
        self,
        preview_paths: list[Path],
        *,
        role_prefix: str,
    ) -> list[ArtifactRef]:
        return [
            ArtifactRef(
                artifact_type="render_image",
                path=str(preview_path),
                format="png",
                role=f"{role_prefix}_{index:02d}",
                metadata_path=None,
            )
            for index, preview_path in enumerate(preview_paths, start=1)
        ]

    def _write_fallback_previews(
        self,
        bundle: dict[str, object],
        output_dir: Path,
        *,
        output_prefix: str,
    ) -> list[Path]:
        width, height = self._preview_size
        preview_paths: list[Path] = []
        total_previews = max(1, self._preview_count)
        for preview_index in range(1, total_previews + 1):
            preview_path = (
                output_dir / f"{output_prefix}.png"
                if preview_index == 1
                else output_dir / f"{output_prefix}_{preview_index:02d}.png"
            )
            self._write_fallback_projection_preview(
                bundle,
                preview_path,
                width,
                height,
                preview_index=preview_index,
                total_previews=total_previews,
            )
            preview_paths.append(preview_path)
        return preview_paths

    def _run_preview_renderer_subprocess(
        self,
        bundle_path: Path,
        output_dir: Path,
        *,
        output_prefix: str,
    ) -> subprocess.CompletedProcess[str]:
        repo_root = Path(__file__).resolve().parents[2]
        width, height = self._preview_size
        command = [
            sys.executable,
            "-m",
            "pat3d.providers.geometry_scene_renderer",
            "--render-bundle",
            str(bundle_path),
            "--output-dir",
            str(output_dir),
            "--width",
            str(width),
            "--height",
            str(height),
            "--preview-count",
            str(self._preview_count),
            "--output-prefix",
            output_prefix,
        ]
        try:
            return subprocess.run(
                command,
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
                timeout=self._preview_timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            stderr = (exc.stderr or "").strip()
            stdout = (exc.stdout or "").strip()
            timeout_note = (
                f"preview subprocess timed out after {self._preview_timeout_seconds:.0f}s"
            )
            if stderr:
                stderr = f"{stderr}\n{timeout_note}"
            else:
                stderr = timeout_note
            return subprocess.CompletedProcess(
                args=command,
                returncode=-9,
                stdout=stdout,
                stderr=stderr,
            )

    def _preview_camera_eyes(self, *, center: np.ndarray, extent: float, count: int) -> list[np.ndarray]:
        horizontal_radius = extent * 2.05
        vertical_offset = extent * 0.95
        start_angle = np.deg2rad(-54.0)
        step = (2.0 * np.pi) / max(count, 1)
        eyes: list[np.ndarray] = []
        for index in range(count):
            angle = start_angle + (step * index)
            eyes.append(
                center
                + np.array(
                    [
                        np.cos(angle) * horizontal_radius,
                        vertical_offset,
                        np.sin(angle) * horizontal_radius,
                    ],
                    dtype=float,
                )
            )
        return eyes

    def _render_geometry_previews_in_process(
        self,
        bundle: dict[str, object],
        output_dir: Path,
        *,
        output_prefix: str,
    ) -> None:
        primary_preview_path = output_dir / f"{output_prefix}.png"
        self._render_geometry_preview(bundle, primary_preview_path)
        if self._preview_count <= 1:
            return

        width, height = self._preview_size
        objects: list[tuple[object, tuple[int, int, int], dict[str, object]]] = []
        min_corner: np.ndarray | None = None
        max_corner: np.ndarray | None = None

        for index, item in enumerate(bundle.get("objects", ())):
            if not isinstance(item, dict):
                continue
            mesh_path = item.get("mesh_obj_path")
            if not isinstance(mesh_path, str):
                continue
            mesh = self._load_preview_mesh(mesh_path)
            if mesh is None:
                continue
            self._apply_bundle_transform(mesh, item)

            bounds = np.asarray(mesh.get_axis_aligned_bounding_box().get_box_points(), dtype=float)
            mesh_min = bounds.min(axis=0)
            mesh_max = bounds.max(axis=0)
            min_corner = mesh_min if min_corner is None else np.minimum(min_corner, mesh_min)
            max_corner = mesh_max if max_corner is None else np.maximum(max_corner, mesh_max)

            color = self._sample_object_color(
                texture_path=item.get("texture_image_path"),
                fallback_index=index,
            )
            objects.append((mesh, color, item))

        if not objects or min_corner is None or max_corner is None:
            return

        import open3d as o3d

        center = (min_corner + max_corner) / 2.0
        extent = max(float(np.max(max_corner - min_corner)), 1.0)
        up = np.array([0.0, 1.0, 0.0], dtype=float)
        orbit_eyes = self._preview_camera_eyes(center=center, extent=extent, count=self._preview_count)
        ground_plane_y = self._numeric_value(bundle.get("applied_ground_plane_y"))

        renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
        try:
            scene = renderer.scene
            scene.set_background(np.array([246 / 255, 241 / 255, 231 / 255, 1.0], dtype=np.float32))
            try:
                scene.set_lighting(
                    o3d.visualization.rendering.Open3DScene.LightingProfile.SOFT_SHADOWS,
                    (0.577, -0.577, -0.577),
                )
            except Exception:
                pass

            for index, (mesh, color, item) in enumerate(objects):
                material = o3d.visualization.rendering.MaterialRecord()
                material.shader = "defaultLit"
                texture_path = item.get("texture_image_path")
                if isinstance(texture_path, str) and Path(texture_path).exists():
                    try:
                        material.albedo_img = o3d.io.read_image(texture_path)
                        material.base_color = (1.0, 1.0, 1.0, 1.0)
                    except Exception:
                        material.base_color = (
                            float(color[0]) / 255.0,
                            float(color[1]) / 255.0,
                            float(color[2]) / 255.0,
                            1.0,
                        )
                else:
                    material.base_color = (
                        float(color[0]) / 255.0,
                        float(color[1]) / 255.0,
                        float(color[2]) / 255.0,
                        1.0,
                    )
                material.base_roughness = 0.82
                material.base_reflectance = 0.08
                scene.add_geometry(f"object_{index}", mesh, material)

            if ground_plane_y is not None:
                self._add_preview_floor(
                    scene,
                    o3d,
                    center=center,
                    extent=extent,
                    ground_plane_y=ground_plane_y,
                )

            for preview_index, eye in enumerate(orbit_eyes, start=1):
                preview_path = (
                    primary_preview_path
                    if preview_index == 1
                    else output_dir / f"{output_prefix}_{preview_index:02d}.png"
                )
                renderer.setup_camera(45.0, center, eye, up)
                image = renderer.render_to_image()
                o3d.io.write_image(str(preview_path), image, 9)
        finally:
            renderer.scene.clear_geometry()

    def _add_preview_floor(self, scene, o3d, *, center: np.ndarray, extent: float, ground_plane_y: float) -> None:
        plane_width = max(extent * 2.4, 2.0)
        plane_depth = max(extent * 2.4, 2.0)
        plane_thickness = max(extent * 0.018, 0.02)
        floor_mesh = o3d.geometry.TriangleMesh.create_box(
            width=plane_width,
            height=plane_thickness,
            depth=plane_depth,
        )
        floor_mesh.translate(
            (
                float(center[0] - (plane_width / 2.0)),
                float(ground_plane_y - (plane_thickness / 2.0)),
                float(center[2] - (plane_depth / 2.0)),
            )
        )
        floor_mesh.compute_vertex_normals()

        material = o3d.visualization.rendering.MaterialRecord()
        material.shader = "defaultLit"
        material.base_color = (0.83, 0.81, 0.76, 1.0)
        material.base_roughness = 0.96
        material.base_reflectance = 0.02
        scene.add_geometry("ground_plane", floor_mesh, material)

    def _write_fallback_projection_preview(
        self,
        bundle: dict[str, object],
        output_path: Path,
        width: int,
        height: int,
        *,
        preview_index: int,
        total_previews: int,
    ) -> None:
        scene_objects, min_corner, max_corner = self._fallback_scene_objects(bundle)
        if not scene_objects or min_corner is None or max_corner is None:
            self._write_empty_preview(bundle, output_path, width, height)
            return

        image = Image.new("RGB", (width, height), color=(246, 241, 231))
        draw = ImageDraw.Draw(image, "RGBA")
        center = (min_corner + max_corner) / 2.0
        extent = max(float(np.max(max_corner - min_corner)), 1.0)
        eye = self._preview_camera_eyes(center=center, extent=extent, count=total_previews)[preview_index - 1]
        target = center
        up = np.array([0.0, 1.0, 0.0], dtype=float)
        projected_floor = self._fallback_floor_polygon(bundle, center=center, extent=extent, eye=eye, target=target, up=up, width=width, height=height)
        if projected_floor is not None:
            draw.polygon(projected_floor, fill=(223, 216, 203, 255), outline=(194, 186, 172, 255))

        face_draws: list[tuple[float, list[tuple[float, float]], tuple[int, int, int], int]] = []
        for obj in scene_objects:
            corners = obj["corners"]
            color = obj["color"]
            face_draws.extend(
                self._fallback_project_box_faces(
                    corners,
                    color=color,
                    eye=eye,
                    target=target,
                    up=up,
                    width=width,
                    height=height,
                )
            )
        face_draws.sort(key=lambda item: item[0], reverse=True)
        for _depth, polygon, fill_color, outline_alpha in face_draws:
            rgba = (fill_color[0], fill_color[1], fill_color[2], 222)
            draw.polygon(polygon, fill=rgba, outline=(42, 42, 42, outline_alpha))

        draw.rounded_rectangle((24, 24, width - 24, height - 24), radius=28, outline=(90, 90, 90), width=2)
        draw.rounded_rectangle((34, 32, 250, 82), radius=18, fill=(255, 255, 255, 210))
        draw.text((52, 44), f"Fallback view {preview_index}/{total_previews}", fill=(38, 38, 38))
        draw.text((52, 64), str(bundle.get("scene_id", "scene")), fill=(82, 82, 82))
        image.save(output_path)

    def _fallback_scene_objects(
        self,
        bundle: dict[str, object],
    ) -> tuple[list[dict[str, object]], np.ndarray | None, np.ndarray | None]:
        scene_objects: list[dict[str, object]] = []
        min_corner: np.ndarray | None = None
        max_corner: np.ndarray | None = None
        for index, item in enumerate(bundle.get("objects", ())):
            if not isinstance(item, dict):
                continue
            mesh_path = item.get("mesh_obj_path")
            if not isinstance(mesh_path, str):
                continue
            try:
                mesh = trimesh.load(mesh_path, force="mesh")
            except Exception:
                continue
            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.dump(concatenate=True)
            if mesh is None or getattr(mesh, "vertices", None) is None or len(mesh.vertices) == 0:
                continue
            bounds = np.asarray(mesh.bounds, dtype=float)
            corners = self._box_corners(bounds[0], bounds[1])
            if not bool(item.get("already_transformed", True)):
                transform = self._bundle_transform_matrix(item)
                corners = self._transform_points(corners, transform)
            obj_min = corners.min(axis=0)
            obj_max = corners.max(axis=0)
            min_corner = obj_min if min_corner is None else np.minimum(min_corner, obj_min)
            max_corner = obj_max if max_corner is None else np.maximum(max_corner, obj_max)
            scene_objects.append(
                {
                    "object_id": str(item.get("object_id") or f"object_{index}"),
                    "corners": corners,
                    "color": self._sample_object_color(
                        texture_path=item.get("texture_image_path"),
                        fallback_index=index,
                    ),
                }
            )
        return scene_objects, min_corner, max_corner

    def _bundle_transform_matrix(self, item: dict[str, object]) -> np.ndarray:
        transform = item.get("transform")
        if not isinstance(transform, dict):
            return np.eye(4, dtype=float)
        translation = tuple(transform.get("translation_xyz") or (0.0, 0.0, 0.0))
        rotation_value = tuple(transform.get("rotation_value") or (1.0, 0.0, 0.0, 0.0))
        scale_value = transform.get("scale_xyz")
        scale_xyz = tuple(scale_value[:3]) if isinstance(scale_value, (list, tuple)) else None
        pose = ObjectPose(
            object_id=str(item.get("object_id") or "object"),
            translation_xyz=tuple(float(value) for value in translation[:3]),
            rotation_type=str(transform.get("rotation_type") or "quaternion"),
            rotation_value=tuple(float(value) for value in rotation_value),
            scale_xyz=(tuple(float(value) for value in scale_xyz) if scale_xyz is not None else None),
        )
        return np.asarray(pose_to_matrix(pose), dtype=float)

    def _box_corners(self, min_corner: np.ndarray, max_corner: np.ndarray) -> np.ndarray:
        xmin, ymin, zmin = [float(value) for value in min_corner]
        xmax, ymax, zmax = [float(value) for value in max_corner]
        return np.asarray(
            [
                [xmin, ymin, zmin],
                [xmax, ymin, zmin],
                [xmin, ymax, zmin],
                [xmax, ymax, zmin],
                [xmin, ymin, zmax],
                [xmax, ymin, zmax],
                [xmin, ymax, zmax],
                [xmax, ymax, zmax],
            ],
            dtype=float,
        )

    def _transform_points(self, points: np.ndarray, transform: np.ndarray) -> np.ndarray:
        homogeneous = np.concatenate([points, np.ones((points.shape[0], 1), dtype=float)], axis=1)
        transformed = homogeneous @ transform.T
        return transformed[:, :3]

    def _project_points(
        self,
        points: np.ndarray,
        *,
        eye: np.ndarray,
        target: np.ndarray,
        up: np.ndarray,
        width: int,
        height: int,
        fov_degrees: float = 45.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        forward = target - eye
        forward = forward / max(np.linalg.norm(forward), 1e-8)
        right = np.cross(forward, up)
        right = right / max(np.linalg.norm(right), 1e-8)
        true_up = np.cross(right, forward)
        camera = np.stack([right, true_up, forward], axis=0)
        camera_points = (points - eye) @ camera.T
        focal = 0.5 * width / np.tan(np.deg2rad(fov_degrees) / 2.0)
        depth = np.maximum(camera_points[:, 2], 1e-3)
        screen_x = (camera_points[:, 0] * focal / depth) + (width / 2.0)
        screen_y = (-camera_points[:, 1] * focal / depth) + (height / 2.0)
        return np.stack([screen_x, screen_y], axis=1), camera_points[:, 2]

    def _fallback_floor_polygon(
        self,
        bundle: dict[str, object],
        *,
        center: np.ndarray,
        extent: float,
        eye: np.ndarray,
        target: np.ndarray,
        up: np.ndarray,
        width: int,
        height: int,
    ) -> list[tuple[float, float]] | None:
        ground_plane_y = self._numeric_value(bundle.get("applied_ground_plane_y"))
        if ground_plane_y is None:
            return None
        half = max(extent * 1.25, 1.0)
        floor = np.asarray(
            [
                [center[0] - half, ground_plane_y, center[2] - half],
                [center[0] + half, ground_plane_y, center[2] - half],
                [center[0] + half, ground_plane_y, center[2] + half],
                [center[0] - half, ground_plane_y, center[2] + half],
            ],
            dtype=float,
        )
        projected, depth = self._project_points(floor, eye=eye, target=target, up=up, width=width, height=height)
        if np.any(depth <= 0.0):
            return None
        return [tuple(point) for point in projected]

    def _fallback_project_box_faces(
        self,
        corners: np.ndarray,
        *,
        color: tuple[int, int, int],
        eye: np.ndarray,
        target: np.ndarray,
        up: np.ndarray,
        width: int,
        height: int,
    ) -> list[tuple[float, list[tuple[float, float]], tuple[int, int, int], int]]:
        face_specs = (
            ((0, 1, 3, 2), 0.86),
            ((4, 5, 7, 6), 1.08),
            ((0, 1, 5, 4), 0.92),
            ((2, 3, 7, 6), 1.00),
            ((1, 3, 7, 5), 0.82),
            ((0, 2, 6, 4), 0.96),
        )
        projected, depth = self._project_points(corners, eye=eye, target=target, up=up, width=width, height=height)
        face_draws: list[tuple[float, list[tuple[float, float]], tuple[int, int, int], int]] = []
        for indices, brightness in face_specs:
            face_points = corners[list(indices)]
            face_center = face_points.mean(axis=0)
            normal = np.cross(face_points[1] - face_points[0], face_points[2] - face_points[0])
            if np.dot(normal, eye - face_center) <= 0.0:
                continue
            if np.any(depth[list(indices)] <= 0.0):
                continue
            polygon = [tuple(projected[i]) for i in indices]
            shaded = tuple(max(0, min(255, int(channel * brightness))) for channel in color)
            face_draws.append((float(depth[list(indices)].mean()), polygon, shaded, 180))
        return face_draws

    def _write_empty_preview(
        self,
        bundle: dict[str, object],
        output_path: Path,
        width: int,
        height: int,
    ) -> None:
        image = Image.new("RGB", (width, height), color=(246, 241, 231))
        draw = ImageDraw.Draw(image, "RGBA")
        draw.rounded_rectangle((24, 24, width - 24, height - 24), radius=24, outline=(90, 90, 90), width=2)
        draw.text((48, 48), f"No geometry available for {bundle.get('scene_id', 'scene')}", fill=(32, 32, 32))
        image.save(output_path)

    def _sample_object_color(self, *, texture_path: object, fallback_index: int) -> tuple[int, int, int]:
        if isinstance(texture_path, str):
            texture_file = Path(texture_path)
            if texture_file.exists():
                try:
                    image = Image.open(texture_file).convert("RGB").resize((1, 1))
                    return tuple(int(channel) for channel in image.getpixel((0, 0)))
                except Exception:
                    pass

        palette = (
            (191, 126, 83),
            (125, 153, 134),
            (170, 140, 88),
            (104, 122, 171),
            (181, 104, 96),
        )
        return palette[fallback_index % len(palette)]

    def _resolve_mesh_artifacts(
        self,
        mesh_path: Path,
        *,
        scene_id: str,
        object_id: str | None = None,
    ) -> tuple[Path | None, Path | None]:
        raw_asset_root = self._raw_asset_root_for_object(scene_id, object_id)
        mtl_path = self._resolve_mtl_path(mesh_path, preferred_root=raw_asset_root)
        texture_path = self._resolve_texture_path(mesh_path, mtl_path, preferred_root=raw_asset_root)
        return mtl_path, texture_path

    def _raw_asset_root_for_object(self, scene_id: str, object_id: str | None) -> Path | None:
        if not object_id:
            return None
        for candidate in self._raw_asset_directory_candidates(scene_id, object_id):
            if candidate.exists():
                return candidate
        return None

    def _resolve_mtl_path(self, mesh_path: Path, *, preferred_root: Path | None = None) -> Path | None:
        candidate_paths: list[Path] = []
        if preferred_root is not None:
            candidate_paths.extend(sorted(preferred_root.glob("*.mtl")))
        for referenced_name in self._read_obj_mtllibs(mesh_path):
            candidate_paths.append(mesh_path.parent / referenced_name)
        candidate_paths.append(mesh_path.with_suffix(".mtl"))

        seen: set[Path] = set()
        for candidate_path in candidate_paths:
            normalized = candidate_path
            if normalized in seen:
                continue
            seen.add(normalized)
            if normalized.exists():
                return normalized
        return None

    def _resolve_texture_path(
        self,
        mesh_path: Path,
        mtl_path: Path | None,
        *,
        preferred_root: Path | None = None,
    ) -> Path | None:
        if preferred_root is not None:
            png_candidates = sorted(preferred_root.glob("*.png"))
            if len(png_candidates) == 1:
                return png_candidates[0]
        if mtl_path is not None:
            for referenced_name in self._read_mtl_texture_refs(mtl_path):
                candidate_path = mtl_path.parent / referenced_name
                if candidate_path.exists():
                    return candidate_path

        sibling_texture = mesh_path.with_suffix(".png")
        if sibling_texture.exists():
            return sibling_texture

        png_candidates = sorted(mesh_path.parent.glob("*.png"))
        if len(png_candidates) == 1:
            return png_candidates[0]
        return None

    def _read_obj_mtllibs(self, mesh_path: Path) -> tuple[str, ...]:
        try:
            source = mesh_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return ()

        referenced: list[str] = []
        for line in source.splitlines():
            stripped = line.strip()
            if not stripped.startswith("mtllib "):
                continue
            referenced.extend(part for part in stripped.split()[1:] if part)
        return tuple(referenced)

    def _read_mtl_texture_refs(self, mtl_path: Path) -> tuple[str, ...]:
        try:
            source = mtl_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return ()

        referenced: list[str] = []
        for line in source.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if not stripped.startswith(("map_Kd ", "map_Ka ", "map_d ")):
                continue
            parts = stripped.split()
            if len(parts) >= 2:
                referenced.append(parts[-1])
        return tuple(referenced)

    def _load_preview_mesh(self, mesh_path: str):
        import open3d as o3d

        mesh = o3d.io.read_triangle_mesh(mesh_path, enable_post_processing=True)
        if mesh.is_empty():
            return None
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()
        return mesh

    def _apply_bundle_transform(self, mesh, item: dict[str, object]) -> None:
        if item.get("already_transformed", True):
            return
        transform = item.get("transform")
        if not isinstance(transform, dict):
            return
        translation = tuple(transform.get("translation_xyz") or (0.0, 0.0, 0.0))
        rotation_value = tuple(transform.get("rotation_value") or (1.0, 0.0, 0.0, 0.0))
        scale_value = transform.get("scale_xyz")
        scale_xyz = tuple(scale_value[:3]) if isinstance(scale_value, (list, tuple)) else None
        pose = ObjectPose(
            object_id=str(item.get("object_id") or "object"),
            translation_xyz=tuple(float(value) for value in translation[:3]),
            rotation_type=str(transform.get("rotation_type") or "quaternion"),
            rotation_value=tuple(float(value) for value in rotation_value),
            scale_xyz=(
                tuple(float(value) for value in scale_xyz)
                if scale_xyz is not None
                else None
            ),
        )
        mesh.transform(pose_to_matrix(pose))


def _render_bundle_cli() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--render-bundle", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--preview-count", type=int, required=True)
    parser.add_argument("--output-prefix", default="geometry_preview")
    args = parser.parse_args()

    bundle_path = Path(args.render_bundle)
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    renderer = GeometrySceneRenderer(
        output_root=str(Path(args.output_dir).parent),
        preview_size=(args.width, args.height),
        preview_count=args.preview_count,
    )
    renderer._render_geometry_previews_in_process(
        bundle,
        Path(args.output_dir),
        output_prefix=args.output_prefix,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_render_bundle_cli())
