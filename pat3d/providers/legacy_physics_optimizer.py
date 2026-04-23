from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
from typing import Callable

import numpy as np

from pat3d.legacy_config import resolve_repo_root
from pat3d.providers._legacy_physics_adapter import LegacySequentialPhysicsAdapter
from pat3d.contracts import PhysicsOptimizer
from pat3d.models import ArtifactRef, ObjectPose, PhysicsOptimizationResult, PhysicsReadyScene
from pat3d.models.pose_utils import pose_to_matrix
from pat3d.runtime.errors import runtime_failure
from pat3d.storage import make_stage_metadata

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_RUNNER_SCRIPT = REPO_ROOT / "pat3d" / "scripts" / "physics" / "run_legacy_physics.py"


class LegacyDiffSimPhysicsOptimizer(PhysicsOptimizer):
    def __init__(
        self,
        *,
        mode: str = "forward_only",
        legacy_arg_overrides: dict[str, str | int | float | bool] | None = None,
        use_subprocess: bool = True,
        recover_static_frames: bool = True,
        optimizer: Callable[[PhysicsReadyScene], PhysicsOptimizationResult | dict] | None = None,
        forward_simulator: Callable[[PhysicsReadyScene], PhysicsOptimizationResult | dict] | None = None,
        default_optimizer: Callable[[PhysicsReadyScene], PhysicsOptimizationResult | dict] | None = None,
        default_forward_simulator: Callable[[PhysicsReadyScene], PhysicsOptimizationResult | dict] | None = None,
        physics_adapter: LegacySequentialPhysicsAdapter | None = None,
        python_executable: str | None = None,
        runner_script: str | None = None,
        subprocess_runner: Callable[[str, str, dict[str, object]], dict[str, object]] | None = None,
        metadata_factory: Callable[..., object] = make_stage_metadata,
        legacy_root: str = "pat3d/legacy_vendor/mask_loss",
    ) -> None:
        adapter = physics_adapter or LegacySequentialPhysicsAdapter(
            legacy_arg_overrides=legacy_arg_overrides
        )
        self._adapter = adapter
        self._mode = mode
        self._legacy_arg_overrides = dict(legacy_arg_overrides or {})
        self._use_subprocess = bool(use_subprocess)
        self._recover_static_frames = bool(recover_static_frames)
        self._optimizer = optimizer
        self._forward_simulator = forward_simulator
        self._default_optimizer = default_optimizer or adapter.optimize
        self._default_forward_simulator = default_forward_simulator or adapter.simulate_to_static
        self._uses_builtin_default_optimizer = default_optimizer is None
        self._uses_builtin_default_forward_simulator = default_forward_simulator is None
        self._python_executable = python_executable
        self._runner_script = runner_script
        self._subprocess_runner = subprocess_runner
        self._metadata_factory = metadata_factory
        self._legacy_root = legacy_root

    def optimize(self, physics_ready_scene: PhysicsReadyScene) -> PhysicsOptimizationResult:
        use_subprocess = self._should_use_subprocess()
        if self._mode == "forward_only":
            runner = self._forward_simulator or self._default_forward_simulator
        elif self._mode == "optimize_then_forward":
            runner = self._optimizer or self._default_optimizer
        else:
            raise ValueError(
                f"Unsupported legacy physics mode '{self._mode}'. "
                "Expected 'forward_only' or 'optimize_then_forward'."
            )
        fallback_error: Exception | None = None
        try:
            if use_subprocess:
                result = self._run_subprocess(physics_ready_scene)
            else:
                result = runner(physics_ready_scene)
        except Exception as exc:
            if use_subprocess:
                recovered = self._recover_from_saved_frames(physics_ready_scene, failure=exc)
                if recovered is not None:
                    result = recovered
                elif self._subprocess_runner is not None:
                    raise
                else:
                    fallback_error = exc
                    result = None
            else:
                fallback_error = exc
                result = None

        if result is None:
            optimized_object_poses = physics_ready_scene.object_poses
            artifacts = (
                ArtifactRef(
                    artifact_type="legacy_physics_reference",
                    path=str(Path(self._legacy_root) / "diff_sim" / "phys_optim.py"),
                    format="py",
                    role="legacy_source",
                    metadata_path=None,
                ),
            )
            metrics = {"identity_passthrough": 1.0, "backend_failure": 1.0}
        else:
            if isinstance(result, PhysicsOptimizationResult):
                return result
            optimized_object_poses = self._coerce_object_poses(
                result.get("optimized_object_poses", physics_ready_scene.object_poses)
            )
            artifacts = self._coerce_artifacts(result.get("artifacts", ()))
            metrics = dict(result.get("metrics", {}))
            if self._is_zero_frame_forward_result(
                metrics=metrics,
                initial_object_poses=physics_ready_scene.object_poses,
                optimized_object_poses=optimized_object_poses,
            ):
                metrics["forward_diff_simulator_failed"] = 1.0
                metrics["zero_frame_forward_result"] = 1.0
        notes = ()
        if result is None:
            notes = ("identity_passthrough",)
            if fallback_error is not None:
                notes = notes + (f"fallback_reason={type(fallback_error).__name__}: {fallback_error}",)
        else:
            if metrics.get("forward_only_simulation_used", 0.0) > 0.0:
                notes = notes + ("forward_only_simulation_used",)
            if metrics.get("forward_diff_simulator_used", 0.0) > 0.0:
                notes = notes + ("forward_diff_simulator_used",)
            if metrics.get("diff_sim_initialization_used", 0.0) > 0.0:
                notes = notes + ("diff_sim_initialization_used",)
            if metrics.get("diff_sim_hierarchical_tree_used", 0.0) > 0.0:
                notes = notes + ("diff_sim_hierarchical_tree_used",)
            if metrics.get("diff_sim_initialization_failed", 0.0) > 0.0:
                notes = notes + ("diff_sim_initialization_failed",)
            if metrics.get("forward_diff_simulator_failed", 0.0) > 0.0:
                notes = notes + ("forward_diff_simulator_failed",)
            if metrics.get("zero_frame_forward_result", 0.0) > 0.0:
                notes = notes + ("zero_frame_forward_result",)
            if metrics.get("support_contact_projection_count", 0.0) > 0.0:
                notes = notes + ("support_contact_projection_used",)
            if metrics.get("recovered_from_subprocess_failure", 0.0) > 0.0:
                notes = notes + ("recovered_from_subprocess_failure",)
            recovery_reason = result.get("recovery_reason")
            if isinstance(recovery_reason, str) and recovery_reason.strip():
                notes = notes + (f"recovery_reason={recovery_reason}",)

        return PhysicsOptimizationResult(
            scene_id=physics_ready_scene.scene_id,
            initial_scene=physics_ready_scene,
            optimized_object_poses=tuple(optimized_object_poses),
            metrics=metrics,
            artifacts=tuple(artifacts),
            metadata=self._metadata_factory(
                stage_name="physics_optimization",
                provider_name="legacy_diff_sim",
                notes=notes,
            ),
        )

    def _is_zero_frame_forward_result(
        self,
        *,
        metrics: dict[str, object],
        initial_object_poses: tuple[ObjectPose, ...],
        optimized_object_poses: tuple[ObjectPose, ...],
    ) -> bool:
        if float(metrics.get("forward_diff_simulator_used", 0.0) or 0.0) <= 0.0:
            return False
        if float(metrics.get("forward_simulator_final_frame", 0.0) or 0.0) > 0.0:
            return False
        if float(metrics.get("forward_simulator_stopped_static", 0.0) or 0.0) > 0.0:
            return False
        return self._poses_match(initial_object_poses, optimized_object_poses)

    def _poses_match(
        self,
        initial_object_poses: tuple[ObjectPose, ...],
        optimized_object_poses: tuple[ObjectPose, ...],
    ) -> bool:
        if len(initial_object_poses) != len(optimized_object_poses):
            return False
        initial_by_id = {pose.object_id: pose for pose in initial_object_poses}
        optimized_by_id = {pose.object_id: pose for pose in optimized_object_poses}
        if initial_by_id.keys() != optimized_by_id.keys():
            return False
        return all(
            np.allclose(
                pose_to_matrix(initial_by_id[object_id]),
                pose_to_matrix(optimized_by_id[object_id]),
                atol=1e-8,
            )
            for object_id in initial_by_id
        )

    def _coerce_object_poses(self, values: object) -> tuple[ObjectPose, ...]:
        poses: list[ObjectPose] = []
        for item in values or ():
            if isinstance(item, ObjectPose):
                poses.append(item)
                continue
            if isinstance(item, dict):
                scale_xyz = item.get("scale_xyz")
                poses.append(
                    ObjectPose(
                        object_id=str(item["object_id"]),
                        translation_xyz=tuple(float(value) for value in item.get("translation_xyz", (0.0, 0.0, 0.0))),
                        rotation_type=str(item.get("rotation_type", "quaternion")),
                        rotation_value=tuple(float(value) for value in item.get("rotation_value", (1.0, 0.0, 0.0, 0.0))),
                        scale_xyz=(
                            tuple(float(value) for value in scale_xyz)
                            if isinstance(scale_xyz, (list, tuple))
                            else None
                        ),
                    )
                )
        return tuple(poses)

    def _coerce_artifacts(self, values: object) -> tuple[ArtifactRef, ...]:
        artifacts: list[ArtifactRef] = []
        for item in values or ():
            if isinstance(item, ArtifactRef):
                artifacts.append(item)
                continue
            if isinstance(item, dict):
                artifacts.append(
                    ArtifactRef(
                        artifact_type=str(item["artifact_type"]),
                        path=str(item["path"]),
                        format=str(item["format"]),
                        role=str(item["role"]) if item.get("role") is not None else None,
                        metadata_path=(
                            str(item["metadata_path"])
                            if item.get("metadata_path") is not None
                            else None
                        ),
                    )
                )
        return tuple(artifacts)

    def _should_use_subprocess(self) -> bool:
        if not self._use_subprocess:
            return False
        if self._mode == "forward_only":
            return self._forward_simulator is None and self._uses_builtin_default_forward_simulator
        if self._mode == "optimize_then_forward":
            return self._optimizer is None and self._uses_builtin_default_optimizer
        return False

    def _resolve_python_executable(self) -> Path:
        attempted: list[str] = []
        for candidate in self._candidate_python_executables():
            resolved = Path(candidate).expanduser()
            if not resolved.is_absolute():
                resolved = self._resolve_bundle_root() / resolved
            attempted.append(str(resolved))
            if resolved.exists():
                return resolved
        raise runtime_failure(
            phase="provider_execution",
            code="legacy_physics_python_missing",
            user_message="Physics simulation could not start because its Python executable is missing.",
            technical_message=(
                "Legacy physics Python executable does not exist. "
                f"Candidates: {attempted}"
            ),
            provider_kind="legacy_diff_sim",
            retryable=False,
        )

    def _candidate_python_executables(self) -> tuple[str, ...]:
        bundle_root = self._resolve_bundle_root()
        candidates = (
            self._python_executable,
            os.environ.get("PAT3D_LEGACY_PHYSICS_PYTHON"),
            str(bundle_root / ".conda" / "pat3d" / "bin" / "python"),
            str(bundle_root / ".venv" / "bin" / "python"),
            sys.executable,
        )
        ordered: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            if candidate is None:
                continue
            normalized = str(candidate).strip()
            if not normalized or normalized in seen:
                continue
            ordered.append(normalized)
            seen.add(normalized)
        return tuple(ordered)

    def _resolve_runner_script(self) -> Path:
        candidate = Path(self._runner_script) if self._runner_script else DEFAULT_RUNNER_SCRIPT
        resolved = candidate.expanduser()
        if not resolved.is_absolute():
            resolved = self._resolve_bundle_root() / resolved
        if not resolved.exists():
            raise runtime_failure(
                phase="provider_execution",
                code="legacy_physics_runner_missing",
                user_message="Physics simulation could not start because its runner script is missing.",
                technical_message=f"Legacy physics runner script does not exist: {resolved}",
                provider_kind="legacy_diff_sim",
                retryable=False,
            )
        return resolved

    def _run_subprocess(self, physics_ready_scene: PhysicsReadyScene) -> dict[str, object]:
        python_executable = self._resolve_python_executable()
        runner_script = self._resolve_runner_script()
        self._clear_previous_outputs(physics_ready_scene)
        payload = {
            "mode": self._mode,
            "legacy_arg_overrides": dict(self._legacy_arg_overrides),
            "physics_ready_scene": physics_ready_scene.to_dict(),
        }
        if self._subprocess_runner is not None:
            try:
                return self._subprocess_runner(str(python_executable), str(runner_script), payload)
            except Exception as exc:
                raise runtime_failure(
                    phase="provider_execution",
                    code="legacy_physics_subprocess_failed",
                    user_message="Physics simulation crashed before producing a stable result.",
                    technical_message=str(exc) or exc.__class__.__name__,
                    provider_kind="legacy_diff_sim",
                    retryable=True,
                    details={"mode": self._mode, "scene_id": physics_ready_scene.scene_id},
                ) from exc

        with tempfile.TemporaryDirectory(prefix="pat3d-legacy-physics-") as temp_dir:
            temp_root = Path(temp_dir)
            input_path = temp_root / "input.json"
            output_path = temp_root / "output.json"
            input_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            bundle_root = self._resolve_bundle_root()
            process = subprocess.run(
                [
                    str(python_executable),
                    str(runner_script),
                    "--input-json",
                    str(input_path),
                    "--output-json",
                    str(output_path),
                ],
                cwd=str(bundle_root),
                capture_output=True,
                text=True,
            )
            if process.returncode == 0:
                if not output_path.exists():
                    raise runtime_failure(
                        phase="provider_execution",
                        code="legacy_physics_output_missing",
                        user_message="Physics simulation finished without producing an output payload.",
                        technical_message="Legacy physics subprocess exited successfully but output.json was not created.",
                        provider_kind="legacy_diff_sim",
                        retryable=False,
                    )
                return json.loads(output_path.read_text(encoding="utf-8"))

        raise runtime_failure(
            phase="provider_execution",
            code="legacy_physics_subprocess_failed",
            user_message="Physics simulation crashed before producing a stable result.",
            technical_message=self._format_process_failure(process),
            provider_kind="legacy_diff_sim",
            retryable=True,
            details={"mode": self._mode, "scene_id": physics_ready_scene.scene_id},
        )

    def _clear_previous_outputs(self, physics_ready_scene: PhysicsReadyScene) -> None:
        args = self._adapter._build_args(physics_ready_scene)
        scene_root = Path(args.phys_result_folder) / args.exp_name
        for folder_name in ("transform", "param"):
            candidate = scene_root / folder_name
            if candidate.exists():
                shutil.rmtree(candidate)
        for candidate in scene_root.glob("progressive_pass_*"):
            if candidate.is_dir():
                shutil.rmtree(candidate)
        for stale_file in (
            scene_root / "diff_init_report.json",
            scene_root / "loss_history.json",
        ):
            if stale_file.exists():
                stale_file.unlink()
        for stale_file in scene_root.glob("loss_history_pass_*.json"):
            if stale_file.exists():
                stale_file.unlink()

    def _resolve_bundle_root(self) -> Path:
        try:
            return resolve_repo_root()
        except RuntimeError:
            return REPO_ROOT

    def _format_process_failure(self, process: subprocess.CompletedProcess[str]) -> str:
        if process.returncode < 0:
            signal_number = -int(process.returncode)
            try:
                signal_name = signal.Signals(signal_number).name
            except ValueError:
                signal_name = f"SIG{signal_number}"
            reason = f"terminated by signal {signal_name}"
        else:
            reason = f"exit code {process.returncode}"
        stderr = (process.stderr or "").strip()
        stdout = (process.stdout or "").strip()
        extras = []
        if stderr:
            extras.append(f"stderr={stderr}")
        if stdout:
            extras.append(f"stdout={stdout}")
        suffix = f" ({'; '.join(extras)})" if extras else ""
        return f"{reason}{suffix}"

    def _recover_from_saved_frames(
        self,
        physics_ready_scene: PhysicsReadyScene,
        *,
        failure: Exception,
    ) -> dict[str, object] | None:
        if not self._recover_static_frames or self._mode != "forward_only":
            return None
        args = self._adapter._build_args(physics_ready_scene)
        transform_dir = Path(args.phys_result_folder) / args.exp_name / "transform"
        frame_paths = sorted(transform_dir.glob("frame_*.npz"), key=self._snapshot_sort_key)
        if not frame_paths:
            return None

        stable_frame_path = self._last_stable_frame_path(frame_paths, args)
        if stable_frame_path is None:
            return None

        transforms = np.load(stable_frame_path, allow_pickle=True)
        transformations = self._normalize_transformations(transforms, physics_ready_scene)
        optimized_object_poses = tuple(
            self._adapter._apply_transform_delta(pose, transformations.get(pose.object_id))
            for pose in physics_ready_scene.object_poses
        )
        return {
            "optimized_object_poses": optimized_object_poses,
            "artifacts": self._adapter._collect_artifacts(args),
            "metrics": {
                "legacy_diff_sim_adapter_used": 1.0,
                "optimized_object_count": float(len(optimized_object_poses)),
                "forward_only_simulation_used": 1.0,
                "forward_diff_simulator_used": 1.0,
                "forward_diff_simulator_failed": 0.0,
                "forward_simulator_final_frame": float(self._frame_number(stable_frame_path)),
                "forward_simulator_stopped_static": 1.0,
                "recovered_from_subprocess_failure": 1.0,
                "support_contact_projection_count": 0.0,
            },
            "recovery_reason": f"{type(failure).__name__}: {failure}",
        }

    def _normalize_transformations(
        self,
        transforms: object,
        physics_ready_scene: PhysicsReadyScene,
    ) -> dict[str, list[list[float]]]:
        raw = transforms if isinstance(transforms, dict) else dict(transforms)
        object_ids = tuple(pose.object_id for pose in physics_ready_scene.object_poses)
        return self._adapter._normalize_transformations(raw, object_ids)

    def _last_stable_frame_path(self, frame_paths: list[Path], args: object) -> Path | None:
        translation_tol = float(getattr(args, "static_translation_tol", 1e-4))
        rotation_tol = float(getattr(args, "static_rotation_tol", 1e-4))
        consecutive_required = int(getattr(args, "static_consecutive_frames", 10))
        min_static_start_frame = int(getattr(args, "min_static_start_frame", 1))
        if len(frame_paths) < consecutive_required + 1:
            return None

        previous = None
        streak = 0
        stable_frame: Path | None = None
        for frame_path in frame_paths:
            current = np.load(frame_path, allow_pickle=True)
            current_map = {
                key: np.asarray(current[key], dtype=np.float64)
                for key in current.files
            }
            if previous is not None:
                max_translation_delta = 0.0
                max_rotation_delta = 0.0
                for object_id, matrix in current_map.items():
                    prior = previous.get(object_id)
                    if prior is None:
                        streak = 0
                        break
                    max_translation_delta = max(
                        max_translation_delta,
                        float(np.max(np.abs(matrix[:3, 3] - prior[:3, 3]))),
                    )
                    max_rotation_delta = max(
                        max_rotation_delta,
                        float(np.max(np.abs(matrix[:3, :3] - prior[:3, :3]))),
                    )
                else:
                    if (
                        self._frame_number(frame_path) >= min_static_start_frame
                        and max_translation_delta <= translation_tol
                        and max_rotation_delta <= rotation_tol
                    ):
                        streak += 1
                    else:
                        streak = 0
                    if streak >= consecutive_required:
                        stable_frame = frame_path
            previous = current_map
        return stable_frame

    def _snapshot_sort_key(self, path: Path) -> tuple[int, str]:
        return (self._frame_number(path), path.name)

    def _frame_number(self, path: Path) -> int:
        match = re.search(r"(\d+)$", path.stem)
        if match is None:
            return 0
        return int(match.group(1))
