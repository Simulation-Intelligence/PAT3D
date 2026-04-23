from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Callable, Mapping

from pat3d.contracts import TextTo3DProvider
from pat3d.models import ArtifactRef, GeneratedObjectAsset, ObjectDescription, ObjectPose
from pat3d.runtime.errors import runtime_failure
from pat3d.storage import make_stage_metadata

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONDA_ENV_PYTHONS = (
    Path.home() / ".conda" / "envs" / "sam3d-objects" / "bin" / "python",
    Path.home() / "anaconda3" / "envs" / "sam3d-objects" / "bin" / "python",
    Path.home() / "miniconda3" / "envs" / "sam3d-objects" / "bin" / "python",
)
DEFAULT_PAT3D_PYTHON = REPO_ROOT / ".conda" / "pat3d" / "bin" / "python"
DEFAULT_RUNNER_SCRIPT = REPO_ROOT / "pat3d" / "scripts" / "object_generation" / "_run_sam3d_object_impl.py"
DEFAULT_SAM3D_ROOT = REPO_ROOT / "extern" / "sam-3d-objects"
SAM3D_ROOT_ENV_VAR = "PAT3D_SAM3D_ROOT"
SAM3D_LOCK_PATH_ENV_VAR = "PAT3D_SAM3D_LOCK_PATH"


def _repo_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return REPO_ROOT / candidate


def _is_cusolver_internal_error(detail: str) -> bool:
    normalized = str(detail or "").lower()
    return "cusolver_status_internal_error" in normalized or "cusolver error" in normalized


def _is_cuda_oom_error(detail: str) -> bool:
    normalized = str(detail or "").lower()
    return (
        "cuda out of memory" in normalized
        or "outofmemoryerror" in normalized
        or "cuda failed with error 2 out of memory" in normalized
        or "error 2 out of memory" in normalized
    )


def _default_sam3d_lock_path() -> Path:
    configured = os.environ.get(SAM3D_LOCK_PATH_ENV_VAR, "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path(tempfile.gettempdir()) / "pat3d-sam3d-gpu.lock"


@contextmanager
def _exclusive_sam3d_gpu_lock():
    lock_path = _default_sam3d_lock_path()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield lock_path
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _object_pose_from_result(
    object_id: str,
    result: Mapping[str, Any],
) -> ObjectPose | None:
    pose_data = result.get("asset_local_pose")
    if not isinstance(pose_data, Mapping):
        return None
    rotation_type = str(pose_data.get("rotation_type") or "quaternion")
    default_rotation_value = {
        "quaternion": (1.0, 0.0, 0.0, 0.0),
        "euler_xyz": (0.0, 0.0, 0.0),
        "matrix4x4": (
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ),
    }.get(rotation_type, ())
    translation_xyz = tuple(float(value) for value in pose_data.get("translation_xyz", (0.0, 0.0, 0.0)))
    rotation_value = tuple(
        float(value) for value in pose_data.get("rotation_value", default_rotation_value)
    )
    scale_xyz_raw = pose_data.get("scale_xyz")
    scale_xyz = (
        tuple(float(value) for value in scale_xyz_raw)
        if scale_xyz_raw is not None
        else None
    )
    return ObjectPose(
        object_id=object_id,
        translation_xyz=translation_xyz,
        rotation_type=rotation_type,
        rotation_value=rotation_value,
        scale_xyz=scale_xyz,
    )


class SAM3DImageTo3DProvider(TextTo3DProvider):
    def __init__(
        self,
        *,
        scene_id: str,
        output_root: str = "data/raw_obj",
        python_executable: str | None = None,
        runner_script: str | None = None,
        sam3d_root: str | None = None,
        config_path: str = "checkpoints/hf/pipeline.yaml",
        device: str | None = None,
        compile: bool = False,
        seed: int = 42,
        crop_completion_enabled: bool = False,
        crop_completion_model: str | None = None,
        subprocess_runner: Callable[[str, str, Mapping[str, Any]], Mapping[str, Any]] | None = None,
        metadata_factory: Callable[..., object] = make_stage_metadata,
    ) -> None:
        _ = crop_completion_enabled, crop_completion_model
        self._scene_id = scene_id
        self._output_root = Path(output_root)
        self._python_executable = python_executable
        self._runner_script = runner_script
        self._sam3d_root = sam3d_root
        self._config_path = config_path
        self._device = device
        self._compile = bool(compile)
        self._seed = int(seed)
        self._subprocess_runner = subprocess_runner
        self._metadata_factory = metadata_factory

    def _provider_kind(self) -> str:
        return "sam3d_image_to_3d"

    def generate(
        self,
        object_description: ObjectDescription,
        *,
        object_reference_image: ArtifactRef | None = None,
    ) -> GeneratedObjectAsset:
        if object_reference_image is None:
            raise RuntimeError(
                "SAM 3D object generation requires an object_reference_image. "
                "Run the paper-core pipeline with segmentation enabled or supply object-reference images explicitly."
            )

        output_dir = self._output_root / self._scene_id / object_description.object_id
        output_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "object_id": object_description.object_id,
            "prompt": object_description.prompt_text,
            "reference_image_path": object_reference_image.path,
            "output_dir": str(output_dir),
            "sam3d_root": str(self._resolve_sam3d_root()),
            "config_path": self._config_path,
            "device": self._device,
            "compile": self._compile,
            "seed": self._seed,
        }
        result = self._run_subprocess(payload)

        mesh_path = Path(str(result["mesh_path"]))
        mesh_format = mesh_path.suffix.lower().lstrip(".") or "glb"
        canonical_mesh_path = result.get("canonical_mesh_path")
        gs_path = result.get("gaussian_path")
        mesh_ply_path = result.get("mesh_ply_path")
        posed_mesh_ply_path = result.get("posed_mesh_ply_path")
        canonical_mesh_note = (
            f"canonical_mesh_path={canonical_mesh_path}"
            if isinstance(canonical_mesh_path, str) and canonical_mesh_path
            else None
        )
        gs_note = f"gaussian_path={gs_path}" if isinstance(gs_path, str) and gs_path else None
        mesh_ply_note = (
            f"mesh_ply_path={mesh_ply_path}"
            if isinstance(mesh_ply_path, str) and mesh_ply_path
            else None
        )
        posed_mesh_ply_note = (
            f"posed_mesh_ply_path={posed_mesh_ply_path}"
            if isinstance(posed_mesh_ply_path, str) and posed_mesh_ply_path
            else None
        )
        asset_local_pose = _object_pose_from_result(object_description.object_id, result)
        mesh_pose_note = "mesh_pose_space=posed" if asset_local_pose is None else None

        return GeneratedObjectAsset(
            object_id=object_description.object_id,
            mesh_obj=ArtifactRef(
                artifact_type="mesh_obj",
                path=str(mesh_path),
                format=mesh_format,
                role="mesh",
                metadata_path=None,
            ),
            preview_image=object_reference_image,
            provider_asset_id=str(result.get("provider_asset_id") or object_description.object_id),
            asset_local_pose=asset_local_pose,
            metadata=self._metadata_factory(
                stage_name="object_asset_generation",
                provider_name="sam3d_image_to_3d",
                notes=tuple(
                    note
                    for note in (
                        f"scene_id={self._scene_id}",
                        f"sam3d_root={self._resolve_sam3d_root()}",
                        f"config_path={self._config_path}",
                        f"seed={self._seed}",
                        f"device={result.get('device') or self._device or ''}",
                        f"cuda_linalg_backend={result.get('cuda_linalg_backend')}"
                        if result.get("cuda_linalg_backend")
                        else None,
                        "asset_local_pose=1" if asset_local_pose is not None else None,
                        mesh_pose_note,
                        canonical_mesh_note,
                        gs_note,
                        mesh_ply_note,
                        posed_mesh_ply_note,
                    )
                    if note
                ),
            ),
        )

    def _python_supports_sam3d(self, python_executable: Path) -> tuple[bool, str | None]:
        env = os.environ.copy()
        env.setdefault("LIDRA_SKIP_INIT", "1")
        try:
            process = subprocess.run(
                [str(python_executable), "-c", "import sam3d_objects"],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                env=env,
            )
        except OSError as exc:
            return False, str(exc)
        if process.returncode == 0:
            return True, None
        detail = process.stderr.strip() or process.stdout.strip() or "unknown import failure"
        return False, detail

    def _resolve_python_executable(self) -> Path:
        candidates: list[Path] = []
        if self._python_executable:
            candidates.append(Path(self._python_executable))
        env_python = os.environ.get("PAT3D_SAM3D_PYTHON", "").strip()
        if env_python:
            candidates.append(Path(env_python))
        candidates.extend(DEFAULT_CONDA_ENV_PYTHONS)
        candidates.extend((DEFAULT_PAT3D_PYTHON, Path(sys.executable)))

        diagnostics: list[str] = []
        seen_candidates: set[Path] = set()
        for candidate in candidates:
            expanded = candidate.expanduser()
            if expanded in seen_candidates:
                continue
            seen_candidates.add(expanded)
            if not expanded.exists():
                continue
            supported, detail = self._python_supports_sam3d(expanded)
            if supported:
                return expanded
            diagnostics.append(f"{expanded}: {detail}")
        detail_suffix = f" Checked: {'; '.join(diagnostics)}" if diagnostics else ""
        raise RuntimeError(
            "Could not find a Python interpreter with SAM 3D Objects installed. "
            "Set PAT3D_SAM3D_PYTHON or install the SAM 3D object dependencies into `.conda/pat3d`."
            f"{detail_suffix}"
        )

    def _resolve_runner_script(self) -> Path:
        candidate = Path(self._runner_script) if self._runner_script else DEFAULT_RUNNER_SCRIPT
        resolved = candidate.expanduser()
        if not resolved.is_absolute():
            resolved = REPO_ROOT / resolved
        if not resolved.exists():
            raise RuntimeError(f"SAM 3D runner script is missing: {resolved}")
        return resolved

    def _resolve_sam3d_root(self) -> Path:
        candidates: list[Path] = []
        if self._sam3d_root is not None:
            candidates.append(Path(self._sam3d_root).expanduser())
        env_root = os.environ.get(SAM3D_ROOT_ENV_VAR, "").strip()
        if env_root:
            candidates.append(Path(env_root).expanduser())
        candidates.append(DEFAULT_SAM3D_ROOT)

        checked: list[str] = []
        for candidate in candidates:
            resolved = candidate if candidate.is_absolute() else REPO_ROOT / candidate
            checked.append(str(resolved))
            if resolved.exists():
                return resolved
        raise RuntimeError(
            "SAM 3D Objects repo is missing. "
            f"Set {SAM3D_ROOT_ENV_VAR} or place the checkout at {DEFAULT_SAM3D_ROOT}. "
            f"Checked: {', '.join(checked)}"
        )

    def _run_subprocess(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        if self._subprocess_runner is not None:
            python_executable = str(self._resolve_python_executable())
            runner_script = str(
                Path(self._runner_script).expanduser()
                if self._runner_script
                else DEFAULT_RUNNER_SCRIPT
            )
            return self._subprocess_runner(python_executable, runner_script, payload)

        python_executable = self._resolve_python_executable()
        runner_script = self._resolve_runner_script()
        env = os.environ.copy()
        env["CONDA_PREFIX"] = str(python_executable.parent.parent)
        env["PATH"] = (
            f"{python_executable.parent}{os.pathsep}{env['PATH']}"
            if env.get("PATH")
            else str(python_executable.parent)
        )
        env.setdefault("LIDRA_SKIP_INIT", "1")
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        attempt_payloads = [dict(payload)]
        last_detail = ""
        with _exclusive_sam3d_gpu_lock() as lock_path:
            for attempt_index, current_payload in enumerate(attempt_payloads, start=1):
                with tempfile.TemporaryDirectory(prefix="pat3d-sam3d-") as temp_dir:
                    temp_root = Path(temp_dir)
                    input_path = temp_root / "input.json"
                    output_path = temp_root / "output.json"
                    input_path.write_text(json.dumps(current_payload, indent=2), encoding="utf-8")
                    process = subprocess.run(
                        [str(python_executable), str(runner_script), "--input-json", str(input_path), "--output-json", str(output_path)],
                        cwd=str(REPO_ROOT),
                        capture_output=True,
                        text=True,
                        env=env,
                    )
                    if process.returncode == 0:
                        if not output_path.exists():
                            raise runtime_failure(
                                phase="provider_execution",
                                code="sam3d_output_missing",
                                user_message="Object asset generation finished without producing an output payload.",
                                technical_message="SAM 3D subprocess exited successfully but output.json was not created.",
                                provider_kind=self._provider_kind(),
                                retryable=False,
                            )
                        return json.loads(output_path.read_text(encoding="utf-8"))

                    detail = process.stderr.strip() or process.stdout.strip() or f"exit code {process.returncode}"
                    last_detail = detail
                    if (
                        attempt_index == 1
                        and _is_cusolver_internal_error(detail)
                        and not current_payload.get("cuda_linalg_backend")
                    ):
                        attempt_payloads.append({**dict(payload), "cuda_linalg_backend": "magma"})
                        continue
                    break

        raise runtime_failure(
            phase="provider_execution",
            code="sam3d_subprocess_failed",
            user_message="Object asset generation crashed before producing meshes.",
            technical_message=f"SAM 3D subprocess failed: {last_detail}",
            provider_kind=self._provider_kind(),
            retryable=_is_cuda_oom_error(last_detail) or _is_cusolver_internal_error(last_detail),
            details={
                "lock_path": str(lock_path),
                "attempts": len(attempt_payloads),
            },
        )


# Backward-compatible alias for older imports.
SAM3DTextTo3DProvider = SAM3DImageTo3DProvider
