from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
from typing import Any, Callable, Mapping

from pat3d.contracts import DepthEstimator
from pat3d.models import ArtifactRef, DepthResult, ReferenceImageResult
from pat3d.providers._legacy_image_inputs import materialize_legacy_scene_image
from pat3d.storage import make_stage_metadata
from pat3d.runtime.errors import runtime_failure


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_RUNNER_SCRIPT = REPO_ROOT / "pat3d" / "scripts" / "object_generation" / "run_current_depth.py"


def _load_default_depth_function() -> Callable[[str, str, str], None]:
    from pat3d.preprocessing.depth import get_depth as current_get_depth

    return current_get_depth


class CurrentDepthEstimator(DepthEstimator):
    def __init__(
        self,
        *,
        output_dir: str = "data/depth",
        legacy_input_dir: str = "data/ref_img",
        depth_function: Callable[[str, str, str], None] | None = None,
        python_executable: str | None = None,
        runner_script: str | None = None,
        subprocess_runner: Callable[[str, str, Mapping[str, Any]], Mapping[str, Any]] | None = None,
        max_retries: int = 1,
        metadata_factory: Callable[..., object] = make_stage_metadata,
    ) -> None:
        self._output_dir = output_dir
        self._legacy_input_dir = legacy_input_dir
        self._depth_function = depth_function
        self._python_executable = python_executable
        self._runner_script = runner_script
        self._subprocess_runner = subprocess_runner
        self._max_retries = max(0, int(max_retries))
        self._metadata_factory = metadata_factory

    def predict(self, reference_image: ReferenceImageResult) -> DepthResult:
        scene_name = reference_image.request.scene_id
        staged_image = materialize_legacy_scene_image(
            reference_image,
            target_root=self._legacy_input_dir,
        )
        image_folder = str(staged_image.parent)
        if self._depth_function is not None:
            self._depth_function(scene_name, image_folder, self._output_dir)
            return self._result_from_scene_dir(
                reference_image=reference_image,
                provider_name="current_depth_pro",
            )

        payload = {
            "scene_name": scene_name,
            "image_folder": image_folder,
            "output_dir": self._output_dir,
            "provider_name": "current_depth",
        }
        result_payload = self._run_subprocess(payload)
        return self._result_from_scene_dir(
            reference_image=reference_image,
            provider_name=str(result_payload.get("provider_name") or "current_depth"),
        )

    def _resolve_python_executable(self) -> Path:
        candidate = self._python_executable or os.environ.get("PAT3D_DEPTH_PYTHON") or sys.executable
        resolved = Path(candidate).expanduser()
        if not resolved.is_absolute():
            resolved = REPO_ROOT / resolved
        if not resolved.exists():
            raise runtime_failure(
                phase="provider_execution",
                code="depth_python_missing",
                user_message="Depth estimation could not start because its Python executable is missing.",
                technical_message=f"Depth Python executable does not exist: {resolved}",
                provider_kind="current_depth",
                retryable=False,
            )
        return resolved

    def _resolve_runner_script(self) -> Path:
        candidate = Path(self._runner_script) if self._runner_script else DEFAULT_RUNNER_SCRIPT
        resolved = candidate.expanduser()
        if not resolved.is_absolute():
            resolved = REPO_ROOT / resolved
        if not resolved.exists():
            raise runtime_failure(
                phase="provider_execution",
                code="depth_runner_missing",
                user_message="Depth estimation could not start because its runner script is missing.",
                technical_message=f"Depth runner script does not exist: {resolved}",
                provider_kind="current_depth",
                retryable=False,
            )
        return resolved

    def _run_subprocess(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        python_executable = self._resolve_python_executable()
        runner_script = self._resolve_runner_script()
        if self._subprocess_runner is not None:
            return self._subprocess_runner(str(python_executable), str(runner_script), payload)

        attempts = self._max_retries + 1
        failures: list[str] = []
        for attempt in range(1, attempts + 1):
            with tempfile.TemporaryDirectory(prefix="pat3d-depth-") as temp_dir:
                temp_root = Path(temp_dir)
                input_path = temp_root / "input.json"
                output_path = temp_root / "output.json"
                input_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                process = subprocess.run(
                    [
                        str(python_executable),
                        str(runner_script),
                        "--input-json",
                        str(input_path),
                        "--output-json",
                        str(output_path),
                    ],
                    cwd=str(REPO_ROOT),
                    capture_output=True,
                    text=True,
                )
                if process.returncode == 0:
                    if not output_path.exists():
                        raise runtime_failure(
                            phase="provider_execution",
                            code="depth_output_missing",
                            user_message="Depth estimation finished without producing an output payload.",
                            technical_message="Depth subprocess exited successfully but output.json was not created.",
                            provider_kind="current_depth",
                            retryable=False,
                        )
                    return json.loads(output_path.read_text(encoding="utf-8"))

                failures.append(self._format_process_failure(process, attempt=attempt, attempts=attempts))

        detail = " | ".join(failures)
        raise runtime_failure(
            phase="provider_execution",
            code="depth_subprocess_failed",
            user_message="Depth estimation crashed before producing depth outputs.",
            technical_message=detail,
            provider_kind="current_depth",
            retryable=True,
            details={"attempts": attempts},
        )

    def _format_process_failure(self, process: subprocess.CompletedProcess[str], *, attempt: int, attempts: int) -> str:
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
        return f"attempt {attempt}/{attempts}: {reason}{suffix}"

    def _result_from_scene_dir(
        self,
        *,
        reference_image: ReferenceImageResult,
        provider_name: str,
    ) -> DepthResult:
        scene_name = reference_image.request.scene_id
        scene_dir = Path(self._output_dir) / scene_name
        depth_array = scene_dir / f"{scene_name}_depth.npy"
        depth_visualization = scene_dir / f"{scene_name}_depth.png"
        point_cloud = scene_dir / f"{scene_name}_point_cloud.npz"

        return DepthResult(
            image=reference_image.image,
            depth_array=ArtifactRef(
                artifact_type="depth_array",
                path=str(depth_array),
                format="npy",
                role="depth",
                metadata_path=None,
            ),
            depth_visualization=ArtifactRef(
                artifact_type="depth_preview",
                path=str(depth_visualization),
                format="png",
                role="depth_preview",
                metadata_path=None,
            )
            if depth_visualization.exists()
            else None,
            point_cloud=ArtifactRef(
                artifact_type="point_cloud",
                path=str(point_cloud),
                format="npz",
                role="point_cloud",
                metadata_path=None,
            )
            if point_cloud.exists()
            else None,
            focal_length_px=None,
            metadata=self._metadata_factory(
                stage_name="depth_estimation",
                provider_name=provider_name,
            ),
        )
