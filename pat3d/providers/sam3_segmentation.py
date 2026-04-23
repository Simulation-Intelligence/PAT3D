from __future__ import annotations

from collections import defaultdict
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from PIL import Image

from pat3d.contracts import Segmenter
from pat3d.models import ArtifactRef, MaskInstance, ReferenceImageResult, SegmentationResult
from pat3d.storage import make_stage_metadata


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONDA_ENV_PYTHONS = (
    Path.home() / ".conda" / "envs" / "pat3d-sam3" / "bin" / "python",
    Path.home() / "anaconda3" / "envs" / "pat3d-sam3" / "bin" / "python",
    Path.home() / "miniconda3" / "envs" / "pat3d-sam3" / "bin" / "python",
)
DEFAULT_LOCAL_VENV_PYTHON = REPO_ROOT / ".venv-sam3" / "bin" / "python"
DEFAULT_RUNNER_SCRIPT = REPO_ROOT / "pat3d" / "scripts" / "segmentation" / "run_sam3_segment.py"


def _repo_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return REPO_ROOT / candidate


def _is_cuda_oom_detail(detail: str) -> bool:
    normalized = str(detail or "").lower()
    return "cuda" in normalized and "out of memory" in normalized


def _should_fallback_to_cpu(device: str | None, detail: str) -> bool:
    requested_device = str(device or "").strip().lower()
    if requested_device and not requested_device.startswith("cuda"):
        return False
    return _is_cuda_oom_detail(detail)


def _clean_prompt(prompt: str) -> str:
    return " ".join(str(prompt or "").strip().split())


def _canonical_prompt(prompt: str) -> str:
    normalized = _clean_prompt(prompt).lower().replace("-", "_").replace(" ", "_")
    cleaned = "".join(ch for ch in normalized if ch.isalnum() or ch == "_").strip("_")
    return cleaned or "object"


def _bbox_int(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def _cutout_from_mask(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> Image.Image:
    x0, y0, x1, y1 = bbox
    crop = image_rgb[y0 : y1 + 1, x0 : x1 + 1].copy()
    crop_mask = mask[y0 : y1 + 1, x0 : x1 + 1]
    white = np.full_like(crop, 255)
    crop = np.where(crop_mask[..., None], crop, white)
    return Image.fromarray(crop)


def _mask_color(index: int) -> np.ndarray:
    palette = (
        (255, 107, 107),
        (77, 171, 247),
        (81, 207, 102),
        (252, 196, 25),
        (132, 94, 247),
        (255, 146, 43),
        (32, 201, 151),
        (240, 101, 149),
        (92, 124, 250),
        (148, 216, 45),
    )
    return np.asarray(palette[index % len(palette)], dtype=np.float32)


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if str(getattr(value, "dtype", "")) in {"torch.bfloat16", "bfloat16"} and hasattr(value, "float"):
        value = value.float()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy())
    return np.asarray(value)


def _normalize_mask_batch(value: Any) -> np.ndarray:
    masks = _to_numpy(value)
    if masks.ndim < 2:
        raise RuntimeError(f"Expected mask tensor with at least 2 dimensions, got shape {masks.shape}.")
    if masks.ndim == 2:
        return masks[None, ...]
    height, width = masks.shape[-2:]
    return masks.reshape(-1, height, width)


def normalize_object_prompts(object_hints: Sequence[str] | None) -> list[str]:
    prompts: list[str] = []
    seen_prompts: set[str] = set()
    for hint in object_hints or ():
        cleaned = _clean_prompt(str(hint))
        if not cleaned:
            continue
        dedupe_key = cleaned.lower()
        if dedupe_key in seen_prompts:
            continue
        seen_prompts.add(dedupe_key)
        prompts.append(cleaned)
    return prompts


def materialize_segmentation_outputs(
    *,
    scene_name: str,
    image_array: np.ndarray,
    prompt_masks: Sequence[Mapping[str, Any]],
    output_dir: str = "data/seg",
) -> dict[str, Any]:
    scene_dir = Path(output_dir) / scene_name
    if scene_dir.exists():
        shutil.rmtree(scene_dir)
    scene_dir.mkdir(parents=True, exist_ok=True)

    overlay = image_array.astype(np.float32).copy()
    counts: dict[str, int] = defaultdict(int)
    instances: list[dict[str, Any]] = []

    for prompt_entry in prompt_masks:
        prompt = _clean_prompt(str(prompt_entry.get("label", "")))
        if not prompt:
            continue

        masks = _normalize_mask_batch(prompt_entry.get("masks", ()))
        scores = _to_numpy(prompt_entry.get("scores", ()))
        if scores.ndim == 0 and masks.shape[0] == 1:
            scores = scores.reshape(1)

        stem_label = _canonical_prompt(prompt)
        for local_index, mask_value in enumerate(masks):
            mask = np.asarray(mask_value).astype(bool)
            area_px = int(np.count_nonzero(mask))
            if area_px <= 0:
                continue

            counts[stem_label] += 1
            instance_name = f"{stem_label}{counts[stem_label]}"
            stem = f"{scene_name}_{instance_name}_{instance_name}"
            bbox_int = _bbox_int(mask)
            cutout_path = scene_dir / f"{stem}.png"
            mask_path = scene_dir / f"{stem}_ann.png"
            _cutout_from_mask(image_array, mask, bbox_int).save(cutout_path)
            Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(mask_path)

            overlay[mask] = overlay[mask] * 0.35 + _mask_color(len(instances)) * 0.65
            score = None
            if local_index < len(scores):
                score = float(scores[local_index])

            instances.append(
                {
                    "instance_id": stem,
                    "label": prompt,
                    "mask_path": str(Path(output_dir) / scene_name / mask_path.name),
                    "bbox_xyxy": [float(value) for value in bbox_int],
                    "confidence": score,
                    "area_px": area_px,
                }
            )

    if not instances:
        raise RuntimeError(
            "SAM 3 did not produce any masks for the requested object hints on this reference image."
        )

    composite_path = scene_dir / f"{scene_name}_segmentation.png"
    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8), mode="RGB").save(composite_path)
    return {
        "provider_name": "sam3_segmenter",
        "instances": instances,
        "composite_visualization_path": str(Path(output_dir) / scene_name / composite_path.name),
    }


class SAM3Segmenter(Segmenter):
    def __init__(
        self,
        *,
        output_dir: str = "data/seg",
        checkpoint_path: str | None = None,
        device: str | None = None,
        confidence_threshold: float = 0.5,
        load_from_hf: bool = True,
        python_executable: str | None = None,
        runner_script: str | None = None,
        subprocess_runner: Callable[[str, str, Mapping[str, Any]], Mapping[str, Any]] | None = None,
        model_builder: Callable[..., Any] | None = None,
        processor_factory: Callable[[Any], Any] | None = None,
        metadata_factory: Callable[..., object] = make_stage_metadata,
    ) -> None:
        self._output_dir = output_dir
        self._checkpoint_path = checkpoint_path
        self._device = device
        self._confidence_threshold = float(confidence_threshold)
        self._load_from_hf = bool(load_from_hf)
        self._python_executable = python_executable
        self._runner_script = runner_script
        self._subprocess_runner = subprocess_runner
        self._model_builder = model_builder
        self._processor_factory = processor_factory
        self._metadata_factory = metadata_factory
        self._processor: Any | None = None
        self._resolved_device: str | None = None

    def _python_supports_sam3(self, python_executable: Path) -> tuple[bool, str | None]:
        try:
            process = subprocess.run(
                [str(python_executable), "-c", "import sam3"],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
            )
        except OSError as exc:
            return False, str(exc)
        if process.returncode == 0:
            return True, None
        detail = process.stderr.strip() or process.stdout.strip() or "unknown import failure"
        return False, detail

    def _resolve_python_executable(self) -> Path:
        candidates = []
        if self._python_executable:
            candidates.append(Path(self._python_executable))
        env_value = os.environ.get("PAT3D_SAM3_PYTHON", "").strip()
        if env_value:
            candidates.append(Path(env_value))
        candidates.extend((*DEFAULT_CONDA_ENV_PYTHONS, DEFAULT_LOCAL_VENV_PYTHON))
        diagnostics: list[str] = []
        for candidate in candidates:
            expanded = candidate.expanduser()
            if not expanded.exists():
                continue
            supports_sam3, detail = self._python_supports_sam3(expanded)
            if supports_sam3:
                return expanded
            diagnostics.append(f"{expanded}: {detail}")
        detail_suffix = f" Checked: {'; '.join(diagnostics)}" if diagnostics else ""
        raise RuntimeError(
            "Could not find a dedicated SAM 3 Python environment. "
            "Set PAT3D_SAM3_PYTHON or create a `pat3d-sam3` conda env with the `sam3` package installed."
            f"{detail_suffix}"
        )

    def _resolve_runner_script(self) -> Path:
        candidate = Path(self._runner_script) if self._runner_script else DEFAULT_RUNNER_SCRIPT
        resolved = candidate.expanduser()
        if not resolved.is_absolute():
            resolved = REPO_ROOT / resolved
        if not resolved.exists():
            raise RuntimeError(f"SAM 3 runner script is missing: {resolved}")
        return resolved

    def _build_processor(self) -> Any:
        if self._processor is not None:
            return self._processor

        model_builder = self._model_builder
        processor_factory = self._processor_factory
        requested_device = self._device or ("cuda" if __import__("torch").cuda.is_available() else "cpu")
        device = requested_device
        checkpoint_path = None
        if self._checkpoint_path:
            checkpoint_path = str(_repo_path(self._checkpoint_path))

        if model_builder is None or processor_factory is None:
            try:
                from sam3.model_builder import build_sam3_image_model
                from sam3.model.sam3_image_processor import Sam3Processor
            except ImportError as exc:
                raise RuntimeError(
                    "SAM 3 segmentation requires the `sam3` package. "
                    "Install Meta's official SAM 3 package before selecting this backend."
                ) from exc

            model_builder = model_builder or build_sam3_image_model

            if processor_factory is None:

                def _default_processor_factory(model: Any) -> Any:
                    return Sam3Processor(
                        model,
                        device=device,
                        confidence_threshold=self._confidence_threshold,
                    )

                processor_factory = _default_processor_factory

        try:
            model = model_builder(
                device=device,
                checkpoint_path=checkpoint_path,
                load_from_HF=self._load_from_hf,
            )
            processor = processor_factory(model)
        except Exception as exc:
            if _should_fallback_to_cpu(requested_device, str(exc)):
                device = "cpu"
                model = model_builder(
                    device=device,
                    checkpoint_path=checkpoint_path,
                    load_from_HF=self._load_from_hf,
                )
                processor = processor_factory(model)
            else:
                raise RuntimeError(
                    "Could not initialize the SAM 3 segmenter. "
                    "Provide a local checkpoint_path or authenticate to Hugging Face for `facebook/sam3`."
                ) from exc

        if hasattr(processor, "set_confidence_threshold"):
            processor.set_confidence_threshold(self._confidence_threshold)
        self._processor = processor
        self._resolved_device = str(device)
        return processor

    def _segment_inprocess(
        self,
        reference_image: ReferenceImageResult,
        prompts: Sequence[str],
    ) -> Mapping[str, Any]:
        processor = self._build_processor()
        image_path = _repo_path(reference_image.image.path)
        image_rgb = Image.open(image_path).convert("RGB")
        image_array = np.asarray(image_rgb)
        state = processor.set_image(image_rgb)
        prompt_masks: list[dict[str, Any]] = []

        for prompt in prompts:
            if hasattr(processor, "reset_all_prompts"):
                processor.reset_all_prompts(state)
            state = processor.set_text_prompt(prompt=prompt, state=state)
            prompt_masks.append(
                {
                    "label": prompt,
                    "masks": _to_numpy(state.get("masks", ())),
                    "scores": _to_numpy(state.get("scores", ())),
                }
            )

        result = materialize_segmentation_outputs(
            scene_name=reference_image.request.scene_id,
            image_array=image_array,
            prompt_masks=prompt_masks,
            output_dir=self._output_dir,
        )
        if self._resolved_device:
            result["device"] = self._resolved_device
        return result

    def _invoke_subprocess(
        self,
        *,
        python_executable: Path,
        runner_script: Path,
        payload: Mapping[str, Any],
    ) -> tuple[int, str, str, dict[str, Any] | None]:
        with tempfile.TemporaryDirectory(prefix="pat3d-sam3-") as temp_dir:
            temp_root = Path(temp_dir)
            input_path = temp_root / "input.json"
            output_path = temp_root / "output.json"
            input_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            process = subprocess.run(
                [str(python_executable), str(runner_script), "--input-json", str(input_path), "--output-json", str(output_path)],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
            )
            output_payload = None
            if process.returncode == 0 and output_path.exists():
                output_payload = json.loads(output_path.read_text(encoding="utf-8"))
            return process.returncode, process.stdout or "", process.stderr or "", output_payload

    def _run_subprocess(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        python_executable = self._resolve_python_executable()
        runner_script = self._resolve_runner_script()
        if self._subprocess_runner is not None:
            return self._subprocess_runner(str(python_executable), str(runner_script), payload)
        returncode, stdout, stderr, output_payload = self._invoke_subprocess(
            python_executable=python_executable,
            runner_script=runner_script,
            payload=payload,
        )
        if returncode == 0:
            if output_payload is None:
                raise RuntimeError("SAM 3 subprocess did not produce an output payload.")
            return output_payload

        detail = stderr.strip() or stdout.strip() or f"exit code {returncode}"
        if _should_fallback_to_cpu(payload.get("device") if isinstance(payload, Mapping) else None, detail):
            cpu_payload = dict(payload)
            cpu_payload["device"] = "cpu"
            returncode, stdout, stderr, output_payload = self._invoke_subprocess(
                python_executable=python_executable,
                runner_script=runner_script,
                payload=cpu_payload,
            )
            if returncode == 0:
                if output_payload is None:
                    raise RuntimeError("SAM 3 subprocess did not produce an output payload.")
                return output_payload
            detail = stderr.strip() or stdout.strip() or f"exit code {returncode}"

        raise RuntimeError(f"SAM 3 subprocess failed: {detail}")

    def _segment_subprocess(
        self,
        reference_image: ReferenceImageResult,
        prompts: Sequence[str],
    ) -> Mapping[str, Any]:
        payload = {
            "scene_name": reference_image.request.scene_id,
            "reference_image_path": str(reference_image.image.path),
            "output_dir": self._output_dir,
            "prompts": list(prompts),
            "device": self._device,
            "confidence_threshold": self._confidence_threshold,
            "load_from_hf": self._load_from_hf,
            "checkpoint_path": self._checkpoint_path,
        }
        return self._run_subprocess(payload)

    def _coerce_result(
        self,
        reference_image: ReferenceImageResult,
        payload: Mapping[str, Any],
        prompts: Sequence[str],
    ) -> SegmentationResult:
        instances: list[MaskInstance] = []
        for entry in payload.get("instances", ()):
            if not isinstance(entry, Mapping):
                continue
            instances.append(
                MaskInstance(
                    instance_id=str(entry["instance_id"]),
                    label=str(entry.get("label") or entry["instance_id"]),
                    mask=ArtifactRef(
                        artifact_type="mask",
                        path=str(entry["mask_path"]),
                        format="png",
                        role="instance_mask",
                        metadata_path=None,
                    ),
                    bbox_xyxy=tuple(float(value) for value in entry.get("bbox_xyxy", (0, 0, 0, 0))),
                    confidence=(
                        float(entry["confidence"])
                        if entry.get("confidence") is not None
                        else None
                    ),
                    area_px=int(entry.get("area_px", 0)),
                )
            )

        composite_path = payload.get("composite_visualization_path")
        return SegmentationResult(
            image=reference_image.image,
            instances=tuple(instances),
            composite_visualization=(
                ArtifactRef(
                    artifact_type="segmentation_visualization",
                    path=str(composite_path),
                    format="png",
                    role="segmentation_visualization",
                    metadata_path=None,
                )
                if composite_path
                else None
            ),
            metadata=self._metadata_factory(
                stage_name="segmentation",
                provider_name=str(payload.get("provider_name") or "sam3_segmenter"),
                notes=tuple(
                    [*(f"prompt={prompt}" for prompt in prompts)]
                    + ([f"device={payload.get('device')}"] if payload.get("device") else [])
                ),
            ),
        )

    def segment(
        self,
        reference_image: ReferenceImageResult,
        object_hints: Sequence[str] | None = None,
    ) -> SegmentationResult:
        prompts = normalize_object_prompts(object_hints)
        if not prompts:
            raise RuntimeError(
                "SAM 3 segmentation requires at least one requested object hint. "
                "Add object names in the dashboard before running this backend."
            )

        if self._model_builder is not None or self._processor_factory is not None:
            payload = self._segment_inprocess(reference_image, prompts)
        else:
            payload = self._segment_subprocess(reference_image, prompts)
        return self._coerce_result(reference_image, payload, prompts)
