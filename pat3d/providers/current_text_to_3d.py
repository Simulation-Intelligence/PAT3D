from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from typing import Callable, Sequence

from pat3d.contracts import TextTo3DProvider
from pat3d.models import ArtifactRef, GeneratedObjectAsset, ObjectDescription
from pat3d.providers._legacy_image_inputs import (
    materialize_legacy_object_reference_image,
    remove_legacy_object_reference_image,
)
from pat3d.providers.hunyuan_prompt_variants import (
    DEFAULT_HUNYUAN_PROMPT_STRATEGY,
    label_for_hunyuan_prompt_strategy,
    resolve_hunyuan_prompt,
)
from pat3d.providers.openai_crop_completion import OpenAIObjectCropCompletion
from pat3d.storage import make_stage_metadata

GeneratorFn = Callable[[str, str, str, str, str], None]
REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_GENERATOR_IMPORT_PATH = "pat3d.preprocessing.obj_gen:text_to_3d"
_WORKER_MODULE = "pat3d.providers._current_text_to_3d_worker"


def _load_generator_from_import_path(import_path: str) -> GeneratorFn:
    module_name, separator, attr_name = str(import_path or "").partition(":")
    if not separator or not module_name.strip() or not attr_name.strip():
        raise ValueError(
            "generator_import_path must look like 'module.submodule:function_name'."
        )
    module = importlib.import_module(module_name.strip())
    generator = getattr(module, attr_name.strip())
    if not callable(generator):
        raise TypeError(f"{import_path!r} did not resolve to a callable generator.")
    return generator


def _run_generator_subprocess(
    *,
    generator_import_path: str,
    prompt: str,
    reference_image_root: str,
    output_root: str,
    scene_id: str,
    object_name: str,
    python_executable: str | None = None,
) -> None:
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    completed = subprocess.run(
        [python_executable or sys.executable, "-m", _WORKER_MODULE],
        input=json.dumps(
            {
                "generator_import_path": generator_import_path,
                "prompt": prompt,
                "reference_image_root": reference_image_root,
                "output_root": output_root,
                "scene_id": scene_id,
                "object_name": object_name,
            }
        ),
        text=True,
        cwd=str(REPO_ROOT),
        env=env,
        check=False,
    )
    if completed.returncode == 0:
        return
    if completed.returncode < 0:
        signal_number = -completed.returncode
        try:
            signal_name = signal.Signals(signal_number).name
        except ValueError:
            signal_name = f"SIG{signal_number}"
        raise RuntimeError(
            f"current_text_to_3d worker for '{object_name}' terminated with signal {signal_name}."
        )
    raise RuntimeError(
        f"current_text_to_3d worker for '{object_name}' exited with code {completed.returncode}."
    )


class CurrentTextTo3DProvider(TextTo3DProvider):
    def __init__(
        self,
        *,
        scene_id: str,
        output_root: str = "data/raw_obj",
        reference_image_root: str = "data/ref_img_obj",
        prompt_strategy: str = DEFAULT_HUNYUAN_PROMPT_STRATEGY,
        crop_completion_enabled: bool = False,
        crop_completion_model: str = "gpt-image-1.5",
        crop_completion_output_root: str = "data/ref_img_obj_completed",
        crop_completion_api_key: str | None = None,
        crop_completion_base_url: str | None = None,
        generator: GeneratorFn | None = None,
        generator_import_path: str = _DEFAULT_GENERATOR_IMPORT_PATH,
        generator_subprocess_enabled: bool = True,
        generator_python_executable: str | None = None,
        crop_completion: OpenAIObjectCropCompletion | None = None,
        metadata_factory: Callable[..., object] = make_stage_metadata,
    ) -> None:
        self._scene_id = scene_id
        self._output_root = output_root
        self._reference_image_root = reference_image_root
        self._prompt_strategy = prompt_strategy
        # Keep the Hunyuan-backed generator out of the main dashboard worker process by
        # default so each object generation can release host/GPU memory on child exit.
        self._generator = generator
        self._generator_import_path = (
            str(generator_import_path).strip() or _DEFAULT_GENERATOR_IMPORT_PATH
        )
        self._generator_subprocess_enabled = bool(generator_subprocess_enabled and generator is None)
        self._generator_python_executable = (
            str(generator_python_executable).strip()
            if generator_python_executable is not None and str(generator_python_executable).strip()
            else None
        )
        if crop_completion is not None:
            self._crop_completion = crop_completion
        elif crop_completion_enabled:
            self._crop_completion = OpenAIObjectCropCompletion(
                api_key=crop_completion_api_key,
                base_url=crop_completion_base_url,
                model=crop_completion_model,
                output_root=crop_completion_output_root,
            )
        else:
            self._crop_completion = None
        self._metadata_factory = metadata_factory

    def generate(
        self,
        object_description: ObjectDescription,
        *,
        object_reference_image: ArtifactRef | None = None,
        scene_object_canonical_names: Sequence[str] = (),
        source_object_count: int = 1,
    ) -> GeneratedObjectAsset:
        object_name = object_description.object_id
        resolved_prompt, resolved_prompt_strategy = resolve_hunyuan_prompt(
            object_description,
            strategy=self._prompt_strategy,
        )
        preview_image = object_reference_image
        reference_image_mode = "text_only"
        completion_notes: tuple[str, ...] = ()
        if object_reference_image is not None:
            if self._crop_completion is not None:
                current_label = " ".join(object_description.canonical_name.replace("_", " ").split()).strip().lower()
                disallowed_object_names: list[str] = []
                seen_disallowed: set[str] = set()
                for raw_name in scene_object_canonical_names:
                    normalized_name = " ".join(str(raw_name or "").replace("_", " ").strip().split())
                    normalized_key = normalized_name.lower()
                    if (
                        not normalized_name
                        or normalized_key == current_label
                        or normalized_key in seen_disallowed
                    ):
                        continue
                    seen_disallowed.add(normalized_key)
                    disallowed_object_names.append(normalized_name)
                completion_result = self._crop_completion.complete(
                    scene_id=self._scene_id,
                    object_name=object_name,
                    object_description=object_description,
                    object_reference_image=object_reference_image,
                    disallowed_object_names=tuple(disallowed_object_names),
                    source_object_count=source_object_count,
                )
                preview_image = completion_result.completed_image
                reference_image_mode = "crop_completion"
                completion_notes = (
                    f"crop_completion_model={completion_result.model}",
                    f"crop_completion_prompt={completion_result.prompt}",
                    f"crop_completion_source_image={completion_result.source_image.path}",
                    f"crop_completion_prepared_input={completion_result.prepared_input_image.path}",
                    f"crop_completion_mask={completion_result.mask_image.path}",
                    f"crop_completion_output={completion_result.completed_image.path}",
                )
            else:
                reference_image_mode = "direct_crop"
            materialize_legacy_object_reference_image(
                scene_id=self._scene_id,
                object_name=object_name,
                object_reference_image=preview_image,
                target_root=self._reference_image_root,
            )
        else:
            remove_legacy_object_reference_image(
                scene_id=self._scene_id,
                object_name=object_name,
                target_root=self._reference_image_root,
            )
        if self._generator is not None:
            self._generator(
                resolved_prompt,
                self._reference_image_root,
                self._output_root,
                self._scene_id,
                object_name,
            )
        elif self._generator_subprocess_enabled:
            _run_generator_subprocess(
                generator_import_path=self._generator_import_path,
                prompt=resolved_prompt,
                reference_image_root=self._reference_image_root,
                output_root=self._output_root,
                scene_id=self._scene_id,
                object_name=object_name,
                python_executable=self._generator_python_executable,
            )
        else:
            generator = _load_generator_from_import_path(self._generator_import_path)
            generator(
                resolved_prompt,
                self._reference_image_root,
                self._output_root,
                self._scene_id,
                object_name,
            )

        mesh_dir = self._resolve_mesh_dir(object_name)
        textured_obj = mesh_dir / f"{object_name}_texture.obj"
        raw_obj = mesh_dir / f"{object_name}_mesh.obj"
        mesh_obj_path = textured_obj if textured_obj.exists() else raw_obj

        mesh_mtl_path = mesh_dir / "material.mtl"
        texture_path = mesh_dir / "material_0.png"

        return GeneratedObjectAsset(
            object_id=object_description.object_id,
            mesh_obj=ArtifactRef(
                artifact_type="mesh_obj",
                path=str(mesh_obj_path),
                format="obj",
                role="textured_mesh" if textured_obj.exists() else "mesh",
                metadata_path=None,
            ),
            mesh_mtl=ArtifactRef(
                artifact_type="material",
                path=str(mesh_mtl_path),
                format="mtl",
                role="material",
                metadata_path=None,
            )
            if mesh_mtl_path.exists()
            else None,
            texture_image=ArtifactRef(
                artifact_type="texture",
                path=str(texture_path),
                format="png",
                role="texture",
                metadata_path=None,
            )
            if texture_path.exists()
            else None,
            preview_image=preview_image,
            provider_asset_id=f"{self._scene_id}:{object_name}",
            metadata=self._metadata_factory(
                stage_name="object_asset_generation",
                provider_name="current_hunyuan3d",
                notes=(
                    f"hunyuan_reference_image_mode={reference_image_mode}",
                    f"hunyuan_prompt_strategy={resolved_prompt_strategy}",
                    f"hunyuan_prompt_strategy_label={label_for_hunyuan_prompt_strategy(resolved_prompt_strategy)}",
                    f"hunyuan_effective_prompt={resolved_prompt}",
                    f"hunyuan_source_prompt={object_description.prompt_text.strip()}",
                    *completion_notes,
                ),
            ),
        )

    def _resolve_mesh_dir(self, object_name: str) -> Path:
        scene_scoped = Path(self._output_root) / self._scene_id / object_name
        if scene_scoped.exists():
            return scene_scoped
        legacy_flat = Path(self._output_root) / object_name
        return legacy_flat if legacy_flat.exists() else scene_scoped
