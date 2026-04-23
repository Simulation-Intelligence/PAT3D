from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Sequence

from pat3d.contracts import TextTo3DProvider
from pat3d.models import ArtifactRef, GeneratedObjectAsset, ObjectDescription, ReferenceImageResult
from pat3d.providers._sam3d_pose_metadata_impl import (
    build_pose_note,
    build_scene_gaussian_note,
    build_scene_manifest_note,
)
from pat3d.providers._sam3d_image_to_3d_impl import (
    DEFAULT_RUNNER_SCRIPT,
    DEFAULT_SAM3D_ROOT,
    REPO_ROOT,
    SAM3DImageTo3DProvider,
)
from pat3d.storage import make_stage_metadata

DEFAULT_MULTI_RUNNER_SCRIPT = REPO_ROOT / "pat3d" / "scripts" / "object_generation" / "_run_sam3d_multi_object_scene_impl.py"


class SAM3DMultiObjectImageTo3DProvider(SAM3DImageTo3DProvider, TextTo3DProvider):
    def __init__(
        self,
        *,
        scene_id: str,
        output_root: str = "data/raw_obj",
        python_executable: str | None = None,
        runner_script: str | None = None,
        single_runner_script: str | None = None,
        sam3d_root: str | None = None,
        config_path: str = "checkpoints/hf/pipeline.yaml",
        device: str | None = None,
        compile: bool = False,
        seed: int = 42,
        crop_completion_enabled: bool = False,
        crop_completion_model: str | None = None,
        subprocess_runner=None,
        metadata_factory=make_stage_metadata,
    ) -> None:
        _ = crop_completion_enabled, crop_completion_model
        super().__init__(
            scene_id=scene_id,
            output_root=output_root,
            python_executable=python_executable,
            runner_script=str(runner_script or DEFAULT_MULTI_RUNNER_SCRIPT),
            sam3d_root=sam3d_root,
            config_path=config_path,
            device=device,
            compile=compile,
            seed=seed,
            subprocess_runner=subprocess_runner,
            metadata_factory=metadata_factory,
        )
        self._single_asset_provider = SAM3DImageTo3DProvider(
            scene_id=scene_id,
            output_root=output_root,
            python_executable=python_executable,
            runner_script=single_runner_script or str(DEFAULT_RUNNER_SCRIPT),
            sam3d_root=sam3d_root or str(DEFAULT_SAM3D_ROOT),
            config_path=config_path,
            device=device,
            compile=compile,
            seed=seed,
            subprocess_runner=subprocess_runner,
            metadata_factory=metadata_factory,
        )
        self._scene_assets: dict[str, GeneratedObjectAsset] = {}

    def _provider_kind(self) -> str:
        return "sam3d_multi_object_image_to_3d"

    def prepare_scene_assets(
        self,
        *,
        asset_requests: Sequence[tuple[ObjectDescription, str]],
        reference_image_result: ReferenceImageResult,
        object_reference_images: Mapping[str, ArtifactRef] | None = None,
        object_mask_artifacts: Mapping[str, Any] | None = None,
    ) -> None:
        if self._scene_assets:
            return

        prepared_requests: list[dict[str, str]] = []
        descriptions_by_id: dict[str, ObjectDescription] = {}
        for description, _source_object_id in asset_requests:
            mask_instance = (object_mask_artifacts or {}).get(description.object_id)
            if mask_instance is None:
                continue
            descriptions_by_id[description.object_id] = description
            preview_image = (object_reference_images or {}).get(description.object_id)
            mask_artifact = getattr(mask_instance, "mask", mask_instance)
            bbox_xyxy = getattr(mask_instance, "bbox_xyxy", None)
            prepared_requests.append(
                {
                    "object_id": description.object_id,
                    "prompt": description.prompt_text,
                    "mask_path": mask_artifact.path,
                    "reference_image_path": preview_image.path if preview_image is not None else "",
                    "bbox_xyxy": (
                        [float(value) for value in bbox_xyxy]
                        if isinstance(bbox_xyxy, (list, tuple)) and len(bbox_xyxy) == 4
                        else []
                    ),
                }
            )

        if not prepared_requests:
            return

        payload = {
            "scene_id": self._scene_id,
            "reference_image_path": reference_image_result.image.path,
            "scene_output_dir": str(self._output_root / self._scene_id),
            "sam3d_root": str(self._resolve_sam3d_root()),
            "config_path": self._config_path,
            "device": self._device,
            "compile": self._compile,
            "seed": self._seed,
            "objects": prepared_requests,
        }
        result = self._run_subprocess(payload)
        scene_manifest_path = str(result.get("scene_manifest_path") or "")
        scene_gaussian_path = str(result.get("scene_gaussian_path") or "")
        object_results = result.get("objects", ())
        if not isinstance(object_results, list):
            raise RuntimeError("SAM 3D multi-object subprocess returned an invalid objects payload.")

        for item in object_results:
            if not isinstance(item, Mapping):
                continue
            object_id = str(item.get("object_id") or "")
            if not object_id:
                continue
            description = descriptions_by_id.get(object_id)
            if description is None:
                continue
            preview_image = (object_reference_images or {}).get(object_id)
            mesh_path = Path(str(item["mesh_path"]))
            mesh_format = mesh_path.suffix.lower().lstrip(".") or "obj"
            notes = [
                f"scene_id={self._scene_id}",
                f"sam3d_root={self._resolve_sam3d_root()}",
                f"config_path={self._config_path}",
                f"seed={self._seed}",
                f"device={result.get('device') or self._device or ''}",
                "mesh_pose_space=posed",
                "sam3d_generation_mode=multi_object_scene",
                build_pose_note(
                    rotation_wxyz=item.get("rotation_wxyz"),
                    translation_xyz=item.get("translation_xyz"),
                    scale_xyz=item.get("scale_xyz"),
                ),
            ]
            canonical_mesh_path = item.get("canonical_mesh_path")
            mesh_ply_path = item.get("mesh_ply_path")
            posed_mesh_ply_path = item.get("posed_mesh_ply_path")
            gaussian_path = item.get("gaussian_path")
            if isinstance(canonical_mesh_path, str) and canonical_mesh_path:
                notes.append(f"canonical_mesh_path={canonical_mesh_path}")
            if isinstance(mesh_ply_path, str) and mesh_ply_path:
                notes.append(f"mesh_ply_path={mesh_ply_path}")
            if isinstance(posed_mesh_ply_path, str) and posed_mesh_ply_path:
                notes.append(f"posed_mesh_ply_path={posed_mesh_ply_path}")
            if isinstance(gaussian_path, str) and gaussian_path:
                notes.append(f"gaussian_path={gaussian_path}")
            if scene_manifest_path:
                notes.append(build_scene_manifest_note(scene_manifest_path))
            if scene_gaussian_path:
                notes.append(build_scene_gaussian_note(scene_gaussian_path))

            self._scene_assets[object_id] = GeneratedObjectAsset(
                object_id=object_id,
                mesh_obj=ArtifactRef(
                    artifact_type="mesh_obj",
                    path=str(mesh_path),
                    format=mesh_format,
                    role="mesh",
                    metadata_path=None,
                ),
                preview_image=preview_image,
                provider_asset_id=str(item.get("provider_asset_id") or object_id),
                metadata=self._metadata_factory(
                    stage_name="object_asset_generation",
                    provider_name="sam3d_multi_object_image_to_3d",
                    notes=tuple(note for note in notes if note),
                ),
            )

    def generate(
        self,
        object_description: ObjectDescription,
        *,
        object_reference_image: ArtifactRef | None = None,
    ) -> GeneratedObjectAsset:
        cached = self._scene_assets.get(object_description.object_id)
        if cached is not None:
            return cached
        return self._single_asset_provider.generate(
            object_description,
            object_reference_image=object_reference_image,
        )
