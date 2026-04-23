from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from pat3d.providers._compat import instantiate_model, scene_id_from_reference, strip_numeric_suffix
from pat3d.storage.artifacts import ArtifactStore


class CurrentObjectTextDescriber:
    def __init__(
        self,
        seg_folder: str = "data/seg",
        prompt_folder: str = "pat3d/preprocessing/gpt_utils",
        api_key_path: str = "pat3d/preprocessing/gpt_utils/apikey.txt",
        output_root: str = "data/descrip",
        artifact_store: ArtifactStore | None = None,
    ) -> None:
        self.seg_folder = seg_folder
        self.prompt_folder = prompt_folder
        self.api_key_path = api_key_path
        self.output_root = output_root
        self.artifact_store = artifact_store

    def describe_objects(
        self,
        object_catalog: Any | None = None,
        reference_image_result: Any | None = None,
        segmentation_result: Any | None = None,
        scene_id: str | None = None,
    ) -> dict[str, Any]:
        from pat3d.preprocessing.obj_descrip import get_obj_descrip

        actual_scene_id = scene_id or scene_id_from_reference(reference_image_result, default="scene")
        args = SimpleNamespace(
            seg_folder=self.seg_folder,
            gpt_prompt_folder=self.prompt_folder,
            gpt_apikey_path=self.api_key_path,
            descrip_folder=self.output_root,
        )
        get_obj_descrip(args, actual_scene_id)

        artifact_store = self.artifact_store or ArtifactStore()
        output_path = Path(self.output_root) / f"{actual_scene_id}.json"
        raw_payload = artifact_store.write_json(output_path, __import__("json").loads(output_path.read_text(encoding="utf-8")))
        data = __import__("json").loads(output_path.read_text(encoding="utf-8"))
        descriptions = []
        for object_id, prompt_text in data.items():
            descriptions.append(
                instantiate_model(
                    "ObjectDescription",
                    {
                        "object_id": object_id,
                        "canonical_name": strip_numeric_suffix(object_id),
                        "prompt_text": prompt_text,
                        "visual_attributes": {},
                        "material_attributes": {},
                        "orientation_hints": {},
                        "metadata": artifact_store.build_stage_metadata(
                            stage_name="object_description",
                            provider_name="current_object_describer",
                        ),
                    },
                )
            )

        return {
            "scene_id": actual_scene_id,
            "descriptions": descriptions,
            "artifact": artifact_store.build_artifact_ref(
                raw_payload, "object_descriptions", "json", "object_descriptions"
            ),
            "metadata": artifact_store.build_stage_metadata(
                stage_name="object_description",
                provider_name="current_object_describer",
            ),
        }
