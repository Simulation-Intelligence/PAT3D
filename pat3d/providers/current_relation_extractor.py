from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from pat3d.providers._compat import instantiate_model, scene_id_from_reference
from pat3d.providers._relation_utils import normalize_scene_relations
from pat3d.storage.artifacts import ArtifactStore


class CurrentRelationExtractor:
    def __init__(
        self,
        ref_image_folder: str = "data/ref_img",
        descrip_folder: str = "data/descrip",
        prompt_folder: str = "pat3d/preprocessing/gpt_utils",
        api_key_path: str = "pat3d/preprocessing/gpt_utils/apikey.txt",
        contain_folder: str = "data/contain",
        contain_on_folder: str = "data/contain_on",
        artifact_store: ArtifactStore | None = None,
    ) -> None:
        self.ref_image_folder = ref_image_folder
        self.descrip_folder = descrip_folder
        self.prompt_folder = prompt_folder
        self.api_key_path = api_key_path
        self.contain_folder = contain_folder
        self.contain_on_folder = contain_on_folder
        self.artifact_store = artifact_store

    def _normalize_relations(
        self,
        raw_payload: Any,
        *,
        object_catalog: Any | None = None,
    ) -> tuple[list[Any], tuple[str, ...]]:
        relations, root_object_ids = normalize_scene_relations(
            raw_payload,
            object_catalog=object_catalog,
        )
        return (
            [
                instantiate_model("ContainmentRelation", relation.to_dict())
                for relation in relations
            ],
            root_object_ids,
        )

    def extract(
        self,
        object_catalog: Any | None = None,
        reference_image_result: Any | None = None,
        segmentation_result: Any | None = None,
        depth_result: Any | None = None,
        scene_id: str | None = None,
        mode: str = "contain_on",
    ) -> Any:
        from pat3d.preprocessing.contain import get_contain_info, get_contain_on_info

        actual_scene_id = scene_id or scene_id_from_reference(reference_image_result, default="scene")
        artifact_store = self.artifact_store or ArtifactStore()
        args = SimpleNamespace(
            descrip_folder=self.descrip_folder,
            gpt_prompt_folder=self.prompt_folder,
            ref_image_folder=self.ref_image_folder,
            gpt_apikey_path=self.api_key_path,
            contain_folder=self.contain_folder,
            contain_on_folder=self.contain_on_folder,
        )

        if mode == "contain":
            get_contain_info(args, actual_scene_id)
            output_path = Path(self.contain_folder) / f"{actual_scene_id}.json"
        else:
            get_contain_on_info(args, actual_scene_id)
            output_path = Path(self.contain_on_folder) / f"{actual_scene_id}.json"

        raw_payload = json.loads(output_path.read_text(encoding="utf-8"))
        relations, root_object_ids = self._normalize_relations(
            raw_payload,
            object_catalog=object_catalog,
        )
        payload = {
            "scene_id": actual_scene_id,
            "relations": relations,
            "root_object_ids": root_object_ids,
            "metadata": artifact_store.build_stage_metadata(
                stage_name="relation_extraction",
                provider_name="current_relation_extractor",
                notes=[f"mode={mode}"],
            ),
        }
        return instantiate_model("SceneRelationGraph", payload)
