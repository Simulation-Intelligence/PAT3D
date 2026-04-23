from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from pat3d.providers._compat import instantiate_model, scene_id_from_reference
from pat3d.storage.artifacts import ArtifactStore


class CurrentSizeInferer:
    def __init__(
        self,
        size_folder: str = "data/size",
        items_folder: str = "data/items",
        ref_image_folder: str = "data/ref_img",
        prompt_folder: str = "pat3d/preprocessing/gpt_utils",
        api_key_path: str = "pat3d/preprocessing/gpt_utils/apikey.txt",
        artifact_store: ArtifactStore | None = None,
    ) -> None:
        self.size_folder = size_folder
        self.items_folder = items_folder
        self.ref_image_folder = ref_image_folder
        self.prompt_folder = prompt_folder
        self.api_key_path = api_key_path
        self.artifact_store = artifact_store

    def _normalize_dimensions(self, value: Any) -> dict[str, float] | None:
        if isinstance(value, dict):
            dims = {}
            for axis in ("x", "y", "z", "width", "height", "depth", "length"):
                if axis in value and isinstance(value[axis], (int, float)):
                    dims[axis] = float(value[axis])
            return dims or None
        if isinstance(value, list) and len(value) >= 3 and all(isinstance(v, (int, float)) for v in value[:3]):
            return {"x": float(value[0]), "y": float(value[1]), "z": float(value[2])}
        return None

    def infer(self, scene_id: str | None = None, reference_image_result: Any | None = None) -> dict[str, Any]:
        from pat3d.preprocessing.size import get_size as current_get_size

        actual_scene_id = scene_id or scene_id_from_reference(reference_image_result, default="scene")
        artifact_store = self.artifact_store or ArtifactStore()
        self_obj = SimpleNamespace(scene_name=actual_scene_id)
        args = SimpleNamespace(
            size_folder=self.size_folder,
            items_folder=self.items_folder,
            ref_image_folder=self.ref_image_folder,
            gpt_prompt_folder=self.prompt_folder,
            gpt_apikey_path=self.api_key_path,
        )
        current_get_size(self_obj, args)

        output_path = Path(self.size_folder) / f"{actual_scene_id}.json"
        raw_payload = json.loads(output_path.read_text(encoding="utf-8"))
        priors = []
        for object_id, value in raw_payload.items():
            priors.append(
                instantiate_model(
                    "SizePrior",
                    {
                        "object_id": object_id,
                        "dimensions_m": self._normalize_dimensions(value),
                        "relative_scale_to_scene": None,
                        "source": "current_size_inferer",
                        "metadata": artifact_store.build_stage_metadata(
                            stage_name="size_inference",
                            provider_name="current_size_inferer",
                        ),
                    },
                )
            )

        return {
            "scene_id": actual_scene_id,
            "priors": priors,
            "artifact": artifact_store.build_artifact_ref(
                output_path, "size_priors", "json", "size_priors"
            ),
            "metadata": artifact_store.build_stage_metadata(
                stage_name="size_inference",
                provider_name="current_size_inferer",
            ),
        }
