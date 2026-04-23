from __future__ import annotations

import re
from typing import Iterable

from pat3d.models import DetectedObject, ObjectCatalog, ReferenceImageResult, SegmentationResult
from pat3d.storage import make_stage_metadata


def _canonicalize_label(label: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", label.strip().lower()).strip("_")
    normalized = re.sub(r"\d+$", "", normalized).strip("_")
    return normalized or "object"


class HeuristicObjectCatalogBuilder:
    def __call__(
        self,
        reference_image_result: ReferenceImageResult,
        segmentation_result: SegmentationResult,
    ) -> ObjectCatalog:
        grouped: dict[str, list[str]] = {}
        display_names: dict[str, str] = {}

        for instance in segmentation_result.instances:
            display_name = instance.label or instance.instance_id
            canonical_name = _canonicalize_label(display_name)
            grouped.setdefault(canonical_name, []).append(instance.instance_id)
            display_names.setdefault(canonical_name, display_name)

        objects = []
        for canonical_name, instance_ids in sorted(grouped.items()):
            object_id = f"{reference_image_result.request.scene_id}:{canonical_name}"
            objects.append(
                DetectedObject(
                    object_id=object_id,
                    canonical_name=canonical_name,
                    display_name=display_names.get(canonical_name),
                    count=len(instance_ids),
                    source_instance_ids=tuple(instance_ids),
                    attributes={},
                )
            )

        return ObjectCatalog(
            scene_id=reference_image_result.request.scene_id,
            objects=tuple(objects),
            metadata=make_stage_metadata(
                stage_name="object_catalog",
                provider_name="heuristic_catalog_builder",
            ),
        )
