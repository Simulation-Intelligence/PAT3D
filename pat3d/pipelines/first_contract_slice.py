from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from pat3d.models import ObjectCatalog, ReferenceImageResult, SceneRequest
from pat3d.stages import (
    ObjectRelationExtractionOutputs,
    ObjectRelationExtractionStage,
    ReferenceImagePassthroughStage,
    SceneUnderstandingOutputs,
    SceneUnderstandingStage,
)


@dataclass(slots=True)
class FirstContractSliceOutputs:
    reference_image_result: ReferenceImageResult
    scene_understanding: SceneUnderstandingOutputs
    object_relation: ObjectRelationExtractionOutputs | None


class FirstContractSlicePipeline:
    def __init__(
        self,
        reference_image_stage: ReferenceImagePassthroughStage,
        scene_understanding_stage: SceneUnderstandingStage,
        object_relation_stage: ObjectRelationExtractionStage | None = None,
    ) -> None:
        self._reference_image_stage = reference_image_stage
        self._scene_understanding_stage = scene_understanding_stage
        self._object_relation_stage = object_relation_stage

    def run(
        self,
        request: SceneRequest,
        object_hints: Sequence[str] | None = None,
        object_catalog: ObjectCatalog | None = None,
    ) -> FirstContractSliceOutputs:
        reference_image_result = self._reference_image_stage.run(request)
        understanding = self._scene_understanding_stage.run(
            reference_image_result,
            object_hints=object_hints,
        )

        object_relation = None
        if self._object_relation_stage is not None:
            object_relation = self._object_relation_stage.run(
                reference_image_result=reference_image_result,
                segmentation_result=understanding.segmentation_result,
                depth_result=understanding.depth_result,
                object_catalog=object_catalog or understanding.object_catalog,
            )

        return FirstContractSliceOutputs(
            reference_image_result=reference_image_result,
            scene_understanding=understanding,
            object_relation=object_relation,
        )
