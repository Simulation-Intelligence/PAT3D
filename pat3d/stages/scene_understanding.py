from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from pat3d.contracts import DepthEstimator, Segmenter
from pat3d.models import DepthResult, ObjectCatalog, ReferenceImageResult, SegmentationResult

ObjectCatalogBuilder = Callable[[ReferenceImageResult, SegmentationResult], ObjectCatalog]


@dataclass(slots=True)
class SceneUnderstandingOutputs:
    reference_image: ReferenceImageResult
    depth_result: DepthResult
    segmentation_result: SegmentationResult
    object_catalog: ObjectCatalog | None = None


class SceneUnderstandingStage:
    stage_name = "scene_understanding"

    def __init__(
        self,
        depth_estimator: DepthEstimator,
        segmenter: Segmenter,
        object_catalog_builder: ObjectCatalogBuilder | None = None,
    ) -> None:
        self._depth_estimator = depth_estimator
        self._segmenter = segmenter
        self._object_catalog_builder = object_catalog_builder

    def run(
        self,
        reference_image_result: ReferenceImageResult,
        object_hints: Sequence[str] | None = None,
    ) -> SceneUnderstandingOutputs:
        depth_result = self._depth_estimator.predict(reference_image_result)
        segmentation_result = self._segmenter.segment(
            reference_image_result,
            object_hints=object_hints,
        )

        object_catalog = None
        if self._object_catalog_builder is not None:
            object_catalog = self._object_catalog_builder(
                reference_image_result,
                segmentation_result,
            )

        return SceneUnderstandingOutputs(
            reference_image=reference_image_result,
            depth_result=depth_result,
            segmentation_result=segmentation_result,
            object_catalog=object_catalog,
        )
