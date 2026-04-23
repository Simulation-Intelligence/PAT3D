from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from pat3d.contracts import ObjectTextDescriber, RelationExtractor, SizePriorEstimator
from pat3d.models import (
    DepthResult,
    ObjectCatalog,
    ObjectDescription,
    ReferenceImageResult,
    SceneRelationGraph,
    SegmentationResult,
    SizeInferenceReport,
    SizeInferenceResult,
    SizePrior,
)
from pat3d.providers._relation_utils import (
    materialize_relation_graph_for_instances,
    materialize_size_priors_for_instances,
)

ObjectCatalogBuilder = Callable[[ReferenceImageResult, SegmentationResult], ObjectCatalog]


@dataclass(slots=True)
class ObjectRelationExtractionOutputs:
    reference_image: ReferenceImageResult
    segmentation_result: SegmentationResult
    object_catalog: ObjectCatalog
    object_descriptions: Sequence[ObjectDescription]
    relation_graph: SceneRelationGraph
    size_priors: Sequence[SizePrior]
    size_inference_report: SizeInferenceReport | None = None
    depth_result: DepthResult | None = None


class ObjectRelationExtractionStage:
    stage_name = "object_relation_extraction"

    def __init__(
        self,
        object_describer: ObjectTextDescriber,
        relation_extractor: RelationExtractor,
        size_inferer: SizePriorEstimator,
        object_catalog_builder: ObjectCatalogBuilder | None = None,
    ) -> None:
        self._object_describer = object_describer
        self._relation_extractor = relation_extractor
        self._size_inferer = size_inferer
        self._object_catalog_builder = object_catalog_builder

    def prepare_without_size_inference(
        self,
        reference_image_result: ReferenceImageResult,
        segmentation_result: SegmentationResult,
        depth_result: DepthResult | None = None,
        object_catalog: ObjectCatalog | None = None,
    ) -> ObjectRelationExtractionOutputs:
        resolved_catalog = self._resolve_object_catalog(
            reference_image_result=reference_image_result,
            segmentation_result=segmentation_result,
            object_catalog=object_catalog,
        )
        object_descriptions = self._object_describer.describe_objects(
            resolved_catalog,
            reference_image_result,
            segmentation_result,
        )
        relation_graph = self._relation_extractor.extract(
            resolved_catalog,
            reference_image_result,
            segmentation_result,
            depth_result=depth_result,
        )
        materialized_relation_graph = materialize_relation_graph_for_instances(
            relation_graph,
            object_catalog=resolved_catalog,
            segmentation_result=segmentation_result,
        )
        return ObjectRelationExtractionOutputs(
            reference_image=reference_image_result,
            segmentation_result=segmentation_result,
            depth_result=depth_result,
            object_catalog=resolved_catalog,
            object_descriptions=object_descriptions,
            relation_graph=materialized_relation_graph,
            size_priors=(),
            size_inference_report=None,
        )

    def run(
        self,
        reference_image_result: ReferenceImageResult,
        segmentation_result: SegmentationResult,
        depth_result: DepthResult | None = None,
        object_catalog: ObjectCatalog | None = None,
    ) -> ObjectRelationExtractionOutputs:
        prepared = self.prepare_without_size_inference(
            reference_image_result,
            segmentation_result,
            depth_result=depth_result,
            object_catalog=object_catalog,
        )
        size_inference = self._normalize_size_inference_result(
            self._size_inferer.estimate(
                prepared.object_catalog,
                reference_image_result,
                segmentation=segmentation_result,
                depth_result=depth_result,
                object_descriptions=prepared.object_descriptions,
                relation_graph=prepared.relation_graph,
            )
        )
        size_priors = materialize_size_priors_for_instances(
            size_inference.size_priors,
            object_catalog=prepared.object_catalog,
        )

        return ObjectRelationExtractionOutputs(
            reference_image=prepared.reference_image,
            segmentation_result=prepared.segmentation_result,
            depth_result=prepared.depth_result,
            object_catalog=prepared.object_catalog,
            object_descriptions=prepared.object_descriptions,
            relation_graph=prepared.relation_graph,
            size_priors=size_priors,
            size_inference_report=size_inference.size_inference_report,
        )

    def _resolve_object_catalog(
        self,
        *,
        reference_image_result: ReferenceImageResult,
        segmentation_result: SegmentationResult,
        object_catalog: ObjectCatalog | None,
    ) -> ObjectCatalog:
        if object_catalog is not None:
            return object_catalog
        if self._object_catalog_builder is None:
            raise ValueError(
                "object_catalog must be provided when no object_catalog_builder is configured."
            )
        return self._object_catalog_builder(reference_image_result, segmentation_result)

    def _normalize_size_inference_result(
        self,
        value: SizeInferenceResult | Sequence[SizePrior],
    ) -> SizeInferenceResult:
        if isinstance(value, SizeInferenceResult):
            return value
        return SizeInferenceResult(size_priors=tuple(value), size_inference_report=None)
