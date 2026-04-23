from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol, Sequence

from pat3d.models import (
    ContainmentRelation,
    DepthResult,
    DetectedObject,
    GeneratedObjectAsset,
    ObjectAssetCatalog,
    ObjectCatalog,
    ReferenceImageResult,
    SceneLayout,
    SceneRelationGraph,
    SegmentationResult,
    SizePrior,
)
from pat3d.stages.object_assets import instance_object_ids


class SceneLayoutBuilder(Protocol):
    def build_layout(
        self,
        *,
        scene_id: str,
        object_assets: ObjectAssetCatalog,
        object_catalog: ObjectCatalog | None = None,
        reference_image_result: ReferenceImageResult | None = None,
        segmentation_result: SegmentationResult | None = None,
        depth_result: DepthResult | None = None,
        relation_graph: SceneRelationGraph | None = None,
        size_priors: Sequence[SizePrior] | None = None,
    ) -> SceneLayout:
        ...


@dataclass(slots=True)
class LayoutInitializationOutputs:
    scene_layout: SceneLayout
    object_assets: ObjectAssetCatalog
    object_catalog: ObjectCatalog | None = None
    relation_graph: SceneRelationGraph | None = None
    size_priors: Sequence[SizePrior] = ()
    reference_image_result: ReferenceImageResult | None = None
    segmentation_result: SegmentationResult | None = None
    depth_result: DepthResult | None = None


class LayoutInitializationStage:
    stage_name = "layout_initialization"

    def __init__(self, layout_builder: SceneLayoutBuilder) -> None:
        self._layout_builder = layout_builder

    def run(
        self,
        *,
        scene_id: str,
        object_assets: ObjectAssetCatalog,
        object_catalog: ObjectCatalog | None = None,
        reference_image_result: ReferenceImageResult | None = None,
        segmentation_result: SegmentationResult | None = None,
        depth_result: DepthResult | None = None,
        relation_graph: SceneRelationGraph | None = None,
        size_priors: Sequence[SizePrior] = (),
    ) -> LayoutInitializationOutputs:
        instance_ids_by_object_id = _expanded_instance_ids_by_object_id(object_catalog)
        expanded_assets = _expand_assets_for_layout(object_assets, object_catalog, instance_ids_by_object_id)
        expanded_size_priors = _expand_size_priors_for_layout(size_priors, object_catalog, instance_ids_by_object_id)
        expanded_relation_graph = _expand_relation_graph_for_layout(
            relation_graph,
            object_catalog,
            instance_ids_by_object_id,
        )
        scene_layout = self._layout_builder.build_layout(
            scene_id=scene_id,
            object_assets=expanded_assets,
            object_catalog=object_catalog,
            reference_image_result=reference_image_result,
            segmentation_result=segmentation_result,
            depth_result=depth_result,
            relation_graph=expanded_relation_graph,
            size_priors=expanded_size_priors,
        )
        return LayoutInitializationOutputs(
            scene_layout=scene_layout,
            object_assets=expanded_assets,
            object_catalog=object_catalog,
            relation_graph=expanded_relation_graph,
            size_priors=expanded_size_priors,
            reference_image_result=reference_image_result,
            segmentation_result=segmentation_result,
            depth_result=depth_result,
        )


def _expand_assets_for_layout(
    object_assets: ObjectAssetCatalog,
    object_catalog: ObjectCatalog | None,
    instance_ids_by_object_id: dict[str, tuple[str, ...]] | None = None,
) -> ObjectAssetCatalog:
    if object_catalog is None or all(detected_object.count <= 1 for detected_object in object_catalog.objects):
        return object_assets

    expanded_ids = instance_ids_by_object_id or _expanded_instance_ids_by_object_id(object_catalog)
    asset_by_id = {asset.object_id: asset for asset in object_assets.assets}
    expanded_assets: list[GeneratedObjectAsset] = []
    consumed_asset_ids: set[str] = set()
    used_ids: set[str] = set()

    for detected_object in object_catalog.objects:
        instance_ids = instance_object_ids(detected_object, used_ids)
        asset = asset_by_id.get(detected_object.object_id)
        if asset is not None:
            for instance_id in instance_ids:
                expanded_assets.append(
                    asset if instance_id == asset.object_id else replace(asset, object_id=instance_id)
                )
                used_ids.add(instance_id)
            consumed_asset_ids.add(detected_object.object_id)
            continue

        instance_assets = [asset_by_id.get(instance_id) for instance_id in instance_ids]
        if all(instance_asset is not None for instance_asset in instance_assets):
            for instance_id, instance_asset in zip(instance_ids, instance_assets):
                expanded_assets.append(instance_asset)
                used_ids.add(instance_id)
                consumed_asset_ids.add(instance_id)
            continue

        raise ValueError(
            f"missing GeneratedObjectAsset for repeated-object layout expansion: {detected_object.object_id}"
        )

    for asset in object_assets.assets:
        if asset.object_id not in consumed_asset_ids:
            expanded_assets.append(asset)

    return ObjectAssetCatalog(
        scene_id=object_assets.scene_id,
        assets=tuple(expanded_assets),
        metadata=object_assets.metadata,
    )


def _expand_size_priors_for_layout(
    size_priors: Sequence[SizePrior],
    object_catalog: ObjectCatalog | None,
    instance_ids_by_object_id: dict[str, tuple[str, ...]] | None = None,
) -> tuple[SizePrior, ...]:
    if object_catalog is None or all(detected_object.count <= 1 for detected_object in object_catalog.objects):
        return tuple(size_priors)

    expanded_ids = instance_ids_by_object_id or _expanded_instance_ids_by_object_id(object_catalog)
    prior_by_id = {prior.object_id: prior for prior in size_priors}
    expanded_priors: list[SizePrior] = []
    consumed_prior_ids: set[str] = set()
    used_ids: set[str] = set()

    for detected_object in object_catalog.objects:
        prior = prior_by_id.get(detected_object.object_id)
        if prior is None:
            continue
        for instance_id in instance_object_ids(detected_object, used_ids):
            expanded_priors.append(prior if instance_id == prior.object_id else replace(prior, object_id=instance_id))
            used_ids.add(instance_id)
        consumed_prior_ids.add(detected_object.object_id)

    for prior in size_priors:
        if prior.object_id not in consumed_prior_ids:
            expanded_priors.append(prior)

    return tuple(expanded_priors)

def _expand_relation_graph_for_layout(
    relation_graph: SceneRelationGraph | None,
    object_catalog: ObjectCatalog | None,
    instance_ids_by_object_id: dict[str, tuple[str, ...]] | None = None,
) -> SceneRelationGraph | None:
    if (
        relation_graph is None
        or object_catalog is None
        or all(detected_object.count <= 1 for detected_object in object_catalog.objects)
    ):
        return relation_graph

    expanded_ids = instance_ids_by_object_id or _expanded_instance_ids_by_object_id(object_catalog)

    def expand_ids(object_id: str) -> tuple[str, ...]:
        return expanded_ids.get(object_id, (object_id,))

    expanded_relations: list[ContainmentRelation] = []
    seen_relation_keys: set[tuple[str, str, str, float | None, str | None]] = set()
    for relation in relation_graph.relations:
        parent_ids = expand_ids(relation.parent_object_id)
        child_ids = expand_ids(relation.child_object_id)
        if len(parent_ids) > 1 and len(child_ids) > 1 and len(parent_ids) == len(child_ids):
            pairs = zip(parent_ids, child_ids)
        else:
            pairs = (
                (parent_id, child_id)
                for parent_id in parent_ids
                for child_id in child_ids
            )
        for parent_id, child_id in pairs:
            if parent_id == child_id:
                continue
            relation_key = (
                parent_id,
                child_id,
                str(relation.relation_type),
                relation.confidence,
                relation.evidence,
            )
            if relation_key in seen_relation_keys:
                continue
            seen_relation_keys.add(relation_key)
            expanded_relations.append(
                replace(
                    relation,
                    parent_object_id=parent_id,
                    child_object_id=child_id,
                )
            )

    expanded_roots = _dedupe_preserving_order(
        expanded_root_id
        for root_id in relation_graph.root_object_ids
        for expanded_root_id in expand_ids(root_id)
    )

    return SceneRelationGraph(
        scene_id=relation_graph.scene_id,
        relations=tuple(expanded_relations),
        root_object_ids=tuple(expanded_roots),
        metadata=relation_graph.metadata,
    )


def _expanded_instance_ids_by_object_id(
    object_catalog: ObjectCatalog | None,
) -> dict[str, tuple[str, ...]]:
    if object_catalog is None:
        return {}
    used_ids: set[str] = set()
    expanded_ids: dict[str, tuple[str, ...]] = {}
    for detected_object in object_catalog.objects:
        instance_ids = instance_object_ids(detected_object, used_ids)
        expanded_ids[detected_object.object_id] = instance_ids
        used_ids.update(instance_ids)
    return expanded_ids


def _dedupe_preserving_order(values: Sequence[str]) -> tuple[str, ...]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return tuple(ordered)
