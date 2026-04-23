from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
import re
from typing import Any

from pat3d.models import (
    ContainmentRelation,
    MaskInstance,
    ObjectCatalog,
    RelationType,
    SceneRelationGraph,
    SegmentationResult,
    SizePrior,
)


_CHILD_TO_PARENT_TYPES = {
    "on": RelationType.SUPPORTS,
    "supports": RelationType.SUPPORTS,
    "supported_by": RelationType.SUPPORTS,
    "in": RelationType.CONTAINS,
    "inside": RelationType.CONTAINS,
    "contains": RelationType.CONTAINS,
    "contained_by": RelationType.CONTAINS,
}
_PARENT_TO_CHILD_TYPES = {
    "supports": RelationType.SUPPORTS,
    "contains": RelationType.CONTAINS,
}
_LEGACY_DEFAULT_RELATION = RelationType.SUPPORTS


def match_object_id(object_catalog: ObjectCatalog | None, key: str) -> str | None:
    normalized = key.strip().lower()
    if not normalized:
        return None
    if object_catalog is None:
        return key.strip()

    candidates = _candidate_object_key_matches(normalized)
    expanded_ids = expanded_instance_ids_by_object_id(object_catalog)
    canonicalized_candidates = {_canonicalize_object_id(candidate) for candidate in candidates}
    canonicalized_candidates.discard("")
    candidate_set = set(candidates)

    for obj in object_catalog.objects:
        instance_ids_lower = {instance_id.lower() for instance_id in expanded_ids.get(obj.object_id, ())}
        if (
            bool(candidate_set.intersection({obj.object_id.lower(), obj.canonical_name.lower(), *instance_ids_lower}))
            or _canonicalize_object_id(obj.object_id) in canonicalized_candidates
            or _canonicalize_object_id(obj.canonical_name) in canonicalized_candidates
        ):
            return obj.object_id
    return None


def _candidate_object_key_matches(key: str) -> tuple[str, ...]:
    normalized = key.strip().lower()
    if not normalized:
        return ()

    candidates: set[str] = {normalized}
    if ":" in normalized:
        candidates.add(normalized.split(":")[-1])
    if "::" in normalized:
        candidates.add(normalized.rsplit("::", 1)[-1])

    if "/" in normalized:
        candidates.update(part for part in normalized.split("/") if part)
    if "\\" in normalized:
        candidates.update(part for part in normalized.split("\\") if part)

    trimmed = _strip_trailing_instance_suffix(normalized)
    if trimmed:
        candidates.add(trimmed)

    return tuple(candidates)


def _size_fallback_dimensions(canonical_name: str) -> dict[str, float] | None:
    """Return conservative fallback dimensions for known object names.

    This is intentionally small and local, and only used when the upstream
    estimator does not provide a size prior for a detected object.
    """

    def _tokens(name: str) -> tuple[str, ...]:
        raw_tokens = [token for token in re.split(r"[^a-z0-9]+", name.lower()) if token]
        normalized: list[str] = []
        seen: set[str] = set()
        for token in raw_tokens:
            normalized.append(token)
            if token.endswith("es") and len(token) > 3:
                normalized.append(token[:-2])
            elif token.endswith("s") and len(token) > 2:
                normalized.append(token[:-1])
            if token not in seen:
                seen.add(token)
        return tuple(normalized)

    normalized = _tokens(canonical_name)
    if not normalized:
        return None

    token_set = set(normalized)

    if "rubik_s_cube" in token_set or "rubik" in token_set or (
        "toy" in token_set and "cube" in token_set
    ):
        return {"x": 0.08, "y": 0.08, "z": 0.08}

    if "book" in token_set:
        return {"x": 0.3, "y": 0.3, "z": 0.06}

    if "table" in token_set:
        return {"x": 1.2, "y": 1.2, "z": 0.08}

    return {"x": 0.1, "y": 0.1, "z": 0.1}


def _strip_trailing_instance_suffix(value: str) -> str:
    trimmed = value.strip().lower()
    while True:
        match = re.match(r"^(.*?)(?:[:-_])(\d+)$", trimmed)
        if match is None:
            return trimmed
        trimmed = match.group(1).rstrip(":-_")
        if not trimmed:
            return value.strip().lower()


def _canonicalize_object_id(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.strip().lower())


def normalize_scene_relations(
    raw_payload: Any,
    *,
    object_catalog: ObjectCatalog | None = None,
    object_resolver=None,
    include_catalog_nodes: bool = True,
) -> tuple[tuple[ContainmentRelation, ...], tuple[str, ...]]:
    relations: list[ContainmentRelation] = []
    seen: set[tuple[str, str, RelationType]] = set()

    def add_relation(parent_name: str, child_name: str, relation_name: str | None, evidence: str | None = None) -> None:
        resolver = object_resolver or match_object_id
        parent_id = resolver(object_catalog, parent_name)
        child_id = resolver(object_catalog, child_name)
        if parent_id is None or child_id is None or parent_id == child_id:
            return
        relation_type = _normalize_relation_type(relation_name)
        key = (parent_id, child_id, relation_type)
        if key in seen:
            return
        seen.add(key)
        relations.append(
            ContainmentRelation(
                parent_object_id=parent_id,
                child_object_id=child_id,
                relation_type=relation_type,
                confidence=None,
                evidence=evidence,
            )
        )

    if isinstance(raw_payload, dict) and _looks_like_single_relation(raw_payload):
        _parse_relation_entry(raw_payload, add_relation)
    elif isinstance(raw_payload, dict) and isinstance(raw_payload.get("relations"), list):
        for relation_entry in raw_payload["relations"]:
            if isinstance(relation_entry, dict):
                _parse_relation_entry(relation_entry, add_relation)
    elif isinstance(raw_payload, list):
        for relation_entry in raw_payload:
            if isinstance(relation_entry, dict):
                _parse_relation_entry(relation_entry, add_relation)
    elif isinstance(raw_payload, dict):
        _parse_legacy_mapping(raw_payload, add_relation)

    filtered_relations, _ = _filter_relations_to_single_parent(relations)
    filtered_node_ids = set(
        {obj.object_id for obj in object_catalog.objects}
        if object_catalog is not None and include_catalog_nodes
        else set()
    )
    filtered_node_ids.update(
        object_id
        for relation in filtered_relations
        for object_id in (relation.parent_object_id, relation.child_object_id)
    )
    root_ids = _resolve_root_object_ids(filtered_node_ids, filtered_relations)
    return tuple(filtered_relations), root_ids


def expand_scene_relation_graph(
    relation_graph: SceneRelationGraph | None,
    *,
    object_catalog: ObjectCatalog | None,
) -> SceneRelationGraph | None:
    if (
        relation_graph is None
        or object_catalog is None
        or all(detected_object.count <= 1 for detected_object in object_catalog.objects)
    ):
        return relation_graph

    expanded_ids = expanded_instance_ids_by_object_id(object_catalog)

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

    expanded_roots = dedupe_preserving_order(
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


def materialize_relation_graph_for_instances(
    relation_graph: SceneRelationGraph | None,
    *,
    object_catalog: ObjectCatalog | None,
    segmentation_result: SegmentationResult | None,
) -> SceneRelationGraph | None:
    expanded_graph = expand_scene_relation_graph(
        relation_graph,
        object_catalog=object_catalog,
    )
    if relation_graph is None or object_catalog is None:
        return _graph_with_single_parent_constraint(expanded_graph)
    if all(detected_object.count <= 1 for detected_object in object_catalog.objects):
        return _graph_with_single_parent_constraint(relation_graph)
    if segmentation_result is None:
        return _graph_with_single_parent_constraint(expanded_graph)

    object_lookup = {
        detected_object.object_id: detected_object
        for detected_object in object_catalog.objects
    }
    instance_lookup = {
        instance.instance_id: instance
        for instance in segmentation_result.instances
    }
    expanded_ids = expanded_instance_ids_by_object_id(object_catalog)
    ordered_instance_ids_by_object_id = {
        detected_object.object_id: tuple(
            instance.instance_id
            for instance in _ordered_instances_for_support_inference(
                expanded_ids.get(detected_object.object_id, ()),
                instance_lookup,
            )
        )
        for detected_object in object_catalog.objects
        if detected_object.count > 1
    }
    existing_relations: list[ContainmentRelation] = []
    seen_relation_keys = {
        (relation.parent_object_id, relation.child_object_id, relation.relation_type): relation
        for relation in existing_relations
    }
    for relation in relation_graph.relations:
        for parent_id, child_id in _expand_relation_instance_pairs(
            relation,
            expanded_ids=expanded_ids,
            object_lookup=object_lookup,
            instance_lookup=instance_lookup,
            ordered_instance_ids_by_object_id=ordered_instance_ids_by_object_id,
        ):
            relation_key = (parent_id, child_id, relation.relation_type)
            if relation_key in seen_relation_keys or parent_id == child_id:
                continue
            expanded_relation = replace(
                relation,
                parent_object_id=parent_id,
                child_object_id=child_id,
            )
            seen_relation_keys[relation_key] = expanded_relation
            existing_relations.append(expanded_relation)

    inferred_any = False
    for detected_object in object_catalog.objects:
        instance_ids = expanded_ids.get(detected_object.object_id, ())
        if len(instance_ids) <= 1:
            continue
        if any(
            relation.parent_object_id in instance_ids and relation.child_object_id in instance_ids
            for relation in existing_relations
        ):
            continue

        ordered_instances = [
            instance_lookup[instance_id]
            for instance_id in ordered_instance_ids_by_object_id.get(detected_object.object_id, ())
            if instance_id in instance_lookup
        ]
        if len(ordered_instances) != len(instance_ids):
            continue

        inferred_relations: list[ContainmentRelation] = []
        valid_stack = True
        for lower_instance, upper_instance in zip(ordered_instances, ordered_instances[1:]):
            overlap_ratio = _horizontal_overlap_ratio(
                lower_instance.bbox_xyxy,
                upper_instance.bbox_xyxy,
            )
            if overlap_ratio < 0.25:
                valid_stack = False
                break
            relation_key = (
                lower_instance.instance_id,
                upper_instance.instance_id,
                RelationType.SUPPORTS,
            )
            if relation_key in seen_relation_keys:
                continue
            inferred_relations.append(
                ContainmentRelation(
                    parent_object_id=lower_instance.instance_id,
                    child_object_id=upper_instance.instance_id,
                    relation_type=RelationType.SUPPORTS,
                    confidence=None,
                    evidence=f"auto_inferred_repeated_instance_stack:x_overlap={overlap_ratio:.3f}",
                )
            )
        if not valid_stack or not inferred_relations:
            continue

        inferred_any = True
        for relation in inferred_relations:
            seen_relation_keys[
                (relation.parent_object_id, relation.child_object_id, relation.relation_type)
            ] = relation
            existing_relations.append(relation)

    root_candidates = list(
        dedupe_preserving_order(
            expanded_root_id
            for root_id in relation_graph.root_object_ids
            for expanded_root_id in expanded_ids.get(root_id, (root_id,))
        )
    )
    if not root_candidates:
        root_candidates = list(
            dedupe_preserving_order(
                expanded_id
                for detected_object in object_catalog.objects
                for expanded_id in expanded_ids.get(detected_object.object_id, (detected_object.object_id,))
            )
        )

    filtered_relations, dropped_relations = _filter_relations_to_single_parent(existing_relations)
    root_node_ids = dedupe_preserving_order(
        (
            *root_candidates,
            *(
                object_id
                for relation in filtered_relations
                for object_id in (relation.parent_object_id, relation.child_object_id)
            ),
        )
    )
    resolved_roots = _resolve_root_object_ids(root_node_ids, filtered_relations)
    metadata = relation_graph.metadata
    metadata_notes = list(metadata.notes) if metadata is not None else []
    if inferred_any:
        metadata_notes.append("auto_inferred_repeated_instance_support")
    if dropped_relations:
        metadata_notes.append(f"single_parent_relation_filter:dropped={len(dropped_relations)}")
    if metadata is not None and metadata_notes:
        metadata = replace(
            metadata,
            notes=tuple(dedupe_preserving_order(metadata_notes)),
        )
    return SceneRelationGraph(
        scene_id=expanded_graph.scene_id,
        relations=tuple(filtered_relations),
        root_object_ids=resolved_roots,
        metadata=metadata,
    )


def _graph_with_single_parent_constraint(
    relation_graph: SceneRelationGraph | None,
) -> SceneRelationGraph | None:
    if relation_graph is None:
        return None
    filtered_relations, dropped_relations = _filter_relations_to_single_parent(relation_graph.relations)
    if not dropped_relations:
        return relation_graph
    node_ids = dedupe_preserving_order(
        (
            *relation_graph.root_object_ids,
            *(
                object_id
                for relation in filtered_relations
                for object_id in (relation.parent_object_id, relation.child_object_id)
            ),
        )
    )
    metadata = relation_graph.metadata
    if metadata is not None:
        metadata = replace(
            metadata,
            notes=tuple(
                dedupe_preserving_order(
                    (*metadata.notes, f"single_parent_relation_filter:dropped={len(dropped_relations)}")
                )
            ),
        )
    return SceneRelationGraph(
        scene_id=relation_graph.scene_id,
        relations=tuple(filtered_relations),
        root_object_ids=_resolve_root_object_ids(node_ids, filtered_relations),
        metadata=metadata,
    )


def _filter_relations_to_single_parent(
    relations: Iterable[ContainmentRelation],
) -> tuple[tuple[ContainmentRelation, ...], tuple[ContainmentRelation, ...]]:
    kept_relations: list[ContainmentRelation] = []
    dropped_relations: list[ContainmentRelation] = []
    seen_child_ids: set[str] = set()
    for relation in relations:
        if relation.child_object_id in seen_child_ids:
            dropped_relations.append(relation)
            continue
        seen_child_ids.add(relation.child_object_id)
        kept_relations.append(relation)
    return tuple(kept_relations), tuple(dropped_relations)


def _resolve_root_object_ids(
    node_ids: Iterable[str],
    relations: Iterable[ContainmentRelation],
) -> tuple[str, ...]:
    child_ids = {relation.child_object_id for relation in relations}
    return tuple(
        object_id for object_id in dedupe_preserving_order(node_ids)
        if object_id not in child_ids
    )


def _expand_relation_instance_pairs(
    relation: ContainmentRelation,
    *,
    expanded_ids: dict[str, tuple[str, ...]],
    object_lookup: dict[str, Any],
    instance_lookup: dict[str, MaskInstance],
    ordered_instance_ids_by_object_id: dict[str, tuple[str, ...]],
) -> tuple[tuple[str, str], ...]:
    parent_ids = expanded_ids.get(relation.parent_object_id, (relation.parent_object_id,))
    child_ids = expanded_ids.get(relation.child_object_id, (relation.child_object_id,))
    if relation.relation_type is RelationType.SUPPORTS:
        return (_select_support_relation_pair(
            relation.parent_object_id,
            relation.child_object_id,
            parent_ids=parent_ids,
            child_ids=child_ids,
            object_lookup=object_lookup,
            instance_lookup=instance_lookup,
            ordered_instance_ids_by_object_id=ordered_instance_ids_by_object_id,
        ),)
    if len(parent_ids) > 1 and len(child_ids) > 1 and len(parent_ids) == len(child_ids):
        return tuple(zip(parent_ids, child_ids, strict=False))
    return tuple(
        (parent_id, child_id)
        for parent_id in parent_ids
        for child_id in child_ids
    )


def _select_support_relation_pair(
    parent_object_id: str,
    child_object_id: str,
    *,
    parent_ids: tuple[str, ...],
    child_ids: tuple[str, ...],
    object_lookup: dict[str, Any],
    instance_lookup: dict[str, MaskInstance],
    ordered_instance_ids_by_object_id: dict[str, tuple[str, ...]],
) -> tuple[str, str]:
    parent_preferred_id = _preferred_support_instance_id(
        parent_object_id,
        candidate_ids=parent_ids,
        relation_role="parent",
        ordered_instance_ids_by_object_id=ordered_instance_ids_by_object_id,
    )
    child_preferred_id = _preferred_support_instance_id(
        child_object_id,
        candidate_ids=child_ids,
        relation_role="child",
        ordered_instance_ids_by_object_id=ordered_instance_ids_by_object_id,
    )
    if len(parent_ids) == 1 and len(child_ids) == 1:
        return parent_ids[0], child_ids[0]

    if len(parent_ids) > 1 and len(child_ids) > 1:
        pair_candidates = [
            (parent_id, child_id)
            for parent_id in parent_ids
            for child_id in child_ids
            if parent_id != child_id
        ]
        if not pair_candidates:
            return parent_ids[0], child_ids[0]
        preferred_pair = (parent_preferred_id, child_preferred_id)
        return max(
            pair_candidates,
            key=lambda pair: _support_pair_score(
                pair[0],
                pair[1],
                instance_lookup=instance_lookup,
                preferred_pair=preferred_pair,
            ),
        )

    if len(parent_ids) > 1:
        child_bbox = _relation_node_bbox(
            child_object_id,
            candidate_ids=child_ids,
            object_lookup=object_lookup,
            instance_lookup=instance_lookup,
        )
        parent_id = _select_support_candidate_id(
            parent_ids,
            relation_role="parent",
            counterpart_bbox=child_bbox,
            preferred_id=parent_preferred_id,
            instance_lookup=instance_lookup,
        )
        return parent_id, child_ids[0]

    parent_bbox = _relation_node_bbox(
        parent_object_id,
        candidate_ids=parent_ids,
        object_lookup=object_lookup,
        instance_lookup=instance_lookup,
    )
    child_id = _select_support_candidate_id(
        child_ids,
        relation_role="child",
        counterpart_bbox=parent_bbox,
        preferred_id=child_preferred_id,
        instance_lookup=instance_lookup,
    )
    return parent_ids[0], child_id


def _preferred_support_instance_id(
    object_id: str,
    *,
    candidate_ids: tuple[str, ...],
    relation_role: str,
    ordered_instance_ids_by_object_id: dict[str, tuple[str, ...]],
) -> str:
    ordered_ids = ordered_instance_ids_by_object_id.get(object_id, ())
    if ordered_ids:
        return ordered_ids[-1] if relation_role == "parent" else ordered_ids[0]
    return candidate_ids[-1] if relation_role == "parent" else candidate_ids[0]


def _select_support_candidate_id(
    candidate_ids: tuple[str, ...],
    *,
    relation_role: str,
    counterpart_bbox: tuple[float, float, float, float] | None,
    preferred_id: str,
    instance_lookup: dict[str, MaskInstance],
) -> str:
    if len(candidate_ids) == 1:
        return candidate_ids[0]
    if counterpart_bbox is None:
        return preferred_id

    return max(
        candidate_ids,
        key=lambda candidate_id: _support_candidate_score(
            candidate_id,
            relation_role=relation_role,
            counterpart_bbox=counterpart_bbox,
            preferred_id=preferred_id,
            instance_lookup=instance_lookup,
        ),
    )


def _support_pair_score(
    parent_id: str,
    child_id: str,
    *,
    instance_lookup: dict[str, MaskInstance],
    preferred_pair: tuple[str, str],
) -> tuple[int, float, int, float]:
    parent_bbox = _bbox_for_instance_id(parent_id, instance_lookup)
    child_bbox = _bbox_for_instance_id(child_id, instance_lookup)
    if parent_bbox is None or child_bbox is None:
        return (0, 0.0, int((parent_id, child_id) == preferred_pair), 0.0)
    return (
        int(_bbox_center_y(parent_bbox) >= _bbox_center_y(child_bbox)),
        _horizontal_overlap_ratio(parent_bbox, child_bbox),
        int((parent_id, child_id) == preferred_pair),
        -_support_contact_gap(parent_bbox, child_bbox),
    )


def _support_candidate_score(
    candidate_id: str,
    *,
    relation_role: str,
    counterpart_bbox: tuple[float, float, float, float],
    preferred_id: str,
    instance_lookup: dict[str, MaskInstance],
) -> tuple[int, float, int, float]:
    candidate_bbox = _bbox_for_instance_id(candidate_id, instance_lookup)
    if candidate_bbox is None:
        return (0, 0.0, int(candidate_id == preferred_id), 0.0)

    if relation_role == "parent":
        orientation_ok = int(_bbox_center_y(candidate_bbox) >= _bbox_center_y(counterpart_bbox))
        overlap = _horizontal_overlap_ratio(candidate_bbox, counterpart_bbox)
        gap = _support_contact_gap(candidate_bbox, counterpart_bbox)
    else:
        orientation_ok = int(_bbox_center_y(candidate_bbox) <= _bbox_center_y(counterpart_bbox))
        overlap = _horizontal_overlap_ratio(counterpart_bbox, candidate_bbox)
        gap = _support_contact_gap(counterpart_bbox, candidate_bbox)

    return (
        orientation_ok,
        overlap,
        int(candidate_id == preferred_id),
        -gap,
    )


def _support_contact_gap(
    parent_bbox: tuple[float, float, float, float],
    child_bbox: tuple[float, float, float, float],
) -> float:
    _, parent_y0, _, _ = parent_bbox
    _, _, _, child_y1 = child_bbox
    return abs(float(parent_y0) - float(child_y1))


def _relation_node_bbox(
    object_id: str,
    *,
    candidate_ids: tuple[str, ...],
    object_lookup: dict[str, Any],
    instance_lookup: dict[str, MaskInstance],
) -> tuple[float, float, float, float] | None:
    for candidate_id in candidate_ids:
        bbox = _bbox_for_instance_id(candidate_id, instance_lookup)
        if bbox is not None:
            return bbox
    detected_object = object_lookup.get(object_id)
    if detected_object is None:
        return None
    for source_instance_id in detected_object.source_instance_ids:
        bbox = _bbox_for_instance_id(source_instance_id, instance_lookup)
        if bbox is not None:
            return bbox
    return None


def _bbox_for_instance_id(
    instance_id: str,
    instance_lookup: dict[str, MaskInstance],
) -> tuple[float, float, float, float] | None:
    instance = instance_lookup.get(instance_id)
    return None if instance is None else instance.bbox_xyxy


def materialize_size_priors_for_instances(
    size_priors: tuple[SizePrior, ...] | list[SizePrior],
    *,
    object_catalog: ObjectCatalog | None,
) -> tuple[SizePrior, ...]:
    if object_catalog is None or all(detected_object.count <= 1 for detected_object in object_catalog.objects):
        return tuple(size_priors)

    expanded_ids = expanded_instance_ids_by_object_id(object_catalog)
    prior_by_id = {}
    for prior in size_priors:
        resolved_object_id = _resolve_object_id_for_size_prior(
            prior.object_id,
            object_catalog,
            expanded_ids,
        )
        prior_by_id[resolved_object_id] = prior
    materialized_priors: list[SizePrior] = []
    consumed_prior_ids: set[str] = set()
    fallback_source_metadata = size_priors[0].metadata if size_priors else None

    for detected_object in object_catalog.objects:
        prior = prior_by_id.get(detected_object.object_id)
        instance_ids = expanded_ids.get(detected_object.object_id, (detected_object.object_id,))

        if prior is None:
            dimensions = _size_fallback_dimensions(detected_object.canonical_name)
            if dimensions is not None:
                for instance_id in instance_ids:
                    materialized_priors.append(
                        SizePrior(
                            object_id=instance_id,
                            dimensions_m=dimensions,
                            relative_scale_to_scene=None,
                            source="materialized_size_fallback",
                            metadata=fallback_source_metadata,
                        )
                    )
            consumed_prior_ids.add(detected_object.object_id)
            continue
        for instance_id in instance_ids:
            materialized_priors.append(
                prior if instance_id == prior.object_id else replace(prior, object_id=instance_id)
            )
        consumed_prior_ids.add(detected_object.object_id)

    for prior in size_priors:
        if _resolve_object_id_for_size_prior(prior.object_id, object_catalog, expanded_ids) not in consumed_prior_ids:
            materialized_priors.append(prior)

    return tuple(materialized_priors)


def _resolve_object_id_for_size_prior(
    prior_object_id: str,
    object_catalog: ObjectCatalog,
    expanded_ids: dict[str, tuple[str, ...]],
) -> str:
    candidates = _candidate_object_key_matches(prior_object_id)
    if not candidates:
        return prior_object_id

    candidate_set = set(candidates)
    canonicalized_candidates = {_canonicalize_object_id(candidate) for candidate in candidates}
    canonicalized_candidates.discard("")

    for detected_object in object_catalog.objects:
        object_id = detected_object.object_id
        object_id_lower = object_id.lower()
        canonical_name_lower = detected_object.canonical_name.lower()
        instance_ids = expanded_ids.get(object_id, ())
        instance_ids_lower = {instance_id.lower() for instance_id in instance_ids}

        if (
            object_id_lower in candidate_set
            or canonical_name_lower in candidate_set
            or bool(candidate_set.intersection(instance_ids_lower))
            or _canonicalize_object_id(object_id) in canonicalized_candidates
            or _canonicalize_object_id(canonical_name_lower) in canonicalized_candidates
            or bool(
                {
                    _canonicalize_object_id(instance_id)
                    for instance_id in instance_ids
                }.intersection(canonicalized_candidates)
            )
        ):
            return object_id

    return prior_object_id


def expanded_instance_ids_by_object_id(
    object_catalog: ObjectCatalog | None,
) -> dict[str, tuple[str, ...]]:
    if object_catalog is None:
        return {}
    used_ids: set[str] = set()
    expanded_ids: dict[str, tuple[str, ...]] = {}
    for detected_object in object_catalog.objects:
        instance_ids = _instance_object_ids(detected_object, used_ids)
        expanded_ids[detected_object.object_id] = instance_ids
        used_ids.update(instance_ids)
    return expanded_ids


def dedupe_preserving_order(values: Iterable[str]) -> tuple[str, ...]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = value.strip() if isinstance(value, str) else str(value).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return tuple(ordered)


def _parse_relation_entry(
    relation_entry: dict[str, Any],
    add_relation,
) -> None:
    child_name = _first_string(
        relation_entry,
        "child",
        "child_object",
        "child_object_id",
        "object",
        "object_id",
    )
    parent_name = _first_string(
        relation_entry,
        "parent",
        "parent_object",
        "parent_object_id",
        "supporter",
        "container",
        "support",
    )
    relation_name = _first_string(
        relation_entry,
        "relation",
        "relation_type",
        "kind",
    )
    if child_name is None or parent_name is None:
        return
    add_relation(parent_name, child_name, relation_name)


def _parse_legacy_mapping(raw_payload: dict[str, Any], add_relation) -> None:
    for object_name, value in raw_payload.items():
        if object_name == "relations":
            continue
        if isinstance(value, str):
            value = [value]
        if isinstance(value, Iterable) and not isinstance(value, dict):
            for parent_name in value:
                if isinstance(parent_name, str):
                    # Legacy prompt examples encode child -> [parent].
                    add_relation(parent_name, object_name, _LEGACY_DEFAULT_RELATION.value, "legacy_untyped_parent_list")
            continue
        if not isinstance(value, dict):
            continue
        for relation_name, related_names in value.items():
            if isinstance(related_names, str):
                related_names = [related_names]
            if not isinstance(related_names, Iterable) or isinstance(related_names, dict):
                continue
            normalized_name = relation_name.strip().lower()
            if normalized_name in _CHILD_TO_PARENT_TYPES:
                for parent_name in related_names:
                    if isinstance(parent_name, str):
                        add_relation(parent_name, object_name, normalized_name)
            elif normalized_name in _PARENT_TO_CHILD_TYPES:
                for child_name in related_names:
                    if isinstance(child_name, str):
                        add_relation(object_name, child_name, normalized_name)


def _first_string(mapping: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _looks_like_single_relation(raw_payload: dict[str, Any]) -> bool:
    keys = {key.lower() for key in raw_payload}
    return (
        {"child", "parent"} <= keys
        or {"child_object_id", "parent_object_id"} <= keys
        or {"child_object", "parent_object"} <= keys
    )


def _normalize_relation_type(value: str | None) -> RelationType:
    if value is None:
        return _LEGACY_DEFAULT_RELATION
    normalized = value.strip().lower()
    if normalized in _CHILD_TO_PARENT_TYPES:
        return _CHILD_TO_PARENT_TYPES[normalized]
    if normalized in _PARENT_TO_CHILD_TYPES:
        return _PARENT_TO_CHILD_TYPES[normalized]
    return _LEGACY_DEFAULT_RELATION


def _instance_object_ids(detected_object, used_ids: set[str] | None = None) -> tuple[str, ...]:
    if detected_object.count <= 1:
        return (detected_object.object_id,)

    reserved_ids = used_ids if used_ids is not None else set()
    source_ids = tuple(
        source_id
        for source_id in detected_object.source_instance_ids
        if source_id.strip()
    )
    resolved_ids: list[str] = []
    for index in range(detected_object.count):
        candidate = (
            source_ids[index]
            if index < len(source_ids)
            else f"{detected_object.object_id}::{index + 1}"
        )
        if candidate in reserved_ids or candidate in resolved_ids:
            candidate = f"{detected_object.object_id}::{index + 1}"
        suffix = 1
        unique_candidate = candidate
        while unique_candidate in reserved_ids or unique_candidate in resolved_ids:
            suffix += 1
            unique_candidate = f"{candidate}:{suffix}"
        resolved_ids.append(unique_candidate)
    return tuple(resolved_ids)


def _ordered_instances_for_support_inference(
    instance_ids: tuple[str, ...],
    instance_lookup: dict[str, MaskInstance],
) -> list[MaskInstance]:
    instances = [
        instance_lookup[instance_id]
        for instance_id in instance_ids
        if instance_id in instance_lookup
    ]
    return sorted(
        instances,
        key=lambda instance: (
            _bbox_center_y(instance.bbox_xyxy),
            -_bbox_center_x(instance.bbox_xyxy),
        ),
        reverse=True,
    )


def _bbox_center_x(bbox_xyxy: tuple[float, float, float, float]) -> float:
    x0, _, x1, _ = bbox_xyxy
    return (float(x0) + float(x1)) / 2.0


def _bbox_center_y(bbox_xyxy: tuple[float, float, float, float]) -> float:
    _, y0, _, y1 = bbox_xyxy
    return (float(y0) + float(y1)) / 2.0


def _horizontal_overlap_ratio(
    left_bbox: tuple[float, float, float, float],
    right_bbox: tuple[float, float, float, float],
) -> float:
    left_x0, _, left_x1, _ = left_bbox
    right_x0, _, right_x1, _ = right_bbox
    overlap = max(0.0, min(float(left_x1), float(right_x1)) - max(float(left_x0), float(right_x0)))
    left_width = max(1e-6, float(left_x1) - float(left_x0))
    right_width = max(1e-6, float(right_x1) - float(right_x0))
    return overlap / min(left_width, right_width)
