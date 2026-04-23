from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path
from typing import Sequence

from pat3d.contracts import ObjectTextDescriber, RelationExtractor, SizePriorEstimator, StructuredLLM
from pat3d.models import (
    ArtifactRef,
    ObjectCatalog,
    ObjectDescription,
    ReferenceImageResult,
    SceneRelationGraph,
    SegmentationResult,
    SizeInferenceReport,
    SizeInferenceResult,
    SizePrior,
    StructuredPromptRequest,
)
from pat3d.providers._relation_utils import (
    expanded_instance_ids_by_object_id,
    match_object_id,
    normalize_scene_relations,
)
from pat3d.runtime.errors import Pat3DRuntimeError, runtime_failure
from pat3d.storage import make_stage_metadata


_FURNITURE_LONGEST_SIDE_M = {
    "table": 1.2,
    "desk": 1.4,
    "chair": 0.9,
    "sofa": 2.0,
    "bed": 2.0,
    "cabinet": 1.2,
    "shelf": 1.2,
    "bookshelf": 1.8,
    "nightstand": 0.6,
    "counter": 1.5,
}

_SMALL_OBJECT_LONGEST_SIDE_M = {
    "apple": 0.09,
    "orange": 0.09,
    "banana": 0.22,
    "pear": 0.11,
    "peach": 0.09,
    "cup": 0.1,
    "mug": 0.11,
    "bottle": 0.3,
    "plate": 0.3,
    "book": 0.3,
    "phone": 0.16,
    "remote": 0.2,
    "mouse": 0.11,
}

_TOY_CUBE_LONGEST_SIDE_M = {
    "rubik_s_cube": 0.08,
    "toy_cube": 0.08,
}

_EXPLICIT_FALLBACK_DIMENSIONS_M = {
    "apple": {"x": 0.08, "y": 0.08, "z": 0.08},
    "orange": {"x": 0.08, "y": 0.08, "z": 0.08},
    "banana": {"x": 0.22, "y": 0.04, "z": 0.04},
    "pear": {"x": 0.08, "y": 0.08, "z": 0.11},
    "peach": {"x": 0.08, "y": 0.08, "z": 0.08},
    "cup": {"x": 0.09, "y": 0.09, "z": 0.1},
    "mug": {"x": 0.09, "y": 0.09, "z": 0.11},
    "bottle": {"x": 0.08, "y": 0.08, "z": 0.3},
    "plate": {"x": 0.26, "y": 0.26, "z": 0.03},
    "book": {"x": 0.24, "y": 0.17, "z": 0.04},
    "phone": {"x": 0.075, "y": 0.155, "z": 0.009},
    "remote": {"x": 0.045, "y": 0.18, "z": 0.02},
    "mouse": {"x": 0.065, "y": 0.11, "z": 0.04},
    "basket": {"x": 0.25, "y": 0.25, "z": 0.085},
    "bowl": {"x": 0.16, "y": 0.16, "z": 0.07},
    "toothbrush": {"x": 0.19, "y": 0.02, "z": 0.03},
    "pencil": {"x": 0.19, "y": 0.008, "z": 0.008},
    "ruler": {"x": 0.3, "y": 0.03, "z": 0.003},
    "scissors": {"x": 0.18, "y": 0.08, "z": 0.01},
    "radio": {"x": 0.22, "y": 0.12, "z": 0.08},
}


def _load_prompt(prompt_path: str) -> str:
    return Path(prompt_path).read_text().strip()


def _flatten_single_value(payload: dict) -> str:
    value = payload
    while isinstance(value, dict) and len(value) == 1:
        value = next(iter(value.values()))
    if isinstance(value, str):
        return value
    return str(value)


def _is_structured_parse_failure(error: Exception) -> bool:
    return isinstance(error, Pat3DRuntimeError) and error.payload.code == "structured_response_parse_failed"


def _normalized_name_tokens(name: str) -> tuple[str, ...]:
    raw_tokens = [token for token in re.split(r"[^a-z0-9]+", str(name).lower()) if token]
    normalized: list[str] = []
    for token in raw_tokens:
        normalized.append(token)
        if token.endswith("es") and len(token) > 3:
            normalized.append(token[:-2])
        elif token.endswith("s") and len(token) > 2:
            normalized.append(token[:-1])
    deduped: list[str] = []
    seen: set[str] = set()
    for token in normalized:
        if token in seen:
            continue
        seen.add(token)
        deduped.append(token)
    return tuple(deduped)


def _match_size_category(name: str, candidates: dict[str, float]) -> tuple[str, float] | None:
    tokens = _normalized_name_tokens(name)
    normalized_name = "_".join(tokens)
    for candidate, value in candidates.items():
        if candidate == normalized_name or candidate in tokens:
            return candidate, value
    return None


def _match_dimension_category(name: str, candidates: dict[str, dict[str, float]]) -> tuple[str, dict[str, float]] | None:
    tokens = _normalized_name_tokens(name)
    normalized_name = "_".join(tokens)
    for candidate, value in candidates.items():
        if candidate == normalized_name or candidate in tokens:
            return candidate, value
    return None


def _build_size_reasonableness_guidance(object_catalog: ObjectCatalog) -> str:
    guidance_lines = [
        "Estimate all object sizes jointly so their relative scale is physically plausible across the whole scene.",
    ]

    matched_furniture = []
    matched_small_objects = []
    matched_toy_cubes = []
    for detected_object in object_catalog.objects:
        furniture_match = _match_size_category(detected_object.canonical_name, _FURNITURE_LONGEST_SIDE_M)
        if furniture_match is not None:
            matched_furniture.append(furniture_match[0])
        small_match = _match_size_category(detected_object.canonical_name, _SMALL_OBJECT_LONGEST_SIDE_M)
        if small_match is not None:
            matched_small_objects.append(small_match[0])
        toy_cube_match = _match_size_category(detected_object.canonical_name, _TOY_CUBE_LONGEST_SIDE_M)
        if toy_cube_match is not None:
            matched_toy_cubes.append(toy_cube_match[0])

    if matched_furniture:
        unique_furniture = ", ".join(dict.fromkeys(matched_furniture))
        guidance_lines.append(
            f"Treat {unique_furniture} as furniture-scale objects, typically around 0.7 m to 2.0 m on the longest side."
        )
    if matched_small_objects:
        unique_small_objects = ", ".join(dict.fromkeys(matched_small_objects))
        guidance_lines.append(
            f"Treat {unique_small_objects} as small hand-sized objects, typically around 0.05 m to 0.35 m on the longest side."
        )
    if matched_toy_cubes:
        unique_toy_cubes = ", ".join(dict.fromkeys(matched_toy_cubes))
        guidance_lines.append(
            f"Treat {unique_toy_cubes} as toy objects, typically around 0.08 m on the longest side and rarely smaller than that."
        )
    if matched_furniture and matched_small_objects:
        guidance_lines.append(
            "Small handheld objects on furniture must stay much smaller than the supporting furniture; their longest side should usually be well under one third of the furniture's longest side."
        )

    guidance_lines.append(
        "Before returning the final JSON, compare every object's size against the others and correct any obviously unrealistic ratios."
    )
    return "Relative Size Sanity Check:\n" + "\n".join(f"- {line}" for line in guidance_lines)


def _build_size_prompt(
    object_catalog: ObjectCatalog,
    template: str,
    *,
    object_descriptions: Sequence[ObjectDescription] = (),
    relation_graph: SceneRelationGraph | None = None,
) -> str:
    guidance = _build_size_reasonableness_guidance(object_catalog)
    inventory_lines = ["Object Inventory:"]
    for detected_object in object_catalog.objects:
        display_name = detected_object.display_name or detected_object.canonical_name.replace("_", " ")
        inventory_lines.append(
            "- "
            f"{detected_object.object_id}: canonical_name={detected_object.canonical_name}; "
            f"display_name={display_name}; count={detected_object.count}"
        )

    description_map = {description.object_id: description.prompt_text for description in object_descriptions}
    description_lines = ["Object Descriptions:"]
    for detected_object in object_catalog.objects:
        description_text = description_map.get(detected_object.object_id)
        if description_text is None:
            description_text = f"The object is {detected_object.canonical_name}."
        description_lines.append(f"- {detected_object.object_id}: {description_text}")

    relation_lines = ["Relation Summary:"]
    if relation_graph is None or not relation_graph.relations:
        relation_lines.append("- No explicit relations were provided.")
    else:
        for relation in relation_graph.relations:
            relation_lines.append(
                f"- {relation.parent_object_id} {relation.relation_type.value} {relation.child_object_id}"
            )
        if relation_graph.root_object_ids:
            relation_lines.append(
                "- Root objects: " + ", ".join(relation_graph.root_object_ids)
            )

    return "\n\n".join(("\n".join(inventory_lines), "\n".join(description_lines), "\n".join(relation_lines), guidance))


def _extract_dimensions_from_size_value(value: object) -> tuple[dict[str, float] | None, float | None]:
    dimensions = None
    relative_scale = None

    if isinstance(value, dict):
        dimensions_source = value.get("dimensions_m") if isinstance(value.get("dimensions_m"), dict) else value
        if isinstance(dimensions_source, dict):
            numeric_dims = {
                axis: float(amount)
                for axis, amount in dimensions_source.items()
                if axis in {"x", "y", "z"} and isinstance(amount, (int, float)) and float(amount) > 0.0
            }
            dimensions = numeric_dims or None
        scale_value = value.get("relative_scale_to_scene")
        if isinstance(scale_value, (int, float)) and float(scale_value) > 0.0:
            relative_scale = float(scale_value)
    elif isinstance(value, (list, tuple)) and len(value) >= 3:
        dims = value[:3]
        if all(isinstance(item, (int, float)) for item in dims):
            numeric_dims = {
                axis: float(amount)
                for axis, amount in zip(("x", "y", "z"), dims)
                if float(amount) > 0.0
            }
            dimensions = numeric_dims or None

    return dimensions, relative_scale


def _build_size_inference_report(
    parsed_output: dict[str, object],
    object_catalog: ObjectCatalog,
    metadata,
) -> SizeInferenceReport | None:
    raw_objects = parsed_output.get("objects")
    if not isinstance(raw_objects, dict):
        return None

    normalized_objects: dict[str, dict[str, object]] = {}
    for key, value in raw_objects.items():
        if not isinstance(key, str):
            continue
        resolved_key = match_object_id(object_catalog, key) or key.strip()
        dimensions, relative_scale = _extract_dimensions_from_size_value(value)
        entry: dict[str, object] = {}
        if dimensions is not None:
            entry["dimensions_m"] = dimensions
        if relative_scale is not None:
            entry["relative_scale_to_scene"] = relative_scale
        if entry:
            normalized_objects[resolved_key] = entry

    anchor_object_ids: list[str] = []
    for raw_anchor in parsed_output.get("anchor_object_ids", ()) if isinstance(parsed_output.get("anchor_object_ids"), (list, tuple)) else ():
        if not isinstance(raw_anchor, str):
            continue
        resolved_anchor = match_object_id(object_catalog, raw_anchor) or raw_anchor.strip()
        if resolved_anchor:
            anchor_object_ids.append(resolved_anchor)

    scene_description = parsed_output.get("scene_description")
    scene_scale_summary = parsed_output.get("scene_scale_summary")
    return SizeInferenceReport(
        scene_description=scene_description.strip() if isinstance(scene_description, str) else "",
        anchor_object_ids=tuple(anchor_object_ids),
        scene_scale_summary=scene_scale_summary.strip() if isinstance(scene_scale_summary, str) else "",
        objects=normalized_objects,
        metadata=metadata,
    )


def _parse_size_inference_result(
    parsed_output: dict[str, object],
    *,
    object_catalog: ObjectCatalog,
    metadata,
) -> SizeInferenceResult:
    report = _build_size_inference_report(parsed_output, object_catalog, metadata)

    if report is not None and report.objects:
        source_entries = report.objects.items()
    else:
        source_entries = [
            (key, value)
            for key, value in parsed_output.items()
            if key not in {"scene_description", "anchor_object_ids", "scene_scale_summary", "objects"}
        ]

    priors: list[SizePrior] = []
    for key, value in source_entries:
        resolved_object_id = key if report is not None and report.objects and key in report.objects else match_object_id(object_catalog, key)
        if resolved_object_id is None:
            continue
        dimensions, relative_scale = _extract_dimensions_from_size_value(value)
        if dimensions is None and relative_scale is None:
            continue
        try:
            priors.append(
                SizePrior(
                    object_id=resolved_object_id,
                    dimensions_m=dimensions,
                    relative_scale_to_scene=relative_scale,
                    source="structured_llm",
                    metadata=metadata,
                )
            )
        except ValueError:
            continue

    return SizeInferenceResult(size_priors=tuple(priors), size_inference_report=report)


def _longest_dimension(dimensions: dict[str, float] | None) -> float | None:
    if not dimensions:
        return None
    return max((float(value) for value in dimensions.values() if float(value) > 0.0), default=None)


def _rescaled_dimensions(dimensions: dict[str, float], *, target_longest: float) -> dict[str, float]:
    current_longest = _longest_dimension(dimensions)
    if current_longest is None or current_longest <= 0.0:
        return dimensions
    scale = float(target_longest) / float(current_longest)
    return {axis: float(value) * scale for axis, value in dimensions.items()}


def _reasonableize_size_priors(
    priors: Sequence[SizePrior],
    object_catalog: ObjectCatalog,
) -> tuple[SizePrior, ...]:
    canonical_name_by_id = {
        detected_object.object_id: detected_object.canonical_name
        for detected_object in object_catalog.objects
    }
    anchor_longest = None
    for prior in priors:
        canonical_name = canonical_name_by_id.get(prior.object_id, "")
        if _match_size_category(canonical_name, _FURNITURE_LONGEST_SIDE_M) is None:
            continue
        prior_longest = _longest_dimension(prior.dimensions_m)
        if prior_longest is None:
            continue
        anchor_longest = prior_longest if anchor_longest is None else max(anchor_longest, prior_longest)

    adjusted_priors: list[SizePrior] = []
    for prior in priors:
        canonical_name = canonical_name_by_id.get(prior.object_id, "")
        toy_cube_match = _match_size_category(canonical_name, _TOY_CUBE_LONGEST_SIDE_M)
        if toy_cube_match is not None and prior.dimensions_m is not None:
            current_longest = _longest_dimension(prior.dimensions_m)
            if current_longest is None:
                adjusted_priors.append(prior)
                continue

            _category_name, canonical_longest = toy_cube_match
            min_allowed = canonical_longest
            max_allowed = canonical_longest * 1.5
            if current_longest < min_allowed:
                adjusted_priors.append(
                    SizePrior(
                        object_id=prior.object_id,
                        dimensions_m=_rescaled_dimensions(prior.dimensions_m, target_longest=min_allowed),
                        relative_scale_to_scene=prior.relative_scale_to_scene,
                        source="structured_llm_reasonableized",
                        metadata=prior.metadata,
                    )
                )
                continue
            if current_longest > max_allowed:
                adjusted_priors.append(
                    SizePrior(
                        object_id=prior.object_id,
                        dimensions_m=_rescaled_dimensions(prior.dimensions_m, target_longest=max_allowed),
                        relative_scale_to_scene=prior.relative_scale_to_scene,
                        source="structured_llm_reasonableized",
                        metadata=prior.metadata,
                    )
                )
                continue

            adjusted_priors.append(prior)
            continue

        small_match = _match_size_category(canonical_name, _SMALL_OBJECT_LONGEST_SIDE_M)
        dimensions = prior.dimensions_m
        if small_match is None or dimensions is None:
            adjusted_priors.append(prior)
            continue

        current_longest = _longest_dimension(dimensions)
        if current_longest is None:
            adjusted_priors.append(prior)
            continue

        _category_name, canonical_longest = small_match
        max_allowed = canonical_longest * 4.0
        if anchor_longest is not None:
            max_allowed = min(canonical_longest * 2.5, float(anchor_longest) * 0.25)

        if current_longest <= max_allowed:
            adjusted_priors.append(prior)
            continue

        adjusted_priors.append(
            SizePrior(
                object_id=prior.object_id,
                dimensions_m=_rescaled_dimensions(dimensions, target_longest=max_allowed),
                relative_scale_to_scene=prior.relative_scale_to_scene,
                source="structured_llm_reasonableized",
                metadata=prior.metadata,
            )
        )

    return tuple(adjusted_priors)


def _default_size_dimensions(canonical_name: str) -> dict[str, float] | None:
    """Return a conservative fallback size for an object name.

    This is used when the size LLM fails to provide an entry for a detected
    object. The fallback keeps all instances of a scene represented while avoiding
    downstream layout failures.
    """

    explicit_match = _match_dimension_category(canonical_name, _EXPLICIT_FALLBACK_DIMENSIONS_M)
    if explicit_match is not None:
        _category_name, dimensions = explicit_match
        return {axis: float(value) for axis, value in dimensions.items()}

    candidate_match = _match_size_category(canonical_name, _TOY_CUBE_LONGEST_SIDE_M)
    if candidate_match is not None:
        _category_name, canonical_longest = candidate_match
        return {
            "x": float(canonical_longest),
            "y": float(canonical_longest),
            "z": float(canonical_longest),
        }

    candidate_match = _match_size_category(canonical_name, _SMALL_OBJECT_LONGEST_SIDE_M)
    if candidate_match is not None:
        _category_name, canonical_longest = candidate_match
        return {
            "x": float(canonical_longest),
            "y": float(canonical_longest),
            "z": float(canonical_longest * 0.2),
        }

    candidate_match = _match_size_category(canonical_name, _FURNITURE_LONGEST_SIDE_M)
    if candidate_match is not None:
        _category_name, canonical_longest = candidate_match
        return {
            "x": float(canonical_longest),
            "y": float(canonical_longest * 0.7),
            "z": float(canonical_longest * 0.08),
        }

    # Generic fallback for any unmatched object class.
    return {"x": 0.1, "y": 0.1, "z": 0.1}


def _fill_missing_size_priors(
    priors: Sequence[SizePrior],
    object_catalog: ObjectCatalog,
    *,
    fallback_metadata=None,
) -> tuple[SizePrior, ...]:
    existing_ids = {prior.object_id for prior in priors}
    fallback_priors: list[SizePrior] = []

    for detected_object in object_catalog.objects:
        if detected_object.object_id in existing_ids:
            continue
        dimensions = _default_size_dimensions(detected_object.canonical_name)
        if dimensions is None:
            continue
        if fallback_metadata is not None:
            metadata = make_stage_metadata(
                stage_name="size_inference",
                provider_name=fallback_metadata.provider_name,
                raw_response_artifacts=fallback_metadata.raw_response_artifacts,
                notes=(*fallback_metadata.notes, f"fallback_inferred_size:{detected_object.object_id}"),
            )
        else:
            metadata = make_stage_metadata(
                stage_name="size_inference",
                provider_name="structured_llm_fallback",
                notes=(f"fallback_inferred_size:{detected_object.object_id}",),
            )
        fallback_priors.append(
            SizePrior(
                object_id=detected_object.object_id,
                dimensions_m=dimensions,
                relative_scale_to_scene=None,
                source="structured_llm_fallback",
                metadata=metadata,
            )
        )

    if not fallback_priors:
        return tuple(priors)

    return tuple((*priors, *fallback_priors))


def _context_artifacts_for_object(
    object_id: str,
    object_catalog: ObjectCatalog,
    reference_image: ReferenceImageResult,
    segmentation: SegmentationResult,
) -> tuple[ArtifactRef, ...]:
    detected = next((obj for obj in object_catalog.objects if obj.object_id == object_id), None)
    if detected is None:
        return (reference_image.image,)

    masks = [
        instance.mask
        for instance in segmentation.instances
        if instance.instance_id in detected.source_instance_ids
    ]
    if masks:
        return (reference_image.image, *masks)
    return (reference_image.image,)


@dataclass(frozen=True, slots=True)
class _RelationPromptObject:
    alias: str
    resolved_object_id: str
    display_name: str


def _build_relation_prompt_objects(
    object_catalog: ObjectCatalog,
) -> tuple[_RelationPromptObject, ...]:
    expanded_ids = expanded_instance_ids_by_object_id(object_catalog)
    used_aliases: set[str] = set()
    prompt_objects: list[_RelationPromptObject] = []

    for detected_object in object_catalog.objects:
        resolved_ids = expanded_ids.get(detected_object.object_id, (detected_object.object_id,))
        display_name = detected_object.display_name or detected_object.canonical_name.replace("_", " ")
        for index, resolved_id in enumerate(resolved_ids, start=1):
            alias = _relation_prompt_alias(
                detected_object.canonical_name,
                index=index if len(resolved_ids) > 1 else None,
                used_aliases=used_aliases,
            )
            prompt_objects.append(
                _RelationPromptObject(
                    alias=alias,
                    resolved_object_id=resolved_id,
                    display_name=display_name,
                )
            )
            used_aliases.add(alias)

    return tuple(prompt_objects)


def _relation_prompt_alias(
    canonical_name: str,
    *,
    index: int | None,
    used_aliases: set[str],
) -> str:
    base_alias = canonical_name.strip().lower() or "object"
    candidate = f"{base_alias}{index}" if index is not None else base_alias
    if candidate not in used_aliases:
        return candidate

    suffix = 2
    unique_candidate = f"{candidate}_{suffix}"
    while unique_candidate in used_aliases:
        suffix += 1
        unique_candidate = f"{candidate}_{suffix}"
    return unique_candidate


def _build_relation_prompt_text(
    prompt_objects: Sequence[_RelationPromptObject],
    template: str,
) -> str:
    object_lines = []
    repeated_bases = {
        re.sub(r"\d+$", "", prompt_object.alias)
        for prompt_object in prompt_objects
        if re.search(r"\d+$", prompt_object.alias)
    }
    for prompt_object in prompt_objects:
        object_lines.append(f"- {prompt_object.alias}: {prompt_object.display_name}")

    guidance_lines = [
        "This scene contains the following object instances. Use these exact names in the JSON relations.",
        *object_lines,
    ]
    if repeated_bases:
        repeated_list = ", ".join(sorted(repeated_bases))
        guidance_lines.append(
            "If a category appears with numbered instances, never use the unnumbered category name "
            f"for that category. Use the numbered instance names instead: {repeated_list}."
        )
    guidance_lines.append(
        "Every object instance must have at most one direct parent relation in the final graph. "
        "Do not assign multiple parents to the same child. If an object could plausibly relate to "
        "multiple ancestors, choose only the single most direct physical parent: the object it "
        "directly rests on or is directly inside."
    )
    guidance_lines.append(template)
    return "\n".join(guidance_lines)


def _build_relation_object_resolver(
    prompt_objects: Sequence[_RelationPromptObject],
):
    alias_to_object_id = {
        prompt_object.alias.lower(): prompt_object.resolved_object_id
        for prompt_object in prompt_objects
    }

    def resolve(object_catalog: ObjectCatalog | None, key: str) -> str | None:
        normalized = key.strip().lower()
        if normalized in alias_to_object_id:
            return alias_to_object_id[normalized]
        return match_object_id(object_catalog, key)

    return resolve


def _context_artifacts_for_relation(
    prompt_objects: Sequence[_RelationPromptObject],
    object_catalog: ObjectCatalog,
    reference_image: ReferenceImageResult,
    segmentation: SegmentationResult,
) -> tuple[ArtifactRef, ...]:
    artifacts: list[ArtifactRef] = [reference_image.image]
    if segmentation.composite_visualization is not None:
        artifacts.append(segmentation.composite_visualization)

    instance_lookup = {
        instance.instance_id: instance
        for instance in segmentation.instances
    }
    source_instance_ids_by_object_id = {
        detected_object.object_id: tuple(
            source_instance_id
            for source_instance_id in detected_object.source_instance_ids
            if source_instance_id.strip()
        )
        for detected_object in object_catalog.objects
    }
    for prompt_object in prompt_objects:
        instance = instance_lookup.get(prompt_object.resolved_object_id)
        if instance is None:
            source_instance_ids = source_instance_ids_by_object_id.get(prompt_object.resolved_object_id, ())
            instance = next(
                (
                    instance_lookup[source_instance_id]
                    for source_instance_id in source_instance_ids
                    if source_instance_id in instance_lookup
                ),
                None,
            )
        if instance is not None:
            artifacts.append(instance.mask)

    return tuple(artifacts)


class PromptFileObjectTextDescriber(ObjectTextDescriber):
    def __init__(
        self,
        llm: StructuredLLM,
        *,
        prompt_path: str = "pat3d/preprocessing/gpt_utils/get_obj_descrip.txt",
    ) -> None:
        self._llm = llm
        self._prompt_path = prompt_path

    def describe_objects(
        self,
        object_catalog: ObjectCatalog,
        reference_image: ReferenceImageResult,
        segmentation: SegmentationResult,
    ) -> Sequence[ObjectDescription]:
        template = _load_prompt(self._prompt_path)
        descriptions = []
        for obj in object_catalog.objects:
            prompt_text = f"The object is {obj.canonical_name}. {template}"
            request = StructuredPromptRequest(
                schema_name="object_description",
                prompt_text=prompt_text,
                system_prompt=template,
                context_artifacts=_context_artifacts_for_object(
                    obj.object_id,
                    object_catalog,
                    reference_image,
                    segmentation,
                ),
                metadata={"object_id": obj.object_id},
            )
            try:
                result = self._llm.generate(request)
                description_text = f"The object is {obj.canonical_name}. {_flatten_single_value(result.parsed_output)}"
                metadata = result.metadata
            except Exception as exc:
                if not isinstance(exc, ValueError) and not _is_structured_parse_failure(exc):
                    raise
                description_text = f"The object is {obj.canonical_name}."
                metadata = make_stage_metadata(
                    stage_name="object_description",
                    provider_name="structured_llm_fallback",
                    notes=(f"fallback_reason={exc}",),
                )
            descriptions.append(
                ObjectDescription(
                    object_id=obj.object_id,
                    canonical_name=obj.canonical_name,
                    prompt_text=description_text,
                    visual_attributes={},
                    material_attributes={},
                    orientation_hints={},
                    metadata=metadata,
                )
            )
        return tuple(descriptions)


class PromptFileRelationExtractor(RelationExtractor):
    def __init__(
        self,
        llm: StructuredLLM,
        *,
        prompt_path: str = "pat3d/preprocessing/gpt_utils/get_contain_on.txt",
    ) -> None:
        self._llm = llm
        self._prompt_path = prompt_path

    def extract(
        self,
        object_catalog: ObjectCatalog,
        reference_image: ReferenceImageResult,
        segmentation: SegmentationResult,
        *,
        depth_result=None,
    ) -> SceneRelationGraph:
        _ = depth_result
        template = _load_prompt(self._prompt_path)
        prompt_objects = _build_relation_prompt_objects(object_catalog)
        prompt_text = _build_relation_prompt_text(prompt_objects, template)
        try:
            result = self._llm.generate(
                StructuredPromptRequest(
                    schema_name="scene_relations",
                    prompt_text=prompt_text,
                    system_prompt=template,
                    context_artifacts=_context_artifacts_for_relation(
                        prompt_objects,
                        object_catalog,
                        reference_image,
                        segmentation,
                    ),
                    metadata={"scene_id": object_catalog.scene_id},
                )
            )
            metadata = result.metadata
            parsed_output = result.parsed_output
        except Exception as exc:
            if not isinstance(exc, ValueError) and not _is_structured_parse_failure(exc):
                raise
            metadata = make_stage_metadata(
                stage_name="relation_extraction",
                provider_name="structured_llm_fallback",
                notes=(f"fallback_reason={exc}",),
            )
            parsed_output = {}

        relations, root_ids = normalize_scene_relations(
            parsed_output,
            object_catalog=object_catalog,
            object_resolver=_build_relation_object_resolver(prompt_objects),
            include_catalog_nodes=False,
        )
        return SceneRelationGraph(
            scene_id=object_catalog.scene_id,
            relations=relations,
            root_object_ids=root_ids,
            metadata=metadata,
        )


class PromptFileSizePriorEstimator(SizePriorEstimator):
    def __init__(
        self,
        llm: StructuredLLM,
        *,
        prompt_path: str = "pat3d/preprocessing/gpt_utils/get_size.txt",
        pause_on_failure: bool = False,
    ) -> None:
        self._llm = llm
        self._prompt_path = prompt_path
        self._pause_on_failure = bool(pause_on_failure)

    def estimate(
        self,
        object_catalog: ObjectCatalog,
        reference_image: ReferenceImageResult,
        *,
        segmentation=None,
        depth_result=None,
        object_descriptions: Sequence[ObjectDescription] = (),
        relation_graph: SceneRelationGraph | None = None,
    ) -> SizeInferenceResult:
        _ = segmentation, depth_result
        template = _load_prompt(self._prompt_path)
        prompt_text = _build_size_prompt(
            object_catalog,
            template,
            object_descriptions=object_descriptions,
            relation_graph=relation_graph,
        )
        fallback_failure = None
        try:
            result = self._llm.generate(
                StructuredPromptRequest(
                    schema_name="size_prior",
                    prompt_text=prompt_text,
                    system_prompt=template,
                    context_artifacts=(reference_image.image,),
                    metadata={"scene_id": object_catalog.scene_id},
                )
            )
            metadata = result.metadata
            parsed_output = result.parsed_output
        except Exception as exc:
            if not isinstance(exc, ValueError) and not _is_structured_parse_failure(exc):
                raise
            fallback_failure = exc
            raw_response_artifacts = ()
            notes = []
            raw_response_path = ""
            technical_message = str(exc)
            if isinstance(exc, Pat3DRuntimeError):
                technical_message = exc.payload.technical_message or str(exc)
                notes.append(f"fallback_reason={technical_message}")
                raw_response_path = str(exc.payload.details.get("raw_response_artifact", "")).strip()
                if raw_response_path:
                    suffix = Path(raw_response_path).suffix.lower().lstrip(".") or "txt"
                    raw_response_artifacts = (
                        ArtifactRef(
                            artifact_type="raw_response",
                            path=raw_response_path,
                            format=suffix,
                            role="llm_raw_response",
                            metadata_path=None,
                        ),
                    )
                    notes.append(f"fallback_raw_response_artifact={raw_response_path}")
            else:
                notes.append(f"fallback_reason={exc}")
            metadata = make_stage_metadata(
                stage_name="size_inference",
                provider_name="structured_llm_fallback",
                raw_response_artifacts=raw_response_artifacts,
                notes=tuple(notes),
            )
            if self._pause_on_failure:
                detail_map = dict(exc.payload.details) if isinstance(exc, Pat3DRuntimeError) else {}
                detail_map.update({
                    "scene_id": object_catalog.scene_id,
                    "raw_response_artifact": detail_map.get("raw_response_artifact") or raw_response_path or "",
                })
                raise runtime_failure(
                    phase="size_inference",
                    code="size_inference_requires_input",
                    user_message=(
                        "Structured size inference did not return a usable response. "
                        "Retry stage 03 or enter dimensions manually to continue."
                    ),
                    technical_message=technical_message if isinstance(exc, Pat3DRuntimeError) else str(exc),
                    provider_kind="structured_llm",
                    retryable=True,
                    details=detail_map,
                ) from exc
            parsed_output = {}

        inference_result = _parse_size_inference_result(
            parsed_output,
            object_catalog=object_catalog,
            metadata=metadata,
        )
        priors_with_fallbacks = _fill_missing_size_priors(
            inference_result.size_priors,
            object_catalog,
            fallback_metadata=metadata if metadata.provider_name == "structured_llm_fallback" else None,
        )
        reasonableized_priors = _reasonableize_size_priors(priors_with_fallbacks, object_catalog)
        report = inference_result.size_inference_report
        if report is None and metadata.provider_name == "structured_llm_fallback":
            reason_note = next((note for note in metadata.notes if note.startswith("fallback_reason=")), "")
            raw_artifact_note = next((note for note in metadata.notes if note.startswith("fallback_raw_response_artifact=")), "")
            summary_parts = []
            if reason_note:
                summary_parts.append(reason_note.split("=", 1)[1])
            if raw_artifact_note:
                summary_parts.append(f"raw_response_artifact={raw_artifact_note.split('=', 1)[1]}")
            if not summary_parts and fallback_failure is not None:
                summary_parts.append(str(fallback_failure))
            report = SizeInferenceReport(
                scene_description="Fallback size priors were used because structured size inference did not return a usable JSON result.",
                anchor_object_ids=(),
                scene_scale_summary="; ".join(summary_parts),
                objects={
                    prior.object_id: {"dimensions_m": dict(prior.dimensions_m or {})}
                    for prior in reasonableized_priors
                    if prior.dimensions_m
                },
                metadata=metadata,
            )
        return SizeInferenceResult(
            size_priors=reasonableized_priors,
            size_inference_report=report,
        )
