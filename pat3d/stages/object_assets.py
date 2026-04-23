from __future__ import annotations

from dataclasses import dataclass, replace
import inspect
from typing import Mapping, Sequence

from pat3d.contracts import TextTo3DProvider
from pat3d.models import ArtifactRef, GeneratedObjectAsset, ObjectAssetCatalog, ObjectCatalog, ObjectDescription


@dataclass(slots=True)
class ObjectAssetGenerationOutputs:
    object_catalog: ObjectCatalog
    object_descriptions: Sequence[ObjectDescription]
    object_assets: ObjectAssetCatalog


def clone_generated_asset_for_object_id(
    asset: GeneratedObjectAsset,
    object_id: str,
) -> GeneratedObjectAsset:
    provider_asset_id = asset.provider_asset_id
    if provider_asset_id:
        provider_asset_id = f"{provider_asset_id}#clone:{object_id}"
    return replace(
        asset,
        object_id=object_id,
        provider_asset_id=provider_asset_id,
    )


def instance_object_ids(detected_object, used_ids: set[str] | None = None) -> tuple[str, ...]:
    if detected_object.count <= 1:
        return (detected_object.object_id,)

    reserved_ids = used_ids if used_ids is not None else set()
    source_ids = tuple(source_id for source_id in detected_object.source_instance_ids if source_id.strip())
    resolved_ids: list[str] = []
    for index in range(detected_object.count):
        candidate = source_ids[index] if index < len(source_ids) else f"{detected_object.object_id}::{index + 1}"
        if candidate in reserved_ids or candidate in resolved_ids:
            candidate = f"{detected_object.object_id}::{index + 1}"
        suffix = 1
        unique_candidate = candidate
        while unique_candidate in reserved_ids or unique_candidate in resolved_ids:
            suffix += 1
            unique_candidate = f"{candidate}:{suffix}"
        resolved_ids.append(unique_candidate)
    return tuple(resolved_ids)


def expand_object_asset_requests(
    object_catalog: ObjectCatalog,
    object_descriptions: Sequence[ObjectDescription],
) -> tuple[tuple[ObjectDescription, str, int], ...]:
    descriptions_by_id = {description.object_id: description for description in object_descriptions}
    used_ids: set[str] = set()
    requests: list[tuple[ObjectDescription, str, int]] = []

    for detected_object in object_catalog.objects:
        description = descriptions_by_id.get(detected_object.object_id)
        if description is None:
            raise ValueError(f"missing ObjectDescription for object_id={detected_object.object_id}")

        target_ids = instance_object_ids(detected_object, used_ids)
        for target_id in target_ids:
            if target_id == description.object_id:
                requests.append((description, detected_object.object_id, detected_object.count))
            else:
                requests.append((replace(description, object_id=target_id), detected_object.object_id, detected_object.count))
            used_ids.add(target_id)

    return tuple(requests)


class ObjectAssetGenerationStage:
    stage_name = "object_asset_generation"

    def __init__(self, text_to_3d_provider: TextTo3DProvider) -> None:
        self._text_to_3d_provider = text_to_3d_provider

    def run(
        self,
        object_catalog: ObjectCatalog,
        object_descriptions: Sequence[ObjectDescription],
        *,
        object_reference_images: Mapping[str, ArtifactRef] | None = None,
    ) -> ObjectAssetGenerationOutputs:
        scene_object_canonical_names = tuple(
            dict.fromkeys(
                detected_object.canonical_name.strip()
                for detected_object in object_catalog.objects
                if detected_object.canonical_name.strip()
            )
        )
        generate_signature = inspect.signature(self._text_to_3d_provider.generate)
        supports_scene_object_canonical_names = any(
            parameter.name == "scene_object_canonical_names"
            or parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in generate_signature.parameters.values()
        )
        ordered_assets = []
        expanded_descriptions = []
        cached_assets_by_source_id: dict[str, GeneratedObjectAsset] = {}
        for generated_description, source_object_id, source_object_count in expand_object_asset_requests(
            object_catalog,
            object_descriptions,
        ):
            expanded_descriptions.append(generated_description)
            cached_asset = cached_assets_by_source_id.get(source_object_id)
            if cached_asset is not None:
                ordered_assets.append(
                    clone_generated_asset_for_object_id(cached_asset, generated_description.object_id)
                )
                continue
            generation_kwargs = {
                "object_reference_image": (
                    (object_reference_images or {}).get(generated_description.object_id)
                    or (object_reference_images or {}).get(source_object_id)
                ),
            }
            if any(
                parameter.name == "source_object_count"
                or parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in generate_signature.parameters.values()
            ):
                generation_kwargs["source_object_count"] = source_object_count
            if supports_scene_object_canonical_names:
                generation_kwargs["scene_object_canonical_names"] = scene_object_canonical_names
            generated_asset = self._text_to_3d_provider.generate(generated_description, **generation_kwargs)
            ordered_assets.append(generated_asset)
            cached_assets_by_source_id[source_object_id] = generated_asset

        return ObjectAssetGenerationOutputs(
            object_catalog=object_catalog,
            object_descriptions=tuple(expanded_descriptions),
            object_assets=ObjectAssetCatalog(
                scene_id=object_catalog.scene_id,
                assets=tuple(ordered_assets),
                metadata=object_catalog.metadata,
            ),
        )
