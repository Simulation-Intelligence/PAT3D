from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

import numpy as np
from PIL import Image

from pat3d.models import ArtifactRef, ObjectCatalog, SceneRequest
from pat3d.pipelines.first_contract_slice import (
    FirstContractSliceOutputs,
    FirstContractSlicePipeline,
)
from pat3d.stages import (
    LayoutInitializationOutputs,
    LayoutInitializationStage,
    ObjectAssetGenerationOutputs,
    ObjectAssetGenerationStage,
    PhysicsOptimizationOutputs,
    PhysicsOptimizationStage,
    SimulationPreparationOutputs,
    SimulationPreparationStage,
    VisualizationExportOutputs,
    VisualizationExportStage,
)
from pat3d.stages.object_assets import instance_object_ids


@dataclass(slots=True)
class PaperCorePipelineOutputs:
    first_contract_slice: FirstContractSliceOutputs
    object_assets: ObjectAssetGenerationOutputs
    layout_initialization: LayoutInitializationOutputs
    simulation_preparation: SimulationPreparationOutputs
    physics_optimization: PhysicsOptimizationOutputs | None = None
    visualization: VisualizationExportOutputs | None = None


def _safe_artifact_stem(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value).strip("_") or "object"


def _materialize_object_reference_images(
    *,
    object_catalog,
    reference_image_path: str,
    segmentation_result,
    output_root: str = "data/ref_img_obj",
) -> dict[str, ArtifactRef]:
    if not object_catalog.objects or not segmentation_result.instances:
        return {}

    image = Image.open(reference_image_path).convert("RGBA")
    image_array = np.asarray(image)
    instances_by_id = {instance.instance_id: instance for instance in segmentation_result.instances}
    instances_by_label: dict[str, list] = {}
    for instance in segmentation_result.instances:
        label_key = str(instance.label or "").strip().lower()
        if not label_key:
            continue
        instances_by_label.setdefault(label_key, []).append(instance)

    scene_dir = Path(output_root) / object_catalog.scene_id
    scene_dir.mkdir(parents=True, exist_ok=True)
    generated: dict[str, ArtifactRef] = {}
    used_ids: set[str] = set()

    for detected_object in object_catalog.objects:
        candidate_instances = [
            instance
            for instance_id in detected_object.source_instance_ids
            for instance in (instances_by_id.get(instance_id),)
            if instance is not None
        ]
        if not candidate_instances:
            for label in (
                detected_object.display_name or "",
                detected_object.canonical_name,
            ):
                label_key = label.strip().lower()
                if not label_key:
                    continue
                matches = instances_by_label.get(label_key)
                if matches:
                    candidate_instances = list(matches)
                    break
        if not candidate_instances:
            continue

        target_ids = (
            instance_object_ids(detected_object, used_ids)
            if detected_object.count > 1
            else (detected_object.object_id,)
        )
        for index, target_id in enumerate(target_ids):
            candidate_instance = candidate_instances[min(index, len(candidate_instances) - 1)]
            mask_path = Path(candidate_instance.mask.path)
            if not mask_path.exists():
                continue
            mask_array = np.asarray(Image.open(mask_path).convert("L")) > 0
            if not np.any(mask_array):
                continue
            ys, xs = np.nonzero(mask_array)
            x0 = max(0, int(xs.min()))
            y0 = max(0, int(ys.min()))
            x1 = int(xs.max()) + 1
            y1 = int(ys.max()) + 1

            crop = image_array[y0:y1, x0:x1].copy()
            crop_mask = (mask_array[y0:y1, x0:x1].astype(np.uint8) * 255)
            crop[..., 3] = crop_mask

            target_path = scene_dir / f"{_safe_artifact_stem(target_id)}.png"
            Image.fromarray(crop, mode="RGBA").save(target_path)
            generated[target_id] = ArtifactRef(
                artifact_type="image",
                path=str(target_path),
                format="png",
                role="object_reference_image",
                metadata_path=None,
            )
            used_ids.add(target_id)

    return generated


class PaperCorePipeline:
    def __init__(
        self,
        first_contract_slice_pipeline: FirstContractSlicePipeline,
        object_asset_stage: ObjectAssetGenerationStage,
        layout_initialization_stage: LayoutInitializationStage,
        simulation_preparation_stage: SimulationPreparationStage,
        physics_optimization_stage: PhysicsOptimizationStage | None = None,
        visualization_stage: VisualizationExportStage | None = None,
    ) -> None:
        self._first_contract_slice_pipeline = first_contract_slice_pipeline
        self._object_asset_stage = object_asset_stage
        self._layout_initialization_stage = layout_initialization_stage
        self._simulation_preparation_stage = simulation_preparation_stage
        self._physics_optimization_stage = physics_optimization_stage
        self._visualization_stage = visualization_stage

    def run(
        self,
        request: SceneRequest,
        *,
        object_hints: Optional[list[str]] = None,
        object_catalog: Optional[ObjectCatalog] = None,
        object_reference_images: Mapping[str, ArtifactRef] | None = None,
    ) -> PaperCorePipelineOutputs:
        first_slice = self._first_contract_slice_pipeline.run(
            request,
            object_hints=object_hints,
            object_catalog=object_catalog,
        )

        if first_slice.object_relation is None:
            raise ValueError(
                "PaperCorePipeline requires object-relation outputs from the first contract slice."
            )

        auto_object_reference_images = _materialize_object_reference_images(
            object_catalog=first_slice.object_relation.object_catalog,
            reference_image_path=first_slice.reference_image_result.image.path,
            segmentation_result=first_slice.scene_understanding.segmentation_result,
        )
        resolved_object_reference_images = {
            **auto_object_reference_images,
            **(dict(object_reference_images) if object_reference_images is not None else {}),
        }

        object_assets = self._object_asset_stage.run(
            first_slice.object_relation.object_catalog,
            first_slice.object_relation.object_descriptions,
            object_reference_images=resolved_object_reference_images or None,
        )

        layout_initialization = self._layout_initialization_stage.run(
            scene_id=request.scene_id,
            object_assets=object_assets.object_assets,
            object_catalog=first_slice.object_relation.object_catalog,
            reference_image_result=first_slice.reference_image_result,
            segmentation_result=first_slice.scene_understanding.segmentation_result,
            depth_result=first_slice.scene_understanding.depth_result,
            relation_graph=first_slice.object_relation.relation_graph,
            size_priors=first_slice.object_relation.size_priors,
        )

        simulation_preparation = self._simulation_preparation_stage.run(
            layout_initialization.scene_layout,
            layout_initialization.object_assets,
        )

        physics_optimization = None
        scene_state_for_visualization = simulation_preparation.physics_ready_scene
        if self._physics_optimization_stage is not None:
            physics_optimization = self._physics_optimization_stage.run(
                simulation_preparation.physics_ready_scene
            )
            scene_state_for_visualization = physics_optimization.optimization_result

        visualization = None
        if self._visualization_stage is not None:
            visualization = self._visualization_stage.run(scene_state_for_visualization)

        return PaperCorePipelineOutputs(
            first_contract_slice=first_slice,
            object_assets=object_assets,
            layout_initialization=layout_initialization,
            simulation_preparation=simulation_preparation,
            physics_optimization=physics_optimization,
            visualization=visualization,
        )
