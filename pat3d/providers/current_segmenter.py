from __future__ import annotations

from pathlib import Path
from typing import Any

from pat3d.providers._compat import (
    coerce_mask_instances,
    image_path_from_reference,
    instantiate_model,
    scene_id_from_reference,
)
from pat3d.storage.artifacts import ArtifactStore


class CurrentSegmenter:
    def __init__(
        self,
        output_root: str = "data/seg",
        artifact_store: ArtifactStore | None = None,
    ) -> None:
        self.output_root = output_root
        self.artifact_store = artifact_store

    def _mask_bbox(self, mask_path: Path) -> tuple[list[float], int]:
        from PIL import Image
        import numpy as np

        mask = np.array(Image.open(mask_path))
        nonzero = np.argwhere(mask > 0)
        if nonzero.size == 0:
            return [0.0, 0.0, 0.0, 0.0], 0
        y0, x0 = nonzero.min(axis=0)[:2]
        y1, x1 = nonzero.max(axis=0)[:2]
        return [float(x0), float(y0), float(x1), float(y1)], int(nonzero.shape[0])

    def segment(self, reference_image_result: Any, object_hints: list[str] | None = None) -> Any:
        from pat3d.preprocessing.seg import get_seg

        image_path = Path(image_path_from_reference(reference_image_result))
        scene_id = scene_id_from_reference(reference_image_result, default=image_path.stem)
        hints = object_hints or []
        get_seg(scene_id, str(image_path.parent), hints, self.output_root)

        seg_dir = Path(self.output_root) / scene_id
        artifact_store = self.artifact_store or ArtifactStore()
        instances: list[dict[str, Any]] = []

        for ann_path in sorted(seg_dir.glob("*_ann.png")):
            stem = ann_path.name[:-8]
            label = stem.split("_")[-1] if "_" in stem else stem
            bbox_xyxy, area_px = self._mask_bbox(ann_path)
            instances.append(
                {
                    "instance_id": stem,
                    "label": label,
                    "mask": artifact_store.build_artifact_ref(ann_path, "mask", "png", "mask"),
                    "bbox_xyxy": bbox_xyxy,
                    "confidence": None,
                    "area_px": area_px,
                }
            )

        composite_path = seg_dir / f"{scene_id}_segmentation.png"
        payload = {
            "image": getattr(reference_image_result, "image", None)
            if not isinstance(reference_image_result, dict)
            else reference_image_result.get("image"),
            "instances": coerce_mask_instances(instances),
            "composite_visualization": artifact_store.build_artifact_ref(
                composite_path, "segmentation_visualization", "png", "segmentation_visualization"
            )
            if composite_path.exists()
            else None,
            "metadata": artifact_store.build_stage_metadata(
                stage_name="scene_understanding",
                provider_name="current_segmenter",
            ),
        }
        return instantiate_model("SegmentationResult", payload)
