from __future__ import annotations

from pathlib import Path
import re
from typing import Callable, Sequence

import numpy as np
from PIL import Image

from pat3d.contracts import Segmenter
from pat3d.models import ArtifactRef, MaskInstance, ReferenceImageResult, SegmentationResult
from pat3d.providers._legacy_image_inputs import materialize_legacy_scene_image
from pat3d.storage import make_stage_metadata


def _bbox_from_mask(mask_path: Path) -> tuple[float, float, float, float]:
    array = np.asarray(Image.open(mask_path).convert("L"))
    ys, xs = np.nonzero(array > 0)
    if len(xs) == 0 or len(ys) == 0:
        return (0.0, 0.0, 0.0, 0.0)
    return (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))


def _label_from_stem(stem: str) -> str:
    parts = stem.split("_")
    for width in range(1, (len(parts) // 2) + 1):
        suffix = parts[-width:]
        previous = parts[-(2 * width) : -width]
        if suffix == previous:
            raw = "_".join(suffix)
            return re.sub(r"\d+$", "", raw).strip("_") or raw
    raw = parts[-1] if parts else stem
    return re.sub(r"\d+$", "", raw).strip("_") or raw


def _load_default_segmentation_function() -> Callable[[str, str, Sequence[str], str], None]:
    from pat3d.preprocessing.seg import get_seg as current_get_seg

    return current_get_seg


class CurrentSegmenter(Segmenter):
    def __init__(
        self,
        *,
        output_dir: str = "data/seg",
        legacy_input_dir: str = "data/ref_img",
        segmentation_function: Callable[[str, str, Sequence[str], str], None] | None = None,
        metadata_factory: Callable[..., object] = make_stage_metadata,
    ) -> None:
        self._output_dir = output_dir
        self._legacy_input_dir = legacy_input_dir
        self._segmentation_function = (
            segmentation_function or _load_default_segmentation_function()
        )
        self._metadata_factory = metadata_factory

    def segment(
        self,
        reference_image: ReferenceImageResult,
        object_hints: Sequence[str] | None = None,
    ) -> SegmentationResult:
        scene_name = reference_image.request.scene_id
        staged_image = materialize_legacy_scene_image(
            reference_image,
            target_root=self._legacy_input_dir,
        )
        image_folder = str(staged_image.parent)
        hints = list(object_hints or [])
        self._segmentation_function(scene_name, image_folder, hints, self._output_dir)

        scene_dir = Path(self._output_dir) / scene_name
        instances = []
        for mask_path in sorted(scene_dir.glob("*_ann.png")):
            stem = mask_path.stem.replace("_ann", "")
            label = _label_from_stem(stem)
            bbox = _bbox_from_mask(mask_path)
            array = np.asarray(Image.open(mask_path).convert("L"))
            area_px = int(np.count_nonzero(array > 0))
            instances.append(
                MaskInstance(
                    instance_id=stem,
                    label=label,
                    mask=ArtifactRef(
                        artifact_type="mask",
                        path=str(mask_path),
                        format="png",
                        role="instance_mask",
                        metadata_path=None,
                    ),
                    bbox_xyxy=bbox,
                    confidence=None,
                    area_px=area_px,
                )
            )

        composite_path = scene_dir / f"{scene_name}_segmentation.png"
        return SegmentationResult(
            image=reference_image.image,
            instances=tuple(instances),
            composite_visualization=ArtifactRef(
                artifact_type="segmentation_visualization",
                path=str(composite_path),
                format="png",
                role="segmentation_visualization",
                metadata_path=None,
            )
            if composite_path.exists()
            else None,
            metadata=self._metadata_factory(
                stage_name="segmentation",
                provider_name="current_sfg2_segmenter",
            ),
        )
