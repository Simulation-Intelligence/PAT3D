#!/usr/bin/env python3
from __future__ import annotations

import argparse
import colorsys
import os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
from transformers import Owlv2ForObjectDetection, Owlv2Processor


MODEL_ID = os.environ.get("PAT3D_SEG_MODEL", "google/owlv2-base-patch16-ensemble")
BOX_THRESHOLD = float(os.environ.get("PAT3D_SEG_BOX_THRESHOLD", "0.12"))
TEXT_THRESHOLD = float(os.environ.get("PAT3D_SEG_TEXT_THRESHOLD", "0.0"))
NMS_IOU = float(os.environ.get("PAT3D_SEG_NMS_IOU", "0.45"))
MASK_MIN_AREA_PX = int(os.environ.get("PAT3D_SEG_MIN_AREA_PX", "600"))

_PROCESSOR: Owlv2Processor | None = None
_MODEL: Owlv2ForObjectDetection | None = None

_ALIAS_MAP = {
    "side_table": ("side table", "end table", "nightstand", "small table", "table"),
    "end_table": ("end table", "side table", "nightstand", "small table", "table"),
    "coffee_table": ("coffee table", "table"),
    "dining_table": ("dining table", "table"),
    "table": ("table", "side table", "end table", "coffee table", "dining table"),
    "chair": ("chair", "armchair", "seat"),
    "sofa": ("sofa", "couch", "loveseat"),
    "plant": ("plant", "potted plant"),
    "bookshelf": ("bookshelf", "bookcase", "shelf"),
    "tv": ("tv", "television", "monitor", "screen"),
}


def _canonical_label(raw: str) -> str:
    lowered = raw.strip().lower().replace("-", "_").replace(" ", "_")
    cleaned = "".join(ch for ch in lowered if ch.isalnum() or ch == "_").strip("_")
    return cleaned or "object"


def _rough_singular(label: str) -> str:
    if label.endswith("ies") and len(label) > 3:
        return label[:-3] + "y"
    if label.endswith("ses") and len(label) > 3:
        return label[:-2]
    if label.endswith("s") and not label.endswith("ss") and len(label) > 3:
        return label[:-1]
    return label


def _display_label(label: str) -> str:
    return label.replace("_", " ")


def _hint_aliases(hint: str) -> list[str]:
    canonical = _rough_singular(_canonical_label(hint))
    aliases = list(_ALIAS_MAP.get(canonical, ()))
    aliases.append(_display_label(canonical))
    if "table" in canonical and "table" not in aliases:
        aliases.append("table")
    deduped: list[str] = []
    seen: set[str] = set()
    for alias in aliases:
        normalized = alias.strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped or [_display_label(canonical)]


def _load_model() -> tuple[Owlv2Processor, Owlv2ForObjectDetection]:
    global _PROCESSOR, _MODEL
    if _PROCESSOR is None or _MODEL is None:
        _PROCESSOR = Owlv2Processor.from_pretrained(MODEL_ID)
        _MODEL = Owlv2ForObjectDetection.from_pretrained(MODEL_ID)
        _MODEL.eval()
    return _PROCESSOR, _MODEL


def _box_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x0 = max(float(box_a[0]), float(box_b[0]))
    y0 = max(float(box_a[1]), float(box_b[1]))
    x1 = min(float(box_a[2]), float(box_b[2]))
    y1 = min(float(box_a[3]), float(box_b[3]))
    inter_w = max(0.0, x1 - x0)
    inter_h = max(0.0, y1 - y0)
    intersection = inter_w * inter_h
    if intersection <= 0.0:
        return 0.0
    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(0.0, float(box_a[3] - box_a[1]))
    area_b = max(0.0, float(box_b[2] - box_b[0])) * max(0.0, float(box_b[3] - box_b[1]))
    denom = area_a + area_b - intersection
    if denom <= 0.0:
        return 0.0
    return intersection / denom


def _clip_box(box: np.ndarray, width: int, height: int) -> tuple[int, int, int, int] | None:
    x0, y0, x1, y1 = [int(round(float(value))) for value in box]
    x0 = max(0, min(width - 1, x0))
    y0 = max(0, min(height - 1, y0))
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _largest_component(mask: np.ndarray) -> np.ndarray:
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if component_count <= 1:
        return mask.astype(bool)
    largest_index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == largest_index


def _mask_from_box(image_rgb: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    height, width = image_rgb.shape[:2]
    x0, y0, x1, y1 = box
    rect = (x0, y0, max(1, x1 - x0), max(1, y1 - y0))
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    mask = np.zeros((height, width), dtype=np.uint8)
    bg_model = np.zeros((1, 65), dtype=np.float64)
    fg_model = np.zeros((1, 65), dtype=np.float64)

    try:
        cv2.grabCut(image_bgr, mask, rect, bg_model, fg_model, 5, cv2.GC_INIT_WITH_RECT)
        foreground = np.logical_or(mask == cv2.GC_FGD, mask == cv2.GC_PR_FGD)
    except cv2.error:
        foreground = np.zeros((height, width), dtype=bool)

    if int(np.count_nonzero(foreground)) < MASK_MIN_AREA_PX:
        foreground = np.zeros((height, width), dtype=bool)
        foreground[y0 : y1 + 1, x0 : x1 + 1] = True

    foreground = _largest_component(foreground)
    return foreground


def _cutout_image(image: np.ndarray, mask: np.ndarray, bbox: tuple[int, int, int, int]) -> Image.Image:
    x0, y0, x1, y1 = bbox
    crop = image[y0 : y1 + 1, x0 : x1 + 1].copy()
    crop_mask = mask[y0 : y1 + 1, x0 : x1 + 1]
    white = np.full_like(crop, 255)
    crop = np.where(crop_mask[..., None], crop, white)
    return Image.fromarray(crop)


def _mask_image(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> Image.Image:
    x0, y0, x1, y1 = bbox
    crop_mask = (mask[y0 : y1 + 1, x0 : x1 + 1] * 255).astype(np.uint8)
    return Image.fromarray(crop_mask, mode="L")


def _segment_overlay(image: np.ndarray, masks: list[np.ndarray]) -> Image.Image:
    overlay = image.astype(np.float32).copy()
    for index, mask in enumerate(masks):
        hue = (index * 0.1618) % 1.0
        rgb = np.array(colorsys.hsv_to_rgb(hue, 0.75, 1.0), dtype=np.float32) * 255.0
        overlay[mask] = overlay[mask] * 0.35 + rgb * 0.65
    return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8), mode="RGB")


def _fallback_full_image(
    image: np.ndarray,
    output_dir: Path,
    scene_name: str,
    hints: list[str],
) -> None:
    label = _rough_singular(_canonical_label(hints[0] if hints else "object"))
    instance_name = f"{label}1"
    stem = f"{scene_name}_{instance_name}_{instance_name}"
    Image.fromarray(image).save(output_dir / f"{stem}.png")
    full_mask = np.full(image.shape[:2], 255, dtype=np.uint8)
    Image.fromarray(full_mask, mode="L").save(output_dir / f"{stem}_ann.png")
    Image.fromarray(image).save(output_dir / f"{scene_name}_segmentation.png")


def _detect_objects(image: Image.Image, hints: list[str]) -> dict[str, list[dict[str, object]]]:
    if not hints:
        return {}

    processor, model = _load_model()
    requested_counts: dict[str, int] = defaultdict(int)
    hint_to_queries: dict[str, list[str]] = {}
    flattened_queries: list[str] = []
    query_to_hint: list[str] = []

    for hint in hints:
        canonical = _rough_singular(_canonical_label(hint))
        requested_counts[canonical] += 1
        aliases = _hint_aliases(hint)
        hint_to_queries[canonical] = aliases
        for alias in aliases:
            flattened_queries.append(alias)
            query_to_hint.append(canonical)

    inputs = processor(text=[flattened_queries], images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]], dtype=torch.float32)
    results = processor.post_process_grounded_object_detection(
        outputs=outputs,
        target_sizes=target_sizes,
        threshold=BOX_THRESHOLD,
        text_labels=[flattened_queries],
    )[0]

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for box, score, label_index in zip(results["boxes"], results["scores"], results["labels"]):
        hint = query_to_hint[int(label_index)]
        grouped[hint].append(
            {
                "box": np.asarray(box, dtype=float),
                "score": float(score),
                "query": flattened_queries[int(label_index)],
            }
        )

    selected: dict[str, list[dict[str, object]]] = {}
    for hint in hints:
        canonical = _rough_singular(_canonical_label(hint))
        candidates = sorted(grouped.get(canonical, []), key=lambda item: float(item["score"]), reverse=True)
        kept: list[dict[str, object]] = []
        for candidate in candidates:
            if candidate["score"] < max(BOX_THRESHOLD, TEXT_THRESHOLD):
                continue
            if any(_box_iou(candidate["box"], existing["box"]) > NMS_IOU for existing in kept):
                continue
            kept.append(candidate)
            if len(kept) >= requested_counts[canonical]:
                break
        if kept:
            selected[canonical] = kept
    return selected


def run_segmentation(input_path: Path, output_dir: Path, object_name_list: list[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    scene_name = input_path.stem
    image = np.asarray(Image.open(input_path).convert("RGB"))
    image_pil = Image.fromarray(image)
    height, width = image.shape[:2]

    detected = _detect_objects(image_pil, object_name_list)
    instance_counts: dict[str, int] = defaultdict(int)
    overlay_masks: list[np.ndarray] = []

    for hint in object_name_list:
        canonical = _rough_singular(_canonical_label(hint))
        candidates = detected.get(canonical, [])
        for candidate in candidates:
            clipped_box = _clip_box(candidate["box"], width, height)
            if clipped_box is None:
                continue

            mask = _mask_from_box(image, clipped_box)
            if int(np.count_nonzero(mask)) < MASK_MIN_AREA_PX:
                continue

            ys, xs = np.nonzero(mask)
            bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
            instance_counts[canonical] += 1
            instance_name = f"{canonical}{instance_counts[canonical]}"
            stem = f"{scene_name}_{instance_name}_{instance_name}"

            _cutout_image(image, mask, bbox).save(output_dir / f"{stem}.png")
            _mask_image(mask, bbox).save(output_dir / f"{stem}_ann.png")
            overlay_masks.append(mask)

    if not overlay_masks:
        _fallback_full_image(image, output_dir, scene_name, object_name_list)
        return

    _segment_overlay(image, overlay_masks).save(output_dir / f"{scene_name}_segmentation.png")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--seg_output", required=True)
    parser.add_argument("--object_name_list", nargs="*", default=[])
    args = parser.parse_args()

    run_segmentation(
        input_path=Path(args.input),
        output_dir=Path(args.seg_output),
        object_name_list=list(args.object_name_list),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
