from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw
import trimesh

from pat3d.contracts import DepthEstimator, Segmenter, StructuredLLM, TextTo3DProvider, TextToImageProvider
from pat3d.models import (
    ArtifactRef,
    DepthResult,
    GeneratedObjectAsset,
    MaskInstance,
    ObjectAssetCatalog,
    ObjectCatalog,
    ObjectDescription,
    ObjectPose,
    ReferenceImageResult,
    SceneLayout,
    SceneRelationGraph,
    SceneRequest,
    SegmentationResult,
    StructuredPromptRequest,
    StructuredPromptResult,
)
from pat3d.providers.catalog_builder import _canonicalize_label
from pat3d.storage import make_stage_metadata


_SUPPORT_SURFACE_NAMES = {
    "table",
    "desk",
    "counter",
    "nightstand",
    "shelf",
    "bookshelf",
    "cabinet",
}
_CONTAINER_OBJECT_NAMES = {"basket"}
_ROUND_OBJECT_NAMES = {"apple", "orange", "pear", "peach", "ball"}
_CYLINDER_OBJECT_NAMES = {"mug", "cup", "bottle", "can", "plate", "bowl"}
_BOX_OBJECT_NAMES = {"book", "phone", "remote", "mouse"}
_COLOR_BY_NAME: dict[str, tuple[int, int, int]] = {
    "table": (146, 108, 73),
    "desk": (132, 96, 68),
    "counter": (158, 130, 104),
    "nightstand": (151, 116, 88),
    "shelf": (124, 98, 75),
    "bookshelf": (128, 100, 72),
    "cabinet": (112, 117, 138),
    "apple": (82, 157, 78),
    "orange": (224, 141, 53),
    "pear": (143, 187, 67),
    "peach": (237, 167, 135),
    "mug": (196, 72, 62),
    "cup": (70, 121, 186),
    "bottle": (54, 145, 130),
    "book": (182, 89, 100),
    "basket": (168, 118, 72),
    "plate": (226, 224, 219),
    "bowl": (177, 112, 71),
}
_PRIMITIVE_DIMS_M: dict[str, tuple[float, float, float]] = {
    "table": (1.45, 0.72, 0.92),
    "desk": (1.35, 0.74, 0.72),
    "counter": (1.55, 0.92, 0.72),
    "nightstand": (0.55, 0.62, 0.42),
    "shelf": (1.20, 1.80, 0.36),
    "bookshelf": (1.25, 1.95, 0.36),
    "cabinet": (1.10, 1.60, 0.52),
    "apple": (0.12, 0.12, 0.12),
    "orange": (0.12, 0.12, 0.12),
    "pear": (0.10, 0.14, 0.10),
    "peach": (0.11, 0.11, 0.11),
    "mug": (0.09, 0.10, 0.09),
    "cup": (0.09, 0.10, 0.09),
    "bottle": (0.08, 0.28, 0.08),
    "book": (0.24, 0.04, 0.18),
    "basket": (0.32, 0.18, 0.24),
    "plate": (0.24, 0.03, 0.24),
    "bowl": (0.18, 0.08, 0.18),
}
_DEFAULT_DIMS_M = (0.18, 0.18, 0.18)
_BASKET_WALL_THICKNESS_M = 0.02
_BASKET_BOTTOM_THICKNESS_M = 0.02
_FALLBACK_COLORS = (
    (196, 72, 62),
    (82, 157, 78),
    (70, 121, 186),
    (224, 141, 53),
    (147, 105, 181),
    (67, 149, 161),
)


@dataclass(frozen=True, slots=True)
class _SmokePlacement:
    instance_id: str
    label: str
    bbox_xyxy: tuple[int, int, int, int]
    shape: str
    color: tuple[int, int, int]


def _safe_stem(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value)).strip("_") or "artifact"


def _canonical_name(value: str) -> str:
    return _canonicalize_label(value)


def _display_name(value: str) -> str:
    return str(value).replace("_", " ").strip() or "object"


def _requested_objects_from_prompt(prompt_text: str | None) -> tuple[str, ...]:
    if not prompt_text:
        return ("object",)
    tokens = [_canonical_name(token) for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_]*", prompt_text)]
    resolved = [
        token
        for token in tokens
        if token in _PRIMITIVE_DIMS_M or token in _SUPPORT_SURFACE_NAMES
    ]
    if not resolved:
        return ("object",)
    deduped: list[str] = []
    for token in resolved:
        if token not in deduped:
            deduped.append(token)
    return tuple(deduped)


def _resolve_requested_objects(
    requested_objects: Sequence[str] | None,
    *,
    prompt_text: str | None = None,
) -> tuple[str, ...]:
    if requested_objects:
        normalized = tuple(_canonical_name(name) for name in requested_objects if str(name).strip())
        if normalized:
            return normalized
    return _requested_objects_from_prompt(prompt_text)


def _object_color(name: str, *, fallback_index: int = 0) -> tuple[int, int, int]:
    color = _COLOR_BY_NAME.get(_canonical_name(name))
    if color is not None:
        return color
    return _FALLBACK_COLORS[fallback_index % len(_FALLBACK_COLORS)]


def _primitive_dims(name: str) -> tuple[float, float, float]:
    return _PRIMITIVE_DIMS_M.get(_canonical_name(name), _DEFAULT_DIMS_M)


def _support_surface_name(names: Sequence[str]) -> str | None:
    for name in names:
        canonical = _canonical_name(name)
        if canonical in _SUPPORT_SURFACE_NAMES:
            return canonical
    return None


def _container_name(names: Sequence[str]) -> str | None:
    for name in names:
        canonical = _canonical_name(name)
        if canonical in _CONTAINER_OBJECT_NAMES:
            return canonical
    return None


def _anchor_name(names: Sequence[str]) -> str | None:
    support_name = _support_surface_name(names)
    if support_name is not None:
        return support_name
    return _container_name(names)


def _shape_for_name(name: str) -> str:
    canonical = _canonical_name(name)
    if canonical in _CONTAINER_OBJECT_NAMES:
        return "container"
    if canonical in _SUPPORT_SURFACE_NAMES:
        return "support"
    if canonical in _ROUND_OBJECT_NAMES:
        return "ellipse"
    return "rounded_rect"


def _instance_layouts(
    *,
    scene_id: str,
    object_names: Sequence[str],
    width: int,
    height: int,
) -> tuple[_SmokePlacement, ...]:
    if not object_names:
        object_names = ("object",)

    anchor_name = _anchor_name(object_names)
    support_index = next(
        (index for index, name in enumerate(object_names) if _canonical_name(name) == anchor_name),
        None,
    )

    counts: dict[str, int] = {}
    placements: list[_SmokePlacement] = []
    anchor_bbox: tuple[int, int, int, int] | None = None
    non_anchor_names = [
        name
        for index, name in enumerate(object_names)
        if support_index is None or index != support_index
    ]

    if anchor_name is not None:
        anchor_width = int(width * 0.62)
        anchor_height = int(height * 0.18)
        if anchor_name in {"shelf", "bookshelf", "cabinet"}:
            anchor_width = int(width * 0.34)
            anchor_height = int(height * 0.48)
        center_x = width // 2
        center_y = int(height * 0.67)
        if anchor_name in _CONTAINER_OBJECT_NAMES:
            anchor_width = int(width * 0.28)
            anchor_height = int(height * 0.22)
            center_y = int(height * 0.72)
        anchor_bbox = (
            center_x - (anchor_width // 2),
            center_y - (anchor_height // 2),
            center_x + (anchor_width // 2),
            center_y + (anchor_height // 2),
        )

        counts[anchor_name] = counts.get(anchor_name, 0) + 1
        instance_id = f"{scene_id}_{anchor_name}{counts[anchor_name]}_{anchor_name}{counts[anchor_name]}"
        placements.append(
            _SmokePlacement(
                instance_id=instance_id,
                label=anchor_name,
                bbox_xyxy=anchor_bbox,
                shape=_shape_for_name(anchor_name),
                color=_object_color(anchor_name),
            )
        )

    orbit_count = max(1, len(non_anchor_names))
    for index, name in enumerate(non_anchor_names, start=1):
        canonical = _canonical_name(name)
        counts[canonical] = counts.get(canonical, 0) + 1
        instance_id = f"{scene_id}_{canonical}{counts[canonical]}_{canonical}{counts[canonical]}"

        if anchor_bbox is not None:
            anchor_x0, anchor_y0, anchor_x1, anchor_y1 = anchor_bbox
            span = max(anchor_x1 - anchor_x0, 1)
            center_x = anchor_x0 + int(((index) / (orbit_count + 1)) * span)
            object_size = max(54, min(140, int(width * 0.09)))
            x0 = center_x - (object_size // 2)
            x1 = center_x + (object_size // 2)
            if anchor_name in _CONTAINER_OBJECT_NAMES:
                inner_width = max(int((anchor_x1 - anchor_x0) * 0.56), object_size + 20)
                center_x = ((anchor_x0 + anchor_x1) // 2) + int(
                    ((index - ((orbit_count + 1) / 2.0)) / max(orbit_count, 1)) * max(inner_width * 0.16, 1.0)
                )
                x0 = center_x - (object_size // 2)
                x1 = center_x + (object_size // 2)
                basket_floor_y = anchor_y1 - int((anchor_y1 - anchor_y0) * 0.22)
                y1 = min(anchor_y1 - 10, basket_floor_y)
                y0 = y1 - object_size
            else:
                top_y = anchor_y0 - int(object_size * 0.30)
                y0 = top_y - object_size
                y1 = top_y
        else:
            object_size = max(80, min(180, int(width * 0.13)))
            center_x = int(((index) / (orbit_count + 1)) * width)
            x0 = center_x - (object_size // 2)
            x1 = center_x + (object_size // 2)
            y1 = int(height * 0.78)
            y0 = y1 - object_size

        placements.append(
            _SmokePlacement(
                instance_id=instance_id,
                label=canonical,
                bbox_xyxy=(x0, y0, x1, y1),
                shape=_shape_for_name(canonical),
                color=_object_color(canonical, fallback_index=index - 1),
            )
        )

    return tuple(placements)


def _write_reference_layout_metadata(
    output_path: Path,
    *,
    scene_id: str,
    width: int,
    height: int,
    prompt_text: str | None,
    object_names: Sequence[str],
    placements: Sequence[_SmokePlacement],
) -> None:
    payload = {
        "scene_id": scene_id,
        "width": width,
        "height": height,
        "prompt_text": prompt_text,
        "requested_objects": list(object_names),
        "placements": [
            {
                "instance_id": placement.instance_id,
                "label": placement.label,
                "bbox_xyxy": list(placement.bbox_xyxy),
                "shape": placement.shape,
                "color": list(placement.color),
            }
            for placement in placements
        ],
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_reference_layout(reference_image: ArtifactRef) -> dict[str, object] | None:
    metadata_path = reference_image.metadata_path
    if not metadata_path:
        return None
    path = Path(metadata_path)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _placements_from_metadata(payload: dict[str, object]) -> tuple[_SmokePlacement, ...]:
    raw_placements = payload.get("placements")
    placements: list[_SmokePlacement] = []
    if not isinstance(raw_placements, list):
        return ()
    for index, raw in enumerate(raw_placements):
        if not isinstance(raw, dict):
            continue
        bbox = raw.get("bbox_xyxy")
        color = raw.get("color")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        if not isinstance(color, list) or len(color) != 3:
            color = list(_object_color(str(raw.get("label") or ""), fallback_index=index))
        placements.append(
            _SmokePlacement(
                instance_id=str(raw.get("instance_id") or f"instance_{index + 1}"),
                label=_canonical_name(str(raw.get("label") or "object")),
                bbox_xyxy=tuple(int(value) for value in bbox),
                shape=str(raw.get("shape") or _shape_for_name(str(raw.get("label") or "object"))),
                color=tuple(int(value) for value in color[:3]),
            )
        )
    return tuple(placements)


def _draw_shape(
    draw: ImageDraw.ImageDraw,
    *,
    bbox_xyxy: tuple[int, int, int, int],
    shape: str,
    fill: int | tuple[int, ...],
    outline: int | tuple[int, ...] | None = None,
) -> None:
    if shape == "ellipse":
        draw.ellipse(bbox_xyxy, fill=fill, outline=outline)
        return

    if shape == "container":
        x0, y0, x1, y1 = bbox_xyxy
        radius = max(12, min((x1 - x0) // 4, 28))
        draw.rounded_rectangle((x0, y0 + 10, x1, y1), radius=radius, fill=fill, outline=outline)
        rim_color = outline if outline is not None else fill
        draw.arc((x0, y0, x1, y0 + max(18, (y1 - y0) // 3)), start=0, end=180, fill=rim_color, width=4)
        return

    radius = 24 if shape == "support" else 18
    draw.rounded_rectangle(bbox_xyxy, radius=radius, fill=fill, outline=outline)
    if shape == "support":
        x0, y0, x1, y1 = bbox_xyxy
        leg_width = max(12, (x1 - x0) // 18)
        leg_height = max(24, (y1 - y0) // 2)
        if isinstance(fill, int):
            leg_fill = fill
        else:
            alpha = fill[3] if len(fill) > 3 else 255
            leg_fill = tuple(max(int(component) - 24, 0) for component in fill[:3]) + (alpha,)
        draw.rounded_rectangle(
            (x0 + leg_width, y1 - 8, x0 + (2 * leg_width), y1 + leg_height),
            radius=6,
            fill=leg_fill,
        )
        draw.rounded_rectangle(
            (x1 - (2 * leg_width), y1 - 8, x1 - leg_width, y1 + leg_height),
            radius=6,
            fill=leg_fill,
        )


def _rotation_to_y_axis() -> np.ndarray:
    return trimesh.transformations.rotation_matrix(-math.pi / 2.0, [1.0, 0.0, 0.0])


def _container_geometry(name: str) -> dict[str, float | tuple[float, float]] | None:
    canonical = _canonical_name(name)
    if canonical != "basket":
        return None
    outer_x, outer_y, outer_z = _primitive_dims(canonical)
    inner_x = max(outer_x - (2.0 * _BASKET_WALL_THICKNESS_M), 0.04)
    inner_z = max(outer_z - (2.0 * _BASKET_WALL_THICKNESS_M), 0.04)
    floor_top_offset_y = (-outer_y / 2.0) + _BASKET_BOTTOM_THICKNESS_M
    return {
        "inner_dims_xz": (inner_x, inner_z),
        "bottom_thickness": _BASKET_BOTTOM_THICKNESS_M,
        "floor_top_offset_y": floor_top_offset_y,
    }


def _basket_mesh(dims: tuple[float, float, float]) -> trimesh.Trimesh:
    outer_x, outer_y, outer_z = dims
    wall_thickness = min(_BASKET_WALL_THICKNESS_M, outer_x / 4.0, outer_z / 4.0)
    bottom_thickness = min(_BASKET_BOTTOM_THICKNESS_M, outer_y / 3.0)
    wall_height = max(outer_y - bottom_thickness, bottom_thickness)
    wall_center_y = (-outer_y / 2.0) + bottom_thickness + (wall_height / 2.0)

    floor = trimesh.creation.box(extents=(outer_x, bottom_thickness, outer_z))
    floor.apply_translation((0.0, (-outer_y / 2.0) + (bottom_thickness / 2.0), 0.0))

    left_wall = trimesh.creation.box(extents=(wall_thickness, wall_height, outer_z))
    left_wall.apply_translation((-(outer_x - wall_thickness) / 2.0, wall_center_y, 0.0))
    right_wall = left_wall.copy()
    right_wall.apply_translation((outer_x - wall_thickness, 0.0, 0.0))

    front_back_width = max(outer_x - (2.0 * wall_thickness), wall_thickness)
    front_wall = trimesh.creation.box(extents=(front_back_width, wall_height, wall_thickness))
    front_wall.apply_translation((0.0, wall_center_y, -(outer_z - wall_thickness) / 2.0))
    back_wall = front_wall.copy()
    back_wall.apply_translation((0.0, 0.0, outer_z - wall_thickness))

    return trimesh.util.concatenate((floor, left_wall, right_wall, front_wall, back_wall))


def _primitive_mesh_for_name(name: str) -> trimesh.Trimesh:
    canonical = _canonical_name(name)
    dims = _primitive_dims(canonical)

    if canonical in _CONTAINER_OBJECT_NAMES:
        return _basket_mesh(dims)

    if canonical in _ROUND_OBJECT_NAMES:
        radius = max(dims) / 2.0
        return trimesh.creation.icosphere(subdivisions=2, radius=radius)

    if canonical in _CYLINDER_OBJECT_NAMES:
        radius = max(dims[0], dims[2]) / 2.0
        mesh = trimesh.creation.cylinder(radius=radius, height=dims[1], sections=32)
        mesh.apply_transform(_rotation_to_y_axis())
        if canonical == "bowl":
            mesh.apply_scale((1.0, 0.65, 1.0))
        return mesh

    return trimesh.creation.box(extents=dims)


def _mesh_extents(asset_path: str) -> tuple[float, float, float]:
    mesh = trimesh.load(asset_path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    extents = np.asarray(mesh.extents, dtype=float)
    return float(extents[0]), float(extents[1]), float(extents[2])


def _relation_inventory(prompt_text: str) -> list[tuple[str, str]]:
    inventory: list[tuple[str, str]] = []
    matched_any = False
    for match in re.finditer(
        r"^- ([^:]+): canonical_name=([^;]+);",
        prompt_text,
        flags=re.MULTILINE,
    ):
        object_id = match.group(1).strip()
        canonical_name = _canonical_name(match.group(2))
        inventory.append((object_id, canonical_name))
        matched_any = True
    if matched_any:
        return inventory

    for match in re.finditer(
        r"^- ([^:]+): (.+)$",
        prompt_text,
        flags=re.MULTILINE,
    ):
        object_id = match.group(1).strip()
        if not object_id or object_id.lower() == "root objects":
            continue
        canonical_name = _canonical_name(match.group(2))
        if not canonical_name:
            continue
        inventory.append((object_id, canonical_name))
    return inventory


def _requests_white_background(prompt_text: str | None) -> bool:
    if not prompt_text:
        return False
    return "white background" in str(prompt_text).strip().lower()


class SmokeTextToImageProvider(TextToImageProvider):
    def __init__(
        self,
        *,
        output_root: str = "results/release_smoke/reference_images",
        canvas_size: Sequence[int] = (1280, 768),
        **_kwargs: object,
    ) -> None:
        width, height = tuple(canvas_size[:2]) if len(tuple(canvas_size[:2])) == 2 else (1280, 768)
        self._width = max(640, int(width))
        self._height = max(480, int(height))
        self._output_root = output_root

    def generate(self, request: SceneRequest) -> ReferenceImageResult:
        object_names = _resolve_requested_objects(request.requested_objects, prompt_text=request.text_prompt)
        placements = _instance_layouts(
            scene_id=request.scene_id,
            object_names=object_names,
            width=self._width,
            height=self._height,
        )
        output_dir = Path(self._output_root) / request.scene_id
        output_dir.mkdir(parents=True, exist_ok=True)
        image_path = output_dir / "reference.png"
        metadata_path = output_dir / "reference_layout.json"

        white_background = _requests_white_background(request.text_prompt)
        image = Image.new(
            "RGB",
            (self._width, self._height),
            color=(255, 255, 255) if white_background else (244, 240, 233),
        )
        draw = ImageDraw.Draw(image, "RGBA")
        if not white_background:
            horizon_y = int(self._height * 0.56)
            draw.rectangle((0, 0, self._width, horizon_y), fill=(242, 236, 229, 255))
            draw.rectangle((0, horizon_y, self._width, self._height), fill=(225, 215, 201, 255))
            draw.rounded_rectangle((28, 28, self._width - 28, self._height - 28), radius=36, outline=(84, 84, 84), width=3)
        if request.text_prompt and not white_background:
            draw.rounded_rectangle((42, 42, min(self._width - 42, 1040), 126), radius=22, fill=(255, 255, 255, 224))
            draw.text((64, 62), request.text_prompt, fill=(35, 35, 35))

        for placement in placements:
            fill = (*placement.color, 255)
            outline = (36, 36, 36, 220)
            _draw_shape(draw, bbox_xyxy=placement.bbox_xyxy, shape=placement.shape, fill=fill, outline=outline)
            label_box = (
                placement.bbox_xyxy[0],
                max(placement.bbox_xyxy[1] - 24, 0),
                placement.bbox_xyxy[0] + 170,
                placement.bbox_xyxy[1],
            )
            draw.rounded_rectangle(label_box, radius=10, fill=(255, 255, 255, 214))
            draw.text((label_box[0] + 10, label_box[1] + 4), _display_name(placement.label), fill=(30, 30, 30))

        image.save(image_path)
        _write_reference_layout_metadata(
            metadata_path,
            scene_id=request.scene_id,
            width=self._width,
            height=self._height,
            prompt_text=request.text_prompt,
            object_names=object_names,
            placements=placements,
        )

        return ReferenceImageResult(
            request=request,
            image=ArtifactRef(
                artifact_type="image",
                path=str(image_path),
                format="png",
                role="reference_image",
                metadata_path=str(metadata_path),
            ),
            generation_prompt=request.text_prompt,
            seed=0,
            width=self._width,
            height=self._height,
            metadata=make_stage_metadata(
                stage_name="reference_image",
                provider_name="smoke_text_to_image",
                notes=("release_smoke",),
            ),
        )


class SmokeDepthEstimator(DepthEstimator):
    def __init__(self, *, output_root: str = "results/release_smoke/depth", **_kwargs: object) -> None:
        self._output_root = output_root

    def predict(self, reference_image: ReferenceImageResult) -> DepthResult:
        image = Image.open(reference_image.image.path).convert("RGB")
        width, height = image.size
        depth = np.linspace(1.0, 0.28, num=height, dtype=np.float32)[:, None]
        depth = np.repeat(depth, width, axis=1)

        layout_payload = _load_reference_layout(reference_image.image) or {}
        placements = _placements_from_metadata(layout_payload)
        for placement in placements:
            x0, y0, x1, y1 = placement.bbox_xyxy
            local_depth = 0.62 if placement.label in _SUPPORT_SURFACE_NAMES else 0.38
            region = depth[max(y0, 0):max(y1, 0), max(x0, 0):max(x1, 0)]
            current_min = float(region.min()) if region.size else local_depth
            region[:] = min(local_depth, current_min)

        output_dir = Path(self._output_root) / reference_image.request.scene_id
        output_dir.mkdir(parents=True, exist_ok=True)
        depth_array_path = output_dir / "depth.npy"
        depth_visualization_path = output_dir / "depth.png"
        np.save(depth_array_path, depth)
        normalized = ((depth - float(depth.min())) / max(float(depth.max() - depth.min()), 1e-6) * 255.0).astype(np.uint8)
        Image.fromarray(normalized, mode="L").save(depth_visualization_path)

        return DepthResult(
            image=reference_image.image,
            depth_array=ArtifactRef("depth_array", str(depth_array_path), "npy", "depth_array", None),
            depth_visualization=ArtifactRef("image", str(depth_visualization_path), "png", "depth_visualization", None),
            point_cloud=None,
            focal_length_px=860.0,
            metadata=make_stage_metadata(
                stage_name="depth",
                provider_name="smoke_depth_estimator",
                notes=("release_smoke",),
            ),
        )


class SmokeSegmenter(Segmenter):
    def __init__(self, *, output_root: str = "results/release_smoke/segmentation", **_kwargs: object) -> None:
        self._output_root = output_root

    def segment(
        self,
        reference_image: ReferenceImageResult,
        object_hints: Sequence[str] | None = None,
    ) -> SegmentationResult:
        layout_payload = _load_reference_layout(reference_image.image)
        if layout_payload is None:
            image = Image.open(reference_image.image.path)
            placements = _instance_layouts(
                scene_id=reference_image.request.scene_id,
                object_names=_resolve_requested_objects(object_hints, prompt_text=reference_image.request.text_prompt),
                width=image.width,
                height=image.height,
            )
        else:
            placements = _placements_from_metadata(layout_payload)

        output_dir = Path(self._output_root) / reference_image.request.scene_id
        output_dir.mkdir(parents=True, exist_ok=True)
        overlay = Image.open(reference_image.image.path).convert("RGBA")
        overlay_draw = ImageDraw.Draw(overlay, "RGBA")

        instances: list[MaskInstance] = []
        for index, placement in enumerate(placements):
            mask = Image.new("L", (reference_image.width, reference_image.height), color=0)
            mask_draw = ImageDraw.Draw(mask)
            _draw_shape(
                mask_draw,
                bbox_xyxy=placement.bbox_xyxy,
                shape=placement.shape,
                fill=255,
            )
            mask_path = output_dir / f"{_safe_stem(placement.instance_id)}.png"
            mask.save(mask_path)
            overlay_fill = (*placement.color, 76)
            _draw_shape(
                overlay_draw,
                bbox_xyxy=placement.bbox_xyxy,
                shape=placement.shape,
                fill=overlay_fill,
                outline=(*placement.color, 180),
            )
            x0, y0, x1, y1 = placement.bbox_xyxy
            instances.append(
                MaskInstance(
                    instance_id=placement.instance_id,
                    label=placement.label,
                    mask=ArtifactRef("mask", str(mask_path), "png", "instance_mask", None),
                    bbox_xyxy=(float(x0), float(y0), float(x1), float(y1)),
                    confidence=1.0,
                    area_px=max((x1 - x0), 0) * max((y1 - y0), 0),
                )
            )

        composite_path = output_dir / "composite.png"
        overlay.save(composite_path)
        return SegmentationResult(
            image=reference_image.image,
            instances=tuple(instances),
            composite_visualization=ArtifactRef("image", str(composite_path), "png", "segmentation_overlay", None),
            metadata=make_stage_metadata(
                stage_name="segmentation",
                provider_name="smoke_segmenter",
                notes=("release_smoke",),
            ),
        )


class SmokeStructuredLLM(StructuredLLM):
    def __init__(self, **_kwargs: object) -> None:
        pass

    def generate(self, request: StructuredPromptRequest) -> StructuredPromptResult:
        payload: dict[str, object]
        if request.schema_name == "object_description":
            match = re.search(r"The object is ([a-zA-Z0-9_ -]+)\.", request.prompt_text)
            canonical_name = _canonical_name(match.group(1) if match else "object")
            payload = {
                "description": (
                    f"A stylized {_display_name(canonical_name)} placeholder asset for PAT3D release smoke tests."
                )
            }
        elif request.schema_name == "scene_relations":
            inventory = _relation_inventory(request.prompt_text)
            support_anchor = next(
                (object_id for object_id, canonical_name in inventory if canonical_name in _SUPPORT_SURFACE_NAMES),
                None,
            )
            container_anchor = next(
                (object_id for object_id, canonical_name in inventory if canonical_name in _CONTAINER_OBJECT_NAMES),
                None,
            )
            if support_anchor is None and container_anchor is None:
                payload = {}
            else:
                relations: list[dict[str, str]] = []
                if support_anchor is not None and container_anchor is not None and support_anchor != container_anchor:
                    relations.append(
                        {
                            "parent": support_anchor,
                            "child": container_anchor,
                            "relation": "supports",
                        }
                    )
                container_parent = container_anchor if container_anchor is not None else support_anchor
                support_parent = support_anchor if support_anchor is not None else container_anchor
                for object_id, canonical_name in inventory:
                    if object_id in {support_anchor, container_anchor}:
                        continue
                    if container_anchor is not None and canonical_name not in _SUPPORT_SURFACE_NAMES:
                        parent = container_parent
                        relation_name = "contains"
                    else:
                        parent = support_parent
                        relation_name = "supports"
                    if parent is None:
                        continue
                    relations.append(
                        {
                            "parent": parent,
                            "child": object_id,
                            "relation": relation_name,
                        }
                    )
                payload = {
                    "relations": relations,
                }
        elif request.schema_name == "size_prior":
            inventory = _relation_inventory(request.prompt_text)
            anchor = next(
                (
                    object_id
                    for object_id, canonical_name in inventory
                    if canonical_name in _SUPPORT_SURFACE_NAMES or canonical_name in _CONTAINER_OBJECT_NAMES
                ),
                None,
            )
            payload = {
                "scene_description": "Local PAT3D smoke-test tabletop scene.",
                "scene_scale_summary": "Furniture anchor plus small tabletop props.",
                "anchor_object_ids": [anchor] if anchor is not None else [],
                "objects": {
                    object_id: {
                        "dimensions_m": {
                            "x": dims[0],
                            "y": dims[1],
                            "z": dims[2],
                        }
                    }
                    for object_id, canonical_name in inventory
                    for dims in (_primitive_dims(canonical_name),)
                },
            }
        else:
            payload = {}

        return StructuredPromptResult(
            schema_name=request.schema_name,
            parsed_output=payload,
            metadata=make_stage_metadata(
                stage_name="structured_llm",
                provider_name="smoke_structured_llm",
                notes=("release_smoke",),
            ),
            raw_response_artifact=None,
        )


class PrimitiveTextTo3DProvider(TextTo3DProvider):
    def __init__(self, *, output_root: str = "results/release_smoke/assets", **_kwargs: object) -> None:
        self._output_root = output_root

    def generate(
        self,
        object_description: ObjectDescription,
        *,
        object_reference_image: ArtifactRef | None = None,
        scene_object_canonical_names: Sequence[str] = (),
        source_object_count: int = 1,
    ) -> GeneratedObjectAsset:
        _ = scene_object_canonical_names, source_object_count
        scene_id = object_description.object_id.split(":", 1)[0]
        output_dir = Path(self._output_root) / scene_id
        output_dir.mkdir(parents=True, exist_ok=True)
        mesh_path = output_dir / f"{_safe_stem(object_description.object_id)}.obj"
        preview_path = output_dir / f"{_safe_stem(object_description.object_id)}.png"

        mesh = _primitive_mesh_for_name(object_description.canonical_name)
        mesh.export(mesh_path)

        preview = Image.new(
            "RGB",
            (256, 256),
            color=_object_color(object_description.canonical_name),
        )
        preview_draw = ImageDraw.Draw(preview)
        preview_draw.rounded_rectangle((18, 18, 238, 238), radius=26, outline=(30, 30, 30), width=3)
        preview_draw.text((32, 104), _display_name(object_description.canonical_name), fill=(255, 255, 255))
        preview.save(preview_path)

        return GeneratedObjectAsset(
            object_id=object_description.object_id,
            mesh_obj=ArtifactRef("mesh_obj", str(mesh_path), "obj", "mesh", None),
            preview_image=(
                object_reference_image
                or ArtifactRef("image", str(preview_path), "png", "object_preview", None)
            ),
            provider_asset_id=f"primitive:{object_description.object_id}",
            metadata=make_stage_metadata(
                stage_name="object_asset_generation",
                provider_name="primitive_text_to_3d",
                notes=("release_smoke",),
            ),
        )


class HeuristicLayoutBuilder:
    def __init__(
        self,
        *,
        layout_root: str = "results/release_smoke/layout",
        support_surface_clearance: float = 0.02,
        ground_drop_clearance: float = 0.22,
        **_kwargs: object,
    ) -> None:
        self._layout_root = layout_root
        self._support_surface_clearance = max(0.0, float(support_surface_clearance))
        self._ground_drop_clearance = max(0.0, float(ground_drop_clearance))

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
        size_priors: Sequence[object] | None = None,
    ) -> SceneLayout:
        _ = reference_image_result, segmentation_result, depth_result, size_priors
        scene_dir = Path(self._layout_root) / scene_id
        scene_dir.mkdir(parents=True, exist_ok=True)

        canonical_by_id: dict[str, str] = {}
        if object_catalog is not None:
            for detected_object in object_catalog.objects:
                canonical_by_id[detected_object.object_id] = detected_object.canonical_name
                for source_instance_id in detected_object.source_instance_ids:
                    canonical_by_id[source_instance_id] = detected_object.canonical_name

        extents_by_id = {
            asset.object_id: _mesh_extents(asset.mesh_obj.path)
            for asset in object_assets.assets
        }
        relation_by_child = (
            {
                relation.child_object_id: relation
                for relation in relation_graph.relations
            }
            if relation_graph is not None
            else {}
        )
        relation_children_by_parent: dict[str, list[str]] = {}
        for child_id, relation in relation_by_child.items():
            relation_children_by_parent.setdefault(relation.parent_object_id, []).append(child_id)
        support_assets = [
            asset
            for asset in object_assets.assets
            if canonical_by_id.get(asset.object_id, _canonical_name(asset.object_id.split(":")[-1]))
            in _SUPPORT_SURFACE_NAMES
        ]
        if support_assets:
            ordered_support_assets = support_assets
        else:
            ordered_support_assets = [
                asset
                for asset in object_assets.assets
                if asset.object_id in relation_children_by_parent
                and canonical_by_id.get(asset.object_id, _canonical_name(asset.object_id.split(":")[-1]))
                in _CONTAINER_OBJECT_NAMES
            ]
        ordered_support_assets = ordered_support_assets or object_assets.assets[:1]

        poses: list[ObjectPose] = []
        anchor_pose_by_id: dict[str, tuple[float, float, float, float, float, float]] = {}
        spacing_cursor = 0.0
        for index, asset in enumerate(ordered_support_assets):
            extents = extents_by_id[asset.object_id]
            width, height, depth = extents
            canonical_name = canonical_by_id.get(asset.object_id, _canonical_name(asset.object_id.split(":")[-1]))
            x = spacing_cursor
            y = height / 2.0
            if canonical_name not in _SUPPORT_SURFACE_NAMES:
                y += self._ground_drop_clearance
            z = 0.0
            spacing_cursor += width + 0.65
            poses.append(
                ObjectPose(
                    object_id=asset.object_id,
                    translation_xyz=(x, y, z),
                    rotation_type="quaternion",
                    rotation_value=(1.0, 0.0, 0.0, 0.0),
                    scale_xyz=(1.0, 1.0, 1.0),
                )
            )
            anchor_pose_by_id[asset.object_id] = (x, y, z, width, height, depth)

        primary_anchor = ordered_support_assets[0] if ordered_support_assets else None
        supportees = [asset for asset in object_assets.assets if asset.object_id not in anchor_pose_by_id]
        supportees.sort(
            key=lambda asset: (
                0 if asset.object_id in relation_children_by_parent else 1,
                asset.object_id,
            )
        )
        orbit_total = max(1, len(supportees))
        for orbit_index, asset in enumerate(supportees, start=1):
            extents = extents_by_id[asset.object_id]
            width, height, depth = extents
            relation = relation_by_child.get(asset.object_id)
            parent_id = relation.parent_object_id if relation is not None else None
            relation_type = (
                relation.relation_type.value
                if relation is not None and hasattr(relation.relation_type, "value")
                else str(relation.relation_type).lower()
                if relation is not None
                else ""
            )
            if parent_id is not None and parent_id in anchor_pose_by_id:
                anchor_id = parent_id
            elif primary_anchor is not None:
                anchor_id = primary_anchor.object_id
            else:
                anchor_id = None

            if anchor_id is None:
                x = (orbit_index - ((orbit_total + 1) / 2.0)) * (width + 0.35)
                y = height / 2.0
                z = 0.0
            else:
                anchor_x, anchor_y, anchor_z, anchor_width, anchor_height, anchor_depth = anchor_pose_by_id[anchor_id]
                anchor_canonical = canonical_by_id.get(anchor_id, _canonical_name(anchor_id.split(":")[-1]))
                if relation_type in {"contains", "in"} and anchor_canonical in _CONTAINER_OBJECT_NAMES:
                    container_geometry = _container_geometry(anchor_canonical)
                    inner_dims_xz = (
                        tuple(container_geometry.get("inner_dims_xz", (anchor_width, anchor_depth)))
                        if container_geometry is not None
                        else (anchor_width, anchor_depth)
                    )
                    floor_top_offset_y = (
                        float(container_geometry.get("floor_top_offset_y", -anchor_height / 2.0))
                        if container_geometry is not None
                        else (-anchor_height / 2.0)
                    )
                    siblings = relation_children_by_parent.get(anchor_id, [asset.object_id])
                    sibling_index = siblings.index(asset.object_id) + 1 if asset.object_id in siblings else orbit_index
                    sibling_total = max(1, len(siblings))
                    x_offset = (
                        (sibling_index - ((sibling_total + 1) / 2.0)) / max(sibling_total, 1)
                    ) * min(float(inner_dims_xz[0]) * 0.24, 0.06)
                    z_offset = (0.04 if sibling_index % 2 == 0 else -0.04) * min(float(inner_dims_xz[1]) / max(anchor_depth, 1e-6), 1.0)
                    x = anchor_x + x_offset
                    y = anchor_y + floor_top_offset_y + (height / 2.0) + self._support_surface_clearance
                    z = anchor_z + z_offset
                else:
                    x_offset = ((orbit_index - ((orbit_total + 1) / 2.0)) / max(orbit_total, 1)) * min(anchor_width * 0.42, 0.48)
                    z_offset = (0.08 if orbit_index % 2 == 0 else -0.08) * min(anchor_depth, 1.0)
                    x = anchor_x + x_offset
                    y = (
                        anchor_y
                        + (anchor_height / 2.0)
                        + (height / 2.0)
                        + self._support_surface_clearance
                    )
                    z = anchor_z + z_offset

            poses.append(
                ObjectPose(
                    object_id=asset.object_id,
                    translation_xyz=(x, y, z),
                    rotation_type="quaternion",
                    rotation_value=(1.0, 0.0, 0.0, 0.0),
                    scale_xyz=(1.0, 1.0, 1.0),
                )
            )
            anchor_pose_by_id[asset.object_id] = (x, y, z, width, height, depth)

        summary_path = scene_dir / "layout_summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "scene_id": scene_id,
                    "object_ids": [pose.object_id for pose in poses],
                    "anchor_ids": list(anchor_pose_by_id),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        return SceneLayout(
            scene_id=scene_id,
            object_poses=tuple(poses),
            layout_space="world",
            support_graph=relation_graph,
            artifacts=(
                ArtifactRef("layout_summary", str(summary_path), "json", "layout_summary", None),
            ),
            metadata=make_stage_metadata(
                stage_name="layout_initialization",
                provider_name="heuristic_layout_builder",
                notes=("release_smoke",),
            ),
        )
