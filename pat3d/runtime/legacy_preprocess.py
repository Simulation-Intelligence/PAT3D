from __future__ import annotations

from argparse import Namespace
from collections.abc import Mapping, Sequence
from glob import glob
from pathlib import Path
from typing import Any

from pat3d.legacy_config import load_root_parse_options
from pat3d.models import ArtifactRef, LegacyPreprocessResult, SceneRequest, StageRunStatus
from pat3d.providers._legacy_image_inputs import materialize_legacy_named_image
from pat3d.storage import make_stage_metadata


def build_legacy_preprocess_args(
    scene_request: SceneRequest,
    *,
    items: Sequence[str] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> Namespace:
    parser = load_root_parse_options()(return_parser=True)
    args = parser.parse_args([])
    args.preprocess = True
    args.scene_name = scene_request.scene_id
    args.text_prompt = scene_request.text_prompt

    resolved_items = tuple(items) if items is not None else scene_request.requested_objects
    args.items = list(resolved_items) if resolved_items else None

    for key, value in dict(overrides or {}).items():
        setattr(args, key, value)

    if scene_request.reference_image is not None:
        staged_reference_image = materialize_legacy_named_image(
            source_path=scene_request.reference_image.path,
            target_stem=scene_request.scene_id,
            target_root=str(Path(getattr(args, "ref_image_folder", "data/ref_img"))),
        )
        args.ref_image_folder = str(staged_reference_image.parent)

    return args


def preview_legacy_preprocess(
    scene_request: SceneRequest,
    *,
    items: Sequence[str] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> LegacyPreprocessResult:
    args = build_legacy_preprocess_args(
        scene_request,
        items=items,
        overrides=overrides,
    )
    return preview_legacy_preprocess_from_args(args)


def run_legacy_preprocess(
    scene_request: SceneRequest,
    *,
    items: Sequence[str] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> LegacyPreprocessResult:
    args = build_legacy_preprocess_args(
        scene_request,
        items=items,
        overrides=overrides,
    )
    return run_legacy_preprocess_from_args(args)


def run_legacy_preprocess_from_args(args: Namespace) -> LegacyPreprocessResult:
    from pat3d.preprocess import Preprocessor

    preprocessor = Preprocessor(args)
    preprocessor.preprocess()
    return build_legacy_preprocess_result(args)


def preview_legacy_preprocess_from_args(args: Namespace) -> LegacyPreprocessResult:
    return build_legacy_preprocess_result(
        args,
        include_missing=True,
        status=StageRunStatus.PENDING,
        notes=("validate_only",),
    )


def build_legacy_preprocess_result(
    args: Namespace,
    *,
    include_missing: bool = False,
    status: StageRunStatus = StageRunStatus.COMPLETED,
    notes: Sequence[str] = (),
) -> LegacyPreprocessResult:
    return LegacyPreprocessResult(
        scene_id=args.scene_name,
        text_prompt=args.text_prompt,
        metadata=make_stage_metadata(
            stage_name="legacy_preprocess",
            provider_name="legacy_preprocessor",
            status=status,
            notes=notes,
        ),
        requested_objects=tuple(args.items) if args.items else None,
        artifacts=collect_legacy_preprocess_artifacts(
            args,
            include_missing=include_missing,
        ),
    )


def collect_legacy_preprocess_artifacts(
    args: Namespace,
    *,
    include_missing: bool = False,
) -> tuple[ArtifactRef, ...]:
    if not args.scene_name:
        return ()

    scene_name = args.scene_name
    artifacts: list[ArtifactRef] = []

    reference_image_path = _resolve_reference_image_path(
        args.ref_image_folder,
        scene_name,
        include_missing=include_missing,
    )
    if reference_image_path is not None:
        artifacts.append(
            _make_artifact_ref(
                reference_image_path,
                artifact_type="image",
                role="reference_image",
            )
        )

    for role, artifact_type, path, format_name in _artifact_specs(args, scene_name):
        candidate = Path(path)
        if include_missing or candidate.exists():
            artifacts.append(
                _make_artifact_ref(
                    candidate,
                    artifact_type=artifact_type,
                    role=role,
                    format_name=format_name,
                )
            )

    return tuple(artifacts)


def _artifact_specs(
    args: Namespace, scene_name: str
) -> tuple[tuple[str, str, Path, str], ...]:
    return (
        ("items_json", "scene_items", Path(args.items_folder) / f"{scene_name}.json", "json"),
        ("depth_dir", "depth", Path(args.depth_folder) / scene_name, "directory"),
        ("segmentation_dir", "segmentation", Path(args.seg_folder) / scene_name, "directory"),
        (
            "object_descriptions",
            "object_descriptions",
            Path(args.descrip_folder) / f"{scene_name}.json",
            "json",
        ),
        (
            "object_reference_images_dir",
            "image_directory",
            Path(args.ref_image_obj_folder) / scene_name,
            "directory",
        ),
        ("raw_objects_dir", "mesh_directory", Path(args.raw_obj_folder) / scene_name, "directory"),
        (
            "organized_objects_dir",
            "mesh_directory",
            Path(args.organized_obj_folder) / scene_name,
            "directory",
        ),
        ("layout_dir", "layout", Path(args.layout_folder) / scene_name, "directory"),
        ("low_poly_dir", "mesh_directory", Path(args.low_poly_folder) / scene_name, "directory"),
        ("size_json", "size_priors", Path(args.size_folder) / f"{scene_name}.json", "json"),
        (
            "contain_json",
            "contain_relations",
            Path(args.contain_folder) / f"{scene_name}.json",
            "json",
        ),
        (
            "contain_on_json",
            "contain_relations",
            Path(args.contain_on_folder) / f"{scene_name}.json",
            "json",
        ),
        ("bbox_put_dir", "layout", Path(args.bbox_put_folder) / scene_name, "directory"),
        (
            "preprocess_timing_log",
            "log",
            Path("results") / "preprocess_timing" / f"{scene_name}.txt",
            "txt",
        ),
    )


def _resolve_reference_image_path(
    ref_image_folder: str,
    scene_name: str,
    *,
    include_missing: bool,
) -> Path | None:
    base_dir = Path(ref_image_folder)
    for suffix in ("png", "jpg", "jpeg", "webp"):
        matches = glob(str(base_dir / f"{scene_name}.{suffix}"))
        if matches:
            return Path(matches[0])
    if include_missing:
        return base_dir / f"{scene_name}.png"
    return None


def _make_artifact_ref(
    path: Path,
    *,
    artifact_type: str,
    role: str,
    format_name: str | None = None,
) -> ArtifactRef:
    if format_name is None:
        if path.is_dir():
            format_name = "directory"
        else:
            format_name = path.suffix.lower().lstrip(".") or "file"
    return ArtifactRef(
        artifact_type=artifact_type,
        path=str(path),
        format=format_name,
        role=role,
    )
