from __future__ import annotations

from pathlib import Path
import shutil

from PIL import Image

from pat3d.models import ArtifactRef, ReferenceImageResult


def materialize_legacy_named_image(
    *,
    source_path: str,
    target_stem: str,
    target_root: str,
) -> Path:
    target_dir = Path(target_root)
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / f"{target_stem}.png"
    _write_png(Path(source_path), target_path)
    return target_path


def materialize_legacy_scene_image(
    reference_image: ReferenceImageResult,
    *,
    target_root: str,
) -> Path:
    return materialize_legacy_named_image(
        source_path=reference_image.image.path,
        target_stem=reference_image.request.scene_id,
        target_root=target_root,
    )


def materialize_legacy_object_reference_image(
    *,
    scene_id: str,
    object_name: str,
    object_reference_image: ArtifactRef,
    target_root: str,
) -> Path:
    target_dir = Path(target_root) / scene_id
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / f"{object_name}.png"
    _write_png(Path(object_reference_image.path), target_path)
    return target_path


def remove_legacy_object_reference_image(
    *,
    scene_id: str,
    object_name: str,
    target_root: str,
) -> None:
    target_path = Path(target_root) / scene_id / f"{object_name}.png"
    if target_path.exists():
        target_path.unlink()


def _write_png(source_path: Path, target_path: Path) -> None:
    if not source_path.exists():
        raise FileNotFoundError(source_path)

    try:
        if source_path.resolve() == target_path.resolve():
            return
    except FileNotFoundError:
        pass

    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_suffix(f"{target_path.suffix}.tmp")
    if source_path.suffix.lower() == ".png":
        shutil.copyfile(source_path, tmp_path)
    else:
        with Image.open(source_path) as image:
            mode = "RGBA" if "A" in image.getbands() else "RGB"
            image.convert(mode).save(tmp_path, format="PNG")
    tmp_path.replace(target_path)
