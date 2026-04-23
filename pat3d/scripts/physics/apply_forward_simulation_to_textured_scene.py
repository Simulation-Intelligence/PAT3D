#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Iterable

import numpy as np


WORKSPACE_ROOT = Path("results/workspaces")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_vertices(obj_path: Path) -> np.ndarray:
    vertices: list[list[float]] = []
    with obj_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("v "):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.asarray(vertices, dtype=np.float64)


def _best_fit_rigid_transform(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, str]:
    if len(source) == 0 or len(target) == 0:
        raise ValueError("cannot estimate transform from an empty mesh")

    if len(source) != len(target):
        source_center = source.mean(axis=0)
        target_center = target.mean(axis=0)
        rotation = np.eye(3, dtype=np.float64)
        translation = target_center - source_center
        aligned = source + translation
        rms_error = float(np.sqrt(np.mean(np.sum((aligned - target.mean(axis=0)) ** 2, axis=1))))
        return rotation, translation, rms_error, "centroid_translation_only"

    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    source_zero = source - source_center
    target_zero = target - target_center
    covariance = source_zero.T @ target_zero
    u_matrix, _singular_values, vt_matrix = np.linalg.svd(covariance)
    rotation = vt_matrix.T @ u_matrix.T
    if np.linalg.det(rotation) < 0:
        vt_matrix[-1, :] *= -1
        rotation = vt_matrix.T @ u_matrix.T
    translation = target_center - rotation @ source_center
    aligned = (rotation @ source.T).T + translation
    rms_error = float(np.sqrt(np.mean(np.sum((aligned - target) ** 2, axis=1))))
    return rotation, translation, rms_error, "kabsch_rigid"


def _format_float(value: float) -> str:
    return f"{value:.10g}"


def _transform_obj(source_path: Path, output_path: Path, rotation: np.ndarray, translation: np.ndarray) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with source_path.open("r", encoding="utf-8", errors="ignore") as source, output_path.open(
        "w",
        encoding="utf-8",
    ) as target:
        for line in source:
            if line.startswith("v "):
                parts = line.rstrip("\n").split()
                if len(parts) >= 4:
                    xyz = np.asarray([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float64)
                    transformed = rotation @ xyz + translation
                    target.write(
                        " ".join(
                            [
                                parts[0],
                                _format_float(float(transformed[0])),
                                _format_float(float(transformed[1])),
                                _format_float(float(transformed[2])),
                                *parts[4:],
                            ]
                        )
                        + "\n"
                    )
                    continue
            if line.startswith("vn "):
                parts = line.rstrip("\n").split()
                if len(parts) >= 4:
                    normal = np.asarray([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float64)
                    transformed = rotation @ normal
                    norm = np.linalg.norm(transformed)
                    if norm > 0:
                        transformed = transformed / norm
                    target.write(
                        " ".join(
                            [
                                parts[0],
                                _format_float(float(transformed[0])),
                                _format_float(float(transformed[1])),
                                _format_float(float(transformed[2])),
                                *parts[4:],
                            ]
                        )
                        + "\n"
                    )
                    continue
            target.write(line)


def _read_obj_mtllibs(obj_path: Path) -> tuple[str, ...]:
    mtllibs: list[str] = []
    try:
        source = obj_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ()
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("mtllib "):
            mtllibs.extend(part for part in stripped.split()[1:] if part)
    return tuple(mtllibs)


def _read_mtl_texture_refs(mtl_path: Path) -> tuple[str, ...]:
    refs: list[str] = []
    try:
        source = mtl_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ()
    for line in source.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith(("map_Kd ", "map_Ka ", "map_d ", "bump ", "map_Bump ")):
            parts = stripped.split()
            if len(parts) >= 2:
                refs.append(parts[-1])
    return tuple(refs)


def _copy_if_exists(source: Path, target: Path) -> Path | None:
    if not source.exists():
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != target.resolve():
        shutil.copy2(source, target)
    return target


def _copy_material_sidecars(source_obj: Path, output_obj: Path) -> tuple[Path | None, Path | None]:
    copied_mtl: Path | None = None
    copied_texture: Path | None = None
    mtllibs = _read_obj_mtllibs(source_obj)

    if mtllibs:
        for mtllib in mtllibs:
            source_mtl = source_obj.parent / mtllib
            target_mtl = output_obj.parent / mtllib
            copied = _copy_if_exists(source_mtl, target_mtl)
            if copied is not None and copied_mtl is None:
                copied_mtl = copied
            if copied is None:
                continue
            for texture_ref in _read_mtl_texture_refs(source_mtl):
                copied_texture = _copy_if_exists(source_mtl.parent / texture_ref, target_mtl.parent / texture_ref) or copied_texture
    else:
        source_mtl = source_obj.with_suffix(".mtl")
        target_mtl = output_obj.with_suffix(".mtl")
        copied_mtl = _copy_if_exists(source_mtl, target_mtl)
        if copied_mtl is not None:
            for texture_ref in _read_mtl_texture_refs(source_mtl):
                copied_texture = _copy_if_exists(source_mtl.parent / texture_ref, target_mtl.parent / texture_ref) or copied_texture

    sibling_png = source_obj.with_suffix(".png")
    copied_texture = _copy_if_exists(sibling_png, output_obj.with_suffix(".png")) or copied_texture
    return copied_mtl, copied_texture


def _load_alias_map(layout_scene_dir: Path) -> dict[str, str]:
    alias_path = layout_scene_dir / "alias_map.json"
    if not alias_path.exists():
        return {}
    with alias_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {str(key): str(value) for key, value in payload.items()}


def _candidate_high_meshes(layout_scene_dir: Path, object_id: str, alias_map: dict[str, str]) -> Iterable[Path]:
    seen: set[Path] = set()
    names = []
    if object_id in alias_map:
        names.append(alias_map[object_id])
    if ":" in object_id:
        names.append(object_id.split(":", 1)[1])
    names.append(object_id)
    for name in names:
        candidate = layout_scene_dir / f"{name}.obj"
        if candidate in seen:
            continue
        seen.add(candidate)
        yield candidate


def _resolve_high_mesh(layout_scene_dir: Path, object_id: str, alias_map: dict[str, str]) -> Path:
    for candidate in _candidate_high_meshes(layout_scene_dir, object_id, alias_map):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"could not find a high-poly layout mesh for {object_id}")


def _workspace_scene_root(scene_name: str) -> Path:
    return WORKSPACE_ROOT / scene_name


def _default_layout_scene_dir(scene_name: str) -> Path:
    return _workspace_scene_root(scene_name) / "layout" / "scene" / scene_name


def _default_initial_sim_dir(scene_name: str) -> Path:
    return _workspace_scene_root(scene_name) / "simulation" / "low_poly" / scene_name


def _default_output_scene_dir(scene_name: str) -> Path:
    return _workspace_scene_root(scene_name) / "visualization" / "simulated_scene"


def _select_simulated_dir(
    scene_name: str,
    initial_sim_dir: Path,
    output_scene_dir: Path,
    explicit_dir: Path | None,
) -> tuple[Path, bool, str]:
    if explicit_dir is not None:
        return explicit_dir, explicit_dir.resolve() == initial_sim_dir.resolve(), "explicit"

    workspace_scene_root = _workspace_scene_root(scene_name)
    candidates = [
        workspace_scene_root / "physics" / "forward_sim_layer_layout" / scene_name,
        workspace_scene_root / "physics" / "diff_sim_layer_layout" / scene_name,
        output_scene_dir / "low_poly",
        output_scene_dir / "low_poly_simulated",
        output_scene_dir,
        Path("vis/simulated_scene") / scene_name,
        Path("data/forward_sim_layer_layout") / scene_name,
    ]
    for candidate in candidates:
        if candidate.exists() and any(candidate.glob("*.obj")):
            return candidate, candidate.resolve() == initial_sim_dir.resolve(), "auto_detected"
    return initial_sim_dir, True, "identity_fallback"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply low-poly forward-simulation rigid motion to high-poly textured layout meshes.",
    )
    parser.add_argument("--scene-name", required=True)
    parser.add_argument(
        "--layout-folder",
        default="",
        help="Legacy layout root. Defaults to results/workspaces/<scene>/layout/scene/<scene> when omitted.",
    )
    parser.add_argument(
        "--initial-sim-folder",
        default="",
        help="Legacy low-poly root. Defaults to results/workspaces/<scene>/simulation/low_poly/<scene> when omitted.",
    )
    parser.add_argument("--simulated-sim-dir")
    parser.add_argument(
        "--output-root",
        default="",
        help="Root directory for exported scenes. Defaults to results/workspaces/<scene>/visualization/simulated_scene when omitted.",
    )
    parser.add_argument("--object-id", action="append", dest="object_ids")
    parser.add_argument(
        "--fail-on-identity-fallback",
        action="store_true",
        help="Fail instead of using initial low-poly meshes as a placeholder when no simulated low-poly output is found.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    scene_name = args.scene_name
    layout_scene_dir = (
        Path(args.layout_folder) / scene_name
        if args.layout_folder
        else _default_layout_scene_dir(scene_name)
    )
    initial_sim_dir = (
        Path(args.initial_sim_folder) / scene_name
        if args.initial_sim_folder
        else _default_initial_sim_dir(scene_name)
    )
    output_scene_dir = (
        Path(args.output_root) / scene_name
        if args.output_root
        else _default_output_scene_dir(scene_name)
    )
    explicit_sim_dir = Path(args.simulated_sim_dir) if args.simulated_sim_dir else None
    simulated_sim_dir, identity_fallback, simulated_dir_source = _select_simulated_dir(
        scene_name,
        initial_sim_dir,
        output_scene_dir,
        explicit_sim_dir,
    )

    if args.fail_on_identity_fallback and identity_fallback:
        raise RuntimeError(
            "no simulated low-poly output was found; pass --simulated-sim-dir or rerun without "
            "--fail-on-identity-fallback to export an identity-applied high-poly scene"
        )

    if not layout_scene_dir.exists():
        raise FileNotFoundError(f"layout scene directory does not exist: {layout_scene_dir}")
    if not initial_sim_dir.exists():
        raise FileNotFoundError(f"initial simulation mesh directory does not exist: {initial_sim_dir}")
    if not simulated_sim_dir.exists():
        raise FileNotFoundError(f"simulated simulation mesh directory does not exist: {simulated_sim_dir}")

    alias_map = _load_alias_map(layout_scene_dir)
    requested_object_ids = set(args.object_ids or [])
    output_mesh_dir = output_scene_dir / "high_poly_textured"
    output_mesh_dir.mkdir(parents=True, exist_ok=True)

    object_results: list[dict[str, object]] = []
    bundle_objects: list[dict[str, object]] = []
    warnings: list[str] = []

    for initial_sim_path in sorted(initial_sim_dir.glob("*.obj")):
        object_id = initial_sim_path.stem
        if requested_object_ids and object_id not in requested_object_ids:
            continue
        simulated_path = simulated_sim_dir / initial_sim_path.name
        if not simulated_path.exists():
            warnings.append(f"missing simulated low-poly mesh for {object_id}: {simulated_path}")
            continue
        high_mesh_path = _resolve_high_mesh(layout_scene_dir, object_id, alias_map)
        output_name = alias_map.get(object_id, high_mesh_path.stem)
        output_mesh_path = output_mesh_dir / f"{output_name}.obj"

        initial_vertices = _load_vertices(initial_sim_path)
        simulated_vertices = _load_vertices(simulated_path)
        rotation, translation, rms_error, method = _best_fit_rigid_transform(initial_vertices, simulated_vertices)
        _transform_obj(high_mesh_path, output_mesh_path, rotation, translation)
        copied_mtl, copied_texture = _copy_material_sidecars(high_mesh_path, output_mesh_path)

        transform_matrix = np.eye(4, dtype=np.float64)
        transform_matrix[:3, :3] = rotation
        transform_matrix[:3, 3] = translation
        object_result = {
            "object_id": object_id,
            "initial_low_poly_mesh": str(initial_sim_path),
            "simulated_low_poly_mesh": str(simulated_path),
            "source_high_poly_mesh": str(high_mesh_path),
            "output_high_poly_mesh": str(output_mesh_path),
            "output_mtl": str(copied_mtl) if copied_mtl is not None else None,
            "output_texture": str(copied_texture) if copied_texture is not None else None,
            "transform_estimation_method": method,
            "vertex_count_initial_low_poly": int(len(initial_vertices)),
            "vertex_count_simulated_low_poly": int(len(simulated_vertices)),
            "rms_fit_error": rms_error,
            "transform_matrix": transform_matrix.tolist(),
        }
        object_results.append(object_result)
        bundle_objects.append(
            {
                "object_id": object_id,
                "mesh_obj_path": str(output_mesh_path),
                "mesh_mtl_path": str(copied_mtl) if copied_mtl is not None else None,
                "texture_image_path": str(copied_texture) if copied_texture is not None else None,
                "already_transformed": True,
                "transform": transform_matrix.tolist(),
            }
        )

    if not object_results:
        raise RuntimeError("no objects were exported")

    scene_bundle = {
        "scene_id": scene_name,
        "generated_at": _now_iso(),
        "source_scene_type": "forward_simulation_applied_high_poly_textured_scene",
        "objects": bundle_objects,
    }
    bundle_path = output_scene_dir / "scene_bundle.json"
    bundle_path.write_text(json.dumps(scene_bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    metadata = {
        "scene_name": scene_name,
        "generated_at": _now_iso(),
        "layout_scene_dir": str(layout_scene_dir),
        "initial_low_poly_dir": str(initial_sim_dir),
        "simulated_low_poly_dir": str(simulated_sim_dir),
        "simulated_low_poly_dir_source": simulated_dir_source,
        "identity_fallback_used": identity_fallback,
        "output_mesh_dir": str(output_mesh_dir),
        "scene_bundle": str(bundle_path),
        "objects": object_results,
        "warnings": warnings,
    }
    metadata_path = output_scene_dir / "apply_forward_simulation_result.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Exported {len(object_results)} high-poly object mesh(es) to {output_mesh_dir}")
    print(f"Scene bundle: {bundle_path}")
    print(f"Metadata: {metadata_path}")
    if identity_fallback:
        print("WARNING: identity fallback was used because no simulated low-poly output was found.")
    for warning in warnings:
        print(f"WARNING: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
