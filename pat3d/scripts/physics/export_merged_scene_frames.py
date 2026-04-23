from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import trimesh


def _frame_number(path: Path) -> int:
    match = re.search(r"(\d+)$", path.stem)
    return int(match.group(1)) if match else 0


def _load_mesh(mesh_path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(mesh_path, force="mesh")
    if isinstance(loaded, trimesh.Scene):
        return loaded.dump(concatenate=True)
    return loaded.copy()


def _load_frame_transforms(frame_path: Path) -> dict[str, np.ndarray]:
    transforms = np.load(frame_path, allow_pickle=True)
    return {
        key: np.asarray(transforms[key], dtype=np.float64)
        for key in transforms.files
    }


def export_merged_scene_frames(
    *,
    frames_dir: Path,
    meshes_dir: Path,
    output_dir: Path,
    start_frame: int | None = None,
    end_frame: int | None = None,
    frame_step: int = 1,
    always_include_end: bool = False,
) -> dict[str, Any]:
    frame_paths = sorted(frames_dir.glob("frame_*.npz"), key=_frame_number)
    if start_frame is not None:
        frame_paths = [path for path in frame_paths if _frame_number(path) >= start_frame]
    if end_frame is not None:
        frame_paths = [path for path in frame_paths if _frame_number(path) <= end_frame]
    final_frame_path = frame_paths[-1] if frame_paths else None
    if frame_step > 1:
        frame_paths = frame_paths[::frame_step]
    if always_include_end and final_frame_path is not None and final_frame_path not in frame_paths:
        frame_paths.append(final_frame_path)
        frame_paths.sort(key=_frame_number)
    if not frame_paths:
        raise RuntimeError(f"no frame transforms found in {frames_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    mesh_cache: dict[str, trimesh.Trimesh] = {}
    manifest_frames: list[dict[str, Any]] = []

    for frame_path in frame_paths:
        frame_meshes: list[trimesh.Trimesh] = []
        transforms = _load_frame_transforms(frame_path)
        for object_id, transform in sorted(transforms.items()):
            mesh_path = meshes_dir / f"{object_id}.obj"
            if not mesh_path.is_file():
                raise FileNotFoundError(
                    f"missing mesh for object '{object_id}': expected {mesh_path}"
                )
            cached = mesh_cache.get(object_id)
            if cached is None:
                cached = _load_mesh(mesh_path)
                mesh_cache[object_id] = cached
            mesh = cached.copy()
            mesh.apply_transform(transform)
            frame_meshes.append(mesh)

        merged = trimesh.util.concatenate(frame_meshes)
        export_name = f"{frame_path.stem}_merged.obj"
        export_path = output_dir / export_name
        merged.export(export_path)
        manifest_frames.append(
            {
                "frame": _frame_number(frame_path),
                "transform_file": frame_path.name,
                "merged_obj": export_name,
                "object_count": len(transforms),
            }
        )

    manifest = {
        "frames_dir": str(frames_dir),
        "meshes_dir": str(meshes_dir),
        "output_dir": str(output_dir),
        "frame_count": len(manifest_frames),
        "start_frame": manifest_frames[0]["frame"],
        "end_frame": manifest_frames[-1]["frame"],
        "frame_step": frame_step,
        "always_include_end": bool(always_include_end),
        "frames": manifest_frames,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _cli() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--meshes-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--always-include-end", action="store_true")
    args = parser.parse_args()

    result = export_merged_scene_frames(
        frames_dir=Path(args.frames_dir),
        meshes_dir=Path(args.meshes_dir),
        output_dir=Path(args.output_dir),
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        frame_step=max(1, args.frame_step),
        always_include_end=bool(args.always_include_end),
    )
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
