from __future__ import annotations

import argparse
import functools
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[4]
SEMANTICS_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "metrics" / "renders"
BLENDER_SCRIPT = SEMANTICS_ROOT / "vqa" / "render" / "blender_script_multi_objs.py"
EXPECTED_RENDER_COUNT = 18


def _now_missing_error(path: Path) -> FileNotFoundError:
    return FileNotFoundError(f"Required path does not exist: {path}")


def resolve_blender_bin(blender_bin: str | Path | None = None) -> Path:
    candidates = [
        blender_bin,
        os.environ.get("PAT3D_BLENDER_BIN"),
        shutil.which("blender"),
        "/usr/local/blender/blender",
        "/home/eleven/guying/blender-4.3.0-linux-x64/blender",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if path.exists():
            return path
    raise FileNotFoundError(
        "Blender executable not found. Set PAT3D_BLENDER_BIN or add blender to PATH."
    )


def _rendered_images(output_dir: Path) -> list[Path]:
    return sorted(output_dir.glob("render_*.png"))


def _dedupe_paths(paths: Sequence[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for raw_path in paths:
        path = str(raw_path or "").strip()
        if not path or path in seen:
            continue
        seen.add(path)
        ordered.append(path)
    return ordered


@functools.lru_cache(maxsize=1)
def _system_python_site_packages() -> tuple[str, ...]:
    override = os.environ.get("PAT3D_BLENDER_PYTHONPATH", "").strip()
    if override:
        return tuple(
            path
            for path in _dedupe_paths(override.split(os.pathsep))
            if Path(path).exists()
        )

    python_candidates = _dedupe_paths(
        [
            os.environ.get("PAT3D_BLENDER_SYSTEM_PYTHON", ""),
            shutil.which("python3") or "",
            shutil.which("python") or "",
        ]
    )
    query = (
        "import json, site; "
        "paths = []; "
        "getter = getattr(site, 'getsitepackages', None); "
        "paths.extend(getter() if callable(getter) else []); "
        "user_site = getattr(site, 'getusersitepackages', lambda: '')(); "
        "paths.extend(user_site if isinstance(user_site, list) else [user_site]); "
        "print(json.dumps(paths))"
    )
    for python_bin in python_candidates:
        try:
            completed = subprocess.run(
                [python_bin, "-c", query],
                text=True,
                capture_output=True,
                check=False,
            )
        except OSError:
            continue
        if completed.returncode != 0:
            continue
        try:
            paths = json.loads(completed.stdout or "[]")
        except json.JSONDecodeError:
            continue
        cleaned = tuple(
            path
            for path in _dedupe_paths(paths if isinstance(paths, list) else [])
            if Path(path).exists()
        )
        if cleaned:
            return cleaned
    return ()


def _blender_env() -> dict[str, str]:
    env = dict(os.environ)
    python_paths = list(_system_python_site_packages())
    existing_pythonpath = env.get("PYTHONPATH", "")
    if existing_pythonpath:
        python_paths.extend(existing_pythonpath.split(os.pathsep))
    merged = _dedupe_paths(python_paths)
    if merged:
        env["PYTHONPATH"] = os.pathsep.join(merged)
    return env


def _summarize_blender_failure(stdout: str, stderr: str) -> str | None:
    ignored_prefixes = (
        "Blender ",
        "Blender quit",
        "Read prefs:",
        "Read blend:",
        "Info:",
        "xcb_connection_has_error()",
    )
    for stream in (stderr, stdout):
        lines = [line.strip() for line in stream.splitlines() if line.strip()]
        if not lines:
            continue
        traceback_index = next((index for index, line in enumerate(lines) if line.startswith("Traceback")), None)
        relevant_lines = lines[traceback_index + 1 :] if traceback_index is not None else lines
        for line in reversed(relevant_lines):
            if line.startswith(ignored_prefixes):
                continue
            return line
    return None


def render_scene_views(
    scene_dir: str | Path,
    case_id: str,
    *,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    blender_bin: str | Path | None = None,
    force: bool = False,
    resolution: int = 1024,
    samples: int = 64,
    representative_index: int = 6,
) -> dict[str, Any]:
    scene_dir = Path(scene_dir).expanduser()
    if not scene_dir.is_absolute():
        scene_dir = (Path.cwd() / scene_dir).resolve()
    if not scene_dir.exists():
        raise _now_missing_error(scene_dir)
    if not any(scene_dir.glob("*.obj")):
        raise FileNotFoundError(f"No OBJ files were found for rendering in {scene_dir}")

    output_dir = Path(output_root).expanduser() / case_id
    if not output_dir.is_absolute():
        output_dir = (Path.cwd() / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    existing_images = _rendered_images(output_dir)
    completed: subprocess.CompletedProcess[str] | None = None
    log_path = output_dir / "render_log.txt"
    if force or len(existing_images) < 18:
        for stale_file in existing_images:
            stale_file.unlink()
        blender = resolve_blender_bin(blender_bin)
        command = [
            str(blender),
            "--background",
            "--python-use-system-env",
            "--python",
            str(BLENDER_SCRIPT),
            "--",
            "--object_path",
            str(scene_dir),
            "--output_dir",
            str(output_dir),
            "--camera_preset",
            "semantic_18",
            "--num_renders",
            str(EXPECTED_RENDER_COUNT),
            "--resolution",
            str(int(resolution)),
            "--samples",
            str(int(samples)),
        ]
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=_blender_env(),
            text=True,
            capture_output=True,
            check=False,
        )
        log_path.write_text(
            "\n".join(
                [
                    f"command: {' '.join(command)}",
                    f"returncode: {completed.returncode}",
                    "",
                    "[stdout]",
                    completed.stdout or "",
                    "",
                    "[stderr]",
                    completed.stderr or "",
                ]
            ),
            encoding="utf-8",
        )
        if completed.returncode != 0:
            summary = _summarize_blender_failure(completed.stdout, completed.stderr)
            raise RuntimeError(
                f"Blender render failed with exit code {completed.returncode}"
                f"{': ' + summary if summary else ''}. See {log_path}"
            )

    image_paths = _rendered_images(output_dir)
    if len(image_paths) < EXPECTED_RENDER_COUNT:
        summary = None
        if completed is not None:
            summary = _summarize_blender_failure(completed.stdout, completed.stderr)
        elif log_path.exists():
            log_text = log_path.read_text(encoding="utf-8")
            summary = _summarize_blender_failure(log_text, log_text)
        if summary:
            raise RuntimeError(
                "Blender render script exited without producing metric views: "
                f"{summary}. See {log_path}"
            )
        raise RuntimeError(
            f"Expected {EXPECTED_RENDER_COUNT} rendered images, found {len(image_paths)} in {output_dir}"
        )

    representative_index = max(0, min(int(representative_index), len(image_paths) - 1))
    return {
        "case_id": case_id,
        "scene_dir": str(scene_dir),
        "output_dir": str(output_dir),
        "image_paths": [str(path) for path in image_paths],
        "image_count": len(image_paths),
        "representative_index": representative_index,
        "representative_image": str(image_paths[representative_index]),
        "view_sampling": {
            "depression_angles_degrees": [0, 20, 45],
            "horizontal_view_count": 6,
            "total_view_count": 18,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render 18 metric views for a PAT3D scene directory.")
    parser.add_argument("--scene-dir", required=True, help="Directory containing per-object layout OBJ files.")
    parser.add_argument("--case-id", required=True, help="Case id used as the render output folder name.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Root folder for rendered metric images.")
    parser.add_argument("--blender-bin", default=None, help="Optional Blender executable path.")
    parser.add_argument("--force", action="store_true", help="Re-render even if 18 images already exist.")
    parser.add_argument("--resolution", type=int, default=1024, help="Square render resolution.")
    parser.add_argument("--samples", type=int, default=64, help="Cycles samples.")
    parser.add_argument("--representative-index", type=int, default=6, help="Image index shown in the dashboard.")
    args = parser.parse_args(argv)

    result = render_scene_views(
        args.scene_dir,
        args.case_id,
        output_root=args.output_root,
        blender_bin=args.blender_bin,
        force=args.force,
        resolution=args.resolution,
        samples=args.samples,
        representative_index=args.representative_index,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
