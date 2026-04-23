from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[4]
T2V_REPO = REPO_ROOT / "extern" / "t2v_metrics"
VQA_RUNNER = Path(__file__).resolve().parent / "run_t2v_vqascore.py"
DEFAULT_MODEL = "clip-flant5-xl"
DEFAULT_CACHE_DIR = REPO_ROOT / "data" / "metrics" / "hf_cache"


def _add_t2v_metrics_to_path() -> None:
    configured = os.environ.get("PAT3D_T2V_METRICS_ROOT")
    candidates = [Path(configured).expanduser()] if configured else []
    candidates.append(T2V_REPO)
    for candidate in candidates:
        if candidate.exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
            return


def _common_vqa_python_candidates(home_dir: Path | None = None) -> list[Path]:
    home = home_dir or Path.home()
    return [
        REPO_ROOT / ".venv-t2v" / "bin" / "python",
        home / "anaconda3" / "envs" / "sam3d-objects" / "bin" / "python",
        home / ".conda" / "envs" / "sam3d-objects" / "bin" / "python",
        home / "miniconda3" / "envs" / "sam3d-objects" / "bin" / "python",
    ]


def _inline_vqa_environment_supported() -> bool:
    required_modules = ("torch", "transformers", "sentencepiece")
    if any(importlib.util.find_spec(name) is None for name in required_modules):
        return False
    try:
        import transformers
    except Exception:
        return False
    version = str(getattr(transformers, "__version__", ""))
    major = version.split(".", 1)[0]
    try:
        return int(major) < 5
    except ValueError:
        return False


def _resolve_external_vqa_python() -> Path | None:
    configured = os.environ.get("PAT3D_VQA_PYTHON", "").strip()
    candidates = [Path(configured).expanduser()] if configured else []
    candidates.extend(_common_vqa_python_candidates())
    current_python = Path(sys.executable).expanduser().resolve() if sys.executable else None
    seen: set[str] = set()
    for candidate in candidates:
        resolved = candidate.expanduser()
        if not resolved.exists():
            continue
        try:
            normalized = str(resolved.resolve())
        except OSError:
            normalized = str(resolved)
        if normalized in seen:
            continue
        seen.add(normalized)
        if current_python is not None and resolved.resolve() == current_python and _inline_vqa_environment_supported():
            continue
        return resolved
    return None


def _tensor_to_float_list(values: Any) -> list[float]:
    if hasattr(values, "detach"):
        values = values.detach().cpu()
    if hasattr(values, "flatten"):
        values = values.flatten()
    if hasattr(values, "tolist"):
        values = values.tolist()
    if not isinstance(values, list):
        values = [values]
    flattened: list[float] = []
    for value in values:
        if isinstance(value, list):
            flattened.extend(float(item) for item in value)
        else:
            flattened.append(float(value))
    return flattened


def _run_external_t2v_vqa_score(
    image_paths: Sequence[str | Path],
    prompt: str,
    *,
    python_executable: str | Path,
    model: str,
) -> dict[str, Any]:
    command = [
        str(Path(python_executable).expanduser()),
        str(VQA_RUNNER),
        *[str(path) for path in image_paths],
        "--prompt",
        prompt,
        "--model",
        model,
        "--cache-dir",
        str(Path(os.environ.get("PAT3D_T2V_HF_CACHE_DIR") or DEFAULT_CACHE_DIR)),
    ]
    env = {
        **os.environ,
        "PAT3D_VQA_EXTERNAL_CHILD": "1",
        "PYTHONPATH": os.pathsep.join(
            [
                str(T2V_REPO),
                os.environ.get("PYTHONPATH", ""),
            ]
        ).strip(os.pathsep),
    }
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Strict t2v_metrics VQAScore failed in external Python "
            f"{python_executable}: {(completed.stderr or completed.stdout).strip()}"
        )
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Invalid JSON from external VQAScore: {completed.stdout[:1000]}") from error


def compute_vqa_score(
    image_paths: Sequence[str | Path],
    prompt: str,
    *,
    model: str | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    del api_key
    if not prompt or not prompt.strip():
        raise ValueError("prompt must be non-empty for VQAScore")
    normalized_paths = [str(Path(path).expanduser()) for path in image_paths]
    if not normalized_paths:
        raise ValueError("at least one image is required for VQAScore")

    _add_t2v_metrics_to_path()
    resolved_model = model or os.environ.get("PAT3D_VQA_MODEL") or DEFAULT_MODEL
    external_python = _resolve_external_vqa_python()
    should_use_external = os.environ.get("PAT3D_VQA_EXTERNAL_CHILD") != "1" and (
        os.environ.get("PAT3D_VQA_PYTHON")
        or not _inline_vqa_environment_supported()
    )
    if should_use_external and external_python is not None:
        return _run_external_t2v_vqa_score(
            normalized_paths,
            prompt,
            python_executable=external_python,
            model=resolved_model,
        )

    try:
        import torch
        import t2v_metrics
    except Exception as error:
        raise RuntimeError(
            "Strict VQAScore requires t2v_metrics and its CLIP-FlanT5 dependencies. "
            "Set PAT3D_VQA_PYTHON to the configured VQAScore environment or install the dependencies."
        ) from error

    resolved_device = os.environ.get("PAT3D_VQA_DEVICE") or (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    if resolved_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Strict VQAScore requested CUDA, but torch cannot see a CUDA device.")
    if resolved_device == "cpu" and resolved_model.startswith("clip-flant5"):
        raise RuntimeError(
            "CLIP-FlanT5 VQAScore is configured for GPU execution. "
            "Set PAT3D_VQA_PYTHON to the configured CUDA-visible VQAScore environment."
        )

    cache_dir = Path(os.environ.get("PAT3D_T2V_HF_CACHE_DIR") or DEFAULT_CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    score_model = t2v_metrics.VQAScore(
        model=resolved_model,
        device=resolved_device,
        cache_dir=str(cache_dir),
    )
    raw_scores = score_model(normalized_paths, [prompt])
    scores = _tensor_to_float_list(raw_scores)
    mean_score = sum(scores) / len(scores)
    return {
        "score": mean_score,
        "mean": mean_score,
        "per_image": scores,
        "model": resolved_model,
        "device": resolved_device,
        "cache_dir": str(cache_dir),
        "image_paths": normalized_paths,
        "prompt": prompt,
        "backend": "t2v_metrics_vqascore",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compute VQAScore for rendered images.")
    parser.add_argument("images", nargs="+", help="Rendered image path(s).")
    parser.add_argument("--prompt", required=True, help="Text prompt for the scene.")
    parser.add_argument("--model", default=None, help="VQAScore model name.")
    parser.add_argument("--api-key", default=None, help="Optional API key for GPT-backed VQAScore.")
    parser.add_argument("--output", help="Optional JSON output path.")
    args = parser.parse_args(argv)

    result = compute_vqa_score(args.images, args.prompt, model=args.model, api_key=args.api_key)
    if args.output:
        output_path = Path(args.output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
