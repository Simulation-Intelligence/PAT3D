from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import sys
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[4]
T2V_REPO = REPO_ROOT / "extern" / "t2v_metrics"
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


def _normalize_image_paths(image_paths: Sequence[str | Path]) -> list[str]:
    normalized: list[str] = []
    for image_path in image_paths:
        path = Path(image_path).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"VQAScore image does not exist: {path}")
        normalized.append(str(path))
    if not normalized:
        raise ValueError("at least one image is required for VQAScore")
    return normalized


def compute_t2v_vqa_score(
    image_paths: Sequence[str | Path],
    prompt: str,
    *,
    model: str | None = None,
    device: str | None = None,
    cache_dir: str | Path | None = None,
) -> dict[str, Any]:
    if not prompt or not prompt.strip():
        raise ValueError("prompt must be non-empty for VQAScore")

    normalized_paths = _normalize_image_paths(image_paths)
    resolved_model = model or os.environ.get("PAT3D_VQA_MODEL") or DEFAULT_MODEL
    resolved_cache_dir = Path(
        cache_dir or os.environ.get("PAT3D_T2V_HF_CACHE_DIR") or DEFAULT_CACHE_DIR
    ).expanduser()
    resolved_cache_dir.mkdir(parents=True, exist_ok=True)

    _add_t2v_metrics_to_path()
    import torch
    import t2v_metrics

    resolved_device = device or os.environ.get("PAT3D_VQA_DEVICE") or (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    if resolved_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "Strict VQAScore requires CUDA for CLIP-FlanT5 in this setup, but torch cannot see a CUDA device."
        )
    if resolved_device == "cpu" and resolved_model.startswith("clip-flant5"):
        raise RuntimeError(
            "CLIP-FlanT5 VQAScore is configured for GPU execution. "
            "Run via the dashboard server or set PAT3D_VQA_PYTHON to a CUDA-visible environment."
        )

    score_model = t2v_metrics.VQAScore(
        model=resolved_model,
        device=resolved_device,
        cache_dir=str(resolved_cache_dir),
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
        "cache_dir": str(resolved_cache_dir),
        "image_paths": normalized_paths,
        "prompt": prompt,
        "backend": "t2v_metrics_vqascore",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run strict t2v_metrics VQAScore.")
    parser.add_argument("images", nargs="+", help="Rendered image path(s).")
    parser.add_argument("--prompt", required=True, help="Text prompt for the scene.")
    parser.add_argument("--model", default=None, help=f"VQAScore model. Defaults to {DEFAULT_MODEL}.")
    parser.add_argument("--device", default=None, help="Torch device, e.g. cuda or cuda:0.")
    parser.add_argument("--cache-dir", default=None, help="Hugging Face cache dir for t2v_metrics models.")
    args = parser.parse_args(argv)

    captured_stdout = io.StringIO()
    with contextlib.redirect_stdout(captured_stdout):
        result = compute_t2v_vqa_score(
            args.images,
            args.prompt,
            model=args.model,
            device=args.device,
            cache_dir=args.cache_dir,
        )
    captured = captured_stdout.getvalue().strip()
    if captured:
        result["stdout"] = captured
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
