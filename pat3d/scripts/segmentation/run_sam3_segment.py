from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pat3d.providers.sam3_segmentation import _repo_path, materialize_segmentation_outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-json", required=True)
    return parser


def run(payload: dict[str, Any]) -> dict[str, Any]:
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    import torch

    requested_device = payload.get("device")
    device = str(requested_device).strip() if requested_device else None
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint_path = payload.get("checkpoint_path")
    if isinstance(checkpoint_path, str) and checkpoint_path.strip():
        checkpoint_path = str(_repo_path(checkpoint_path))
    else:
        checkpoint_path = None

    confidence_threshold = float(payload.get("confidence_threshold", 0.5))
    model = build_sam3_image_model(
        device=device,
        checkpoint_path=checkpoint_path,
        load_from_HF=bool(payload.get("load_from_hf", True)),
    )
    processor = Sam3Processor(
        model,
        device=device,
        confidence_threshold=confidence_threshold,
    )

    image_path = _repo_path(str(payload["reference_image_path"]))
    image_rgb = Image.open(image_path).convert("RGB")
    image_array = np.asarray(image_rgb)
    prompts = [str(item).strip() for item in payload.get("prompts", ()) if str(item).strip()]
    def sam3_autocast_context():
        if str(device).startswith("cuda"):
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    with sam3_autocast_context():
        state = processor.set_image(image_rgb)
    prompt_masks: list[dict[str, Any]] = []

    for prompt in prompts:
        if hasattr(processor, "reset_all_prompts"):
            processor.reset_all_prompts(state)
        with sam3_autocast_context():
            state = processor.set_text_prompt(prompt=prompt, state=state)
        prompt_masks.append(
            {
                "label": prompt,
                "masks": state.get("masks", ()),
                "scores": state.get("scores", ()),
            }
        )

    result = materialize_segmentation_outputs(
        scene_name=str(payload["scene_name"]),
        image_array=image_array,
        prompt_masks=prompt_masks,
        output_dir=str(payload.get("output_dir") or "data/seg"),
    )
    result["device"] = device
    return result


def main() -> int:
    args = build_parser().parse_args()
    input_path = Path(args.input_json)
    output_path = Path(args.output_json)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    result = run(payload)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
