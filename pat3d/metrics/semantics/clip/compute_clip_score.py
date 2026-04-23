from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


def _as_float_list(values: Any) -> list[float]:
    if hasattr(values, "detach"):
        values = values.detach().cpu().flatten().tolist()
    if not isinstance(values, list):
        values = [values]
    return [float(value) for value in values]


def compute_clip_score(
    image_paths: Sequence[str | Path],
    prompt: str,
    *,
    model_name: str = "openai/clip-vit-base-patch16",
) -> dict[str, Any]:
    if not prompt or not prompt.strip():
        raise ValueError("prompt must be non-empty for CLIP score")
    normalized_paths = [str(Path(path).expanduser()) for path in image_paths]
    if not normalized_paths:
        raise ValueError("at least one image is required for CLIP score")

    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()

    scores: list[float] = []
    with torch.no_grad():
        for image_path in normalized_paths:
            image = Image.open(image_path).convert("RGB")
            inputs = processor(
                text=[prompt],
                images=image,
                return_tensors="pt",
                padding=True,
            ).to(device)
            outputs = model(**inputs)
            image_features = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
            text_features = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
            cosine = (image_features * text_features).sum(dim=-1)
            scores.extend(_as_float_list(torch.clamp(cosine * 100.0, min=0.0)))

    mean_score = sum(scores) / len(scores)
    return {
        "score": mean_score,
        "mean": mean_score,
        "per_image": scores,
        "model": model_name,
        "image_paths": normalized_paths,
        "prompt": prompt,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compute CLIPScore for rendered images.")
    parser.add_argument("images", nargs="+", help="Rendered image path(s).")
    parser.add_argument("--prompt", required=True, help="Text prompt for the scene.")
    parser.add_argument("--model", default="openai/clip-vit-base-patch16", help="HuggingFace CLIP model.")
    parser.add_argument("--output", help="Optional JSON output path.")
    args = parser.parse_args(argv)

    result = compute_clip_score(args.images, args.prompt, model_name=args.model)
    if args.output:
        output_path = Path(args.output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
