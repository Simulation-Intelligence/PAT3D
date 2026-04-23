from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[4]
SEMANTICS_ROOT = Path(__file__).resolve().parents[1]
if str(SEMANTICS_ROOT) not in sys.path:
    sys.path.insert(0, str(SEMANTICS_ROOT))

from physical_plausibility_score import score_physical_plausibility


def compute_physical_plausibility_score(
    image_paths: Sequence[str | Path],
    prompt: str,
    *,
    model: str | None = None,
) -> dict:
    return score_physical_plausibility(
        image_paths,
        prompt,
        model=model,
        return_details=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compute physical plausibility score for rendered images.")
    parser.add_argument("images", nargs="+", help="Rendered image path(s).")
    parser.add_argument("--prompt", required=True, help="Text prompt for the scene.")
    parser.add_argument("--model", default=None, help="OpenAI model. Defaults to OPENAI_MODEL or gpt-4o.")
    parser.add_argument("--output", help="Optional JSON output path.")
    args = parser.parse_args(argv)

    result = compute_physical_plausibility_score(args.images, args.prompt, model=args.model)
    if args.output:
        output_path = Path(args.output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
