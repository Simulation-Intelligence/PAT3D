from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROMPT_TEMPLATE_PATH = (
    Path(__file__).resolve().parent / "phys_plausibility" / "prompt_description.txt"
)
DEFAULT_API_KEY_PATH = REPO_ROOT / "pat3d" / "preprocessing" / "gpt_utils" / "apikey.txt"
DEFAULT_MODEL = "gpt-4o"


def _load_openai_sdk() -> tuple[type[Exception], Any]:
    try:
        from openai import BadRequestError, OpenAI
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "The openai Python package is required for physical plausibility scoring. "
            "Run this script with the dashboard/PAT3D Python environment or install openai."
        ) from error
    return BadRequestError, OpenAI


def _load_repo_dotenv() -> None:
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$", stripped)
        if match is None:
            continue
        key, value = match.groups()
        if key in os.environ:
            continue
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]
        if value:
            os.environ[key] = value


def _read_text_file(path: str | Path | None) -> str:
    if path is None:
        return ""
    return Path(path).expanduser().read_text(encoding="utf-8").strip()


def _resolve_api_key(api_key: str | None, api_key_path: str | Path | None) -> str | None:
    if api_key:
        return api_key
    if os.environ.get("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"]

    candidate_path = Path(api_key_path).expanduser() if api_key_path else DEFAULT_API_KEY_PATH
    if candidate_path.exists():
        key = candidate_path.read_text(encoding="utf-8").strip()
        return key or None
    return None


def _normalize_image_paths(image_paths: str | Path | Sequence[str | Path]) -> list[Path]:
    if isinstance(image_paths, (str, Path)):
        raw_paths: Sequence[str | Path] = [image_paths]
    else:
        raw_paths = image_paths

    normalized: list[Path] = []
    for raw_path in raw_paths:
        image_path = Path(raw_path).expanduser()
        if not image_path.is_absolute():
            image_path = (Path.cwd() / image_path).resolve()
        if not image_path.exists():
            raise FileNotFoundError(f"Rendered image does not exist: {image_path}")
        if not image_path.is_file():
            raise ValueError(f"Rendered image path is not a file: {image_path}")
        normalized.append(image_path)

    if not normalized:
        raise ValueError("At least one rendered image path is required.")
    return normalized


def _image_to_data_url(image_path: Path) -> str:
    mime_type, _encoding = mimetypes.guess_type(str(image_path))
    if mime_type is None:
        suffix = image_path.suffix.lower().lstrip(".")
        mime_type = "image/jpeg" if suffix in {"jpg", "jpeg"} else "image/png"

    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _build_prompt(
    *,
    scene_description: str | None,
    prompt_template_path: str | Path | None,
) -> str:
    template_path = Path(prompt_template_path or DEFAULT_PROMPT_TEMPLATE_PATH)
    template = template_path.read_text(encoding="utf-8").strip()
    description = (scene_description or "").strip()
    if not description:
        description = "not provided; evaluate only the visible physical relationships."

    prompt = template
    replaced = False
    for placeholder in ("‘...‘", "'...'", '"..."', "..."):
        if placeholder in prompt:
            prompt = prompt.replace(placeholder, description, 1)
            replaced = True
            break
    if not replaced:
        prompt = f"{prompt}\n\nScene semantic description: {description}"

    return (
        f"{prompt}\n\n"
        "If multiple images are provided, treat them as different rendered views of the same scene. "
        "Judge support, contact, gravity, penetration, floating objects, and obvious instability. "
        "Return exactly one JSON object with this schema: "
        '{"score": <number from 0 to 100>, "reasoning": "<brief reason>"}'
    )


def _supports_non_default_temperature(model: str) -> bool:
    return not str(model).strip().lower().startswith("gpt-5")


def _supports_reasoning_effort(model: str) -> bool:
    return str(model).strip().lower().startswith("gpt-5")


def _create_completion_with_compatible_token_limit(
    client: Any,
    *,
    bad_request_error_type: type[Exception],
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    reasoning_effort: str | None,
) -> Any:
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "response_format": {"type": "json_object"},
        "seed": 100,
    }
    if _supports_non_default_temperature(model):
        kwargs["temperature"] = 0
    if reasoning_effort and _supports_reasoning_effort(model):
        kwargs["reasoning_effort"] = reasoning_effort

    payload_variants = (
        {"max_completion_tokens": max_tokens},
        {"max_tokens": max_tokens},
        {},
    )
    last_error: Exception | None = None
    for index, payload in enumerate(payload_variants):
        try:
            return client.chat.completions.create(**{**kwargs, **payload})
        except bad_request_error_type as error:
            last_error = error
            detail = str(error).lower()
            unsupported = "unsupported parameter" in detail
            if not unsupported:
                raise
            if "max_completion_tokens" not in detail and "max_tokens" not in detail:
                raise
            if index + 1 >= len(payload_variants):
                raise
            continue
    raise last_error


def _parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    candidates = [stripped]
    fenced_blocks = re.findall(
        r"```(?:json)?\s*(.*?)```",
        stripped,
        flags=re.DOTALL | re.IGNORECASE,
    )
    candidates.extend(block for block in fenced_blocks if block.strip())

    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if match is not None:
        candidates.append(match.group(0))

    for candidate in candidates:
        payload = candidate.strip()
        if not payload:
            continue
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict):
            raise ValueError("GPT response JSON must be an object.")
        return parsed

    raise ValueError(f"GPT response is not a JSON object: {stripped[:200]}")


def _extract_score(parsed_response: dict[str, Any]) -> float:
    for key in (
        "score",
        "physical_plausibility_score",
        "phys_plausibility_score",
        "plausibility_score",
        "final_score",
    ):
        if key in parsed_response:
            raw_score = parsed_response[key]
            break
    else:
        raise ValueError(f"GPT response is missing a score field: {parsed_response}")

    if isinstance(raw_score, str):
        match = re.search(r"-?\d+(?:\.\d+)?", raw_score)
        if match is None:
            raise ValueError(f"Score is not numeric: {raw_score!r}")
        score = float(match.group(0))
    else:
        score = float(raw_score)

    if not 0 <= score <= 100:
        raise ValueError(f"Physical plausibility score must be between 0 and 100, got {score}")
    return score


def score_physical_plausibility(
    image_paths: str | Path | Sequence[str | Path],
    scene_description: str | None = None,
    *,
    prompt_template_path: str | Path | None = None,
    model: str | None = None,
    api_key: str | None = None,
    api_key_path: str | Path | None = None,
    base_url: str | None = None,
    reasoning_effort: str | None = None,
    max_tokens: int = 500,
    return_details: bool = False,
) -> float | dict[str, Any]:
    _load_repo_dotenv()
    normalized_paths = _normalize_image_paths(image_paths)
    resolved_model = model or os.environ.get("OPENAI_MODEL") or DEFAULT_MODEL
    resolved_api_key = _resolve_api_key(api_key, api_key_path)
    if not resolved_api_key:
        raise ValueError(
            "OPENAI_API_KEY is not set and no API key file was found. "
            "Set OPENAI_API_KEY or pass api_key/api_key_path."
        )

    BadRequestError, OpenAI = _load_openai_sdk()
    client_kwargs: dict[str, Any] = {"api_key": resolved_api_key}
    resolved_base_url = base_url or os.environ.get("OPENAI_BASE_URL")
    if resolved_base_url:
        client_kwargs["base_url"] = resolved_base_url
    client = OpenAI(**client_kwargs)

    prompt = _build_prompt(
        scene_description=scene_description,
        prompt_template_path=prompt_template_path,
    )
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for image_path in normalized_paths:
        content.append({"type": "image_url", "image_url": {"url": _image_to_data_url(image_path)}})

    response = _create_completion_with_compatible_token_limit(
        client,
        bad_request_error_type=BadRequestError,
        model=resolved_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict evaluator for physical plausibility in rendered 3D scenes. "
                    "Return only JSON."
                ),
            },
            {"role": "user", "content": content},
        ],
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
    )
    raw_text = response.choices[0].message.content or ""
    parsed_response = _parse_json_object(raw_text)
    score = _extract_score(parsed_response)

    if not return_details:
        return score
    return {
        "score": score,
        "model": resolved_model,
        "image_paths": [str(path) for path in normalized_paths],
        "scene_description": scene_description or "",
        "prompt_template_path": str(prompt_template_path or DEFAULT_PROMPT_TEMPLATE_PATH),
        "parsed_response": parsed_response,
        "raw_response": raw_text,
    }


def _load_scene_description(args: argparse.Namespace) -> str:
    if args.scene_description_path:
        return _read_text_file(args.scene_description_path)
    return args.scene_description or ""


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score physical plausibility for one or more rendered scene images with GPT."
    )
    parser.add_argument("images", nargs="+", help="Rendered image path(s) for the same scene.")
    parser.add_argument(
        "--scene-description",
        default="",
        help="Semantic text description of the scene.",
    )
    parser.add_argument(
        "--scene-description-path",
        help="Path to a text file containing the semantic scene description.",
    )
    parser.add_argument(
        "--prompt-template-path",
        default=str(DEFAULT_PROMPT_TEMPLATE_PATH),
        help="Prompt template path.",
    )
    parser.add_argument("--model", default=None, help="OpenAI chat model. Defaults to OPENAI_MODEL or gpt-4o.")
    parser.add_argument("--api-key-path", default=None, help="Optional OpenAI API key file path.")
    parser.add_argument("--base-url", default=None, help="Optional OpenAI-compatible API base URL.")
    parser.add_argument("--reasoning-effort", default=None, help="Optional reasoning effort for GPT-5 models.")
    parser.add_argument("--max-tokens", type=int, default=500, help="Maximum output tokens.")
    parser.add_argument("--json", action="store_true", help="Print a detailed JSON result instead of only the score.")
    parser.add_argument("--output", help="Optional path to write the detailed JSON result.")
    args = parser.parse_args(argv)

    details = score_physical_plausibility(
        args.images,
        _load_scene_description(args),
        prompt_template_path=args.prompt_template_path,
        model=args.model,
        api_key_path=args.api_key_path,
        base_url=args.base_url,
        reasoning_effort=args.reasoning_effort,
        max_tokens=args.max_tokens,
        return_details=True,
    )

    if args.output:
        output_path = Path(args.output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(details, indent=2) + "\n", encoding="utf-8")

    if args.json:
        print(json.dumps(details, indent=2))
    else:
        print(details["score"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
