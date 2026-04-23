from __future__ import annotations

from typing import Iterable

from pat3d.models import ObjectDescription

HUNYUAN_TEXT2IMAGE_SUFFIX = ",白色背景,3D风格,最佳质量"
HUNYUAN_PROMPT_CHAR_LIMIT = 60
DEFAULT_HUNYUAN_PROMPT_STRATEGY = "material_noun"

_MATERIAL_PATTERNS: tuple[tuple[str, str], ...] = (
    ("woven wicker", "wicker"),
    ("wicker", "wicker"),
    ("ceramic", "ceramic"),
    ("porcelain", "porcelain"),
    ("wooden", "wooden"),
    ("wood", "wooden"),
    ("metal", "metal"),
    ("steel", "steel"),
    ("glass", "glass"),
    ("plastic", "plastic"),
    ("rubber", "rubber"),
    ("fabric", "fabric"),
    ("cloth", "fabric"),
    ("paper", "paper"),
    ("cardboard", "cardboard"),
    ("stone", "stone"),
    ("marble", "marble"),
    ("leather", "leather"),
)

_SHAPE_PATTERNS: tuple[str, ...] = (
    "empty",
    "round",
    "shallow",
    "deep",
    "flat",
    "tall",
    "short",
    "wide",
    "narrow",
    "cylindrical",
    "oval",
    "square",
    "rectangular",
    "conical",
    "smooth",
    "textured",
    "organic",
)

_PROMPT_VARIANT_SPECS: tuple[tuple[str, str], ...] = (
    ("baseline_structured", "Baseline structured"),
    ("canonical_only", "Canonical name only"),
    ("material_noun", "Material + noun"),
    ("compact_shape_material", "Compact shape + material"),
    ("official_sentence", "Official short sentence"),
)
_PROMPT_VARIANT_LABELS = dict(_PROMPT_VARIANT_SPECS)


def default_hunyuan_prompt_variant_ids() -> tuple[str, ...]:
    return tuple(variant_id for variant_id, _label in _PROMPT_VARIANT_SPECS)


def hunyuan_effective_prompt(prompt: str) -> str:
    normalized = str(prompt or "").strip()
    return normalized[:HUNYUAN_PROMPT_CHAR_LIMIT] + HUNYUAN_TEXT2IMAGE_SUFFIX


def build_hunyuan_prompt_variants(
    description: ObjectDescription,
    *,
    variant_ids: Iterable[str] | None = None,
) -> tuple[dict[str, str], ...]:
    body = _description_body(description.prompt_text, description.canonical_name)
    material_terms = _extract_material_terms(body)
    compact_phrase = _compact_phrase(description.canonical_name, body, material_terms)

    prompt_by_variant = {
        "baseline_structured": description.prompt_text.strip(),
        "canonical_only": description.canonical_name.strip(),
        "material_noun": _material_noun_prompt(description.canonical_name, material_terms),
        "compact_shape_material": compact_phrase,
        "official_sentence": _official_sentence_prompt(compact_phrase),
    }

    requested_ids = tuple(variant_ids or default_hunyuan_prompt_variant_ids())
    variants: list[dict[str, str]] = []
    for variant_id, label in _PROMPT_VARIANT_SPECS:
        if variant_id not in requested_ids:
            continue
        strategy_prompt = prompt_by_variant.get(variant_id, description.prompt_text.strip())
        variants.append(
            {
                "variant_id": variant_id,
                "label": label,
                "strategy_prompt": strategy_prompt,
                "effective_prompt": hunyuan_effective_prompt(strategy_prompt),
            }
        )
    return tuple(variants)


def resolve_hunyuan_prompt(
    description: ObjectDescription,
    *,
    strategy: str = DEFAULT_HUNYUAN_PROMPT_STRATEGY,
) -> tuple[str, str]:
    normalized_strategy = str(strategy or "").strip() or DEFAULT_HUNYUAN_PROMPT_STRATEGY
    if normalized_strategy in {"verbatim", "raw"}:
        return description.prompt_text.strip(), "verbatim"

    variants = build_hunyuan_prompt_variants(description)
    prompt_by_variant = {item["variant_id"]: item["strategy_prompt"] for item in variants}
    if normalized_strategy not in prompt_by_variant:
        supported = ", ".join(["verbatim", *prompt_by_variant.keys()])
        raise ValueError(f"Unsupported Hunyuan prompt strategy '{normalized_strategy}'. Expected one of: {supported}")
    return prompt_by_variant[normalized_strategy], normalized_strategy


def label_for_hunyuan_prompt_strategy(strategy: str) -> str:
    normalized_strategy = str(strategy or "").strip()
    if normalized_strategy in {"verbatim", "raw"}:
        return "Verbatim"
    return _PROMPT_VARIANT_LABELS.get(normalized_strategy, normalized_strategy)


def _description_body(prompt_text: str, canonical_name: str) -> str:
    normalized_prompt = str(prompt_text or "").strip()
    canonical = str(canonical_name or "").strip()
    prefix = f"The object is {canonical}."
    if normalized_prompt.startswith(prefix):
        return normalized_prompt[len(prefix):].strip(" .")
    if "." in normalized_prompt:
        return normalized_prompt.split(".", 1)[1].strip(" .")
    return normalized_prompt


def _extract_material_terms(body: str) -> tuple[str, ...]:
    lowered = body.lower()
    extracted: list[str] = []
    for pattern, replacement in _MATERIAL_PATTERNS:
        if pattern in lowered and replacement not in extracted:
            extracted.append(replacement)
    return tuple(extracted)


def _extract_shape_terms(body: str) -> tuple[str, ...]:
    lowered = body.lower()
    extracted: list[str] = []
    for pattern in _SHAPE_PATTERNS:
        if pattern in lowered and pattern not in extracted:
            extracted.append(pattern)
    return tuple(extracted)


def _compact_phrase(canonical_name: str, body: str, material_terms: tuple[str, ...]) -> str:
    pieces: list[str] = []
    for term in _extract_shape_terms(body):
        if term in {"horizontal", "vertical"}:
            continue
        if term not in pieces:
            pieces.append(term)
    for term in material_terms:
        if term not in pieces:
            pieces.append(term)
    pieces.append(str(canonical_name or "").strip())
    return " ".join(part for part in pieces if part)


def _material_noun_prompt(canonical_name: str, material_terms: tuple[str, ...]) -> str:
    if not material_terms:
        return str(canonical_name or "").strip()
    return " ".join([*material_terms, str(canonical_name or "").strip()]).strip()


def _official_sentence_prompt(compact_phrase: str) -> str:
    normalized = str(compact_phrase or "").strip()
    article = "an" if normalized[:1].lower() in {"a", "e", "i", "o", "u"} else "a"
    return f"A 3D model of {article} {normalized}, white background"
