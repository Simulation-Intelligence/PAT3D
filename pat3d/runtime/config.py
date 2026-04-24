from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


PIPELINE_REQUIRED_PROVIDER_ROLES: dict[str, tuple[str, ...]] = {
    "prompt_backed_contract_slice": ("structured_llm",),
    "paper_core": (
        "structured_llm",
        "text_to_3d_provider",
        "layout_builder",
        "mesh_simplifier",
    ),
}


PIPELINE_OPTIONAL_PROVIDER_ROLES: dict[str, tuple[str, ...]] = {
    "prompt_backed_contract_slice": (
        "text_to_image_provider",
        "depth_estimator",
        "segmenter",
    ),
    "paper_core": (
        "text_to_image_provider",
        "depth_estimator",
        "segmenter",
        "physics_optimizer",
        "scene_renderer",
    ),
}


@dataclass(frozen=True)
class ProviderBinding:
    kind: str
    enabled: bool = True
    options: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProviderBinding":
        return cls(
            kind=str(data["kind"]),
            enabled=bool(data.get("enabled", True)),
            options=dict(data.get("options", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "enabled": self.enabled,
            "options": dict(self.options),
        }


@dataclass(frozen=True)
class RuntimeConfig:
    pipeline: str
    providers: dict[str, ProviderBinding] = field(default_factory=dict)
    pipeline_options: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RuntimeConfig":
        providers_raw = data.get("providers", {})
        providers = {
            str(role): ProviderBinding.from_dict(binding)
            for role, binding in dict(providers_raw).items()
        }
        return cls(
            pipeline=str(data["pipeline"]),
            providers=providers,
            pipeline_options=dict(data.get("pipeline_options", {})),
        )

    @classmethod
    def from_json_file(cls, path: str | Path) -> "RuntimeConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self) -> dict[str, Any]:
        return {
            "pipeline": self.pipeline,
            "providers": {
                role: binding.to_dict()
                for role, binding in sorted(self.providers.items())
            },
            "pipeline_options": dict(self.pipeline_options),
        }

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def required_provider_roles(self) -> tuple[str, ...]:
        return PIPELINE_REQUIRED_PROVIDER_ROLES.get(self.pipeline, ())

    def optional_provider_roles(self) -> tuple[str, ...]:
        return PIPELINE_OPTIONAL_PROVIDER_ROLES.get(self.pipeline, ())

    def validate(self) -> None:
        supported = set(PIPELINE_REQUIRED_PROVIDER_ROLES) | set(
            PIPELINE_OPTIONAL_PROVIDER_ROLES
        )
        if self.pipeline not in supported:
            supported_text = ", ".join(sorted(supported))
            raise ValueError(
                f"Unsupported pipeline '{self.pipeline}'. Supported pipelines: "
                f"{supported_text}"
            )

        missing = [
            role
            for role in self.required_provider_roles()
            if role not in self.providers or not self.providers[role].enabled
        ]
        if missing:
            missing_text = ", ".join(missing)
            raise ValueError(
                f"Pipeline '{self.pipeline}' is missing required enabled providers: "
                f"{missing_text}"
            )

    def enabled_provider_bindings(self) -> dict[str, ProviderBinding]:
        return {
            role: binding
            for role, binding in self.providers.items()
            if binding.enabled
        }


def make_prompt_backed_runtime_config(
    *,
    model: str = "gpt-4o",
    api_key: str | None = None,
    include_text_to_image: bool = False,
    image_model: str = "gpt-image-1",
    reasoning_effort: str | None = None,
    include_depth: bool = True,
    include_segmentation: bool = True,
) -> RuntimeConfig:
    providers: dict[str, ProviderBinding] = {
        "structured_llm": ProviderBinding(
            kind="openai_structured_llm",
            options={
                key: value
                for key, value in {
                    "model": model,
                    "api_key": api_key,
                    "reasoning_effort": reasoning_effort,
                }.items()
                if value is not None
            },
        )
    }
    if include_text_to_image:
        providers["text_to_image_provider"] = ProviderBinding(
            kind="openai_text_to_image",
            options={
                key: value
                for key, value in {
                    "model": image_model,
                    "api_key": api_key,
                }.items()
                if value is not None
            },
        )
    if include_depth:
        providers["depth_estimator"] = ProviderBinding(kind="current_depth")
    if include_segmentation:
        providers["segmenter"] = ProviderBinding(kind="current_segmenter")
    return RuntimeConfig(
        pipeline="prompt_backed_contract_slice",
        providers=providers,
    )


def make_paper_core_runtime_config(
    *,
    scene_id: str,
    model: str = "gpt-4o",
    api_key: str | None = None,
    include_text_to_image: bool = False,
    image_model: str = "gpt-image-1",
    reasoning_effort: str | None = None,
    include_depth: bool = True,
    include_segmentation: bool = True,
    include_physics: bool = True,
    include_renderer: bool = True,
) -> RuntimeConfig:
    providers: dict[str, ProviderBinding] = {
        "structured_llm": ProviderBinding(
            kind="openai_structured_llm",
            options={
                key: value
                for key, value in {
                    "model": model,
                    "api_key": api_key,
                    "reasoning_effort": reasoning_effort,
                }.items()
                if value is not None
            },
        ),
        "text_to_3d_provider": ProviderBinding(
            kind="current_text_to_3d",
            options={"scene_id": scene_id},
        ),
        "layout_builder": ProviderBinding(kind="legacy_layout"),
        "mesh_simplifier": ProviderBinding(kind="current_mesh_simplifier"),
    }
    if include_text_to_image:
        providers["text_to_image_provider"] = ProviderBinding(
            kind="openai_text_to_image",
            options={
                key: value
                for key, value in {
                    "model": image_model,
                    "api_key": api_key,
                }.items()
                if value is not None
            },
        )
    if include_depth:
        providers["depth_estimator"] = ProviderBinding(kind="current_depth")
    if include_segmentation:
        providers["segmenter"] = ProviderBinding(kind="current_segmenter")
    if include_physics:
        providers["physics_optimizer"] = ProviderBinding(kind="passthrough_physics")
    if include_renderer:
        providers["scene_renderer"] = ProviderBinding(kind="geometry_scene_renderer")
    return RuntimeConfig(pipeline="paper_core", providers=providers)
