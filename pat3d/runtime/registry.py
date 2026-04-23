from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

from pat3d.runtime.config import ProviderBinding


class RuntimeRegistryError(KeyError):
    pass


ProviderBuilder = Callable[[ProviderBinding], Any]
PipelineFactory = Callable[..., Any]


class ProviderRegistry:
    def __init__(
        self,
        builders: dict[str, ProviderBuilder] | None = None,
    ) -> None:
        self._builders: dict[str, ProviderBuilder] = dict(builders or {})

    def register(self, kind: str, builder: ProviderBuilder) -> None:
        self._builders[kind] = builder

    def registered_kinds(self) -> tuple[str, ...]:
        return tuple(sorted(self._builders))

    def resolve(self, kind: str) -> ProviderBuilder:
        try:
            return self._builders[kind]
        except KeyError as exc:
            supported = ", ".join(self.registered_kinds())
            raise RuntimeRegistryError(
                f"Unknown provider kind '{kind}'. Supported provider kinds: {supported}"
            ) from exc

    def build(self, binding: ProviderBinding) -> Any:
        return self.resolve(binding.kind)(binding)


class PipelineRegistry:
    def __init__(
        self,
        factories: dict[str, PipelineFactory] | None = None,
    ) -> None:
        self._factories: dict[str, PipelineFactory] = dict(factories or {})

    def register(self, name: str, factory: PipelineFactory) -> None:
        self._factories[name] = factory

    def registered_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))

    def resolve(self, name: str) -> PipelineFactory:
        try:
            return self._factories[name]
        except KeyError as exc:
            supported = ", ".join(self.registered_names())
            raise RuntimeRegistryError(
                f"Unknown pipeline '{name}'. Supported pipelines: {supported}"
            ) from exc

    def build(self, name: str, **kwargs: Any) -> Any:
        factory = self.resolve(name)
        signature = inspect.signature(factory)
        try:
            signature.bind(**kwargs)
        except TypeError as exc:
            raise ValueError(
                f"Invalid runtime wiring for pipeline '{name}': {exc}"
            ) from exc
        return factory(**kwargs)
