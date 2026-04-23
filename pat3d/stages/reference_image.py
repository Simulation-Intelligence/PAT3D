from __future__ import annotations

from collections.abc import Callable, Mapping

from pat3d.contracts import TextToImageProvider
from pat3d.models import ArtifactRef, ReferenceImageResult, SceneRequest, StageRunMetadata

MetadataFactory = Callable[..., StageRunMetadata]
ImageInspector = Callable[[str], tuple[int, int]]


class ReferenceImagePassthroughStage:
    stage_name = "reference_image"

    def __init__(
        self,
        metadata_factory: MetadataFactory,
        image_inspector: ImageInspector | None = None,
    ) -> None:
        self._metadata_factory = metadata_factory
        self._image_inspector = image_inspector

    def run(self, request: SceneRequest) -> ReferenceImageResult:
        if request.reference_image is None:
            raise ValueError("SceneRequest.reference_image is required for passthrough mode.")
        width, height = self._resolve_dimensions(
            reference_image=request.reference_image,
            constraints=request.constraints,
        )

        metadata = self._metadata_factory(
            stage_name=self.stage_name,
            provider_name="passthrough",
        )
        return ReferenceImageResult(
            request=request,
            image=request.reference_image,
            generation_prompt=None,
            seed=None,
            width=width,
            height=height,
            metadata=metadata,
        )

    def _resolve_dimensions(
        self,
        *,
        reference_image: ArtifactRef,
        constraints: Mapping[str, object] | None,
    ) -> tuple[int, int]:
        constraint_dimensions = self._dimensions_from_constraints(constraints)
        if constraint_dimensions is not None:
            return constraint_dimensions
        if self._image_inspector is not None:
            return self._validate_dimensions(self._image_inspector(reference_image.path))
        try:
            from PIL import Image

            with Image.open(reference_image.path) as image:
                return self._validate_dimensions(tuple(image.size))
        except Exception as exc:  # pragma: no cover - failure path
            raise ValueError(
                "reference image dimensions are required; provide them in "
                "SceneRequest.constraints, configure image_inspector, or supply a readable image path"
            ) from exc
        raise ValueError(
            "reference image dimensions are required; provide them in "
            "SceneRequest.constraints or configure image_inspector"
        )

    @classmethod
    def _dimensions_from_constraints(
        cls,
        constraints: Mapping[str, object] | None,
    ) -> tuple[int, int] | None:
        if not constraints:
            return None

        nested_dimensions = constraints.get("reference_image_size")
        if isinstance(nested_dimensions, Mapping):
            width = nested_dimensions.get("width")
            height = nested_dimensions.get("height")
            if isinstance(width, (int, float)) and isinstance(height, (int, float)):
                return cls._validate_dimensions((int(width), int(height)))

        width = constraints.get("reference_image_width")
        height = constraints.get("reference_image_height")
        if isinstance(width, (int, float)) and isinstance(height, (int, float)):
            return cls._validate_dimensions((int(width), int(height)))
        return None

    @staticmethod
    def _validate_dimensions(dimensions: tuple[int, int]) -> tuple[int, int]:
        width, height = dimensions
        if width <= 0 or height <= 0:
            raise ValueError("reference image dimensions must be positive")
        return width, height


class ReferenceImageStage:
    stage_name = "reference_image"

    def __init__(
        self,
        metadata_factory: MetadataFactory,
        *,
        text_to_image_provider: TextToImageProvider | None = None,
        image_inspector: ImageInspector | None = None,
    ) -> None:
        self._text_to_image_provider = text_to_image_provider
        self._passthrough = ReferenceImagePassthroughStage(
            metadata_factory=metadata_factory,
            image_inspector=image_inspector,
        )

    def run(self, request: SceneRequest) -> ReferenceImageResult:
        if request.reference_image is not None:
            return self._passthrough.run(request)
        if self._text_to_image_provider is None:
            raise ValueError(
                "SceneRequest.reference_image is required unless text_to_image_provider "
                "is configured."
            )
        return self._text_to_image_provider.generate(request)
