from __future__ import annotations

import base64
from pathlib import Path
import time
from typing import Any, Callable, Mapping

from pat3d.contracts import TextToImageProvider
from pat3d.models import ArtifactRef, ReferenceImageResult, SceneRequest
from pat3d.runtime.errors import runtime_failure
from pat3d.storage import make_stage_metadata


class OpenAITextToImageProvider(TextToImageProvider):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = "gpt-image-1",
        output_dir: str = "data/ref_img",
        max_attempts: int = 3,
        retry_backoff_seconds: float = 1.5,
        request_timeout_seconds: float = 60.0,
        client: Any | None = None,
        metadata_factory: Callable[..., Any] = make_stage_metadata,
    ) -> None:
        self._model = model
        self._output_dir = Path(output_dir)
        self._max_attempts = max(1, int(max_attempts))
        self._retry_backoff_seconds = max(0.0, float(retry_backoff_seconds))
        self._metadata_factory = metadata_factory
        if client is not None:
            self._client = client
        else:
            from openai import OpenAI

            client_kwargs: dict[str, Any] = {}
            if api_key is not None:
                client_kwargs["api_key"] = api_key
            if base_url is not None:
                client_kwargs["base_url"] = base_url
            client_kwargs["timeout"] = request_timeout_seconds
            self._client = OpenAI(**client_kwargs)

    def generate(self, request: SceneRequest) -> ReferenceImageResult:
        if request.text_prompt is None:
            raise ValueError("SceneRequest.text_prompt is required for image generation.")

        width, height = self._resolve_dimensions(request.constraints)
        response = self._generate_with_retry(
            prompt=request.text_prompt,
            width=width,
            height=height,
        )
        image_bytes = self._extract_image_bytes(response)

        self._output_dir.mkdir(parents=True, exist_ok=True)
        image_path = self._output_dir / f"{request.scene_id}.png"
        temp_path = image_path.with_suffix(".png.tmp")
        temp_path.write_bytes(image_bytes)
        temp_path.replace(image_path)

        metadata = self._metadata_factory(
            stage_name="reference_image",
            provider_name=f"openai:{self._model}",
        )
        return ReferenceImageResult(
            request=request,
            image=ArtifactRef(
                artifact_type="image",
                path=str(image_path),
                format="png",
                role="reference_image",
            ),
            generation_prompt=request.text_prompt,
            seed=None,
            width=width,
            height=height,
            metadata=metadata,
        )

    def _generate_with_retry(self, *, prompt: str, width: int, height: int) -> Any:
        last_error: Exception | None = None
        model_candidates = self._resolve_model_candidates()
        for attempt in range(1, self._max_attempts + 1):
            for index, candidate_model in enumerate(model_candidates):
                try:
                    response = self._client.images.generate(
                        model=candidate_model,
                        prompt=prompt,
                        size=f"{width}x{height}",
                    )
                    self._model = candidate_model
                    return response
                except Exception as exc:
                    last_error = exc
                    if self._should_try_next_candidate(candidate_model, exc) and index + 1 < len(model_candidates):
                        continue
                    normalized = self._normalize_provider_error(exc, model=candidate_model)
                    last_error = normalized
                    if not normalized.payload.retryable or attempt >= self._max_attempts:
                        break
                    last_error = exc
                    time.sleep(self._retry_backoff_seconds * attempt)
            else:
                continue
            break
        if isinstance(last_error, Exception):
            raise last_error
        raise runtime_failure(
            phase="reference_image_generation",
            code="image_generation_failed",
            user_message="Reference image generation failed.",
            technical_message="The image provider returned no response.",
            retryable=False,
            provider_kind=f"openai:{self._model}",
        )

    @staticmethod
    def _extract_image_bytes(response: Any) -> bytes:
        data = getattr(response, "data", None)
        if not data:
            raise ValueError("Text-to-image provider returned no image data.")

        first_item = data[0]
        b64_json = getattr(first_item, "b64_json", None)
        if not b64_json:
            raise ValueError("Text-to-image provider response does not include b64_json.")
        return base64.b64decode(b64_json)

    def _normalize_image_model(self, model: str) -> str:
        normalized = model.strip().lower()
        if normalized in {"gpt-image-1_5", "gpt image 1.5"}:
            return "gpt-image-1.5"
        if not normalized:
            return "gpt-image-1"
        return normalized

    def _resolve_model_candidates(self) -> list[str]:
        base_model = self._normalize_image_model(self._model)
        if base_model == "gpt-image-1.5":
            return ["gpt-image-1.5", "gpt-image-1"]
        return [base_model]

    @staticmethod
    def _should_try_next_candidate(model: str, error: Exception) -> bool:
        message = str(error).lower()
        return (
            model.strip().lower() in {"gpt-image-1.5", "gpt-image-1_5"}
            and (
                "unsupported value" in message
                or "invalid value" in message
                or "does not exist" in message
                or "not supported" in message
                or "unknown model" in message
                or "resource not found" in message
            )
        )

    def _normalize_provider_error(self, error: Exception, model: str):
        from openai import (
            APIConnectionError,
            APIStatusError,
            APITimeoutError,
            AuthenticationError,
            BadRequestError,
            InternalServerError,
            PermissionDeniedError,
            RateLimitError,
        )

        details = {"model": model, "exception_type": error.__class__.__name__}
        status_code = getattr(error, "status_code", None)
        if status_code is not None:
            details["status_code"] = status_code
        text = str(error).strip() or error.__class__.__name__
        lowered = text.lower()
        body = getattr(error, "body", None)
        body_error = body.get("error") if isinstance(body, Mapping) else None
        body_code = str(body_error.get("code", "")).strip().lower() if isinstance(body_error, Mapping) else ""
        body_type = str(body_error.get("type", "")).strip().lower() if isinstance(body_error, Mapping) else ""

        if isinstance(error, (APIConnectionError, APITimeoutError)):
            return runtime_failure(
                phase="reference_image_generation",
                code="provider_unreachable",
                user_message="Could not reach the configured image provider. Check network or DNS access and retry.",
                technical_message=text,
                retryable=True,
                provider_kind=f"openai:{model}",
                details=details,
            )
        if isinstance(error, AuthenticationError):
            return runtime_failure(
                phase="reference_image_generation",
                code="provider_auth_failed",
                user_message="The image provider rejected the configured API credentials.",
                technical_message=text,
                retryable=False,
                provider_kind=f"openai:{model}",
                details=details,
            )
        if isinstance(error, PermissionDeniedError):
            return runtime_failure(
                phase="reference_image_generation",
                code="provider_permission_denied",
                user_message="The image provider denied access to the requested model or capability.",
                technical_message=text,
                retryable=False,
                provider_kind=f"openai:{model}",
                details=details,
            )
        if isinstance(error, RateLimitError):
            retryable = not any(
                token in lowered
                for token in ("insufficient_quota", "billing", "hard_limit", "quota")
            )
            return runtime_failure(
                phase="reference_image_generation",
                code="provider_rate_limited",
                user_message="The image provider rejected the request due to rate limits or quota.",
                technical_message=text,
                retryable=retryable,
                provider_kind=f"openai:{model}",
                details=details,
            )
        if isinstance(error, BadRequestError):
            if body_code == "moderation_blocked" or body_type == "image_generation_user_error":
                return runtime_failure(
                    phase="reference_image_generation",
                    code="provider_content_blocked",
                    user_message=(
                        "The image provider blocked this prompt for safety reasons. "
                        "Rephrase the scene description and try again."
                    ),
                    technical_message=text,
                    retryable=False,
                    provider_kind=f"openai:{model}",
                    details=details,
                )
            return runtime_failure(
                phase="reference_image_generation",
                code="provider_bad_request",
                user_message="The image provider rejected the request payload.",
                technical_message=text,
                retryable=False,
                provider_kind=f"openai:{model}",
                details=details,
            )
        if isinstance(error, (InternalServerError, APIStatusError)):
            retryable = bool(status_code is None or int(status_code) >= 500)
            return runtime_failure(
                phase="reference_image_generation",
                code="provider_api_error",
                user_message="The image provider returned an upstream API error.",
                technical_message=text,
                retryable=retryable,
                provider_kind=f"openai:{model}",
                details=details,
            )
        return runtime_failure(
            phase="reference_image_generation",
            code="provider_error",
            user_message="Reference image generation failed.",
            technical_message=text,
            retryable=False,
            provider_kind=f"openai:{model}",
            details=details,
        )

    @staticmethod
    def _resolve_dimensions(
        constraints: Mapping[str, object] | None,
    ) -> tuple[int, int]:
        parsed = None
        if constraints:
            nested = constraints.get("reference_image_size")
            if isinstance(nested, Mapping):
                width = nested.get("width")
                height = nested.get("height")
                if isinstance(width, (int, float)) and isinstance(height, (int, float)):
                    parsed = (int(width), int(height))

            if parsed is None:
                width = constraints.get("reference_image_width")
                height = constraints.get("reference_image_height")
                if isinstance(width, (int, float)) and isinstance(height, (int, float)):
                    parsed = (int(width), int(height))

        if parsed is None:
            return (1024, 1024)

        width, height = parsed
        if width <= 0 or height <= 0:
            raise ValueError("reference image dimensions must be positive")
        return (width, height)
