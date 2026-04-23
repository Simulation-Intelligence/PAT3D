from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from PIL import Image

from pat3d.models import ArtifactRef, ObjectDescription
from pat3d.runtime.errors import runtime_failure


def _safe_artifact_stem(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value).strip("_") or "object"


@dataclass(frozen=True, slots=True)
class CropCompletionResult:
    completed_image: ArtifactRef
    source_image: ArtifactRef
    prepared_input_image: ArtifactRef
    mask_image: ArtifactRef
    prompt: str
    model: str


class OpenAIObjectCropCompletion:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = "gpt-image-1.5",
        output_root: str = "data/ref_img_obj_completed",
        canvas_size: int = 1024,
        padding_ratio: float = 0.15,
        max_attempts: int = 3,
        retry_backoff_seconds: float = 1.5,
        request_timeout_seconds: float = 60.0,
        quality: str = "medium",
        client: Any | None = None,
    ) -> None:
        self._model = model
        self._output_root = Path(output_root)
        self._canvas_size = max(256, int(canvas_size))
        self._padding_ratio = min(max(float(padding_ratio), 0.0), 0.45)
        self._max_attempts = max(1, int(max_attempts))
        self._retry_backoff_seconds = max(0.0, float(retry_backoff_seconds))
        self._quality = quality
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

    def complete(
        self,
        *,
        scene_id: str,
        object_name: str,
        object_description: ObjectDescription,
        object_reference_image: ArtifactRef,
        disallowed_object_names: Sequence[str] = (),
        source_object_count: int = 1,
    ) -> CropCompletionResult:
        source_path = Path(object_reference_image.path)
        if not source_path.exists():
            raise FileNotFoundError(source_path)

        scene_dir = self._output_root / scene_id
        scene_dir.mkdir(parents=True, exist_ok=True)
        stem = _safe_artifact_stem(object_name)
        source_copy_path = scene_dir / f"{stem}.source.png"
        prepared_path = scene_dir / f"{stem}.input.png"
        mask_path = scene_dir / f"{stem}.mask.png"
        completed_path = scene_dir / f"{stem}.png"

        with Image.open(source_path) as source_image:
            self._write_png(source_image.convert("RGBA"), source_copy_path)
        self._prepare_canvas(source_path, prepared_path, mask_path)
        prompt = self._build_prompt(
            object_description.prompt_text,
            target_object_name=object_description.canonical_name or object_name,
            disallowed_object_names=disallowed_object_names,
            source_object_count=source_object_count,
        )
        response = self._edit_with_retry(
            image_path=prepared_path,
            mask_path=mask_path,
            prompt=prompt,
        )
        image_bytes = self._extract_image_bytes(response)
        temp_path = completed_path.with_suffix(".png.tmp")
        temp_path.write_bytes(image_bytes)
        temp_path.replace(completed_path)

        used_model = self._normalize_image_model(self._model)
        return CropCompletionResult(
            completed_image=ArtifactRef(
                artifact_type="image",
                path=str(completed_path),
                format="png",
                role="completed_object_reference_image",
                metadata_path=None,
            ),
            source_image=ArtifactRef(
                artifact_type="image",
                path=str(source_copy_path),
                format="png",
                role="crop_completion_source_image",
                metadata_path=None,
            ),
            prepared_input_image=ArtifactRef(
                artifact_type="image",
                path=str(prepared_path),
                format="png",
                role="crop_completion_input",
                metadata_path=None,
            ),
            mask_image=ArtifactRef(
                artifact_type="image",
                path=str(mask_path),
                format="png",
                role="crop_completion_mask",
                metadata_path=None,
            ),
            prompt=prompt,
            model=used_model,
        )

    def _prepare_canvas(self, source_path: Path, prepared_path: Path, mask_path: Path) -> None:
        with Image.open(source_path) as source_image:
            source_rgba = source_image.convert("RGBA")
        object_rgba, object_alpha = self._extract_object_crop(source_rgba)
        target_size = self._fit_size(object_rgba.size)
        resized_object = object_rgba.resize(target_size, Image.Resampling.LANCZOS)
        resized_alpha = object_alpha.resize(target_size, Image.Resampling.LANCZOS)

        canvas = Image.new("RGBA", (self._canvas_size, self._canvas_size), (255, 255, 255, 255))
        offset_x = max(0, (self._canvas_size - target_size[0]) // 2)
        offset_y = max(0, (self._canvas_size - target_size[1]) // 2)
        canvas.alpha_composite(resized_object, dest=(offset_x, offset_y))

        preserve_mask = resized_alpha.point(lambda value: 255 if value > 8 else 0, mode="L")
        mask = Image.new("RGBA", (self._canvas_size, self._canvas_size), (255, 255, 255, 0))
        opaque_region = Image.new("RGBA", preserve_mask.size, (255, 255, 255, 255))
        mask.paste(opaque_region, (offset_x, offset_y), preserve_mask)

        self._write_png(canvas, prepared_path)
        self._write_png(mask, mask_path)

    @staticmethod
    def _extract_object_crop(source_rgba: Image.Image) -> tuple[Image.Image, Image.Image]:
        if "A" in source_rgba.getbands():
            alpha = source_rgba.getchannel("A")
            bbox = alpha.getbbox()
            if bbox is not None:
                return source_rgba.crop(bbox), alpha.crop(bbox)
        bbox = source_rgba.getbbox() or (0, 0, source_rgba.width, source_rgba.height)
        opaque_alpha = Image.new("L", source_rgba.size, color=255)
        return source_rgba.crop(bbox), opaque_alpha.crop(bbox)

    def _fit_size(self, size: tuple[int, int]) -> tuple[int, int]:
        width, height = size
        max_content = max(1, int(round(self._canvas_size * (1.0 - (2.0 * self._padding_ratio)))))
        scale = min(max_content / max(width, 1), max_content / max(height, 1))
        return (
            max(1, int(round(width * scale))),
            max(1, int(round(height * scale))),
        )

    @staticmethod
    def _write_png(image: Image.Image, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(f"{path.suffix}.tmp")
        image.save(tmp_path, format="PNG")
        tmp_path.replace(path)

    @staticmethod
    def _normalize_prompt_name(value: str) -> str:
        return " ".join(str(value or "").replace("_", " ").strip().split())

    def _build_prompt(
        self,
        prompt_text: str,
        *,
        target_object_name: str = "",
        disallowed_object_names: Sequence[str] = (),
        source_object_count: int = 1,
    ) -> str:
        normalized = " ".join(str(prompt_text or "").strip().split())
        if normalized and not normalized.endswith("."):
            normalized = f"{normalized}."
        if "white background" not in normalized.lower():
            normalized = f"{normalized} White background.".strip()
        target_label = self._normalize_prompt_name(target_object_name)
        target_key = target_label.lower()
        blocked_labels: list[str] = []
        seen_blocked: set[str] = set()
        for raw_name in disallowed_object_names:
            label = self._normalize_prompt_name(raw_name)
            key = label.lower()
            if not label or key == target_key or key in seen_blocked:
                continue
            seen_blocked.add(key)
            blocked_labels.append(label)
        blocked_clause = ""
        if blocked_labels:
            blocked_clause = (
                " Do not generate any separate items such as "
                f"{', '.join(blocked_labels)}. "
                "Those categories are handled as separate scene objects and must stay absent here."
            )
        single_instance_clause = ""
        if int(source_object_count or 1) > 1:
            single_instance_clause = (
                " The scene may contain multiple instances of this category, "
                "but this completion must reconstruct exactly one isolated instance of the target object. "
                "Do not duplicate the object or recreate the full repeated set."
            )
        return (
            "Complete the full isolated object from the visible crop. "
            f"{f'The target object category is {target_label}. ' if target_label else ''}"
            "Preserve the visible target object appearance, remove occlusion artifacts, and infer only the missing parts of that same object. "
            "Do not add extra objects, supports, companions, accessories, attachments, furniture, decor, or scene context that are not part of the target object itself."
            f"{blocked_clause}{single_instance_clause} "
            "Return only the single object on a white background. "
            f"{normalized}"
        ).strip()

    def _edit_with_retry(self, *, image_path: Path, mask_path: Path, prompt: str) -> Any:
        last_error: Exception | None = None
        model_candidates = self._resolve_model_candidates()
        for attempt in range(1, self._max_attempts + 1):
            for index, candidate_model in enumerate(model_candidates):
                try:
                    with image_path.open("rb") as image_handle, mask_path.open("rb") as mask_handle:
                        request_kwargs: dict[str, Any] = {
                            "model": candidate_model,
                            "image": image_handle,
                            "mask": mask_handle,
                            "prompt": prompt,
                            "size": f"{self._canvas_size}x{self._canvas_size}",
                        }
                        if candidate_model.strip().lower() == "gpt-image-1":
                            request_kwargs["input_fidelity"] = "high"
                        response = self._client.images.edit(**request_kwargs)
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
            phase="object_reference_completion",
            code="image_edit_failed",
            user_message="Object crop completion failed.",
            technical_message="The image edit provider returned no response.",
            retryable=False,
            provider_kind=f"openai:{self._model}",
        )

    @staticmethod
    def _extract_image_bytes(response: Any) -> bytes:
        data = getattr(response, "data", None)
        if not data:
            raise ValueError("Image edit provider returned no image data.")
        first_item = data[0]
        b64_json = getattr(first_item, "b64_json", None)
        if not b64_json:
            raise ValueError("Image edit provider response does not include b64_json.")
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
                phase="object_reference_completion",
                code="provider_unreachable",
                user_message="Could not reach the configured image edit provider. Check network or DNS access and retry.",
                technical_message=text,
                retryable=True,
                provider_kind=f"openai:{model}",
                details=details,
            )
        if isinstance(error, AuthenticationError):
            return runtime_failure(
                phase="object_reference_completion",
                code="provider_auth_failed",
                user_message="The image edit provider rejected the configured API credentials.",
                technical_message=text,
                retryable=False,
                provider_kind=f"openai:{model}",
                details=details,
            )
        if isinstance(error, PermissionDeniedError):
            return runtime_failure(
                phase="object_reference_completion",
                code="provider_permission_denied",
                user_message="The image edit provider denied access to the requested model or capability.",
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
                phase="object_reference_completion",
                code="provider_rate_limited",
                user_message="The image edit provider rejected the request due to rate limits or quota.",
                technical_message=text,
                retryable=retryable,
                provider_kind=f"openai:{model}",
                details=details,
            )
        if isinstance(error, BadRequestError):
            if body_code == "moderation_blocked" or body_type == "image_generation_user_error":
                return runtime_failure(
                    phase="object_reference_completion",
                    code="provider_content_blocked",
                    user_message=(
                        "The image edit provider blocked this completion prompt for safety reasons. "
                        "Rephrase the object description and try again."
                    ),
                    technical_message=text,
                    retryable=False,
                    provider_kind=f"openai:{model}",
                    details=details,
                )
            return runtime_failure(
                phase="object_reference_completion",
                code="provider_bad_request",
                user_message="The image edit provider rejected the request payload.",
                technical_message=text,
                retryable=False,
                provider_kind=f"openai:{model}",
                details=details,
            )
        if isinstance(error, (InternalServerError, APIStatusError)):
            retryable = bool(status_code is None or int(status_code) >= 500)
            return runtime_failure(
                phase="object_reference_completion",
                code="provider_api_error",
                user_message="The image edit provider returned an upstream API error.",
                technical_message=text,
                retryable=retryable,
                provider_kind=f"openai:{model}",
                details=details,
            )
        return runtime_failure(
            phase="object_reference_completion",
            code="provider_error",
            user_message="The image edit provider failed unexpectedly.",
            technical_message=text,
            retryable=False,
            provider_kind=f"openai:{model}",
            details=details,
        )
