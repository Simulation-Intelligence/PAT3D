from __future__ import annotations

import ast
import base64
import importlib
import json
import re
import time
from pathlib import Path
from typing import Any, Callable

class _MissingOpenAIError(Exception):
    """Fallback base for optional OpenAI exception types."""


class BadRequestError(_MissingOpenAIError):
    """Fallback for OpenAI-compatible clients that only expose OpenAI."""


_DEFAULT_BAD_REQUEST_ERROR = BadRequestError

from pat3d.contracts import StructuredLLM
from pat3d.models import ArtifactRef, StructuredPromptRequest, StructuredPromptResult
from pat3d.runtime.errors import runtime_failure
from pat3d.storage import make_stage_metadata
from pat3d.storage.runs import make_run_id


def _get_openai_module():
    return importlib.import_module("openai")


def _get_openai_attr(name: str, default: Any = None) -> Any:
    try:
        return getattr(_get_openai_module(), name)
    except (ImportError, AttributeError):
        return default


def _get_openai_exception_type(name: str, fallback: type[BaseException] = _MissingOpenAIError) -> type[BaseException]:
    value = _get_openai_attr(name)
    if isinstance(value, type) and issubclass(value, BaseException):
        return value
    return fallback


def _get_bad_request_error_type() -> type[BaseException]:
    if (
        isinstance(BadRequestError, type)
        and issubclass(BadRequestError, BaseException)
        and BadRequestError is not _DEFAULT_BAD_REQUEST_ERROR
    ):
        return BadRequestError
    return _get_openai_exception_type("BadRequestError", BadRequestError)


def _is_image_artifact(artifact: ArtifactRef) -> bool:
    return artifact.format.lower() in {"png", "jpg", "jpeg", "webp"}


def _encode_image(image_path: str) -> str:
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except FileNotFoundError as error:
        raise runtime_failure(
            phase="structured_llm_request",
            code="context_artifact_missing",
            user_message=f"Structured LLM context image '{image_path}' does not exist.",
            technical_message=str(error),
            retryable=False,
            details={"path": image_path},
        ) from error


def _parse_json_like_payload(text: str) -> Any:
    stripped = text.strip()
    candidates: list[str] = [stripped]

    fenced_blocks = re.findall(r"```(?:json)?\s*(.*?)```", stripped, flags=re.DOTALL | re.IGNORECASE)
    candidates.extend(block for block in fenced_blocks if block.strip())

    for pattern in (r"(\{.*\})", r"(\[.*\])"):
        match = re.search(pattern, stripped, flags=re.DOTALL)
        if match is not None:
            candidates.append(match.group(1))

    seen: set[str] = set()
    for candidate in candidates:
        payload = candidate.strip()
        if not payload or payload in seen:
            continue
        seen.add(payload)
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            pass
        try:
            return ast.literal_eval(payload)
        except (ValueError, SyntaxError):
            pass

    raise ValueError("No JSON-like payload found in provider response.")


def _extract_json_payload(text: str, *, schema_name: str) -> dict[str, Any]:
    parsed = _parse_json_like_payload(text)
    if isinstance(parsed, dict):
        return parsed
    if schema_name == "object_description":
        return {"result": str(parsed)}
    raise ValueError("Provider response did not resolve to a JSON object.")


def _content_part_text(part: Any) -> str:
    if isinstance(part, str):
        return part
    if isinstance(part, dict):
        for key in ("text", "output_text", "content"):
            value = part.get(key)
            if isinstance(value, str) and value.strip():
                return value
            if isinstance(value, list):
                joined = "\n".join(_content_part_text(item) for item in value).strip()
                if joined:
                    return joined
        return ""
    text = getattr(part, "text", None)
    if isinstance(text, str) and text.strip():
        return text
    output_text = getattr(part, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text
    content = getattr(part, "content", None)
    if isinstance(content, str) and content.strip():
        return content
    if isinstance(content, list):
        return "\n".join(_content_part_text(item) for item in content).strip()
    return ""


def _response_message_text(response: Any) -> str:
    choices = getattr(response, "choices", None)
    if isinstance(choices, list) and choices:
        message = getattr(choices[0], "message", None)
        if message is not None:
            content = getattr(message, "content", None)
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                joined = "\n".join(_content_part_text(item) for item in content).strip()
                if joined:
                    return joined
            refusal = getattr(message, "refusal", None)
            if isinstance(refusal, str) and refusal.strip():
                return refusal
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str):
        return output_text
    return ""


def _response_finish_reason(response: Any) -> str:
    choices = getattr(response, "choices", None)
    if isinstance(choices, list) and choices:
        finish_reason = getattr(choices[0], "finish_reason", None)
        if isinstance(finish_reason, str):
            return finish_reason
    return ""


def _supports_non_default_temperature(model: str) -> bool:
    return not str(model).strip().lower().startswith("gpt-5")


class OpenAIStructuredLLM(StructuredLLM):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = "gpt-4o",
        reasoning_effort: str | None = None,
        max_parse_retries: int | None = None,
        max_attempts: int | None = None,
        max_completion_tokens: int | None = None,
        client: Any | None = None,
        metadata_factory: Callable[..., Any] = make_stage_metadata,
    ) -> None:
        self._model = model
        self._reasoning_effort = reasoning_effort
        resolved_max_attempts = max_attempts
        if resolved_max_attempts is None:
            resolved_retries = 2 if max_parse_retries is None else int(max_parse_retries)
            resolved_max_attempts = resolved_retries + 1
        self._max_attempts = max(1, int(resolved_max_attempts))
        self._metadata_factory = metadata_factory
        self._configured_max_completion_tokens = (
            max(1, int(max_completion_tokens)) if max_completion_tokens is not None else None
        )
        if client is not None:
            self._client = client
        else:
            openai_client_class = _get_openai_attr("OpenAI")
            if openai_client_class is None:
                raise ImportError("openai.OpenAI is not available")
            client_kwargs: dict[str, Any] = {}
            if api_key is not None:
                client_kwargs["api_key"] = api_key
            if base_url is not None:
                client_kwargs["base_url"] = base_url
            self._client = openai_client_class(**client_kwargs)

    def generate(self, request: StructuredPromptRequest) -> StructuredPromptResult:
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})

        content: list[dict[str, Any]] = [{"type": "text", "text": request.prompt_text}]
        for artifact in request.context_artifacts:
            if not _is_image_artifact(artifact):
                continue
            encoded_image = _encode_image(artifact.path)
            suffix = Path(artifact.path).suffix.lower().lstrip(".") or artifact.format
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/{suffix};base64,{encoded_image}"},
                }
            )

        messages.append({"role": "user", "content": content})
        attempt_count = self._max_attempts
        raw_text = ""
        last_parse_error: Exception | None = None
        max_tokens = self._initial_max_tokens(request)
        for attempt_index in range(attempt_count):
            try:
                response = self._create_completion_with_compatible_token_limit(
                    messages=messages,
                    max_tokens=max_tokens,
                )
            except Exception as error:
                normalized_error = self._normalize_provider_error(error)
                if attempt_index + 1 >= attempt_count or not normalized_error.payload.retryable:
                    raise normalized_error from error
                time.sleep(0.4 * (attempt_index + 1))
                continue

            raw_text = _response_message_text(response)
            finish_reason = _response_finish_reason(response)
            try:
                parsed_output = _extract_json_payload(raw_text, schema_name=request.schema_name)
                break
            except Exception as error:
                last_parse_error = error
                if finish_reason == "length" and not raw_text.strip():
                    max_tokens = self._expanded_max_tokens(max_tokens)
                if attempt_index + 1 >= attempt_count or not self._should_retry_response_parse(
                    raw_text,
                    finish_reason=finish_reason,
                ):
                    raw_artifact = self._write_raw_response_artifact(request, raw_text)
                    raise runtime_failure(
                        phase="structured_llm_response_parse",
                        code="structured_response_parse_failed",
                        user_message=f"The structured LLM response for schema '{request.schema_name}' could not be parsed.",
                        technical_message=str(error),
                        retryable=False,
                        provider_kind=f"openai:{self._model}",
                        details={
                            "schema_name": request.schema_name,
                            "raw_response_artifact": raw_artifact.path,
                            "parse_attempts": attempt_index + 1,
                            "max_attempts": attempt_count,
                            "finish_reason": finish_reason,
                            "max_completion_tokens": max_tokens,
                        },
                    ) from error
                time.sleep(0.2 * (attempt_index + 1))
        else:  # pragma: no cover - guarded by the parse failure branch above
            raise runtime_failure(
                phase="structured_llm_response_parse",
                code="structured_response_parse_failed",
                user_message=f"The structured LLM response for schema '{request.schema_name}' could not be parsed.",
                technical_message=str(last_parse_error) if last_parse_error else "Unknown parse failure.",
                retryable=False,
                provider_kind=f"openai:{self._model}",
                details={"schema_name": request.schema_name, "parse_attempts": attempt_count, "max_attempts": attempt_count},
            )
        metadata = self._metadata_factory(
            stage_name="structured_llm",
            provider_name=f"openai:{self._model}",
            notes=(f"schema={request.schema_name}",),
        )
        return StructuredPromptResult(
            schema_name=request.schema_name,
            parsed_output=parsed_output,
            metadata=metadata,
            raw_response_artifact=None,
        )

    def _should_retry_response_parse(self, raw_text: str, *, finish_reason: str = "") -> bool:
        _ = finish_reason
        return not raw_text.strip()

    def _initial_max_tokens(self, request: StructuredPromptRequest) -> int:
        if self._configured_max_completion_tokens is not None:
            return self._configured_max_completion_tokens
        if request.schema_name == "size_prior" and str(self._model).strip().startswith("gpt-5"):
            return 1200
        return 800

    def _expanded_max_tokens(self, current: int) -> int:
        if self._configured_max_completion_tokens is not None:
            return self._configured_max_completion_tokens
        if current < 1600:
            return 1600
        if current < 2400:
            return 2400
        return min(current * 2, 3200)

    def _create_completion_with_compatible_token_limit(
        self,
        *,
        messages: list[dict[str, Any]],
        max_tokens: int,
    ):
        completion_kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "seed": 100,
            "response_format": {"type": "json_object"},
        }
        if _supports_non_default_temperature(self._model):
            completion_kwargs["temperature"] = 0
        if self._should_send_reasoning_effort():
            completion_kwargs["reasoning_effort"] = self._reasoning_effort

        payload_variants = (
            {"max_completion_tokens": max_tokens},
            {"max_tokens": max_tokens},
            {},
        )
        bad_request_error_type = _get_bad_request_error_type()
        last_error: Exception | None = None
        for index, payload in enumerate(payload_variants):
            try:
                return self._client.chat.completions.create(
                    **{**completion_kwargs, **payload},
                )
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

    def _should_send_reasoning_effort(self) -> bool:
        if self._reasoning_effort is None:
            return False
        if str(self._reasoning_effort).strip().lower() == "auto":
            return False
        return str(self._model).strip().startswith("gpt-5")

    def _write_raw_response_artifact(
        self,
        request: StructuredPromptRequest,
        raw_text: str,
    ) -> ArtifactRef:
        output_dir = Path("results") / "runtime" / "raw_responses"
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{request.schema_name}-{make_run_id('raw-response')}.txt"
        path = output_dir / filename
        path.write_text(raw_text, encoding="utf-8")
        return ArtifactRef(
            artifact_type="raw_response",
            path=str(path),
            format="txt",
            role=request.schema_name,
        )

    def _normalize_provider_error(self, error: Exception):
        api_connection_error = _get_openai_exception_type("APIConnectionError")
        api_status_error = _get_openai_exception_type("APIStatusError")
        api_timeout_error = _get_openai_exception_type("APITimeoutError")
        authentication_error = _get_openai_exception_type("AuthenticationError")
        internal_server_error = _get_openai_exception_type("InternalServerError")
        permission_denied_error = _get_openai_exception_type("PermissionDeniedError")
        rate_limit_error = _get_openai_exception_type("RateLimitError")
        bad_request_error = _get_bad_request_error_type()

        details = {"model": self._model, "exception_type": error.__class__.__name__}
        status_code = getattr(error, "status_code", None)
        if status_code is not None:
            details["status_code"] = status_code
        text = str(error).strip() or error.__class__.__name__
        lowered = text.lower()

        if isinstance(error, (api_connection_error, api_timeout_error)):
            return runtime_failure(
                phase="structured_llm_request",
                code="provider_unreachable",
                user_message="Could not reach the configured structured LLM provider. Check network or DNS access and retry.",
                technical_message=text,
                retryable=True,
                provider_kind=f"openai:{self._model}",
                details=details,
            )
        if isinstance(error, authentication_error):
            return runtime_failure(
                phase="structured_llm_request",
                code="provider_auth_failed",
                user_message="The structured LLM provider rejected the configured API credentials.",
                technical_message=text,
                retryable=False,
                provider_kind=f"openai:{self._model}",
                details=details,
            )
        if isinstance(error, permission_denied_error):
            return runtime_failure(
                phase="structured_llm_request",
                code="provider_permission_denied",
                user_message="The structured LLM provider denied access to the requested model or capability.",
                technical_message=text,
                retryable=False,
                provider_kind=f"openai:{self._model}",
                details=details,
            )
        if isinstance(error, rate_limit_error):
            retryable = not any(
                token in lowered
                for token in ("insufficient_quota", "billing", "hard_limit", "quota")
            )
            return runtime_failure(
                phase="structured_llm_request",
                code="provider_rate_limited",
                user_message="The structured LLM provider rejected the request due to rate limits or quota.",
                technical_message=text,
                retryable=retryable,
                provider_kind=f"openai:{self._model}",
                details=details,
            )
        if isinstance(error, bad_request_error):
            return runtime_failure(
                phase="structured_llm_request",
                code="provider_bad_request",
                user_message="The structured LLM provider rejected the request payload.",
                technical_message=text,
                retryable=False,
                provider_kind=f"openai:{self._model}",
                details=details,
            )
        if isinstance(error, (internal_server_error, api_status_error)):
            retryable = bool(status_code is None or int(status_code) >= 500)
            return runtime_failure(
                phase="structured_llm_request",
                code="provider_api_error",
                user_message="The structured LLM provider returned an upstream API error.",
                technical_message=text,
                retryable=retryable,
                provider_kind=f"openai:{self._model}",
                details=details,
            )
        return runtime_failure(
            phase="structured_llm_request",
            code="provider_error",
            user_message="Structured LLM generation failed.",
            technical_message=text,
            retryable=False,
            provider_kind=f"openai:{self._model}",
            details=details,
        )
