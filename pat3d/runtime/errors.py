from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class RuntimeFailurePayload:
    phase: str
    user_message: str
    technical_message: str
    code: str = "runtime_error"
    provider_role: str | None = None
    provider_kind: str | None = None
    retryable: bool = False
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "code": self.code,
            "user_message": self.user_message,
            "technical_message": self.technical_message,
            "provider_role": self.provider_role,
            "provider_kind": self.provider_kind,
            "retryable": self.retryable,
            "details": dict(self.details),
        }


class Pat3DRuntimeError(RuntimeError):
    def __init__(self, payload: RuntimeFailurePayload) -> None:
        super().__init__(payload.user_message)
        self.payload = payload

    def to_dict(self) -> dict[str, Any]:
        return self.payload.to_dict()

    def to_result(self) -> dict[str, Any]:
        return {"status": "failed", "error": self.to_dict()}


def runtime_failure(
    *,
    phase: str,
    user_message: str,
    technical_message: str,
    code: str = "runtime_error",
    provider_role: str | None = None,
    provider_kind: str | None = None,
    retryable: bool = False,
    details: Mapping[str, Any] | None = None,
) -> Pat3DRuntimeError:
    return Pat3DRuntimeError(
        RuntimeFailurePayload(
            phase=phase,
            code=code,
            user_message=user_message,
            technical_message=technical_message,
            provider_role=provider_role,
            provider_kind=provider_kind,
            retryable=retryable,
            details=dict(details or {}),
        )
    )


def coerce_runtime_failure(
    error: Exception,
    *,
    phase: str,
    user_message: str,
    technical_message: str | None = None,
    code: str = "runtime_error",
    provider_role: str | None = None,
    provider_kind: str | None = None,
    retryable: bool = False,
    details: Mapping[str, Any] | None = None,
) -> Pat3DRuntimeError:
    if isinstance(error, Pat3DRuntimeError):
        return error
    resolved_technical_message = technical_message or str(error) or error.__class__.__name__
    resolved_details = {"exception_type": error.__class__.__name__}
    resolved_details.update(dict(details or {}))
    return runtime_failure(
        phase=phase,
        user_message=user_message,
        technical_message=resolved_technical_message,
        code=code,
        provider_role=provider_role,
        provider_kind=provider_kind,
        retryable=retryable,
        details=resolved_details,
    )

