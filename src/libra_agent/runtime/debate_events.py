from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from contextvars import ContextVar
from typing import Any

DebateEventPublisher = Callable[[str, dict[str, Any]], None]

MAX_EVENT_TEXT_CHARS = 40_000
MAX_EVENT_LIST_ITEMS = 80
SENSITIVE_KEY_PARTS = (
    "api_key",
    "apikey",
    "app_key",
    "appkey",
    "app_secret",
    "appsecret",
    "secretkey",
    "secret",
    "authorization",
    "password",
    "credential",
    "database_url",
)
SENSITIVE_EXACT_KEYS = {"token", "access_token", "refresh_token"}
SENSITIVE_ASSIGNMENT = re.compile(
    r"(?im)^([A-Z0-9_./-]*(?:API[_-]?KEY|APP[_-]?KEY|APP[_-]?SECRET|SECRET|TOKEN|AUTHORIZATION|PASSWORD|DATABASE_URL)[A-Z0-9_./-]*\s*[:=]\s*)(.+)$"
)

debate_event_publisher: ContextVar[DebateEventPublisher | None] = ContextVar(
    "debate_event_publisher",
    default=None,
)


def publish_debate_event(event: str, payload: dict[str, Any]) -> None:
    publisher = debate_event_publisher.get()
    if publisher is not None:
        publisher(event, sanitize_event_payload(payload))


def publish_tool_observation(
    *,
    actor: str,
    phase: str,
    tools: Sequence[Any],
) -> None:
    publish_debate_event(
        "tool_observation",
        {
            "actor": actor,
            "phase": phase,
            "tools": [_object_to_payload(tool) for tool in tools],
        },
    )


def publish_llm_prompt(
    *,
    actor: str,
    phase: str,
    model: str | None,
    system_prompt: str,
    user_prompt: str,
    temperature: float | None = None,
    tool_name: str | None = None,
    tool_description: str | None = None,
    input_schema: Mapping[str, Any] | None = None,
) -> None:
    publish_debate_event(
        "llm_prompt",
        {
            "actor": actor,
            "phase": phase,
            "model": model,
            "temperature": temperature,
            "tool_name": tool_name,
            "tool_description": tool_description,
            "input_schema": dict(input_schema) if input_schema is not None else None,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
        },
    )


def publish_llm_response(
    *,
    actor: str,
    phase: str,
    model: str | None,
    output: Any,
    tool_name: str | None = None,
) -> None:
    publish_debate_event(
        "llm_response",
        {
            "actor": actor,
            "phase": phase,
            "model": model,
            "tool_name": tool_name,
            "output": _object_to_payload(output),
        },
    )


def publish_llm_error(
    *,
    actor: str,
    phase: str,
    model: str | None,
    error: Exception | str,
    tool_name: str | None = None,
) -> None:
    publish_debate_event(
        "llm_error",
        {
            "actor": actor,
            "phase": phase,
            "model": model,
            "tool_name": tool_name,
            "error": str(error),
        },
    )


def publish_llm_skipped(
    *,
    actor: str,
    phase: str,
    reason: str,
    context: Mapping[str, Any] | None = None,
) -> None:
    publish_debate_event(
        "llm_skipped",
        {
            "actor": actor,
            "phase": phase,
            "reason": reason,
            "context": dict(context or {}),
        },
    )


def sanitize_event_payload(payload: Any) -> Any:
    return _sanitize_value(payload, depth=0)


def _sanitize_value(value: Any, *, depth: int) -> Any:
    if depth > 8:
        return "<truncated-depth>"
    if isinstance(value, Mapping):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if _is_sensitive_key(key_text):
                sanitized[key_text] = "<redacted>"
            else:
                sanitized[key_text] = _sanitize_value(item, depth=depth + 1)
        return sanitized
    if isinstance(value, str):
        return _limit_text(_redact_text(value))
    if isinstance(value, Sequence) and not isinstance(value, bytes | bytearray):
        items = list(value[:MAX_EVENT_LIST_ITEMS])
        result = [_sanitize_value(item, depth=depth + 1) for item in items]
        if len(value) > MAX_EVENT_LIST_ITEMS:
            result.append(f"<truncated-list {len(value) - MAX_EVENT_LIST_ITEMS} more items>")
        return result
    return value


def _object_to_payload(value: Any) -> Any:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return value


def _is_sensitive_key(key: str) -> bool:
    lowered = key.strip().lower()
    if lowered in SENSITIVE_EXACT_KEYS:
        return True
    return any(part in lowered for part in SENSITIVE_KEY_PARTS)


def _redact_text(value: str) -> str:
    return SENSITIVE_ASSIGNMENT.sub(r"\1<redacted>", value)


def _limit_text(value: str) -> str:
    if len(value) <= MAX_EVENT_TEXT_CHARS:
        return value
    omitted = len(value) - MAX_EVENT_TEXT_CHARS
    return value[:MAX_EVENT_TEXT_CHARS] + f"\n...[truncated {omitted} chars]"
