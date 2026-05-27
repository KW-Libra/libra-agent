from __future__ import annotations

import json
import os
from collections.abc import Mapping
from datetime import UTC, datetime
import threading
from typing import Any

import httpx

from .errors import ChatClientError

DEFAULT_REQUEST_TIMEOUT_SECONDS = 45.0
_USAGE_LOG_LOCK = threading.Lock()


class AnthropicClientError(ChatClientError):
    pass


class AnthropicChatClient:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = "https://api.anthropic.com",
        anthropic_version: str = "2023-06-01",
        max_tokens: int = 4096,
        timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.api_key = api_key.strip()
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.anthropic_version = anthropic_version
        self.max_tokens = int(max_tokens)
        self.timeout_seconds = timeout_seconds
        self.transport = transport

    def chat_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
    ) -> dict[str, Any]:
        self._validate_api_key()
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": temperature,
            "system": system_prompt,
            "messages": [
                {"role": "user", "content": user_prompt},
            ],
        }
        last_error: Exception | None = None
        for attempt in range(self._chat_json_attempts()):
            try:
                with self._http_client() as client:
                    response = client.post(
                        f"{self.base_url}/v1/messages", json=payload, headers=self._headers()
                    )
                    response.raise_for_status()
                data = response.json()
                self._log_usage(data)
                content = data.get("content")
                text = self._extract_text(content)
                if not text.strip():
                    raise AnthropicClientError("Anthropic returned an empty response body.")
                return self._decode_json(text)
            except httpx.HTTPStatusError as exc:
                last_error = exc
                if not self._should_retry_http_status(exc) or attempt + 1 >= self._chat_json_attempts():
                    break
            except (httpx.HTTPError, AnthropicClientError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt + 1 >= self._chat_json_attempts():
                    break

        if isinstance(last_error, AnthropicClientError):
            raise last_error
        raise AnthropicClientError(f"Failed to call Anthropic Messages API: {last_error}") from last_error

    def chat_json_tool(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        tool_name: str,
        tool_description: str,
        input_schema: Mapping[str, Any],
        temperature: float = 0.1,
    ) -> dict[str, Any]:
        self._validate_api_key()
        if not tool_name.strip():
            raise AnthropicClientError("tool_name is required for Anthropic tool JSON calls.")
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": temperature,
            "system": system_prompt,
            "messages": [
                {"role": "user", "content": user_prompt},
            ],
            "tools": [
                {
                    "name": tool_name,
                    "description": tool_description,
                    "input_schema": dict(input_schema),
                }
            ],
            "tool_choice": {"type": "tool", "name": tool_name},
        }
        try:
            with self._http_client() as client:
                response = client.post(
                    f"{self.base_url}/v1/messages", json=payload, headers=self._headers()
                )
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise AnthropicClientError(f"Failed to call Anthropic Messages API: {exc}") from exc

        data = response.json()
        self._log_usage(data)
        return self._extract_tool_input(data.get("content"), tool_name=tool_name)

    def ensure_available(self) -> None:
        self._validate_api_key()
        try:
            with self._http_client(timeout=min(self.timeout_seconds, 10.0)) as client:
                response = client.get(f"{self.base_url}/v1/models", headers=self._headers())
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise AnthropicClientError(
                f"Anthropic API is not reachable at {self.base_url}: {exc}"
            ) from exc

    def _validate_api_key(self) -> None:
        if not self.api_key:
            raise AnthropicClientError("ANTHROPIC_API_KEY is required for the Anthropic backend.")

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self.api_key,
            "anthropic-version": self.anthropic_version,
            "content-type": "application/json",
        }

    def _http_client(self, *, timeout: float | None = None) -> httpx.Client:
        return httpx.Client(
            timeout=self._effective_timeout(timeout),
            transport=self.transport,
        )

    def _effective_timeout(self, timeout: float | None = None) -> float:
        base_timeout = self.timeout_seconds if timeout is None else timeout
        try:
            base_timeout = float(base_timeout)
        except (TypeError, ValueError):
            base_timeout = DEFAULT_REQUEST_TIMEOUT_SECONDS
        return max(1.0, min(base_timeout, self._request_timeout_cap()))

    def _request_timeout_cap(self) -> float:
        raw = os.environ.get("LIBRA_LLM_REQUEST_TIMEOUT_SECONDS")
        if raw is None or not raw.strip():
            return DEFAULT_REQUEST_TIMEOUT_SECONDS
        try:
            return max(1.0, float(raw))
        except ValueError:
            return DEFAULT_REQUEST_TIMEOUT_SECONDS

    def _chat_json_attempts(self) -> int:
        raw = os.environ.get("LIBRA_ANTHROPIC_CHAT_JSON_ATTEMPTS", "2")
        try:
            return max(1, min(int(raw), 5))
        except ValueError:
            return 2

    def _should_retry_http_status(self, exc: httpx.HTTPStatusError) -> bool:
        status_code = exc.response.status_code
        return status_code == 408 or status_code == 409 or status_code == 429 or status_code >= 500

    def _log_usage(self, payload: Any) -> None:
        log_path = os.environ.get("LIBRA_LLM_USAGE_LOG")
        if not log_path or not log_path.strip() or not isinstance(payload, Mapping):
            return
        try:
            log_dir = os.path.dirname(log_path)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            record = {
                "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "provider": "anthropic",
                "model": self._string_or_none(payload.get("model")) or self.model,
                "id": self._string_or_none(payload.get("id")),
                "stop_reason": self._string_or_none(payload.get("stop_reason")),
                "usage": self._sanitize_usage(payload.get("usage")),
            }
            line = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            with _USAGE_LOG_LOCK:
                with open(log_path, "a", encoding="utf-8") as handle:
                    handle.write(line)
        except Exception:
            return

    def _string_or_none(self, value: Any) -> str | None:
        return value if isinstance(value, str) else None

    def _sanitize_usage(self, value: Any) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            return {}
        sanitized = self._sanitize_json_value(value)
        return dict(sanitized) if isinstance(sanitized, Mapping) else {}

    def _sanitize_json_value(self, value: Any) -> Any:
        if isinstance(value, Mapping):
            return {
                str(key): self._sanitize_json_value(item)
                for key, item in value.items()
                if isinstance(key, (str, int, float, bool))
            }
        if isinstance(value, list):
            return [self._sanitize_json_value(item) for item in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    def _extract_text(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            raise AnthropicClientError("Anthropic response did not contain content blocks.")
        text_parts: list[str] = []
        for item in content:
            if not isinstance(item, Mapping):
                continue
            if item.get("type") == "text" and isinstance(item.get("text"), str):
                text_parts.append(str(item["text"]))
            elif isinstance(item.get("text"), str):
                text_parts.append(str(item["text"]))
        return "".join(text_parts)

    def _extract_tool_input(self, content: Any, *, tool_name: str) -> dict[str, Any]:
        if not isinstance(content, list):
            raise AnthropicClientError("Anthropic tool response did not contain content blocks.")
        for item in content:
            if not isinstance(item, Mapping):
                continue
            if item.get("type") != "tool_use" or item.get("name") != tool_name:
                continue
            tool_input = item.get("input")
            if isinstance(tool_input, Mapping):
                return dict(tool_input)
            if isinstance(tool_input, str):
                return self._decode_json(tool_input)
            raise AnthropicClientError("Anthropic tool_use input was not an object.")
        raise AnthropicClientError(
            f"Anthropic response did not contain expected tool_use block: {tool_name}."
        )

    def _decode_json(self, raw_text: str) -> dict[str, Any]:
        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError:
            payload = self._extract_first_object(raw_text)
        if not isinstance(payload, Mapping):
            raise AnthropicClientError("Anthropic JSON response was not an object.")
        return dict(payload)

    def _extract_first_object(self, raw_text: str) -> dict[str, Any]:
        start = raw_text.find("{")
        if start < 0:
            raise AnthropicClientError("Anthropic response did not contain JSON.")
        depth = 0
        for index in range(start, len(raw_text)):
            char = raw_text[index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    chunk = raw_text[start : index + 1]
                    try:
                        payload = json.loads(chunk)
                    except json.JSONDecodeError as exc:
                        raise AnthropicClientError(
                            f"Failed to parse Anthropic JSON chunk: {exc}"
                        ) from exc
                    if not isinstance(payload, Mapping):
                        raise AnthropicClientError("Anthropic JSON chunk was not an object.")
                    return dict(payload)
        raise AnthropicClientError("Anthropic response contained an unterminated JSON object.")
