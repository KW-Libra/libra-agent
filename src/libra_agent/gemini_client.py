from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Mapping
from typing import Any

import httpx

from .errors import ChatClientError

DEFAULT_REQUEST_TIMEOUT_SECONDS = 45.0


class GeminiClientError(ChatClientError):
    pass


class GeminiChatClient:
    _throttle_lock = threading.Lock()
    _last_generate_at = 0.0

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "gemini-2.5-flash",
        base_url: str = "https://generativelanguage.googleapis.com",
        max_tokens: int = 4096,
        timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.api_key = api_key.strip()
        self.model = model.strip() or "gemini-2.5-flash"
        self.base_url = base_url.rstrip("/")
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
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": self.max_tokens,
                "responseMimeType": "application/json",
            },
        }
        retry_attempts = self._retry_attempts()
        for attempt in range(retry_attempts + 1):
            self._throttle_generate()
            try:
                with self._http_client() as client:
                    response = client.post(self._generate_url(), json=payload)
                    response.raise_for_status()
                break
            except httpx.HTTPStatusError as exc:
                if not self._should_retry_status(exc.response.status_code, attempt, retry_attempts):
                    raise GeminiClientError(
                        f"Failed to call Gemini API: {self._redact_api_key(exc)}"
                    ) from exc
                time.sleep(self._retry_delay_seconds(attempt))
            except httpx.HTTPError as exc:
                if attempt >= retry_attempts:
                    raise GeminiClientError(
                        f"Failed to call Gemini API: {self._redact_api_key(exc)}"
                    ) from exc
                time.sleep(self._retry_delay_seconds(attempt))
        else:  # pragma: no cover - loop always breaks or raises
            raise GeminiClientError("Failed to call Gemini API after retries.")

        text = self._extract_text(response.json())
        if not text.strip():
            raise GeminiClientError("Gemini returned an empty response body.")
        return self._decode_json(text)

    def ensure_available(self) -> None:
        self._validate_api_key()
        try:
            with self._http_client(timeout=min(self.timeout_seconds, 10.0)) as client:
                response = client.get(self._model_url())
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise GeminiClientError(
                f"Gemini API is not reachable at {self.base_url}: {self._redact_api_key(exc)}"
            ) from exc

    def _validate_api_key(self) -> None:
        if not self.api_key:
            raise GeminiClientError("GEMINI_API_KEY is required for the Gemini backend.")

    def _throttle_generate(self) -> None:
        delay_seconds = self._throttle_seconds()
        if delay_seconds <= 0:
            return
        with self._throttle_lock:
            now = time.monotonic()
            wait_seconds = delay_seconds - (now - self.__class__._last_generate_at)
            if wait_seconds > 0:
                time.sleep(wait_seconds)
            self.__class__._last_generate_at = time.monotonic()

    def _throttle_seconds(self) -> float:
        raw = os.environ.get("LIBRA_GEMINI_THROTTLE_SECONDS")
        if raw is not None:
            try:
                return max(0.0, float(raw))
            except ValueError:
                return 0.0
        free_tier = os.environ.get("LIBRA_GEMINI_FREE_TIER", "true").lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
        provider = os.environ.get("LIBRA_LLM_PROVIDER", "").lower()
        if free_tier and provider == "gemini":
            return 13.0
        return 0.0

    def _retry_attempts(self) -> int:
        raw = os.environ.get("LIBRA_GEMINI_RETRY_ATTEMPTS")
        if raw is None:
            return 2
        try:
            return max(0, int(raw))
        except ValueError:
            return 2

    def _retry_delay_seconds(self, attempt: int) -> float:
        raw = os.environ.get("LIBRA_GEMINI_RETRY_BASE_DELAY_SECONDS")
        try:
            base = max(0.0, float(raw)) if raw is not None else 2.0
        except ValueError:
            base = 2.0
        return base * (attempt + 1)

    def _should_retry_status(self, status_code: int, attempt: int, retry_attempts: int) -> bool:
        return attempt < retry_attempts and status_code in {429, 500, 502, 503, 504}

    def _redact_api_key(self, value: Any) -> str:
        text = str(value)
        if self.api_key:
            text = text.replace(self.api_key, "<redacted>")
        return text

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

    def _model_url(self) -> str:
        return f"{self.base_url}/v1beta/models/{self.model}?key={self.api_key}"

    def _generate_url(self) -> str:
        return f"{self.base_url}/v1beta/models/{self.model}:generateContent?key={self.api_key}"

    def _extract_text(self, payload: Any) -> str:
        if not isinstance(payload, Mapping):
            raise GeminiClientError("Gemini JSON response was not an object.")
        candidates = payload.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise GeminiClientError("Gemini response did not contain candidates.")
        content = candidates[0].get("content") if isinstance(candidates[0], Mapping) else None
        parts = content.get("parts") if isinstance(content, Mapping) else None
        if not isinstance(parts, list):
            raise GeminiClientError("Gemini response did not contain text parts.")
        text_parts: list[str] = []
        for item in parts:
            if isinstance(item, Mapping) and isinstance(item.get("text"), str):
                text_parts.append(str(item["text"]))
        return "".join(text_parts)

    def _decode_json(self, raw_text: str) -> dict[str, Any]:
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = self._strip_fence(cleaned)
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError:
            payload = self._extract_first_object(cleaned)
        if not isinstance(payload, Mapping):
            raise GeminiClientError("Gemini JSON response was not an object.")
        return dict(payload)

    def _strip_fence(self, raw_text: str) -> str:
        lines = raw_text.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()

    def _extract_first_object(self, raw_text: str) -> dict[str, Any]:
        start = raw_text.find("{")
        if start < 0:
            raise GeminiClientError("Gemini response did not contain JSON.")
        depth = 0
        in_string = False
        escape = False
        for index in range(start, len(raw_text)):
            char = raw_text[index]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    chunk = raw_text[start : index + 1]
                    try:
                        payload = json.loads(chunk)
                    except json.JSONDecodeError as exc:
                        raise GeminiClientError(
                            f"Failed to parse Gemini JSON chunk: {exc}"
                        ) from exc
                    if not isinstance(payload, Mapping):
                        raise GeminiClientError("Gemini JSON chunk was not an object.")
                    return dict(payload)
        raise GeminiClientError("Gemini response contained an unterminated JSON object.")
