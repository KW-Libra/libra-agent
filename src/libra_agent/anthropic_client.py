from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import httpx

from .errors import ChatClientError


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
        timeout_seconds: float = 180.0,
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
        try:
            with self._http_client() as client:
                response = client.post(
                    f"{self.base_url}/v1/messages", json=payload, headers=self._headers()
                )
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise AnthropicClientError(f"Failed to call Anthropic Messages API: {exc}") from exc

        data = response.json()
        content = data.get("content")
        text = self._extract_text(content)
        if not text.strip():
            raise AnthropicClientError("Anthropic returned an empty response body.")
        return self._decode_json(text)

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

        return self._extract_tool_input(response.json().get("content"), tool_name=tool_name)

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
            timeout=self.timeout_seconds if timeout is None else timeout,
            transport=self.transport,
        )

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
