from __future__ import annotations

import json
from typing import Any, Mapping

import httpx

from .errors import ChatClientError


class OllamaClientError(ChatClientError):
    pass


class OllamaChatClient:
    def __init__(
        self,
        *,
        model: str,
        host: str = "http://127.0.0.1:11434",
        timeout_seconds: float = 180.0,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def chat_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
    ) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "format": "json",
            "options": {
                "temperature": temperature,
            },
        }
        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                response = client.post(f"{self.host}/api/chat", json=payload)
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise OllamaClientError(f"Failed to call Ollama chat API: {exc}") from exc

        data = response.json()
        message = data.get("message", {})
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise OllamaClientError("Ollama returned an empty response body.")
        return self._decode_json(content)

    def ensure_available(self) -> None:
        try:
            with httpx.Client(timeout=min(self.timeout_seconds, 10.0)) as client:
                response = client.get(f"{self.host}/api/tags")
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise OllamaClientError(f"Ollama is not reachable at {self.host}: {exc}") from exc

    def _decode_json(self, raw_text: str) -> dict[str, Any]:
        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError:
            payload = self._extract_first_object(raw_text)
        if not isinstance(payload, Mapping):
            raise OllamaClientError("Ollama JSON response was not an object.")
        return dict(payload)

    def _extract_first_object(self, raw_text: str) -> dict[str, Any]:
        start = raw_text.find("{")
        if start < 0:
            raise OllamaClientError("Ollama response did not contain JSON.")
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
                        raise OllamaClientError(f"Failed to parse Ollama JSON chunk: {exc}") from exc
                    if not isinstance(payload, Mapping):
                        raise OllamaClientError("Ollama JSON chunk was not an object.")
                    return dict(payload)
        raise OllamaClientError("Ollama response contained an unterminated JSON object.")
