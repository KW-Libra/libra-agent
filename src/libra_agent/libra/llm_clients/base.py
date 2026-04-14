from __future__ import annotations

from typing import Any, Protocol


class ChatClientError(RuntimeError):
    pass


class ChatClientProtocol(Protocol):
    model: str

    def chat_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
    ) -> dict[str, Any]:
        ...

    def ensure_available(self) -> None:
        ...
