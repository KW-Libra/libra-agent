from __future__ import annotations

from typing import Any, Protocol

from libra_agent.errors import ChatClientError


class ChatClientProtocol(Protocol):
    model: str

    def chat_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
    ) -> dict[str, Any]: ...

    def chat_json_tool(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        tool_name: str,
        tool_description: str,
        input_schema: dict[str, Any],
        temperature: float = 0.1,
    ) -> dict[str, Any]: ...

    def ensure_available(self) -> None: ...


__all__ = ["ChatClientError", "ChatClientProtocol"]
