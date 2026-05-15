from __future__ import annotations

from collections.abc import Callable
from contextvars import ContextVar
from typing import Any

DebateEventPublisher = Callable[[str, dict[str, Any]], None]

debate_event_publisher: ContextVar[DebateEventPublisher | None] = ContextVar(
    "debate_event_publisher",
    default=None,
)


def publish_debate_event(event: str, payload: dict[str, Any]) -> None:
    publisher = debate_event_publisher.get()
    if publisher is not None:
        publisher(event, payload)
