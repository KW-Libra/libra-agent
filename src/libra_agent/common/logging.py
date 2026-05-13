"""structlog JSON 구조화 로깅.

`LOG_FORMAT=json` 이면 JSON Lines (운영), 아니면 콘솔 (dev).
traceId 는 ContextVar 에서 자동 추출되어 모든 로그에 박힘.
"""
from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

from libra_agent.common.correlation import trace_id_var
from libra_agent.config import settings


def _inject_trace_id(_logger: Any, _method: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    tid = trace_id_var.get()
    if tid:
        event_dict["traceId"] = tid
    event_dict.setdefault("service", "libra-agent")
    return event_dict


def configure_logging() -> None:
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        _inject_trace_id,
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    if settings.log_format == "json":
        renderer: Any = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, settings.log_level)
        ),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=settings.log_level,
    )


def get_logger(name: str = __name__) -> Any:
    return structlog.get_logger(name)
