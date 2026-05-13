"""traceId 컨텍스트 변수.

Spring 의 `X-Trace-Id` 헤더로 들어옴. 미들웨어에서 ContextVar 에 박아
같은 요청을 처리하는 async task 들이 자동으로 trace 컨텍스트 상속.
"""
from __future__ import annotations

from contextvars import ContextVar
from uuid import uuid4

trace_id_var: ContextVar[str] = ContextVar("trace_id", default="")


def new_trace_id() -> str:
    return str(uuid4())
