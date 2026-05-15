"""FastAPI exception handlers — RFC 7807 ProblemDetail 응답.

Spring 의 ErrorCode enum 과 이름 동기화. 가능한 한 같은 카탈로그 유지.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from libra_agent.common.correlation import trace_id_var
from libra_agent.common.logging import get_logger

log = get_logger(__name__)


class ErrorCode(StrEnum):
    VALIDATION_FAILED = "VALIDATION_FAILED"
    RESOURCE_NOT_FOUND = "RESOURCE_NOT_FOUND"
    RUN_NOT_FOUND = "RUN_NOT_FOUND"
    RUN_ALREADY_TERMINATED = "RUN_ALREADY_TERMINATED"
    AGENT_UPSTREAM_ERROR = "AGENT_UPSTREAM_ERROR"
    INTERNAL_ERROR = "INTERNAL_ERROR"


_STATUS_BY_CODE: dict[ErrorCode, int] = {
    ErrorCode.VALIDATION_FAILED: 400,
    ErrorCode.RESOURCE_NOT_FOUND: 404,
    ErrorCode.RUN_NOT_FOUND: 404,
    ErrorCode.RUN_ALREADY_TERMINATED: 409,
    ErrorCode.AGENT_UPSTREAM_ERROR: 502,
    ErrorCode.INTERNAL_ERROR: 500,
}


class ApiError(Exception):
    """도메인 예외. handler 가 ProblemDetail JSON 으로 변환."""

    def __init__(self, code: ErrorCode, detail: str | None = None):
        self.code = code
        self.detail = detail or code.value
        super().__init__(self.detail)


def _problem(status: int, code: str, detail: str, request: Request) -> dict[str, Any]:
    return {
        "type": "about:blank",
        "title": code,
        "status": status,
        "detail": detail,
        "instance": str(request.url.path),
        "code": code,
        "traceId": trace_id_var.get() or None,
    }


async def api_error_handler(request: Request, exc: ApiError) -> JSONResponse:
    status = _STATUS_BY_CODE.get(exc.code, 500)
    log.warning("api_error", code=exc.code.value, detail=exc.detail, path=request.url.path)
    return JSONResponse(
        status_code=status,
        content=_problem(status, exc.code.value, exc.detail, request),
    )


async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    detail = "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors())
    log.warning("validation_failed", detail=detail, path=request.url.path)
    return JSONResponse(
        status_code=400,
        content=_problem(400, ErrorCode.VALIDATION_FAILED.value, detail, request),
    )


async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    log.exception("unhandled_error", path=request.url.path)
    return JSONResponse(
        status_code=500,
        content=_problem(
            500,
            ErrorCode.INTERNAL_ERROR.value,
            "내부 오류가 발생했습니다",
            request,
        ),
    )
