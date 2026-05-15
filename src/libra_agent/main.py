"""FastAPI 진입점.

구성:
  - lifespan: structlog 초기화 → AsyncPostgresSaver setup → (종료 시 풀 close)
  - CORS 미들웨어
  - traceId 미들웨어 (Spring 의 X-Trace-Id 또는 새 발급)
  - exception handlers (ProblemDetail 응답)
  - routes 등록
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware

from libra_agent.api.routes import router as api_router
from libra_agent.common.correlation import new_trace_id, trace_id_var
from libra_agent.common.errors import (
    ApiError,
    api_error_handler,
    unhandled_error_handler,
    validation_error_handler,
)
from libra_agent.common.logging import configure_logging, get_logger
from libra_agent.config import settings
from libra_agent.runtime.checkpointer import close_checkpointer, init_checkpointer

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    configure_logging()
    log.info("startup", host=settings.host, port=settings.port)
    await init_checkpointer()
    try:
        yield
    finally:
        await close_checkpointer()
        log.info("shutdown")


app = FastAPI(
    title="libra-agent",
    version="0.1.0",
    description="Libra 멀티에이전트 의사결정 거버넌스 — LangGraph + FastAPI",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Trace-Id"],
)


@app.middleware("http")
async def correlation_middleware(request: Request, call_next):
    trace_id = request.headers.get("X-Trace-Id") or new_trace_id()
    token = trace_id_var.set(trace_id)
    try:
        response = await call_next(request)
    finally:
        trace_id_var.reset(token)
    response.headers["X-Trace-Id"] = trace_id
    return response


app.add_exception_handler(ApiError, api_error_handler)
app.add_exception_handler(RequestValidationError, validation_error_handler)
app.add_exception_handler(Exception, unhandled_error_handler)

app.include_router(api_router)


def main() -> None:
    """Console-script entrypoint for the deployed FastAPI app."""
    import uvicorn

    uvicorn.run("libra_agent.main:app", host=settings.host, port=settings.port)
