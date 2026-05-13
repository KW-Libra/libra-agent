"""HTTP / SSE 라우트 — 골격."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Body
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from libra_agent.api.sse import resume_and_stream, run_and_stream
from libra_agent.common.errors import ApiError, ErrorCode
from libra_agent.common.logging import get_logger

router = APIRouter()
log = get_logger(__name__)


# --- DTO (다음 단계에서 schemas/ 로 옮기고 정식 schema 로 대체) -----------------

class RunStartRequest(BaseModel):
    """현재 골격용. 다음 단계에서 JudgeRunRequest (contracts 차용 + spec §2) 로 대체."""
    query: str = Field(..., min_length=1)
    portfolio: dict[str, Any] | None = None
    trigger: str = Field(default="pull")
    thread_id: str | None = None


class ResumeRequest(BaseModel):
    """contracts/user-approval-response.schema.json 차용 — 다음 단계에서 정식 schema 로."""
    approved: bool
    decision: str | None = None             # APPROVE / REJECT / REVISE / DEFER
    option_index: int | None = None         # 0..2 (UDR 3옵션)
    override_plan: dict[str, float] | None = None
    note: str | None = None


# --- Routes -----------------------------------------------------------------

@router.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "UP",
        "service": "libra-agent",
        "now": datetime.now(timezone.utc).isoformat(),
    }


@router.post("/api/runs")
async def start_run(body: RunStartRequest) -> EventSourceResponse:
    thread_id = body.thread_id or str(uuid4())
    log.info("run.start", thread_id=thread_id, trigger=body.trigger, query=body.query[:80])
    return EventSourceResponse(run_and_stream(thread_id=thread_id, request=body))


@router.post("/api/runs/{thread_id}/resume")
async def resume_run(thread_id: str, body: ResumeRequest = Body(...)) -> EventSourceResponse:
    if not thread_id:
        raise ApiError(ErrorCode.VALIDATION_FAILED, "thread_id required")
    log.info(
        "run.resume",
        thread_id=thread_id,
        approved=body.approved,
        option_index=body.option_index,
    )
    return EventSourceResponse(resume_and_stream(thread_id=thread_id, request=body))
