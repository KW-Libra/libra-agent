"""HTTP / SSE routes for the deployed agent API."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Any, Literal
from uuid import uuid4

from fastapi import APIRouter, Body
from pydantic import BaseModel, ConfigDict, Field
from sse_starlette.sse import EventSourceResponse

from libra_agent.api.sse import resume_and_stream, run_and_stream
from libra_agent.common.errors import ApiError, ErrorCode
from libra_agent.common.logging import get_logger
from libra_agent.knowledge import KnowledgeReader, build_domain_inputs

router = APIRouter()
log = get_logger(__name__)


class RunStartRequest(BaseModel):
    """Request accepted by the SSE `/api/runs` endpoint.

    `approval_required` is kept as a legacy alias while backend/frontend code
    moves to the contract field `enable_human_interrupts`.
    """

    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1)
    portfolio: dict[str, Any] = Field(...)
    knowledge_sources: dict[str, Any] | None = None
    knowledge_base: dict[str, Any] | None = None
    portfolio_definition: dict[str, Any] | None = None
    trigger_event: dict[str, Any] | None = None
    governance_v1: dict[str, Any] | None = None
    trigger: Literal["pull", "push"] = Field(default="pull")
    depth: Literal["shallow", "medium", "deep"] = Field(default="medium")
    deadline_seconds: int | None = Field(default=None, ge=1)
    thread_id: str | None = Field(default=None, min_length=1)
    enable_human_interrupts: bool = False
    approval_required: bool = False

    def human_review_enabled(self) -> bool:
        return bool(self.enable_human_interrupts or self.approval_required)


class ResumeRequest(BaseModel):
    """User approval payload.

    The schema intentionally allows additive client metadata. `option_index`
    remains for the current UI event contract, while `override_decision` matches
    the JSON contract.
    """

    model_config = ConfigDict(extra="allow")

    approved: bool
    decision: str | None = None  # APPROVE / REJECT / REVISE / DEFER
    interrupt_id: str | None = None
    option_index: int | None = None  # 0..2 (UDR 3옵션)
    override_decision: str | None = None
    override_plan: dict[str, float] | None = None
    note: str | None = None
    effective_at: str | None = None
    responder: str | None = None
    metadata: dict[str, Any] | None = None


# --- Routes -----------------------------------------------------------------


@router.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "UP",
        "service": "libra-agent",
        "now": datetime.now(UTC).isoformat(),
    }


@router.get("/internal/knowledge/current")
async def current_knowledge(include_payloads: bool = False) -> dict[str, Any]:
    snapshot = KnowledgeReader.from_settings().load_current()
    return snapshot.to_dict(include_payloads=include_payloads)


@router.get("/internal/knowledge/domain-inputs")
async def knowledge_domain_inputs() -> dict[str, Any]:
    snapshot = KnowledgeReader.from_settings().load_current().to_dict(include_payloads=True)
    return build_domain_inputs(snapshot)


@router.post("/api/runs")
async def start_run(body: RunStartRequest) -> EventSourceResponse:
    thread_id = body.thread_id or str(uuid4())
    log.info("run.start", thread_id=thread_id, trigger=body.trigger, query=body.query[:80])
    return EventSourceResponse(run_and_stream(thread_id=thread_id, request=body))


@router.post("/api/runs/{thread_id}/resume")
async def resume_run(
    thread_id: str,
    body: Annotated[ResumeRequest, Body()],
) -> EventSourceResponse:
    if not thread_id:
        raise ApiError(ErrorCode.VALIDATION_FAILED, "thread_id required")
    log.info(
        "run.resume",
        thread_id=thread_id,
        approved=body.approved,
        option_index=body.option_index,
    )
    return EventSourceResponse(resume_and_stream(thread_id=thread_id, request=body))
