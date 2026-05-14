"""LangGraph 그래프 스켈레톤.

design_spec_v1.md §1.1 mermaid 흐름:
  INIT → compliance_before → round1 (병렬 11) → mediator → [round2 표적]
       → final_judge_tentative → compliance_after → final_judge_branch
       → END (or interrupt for HITL)

본 파일은 **노드 stub**. 다음 단계에서:
  - round1: 11 에이전트 병렬 (Send API)
  - mediator: Haiku + tool_use
  - final_judge: Sonnet + 4분기 branch 룰
  - compliance_*: 코드 (10 룰)
  - interrupt(): COMPLIANCE_VETO / STRONG_CONFLICT 분기에서
"""
from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from libra_agent.common.logging import get_logger
from libra_agent.knowledge import KnowledgeReader
from libra_agent.runtime.checkpointer import get_checkpointer

log = get_logger(__name__)


class GraphState(TypedDict, total=False):
    """그래프 state. 다음 단계에서 schemas/ 들로 강타입화."""
    thread_id: str
    trigger: str
    query: str
    portfolio: dict[str, Any]
    approval_required: bool
    knowledge_snapshot: dict[str, Any]
    round1_opinions: list[dict[str, Any]]
    round2_opinions: list[dict[str, Any]]
    mediator_decision: dict[str, Any]
    tentative_trades: list[dict[str, Any]]
    compliance_before: dict[str, Any]
    compliance_after: dict[str, Any]
    final_decision: dict[str, Any]
    approval_request: dict[str, Any]
    approval_response: dict[str, Any]
    run_status: str
    error: str | None


async def _node_compliance_before(state: GraphState) -> dict[str, Any]:
    log.info("node.compliance_before", thread_id=state.get("thread_id"))
    return {"compliance_before": {"can_proceed": True, "violations": [], "state": "BEFORE"}}


async def _node_round1(state: GraphState) -> dict[str, Any]:
    log.info("node.round1", thread_id=state.get("thread_id"))
    knowledge_snapshot = KnowledgeReader.from_settings().load_current().to_dict(include_payloads=True)
    log.info(
        "node.round1.knowledge_loaded",
        thread_id=state.get("thread_id"),
        available=knowledge_snapshot["summary"]["available"],
        source=knowledge_snapshot["summary"]["source"],
    )
    # 다음 단계: Send API 로 11개 에이전트 병렬 호출
    return {"knowledge_snapshot": knowledge_snapshot, "round1_opinions": []}


async def _node_mediator(state: GraphState) -> dict[str, Any]:
    log.info("node.mediator", thread_id=state.get("thread_id"))
    return {
        "mediator_decision": {
            "targets_to_recall": [],
            "skip_round_2": True,
            "rationale": "(stub) — 다음 단계에서 채움",
        }
    }


async def _node_final_judge(state: GraphState) -> dict[str, Any]:
    log.info("node.final_judge", thread_id=state.get("thread_id"))
    approval_required = bool(state.get("approval_required"))
    return {
        "compliance_after": {"can_proceed": True, "violations": [], "state": "AFTER"},
        "final_decision": {
            "decision": "HOLD",
            "branch": "USER_APPROVAL_REQUIRED" if approval_required else "CONSENSUS",
            "requires_approval": approval_required,
            "trades": [],
            "reasoning": "(stub) — 다음 단계에서 채움",
        },
    }


async def _node_human_review(state: GraphState) -> dict[str, Any]:
    log.info("node.human_review", thread_id=state.get("thread_id"))
    final_decision = state.get("final_decision") or {}
    if not final_decision.get("requires_approval"):
        return {"run_status": "completed"}

    approval_request = {
        "type": "human_approval",
        "reason": "approval_required",
        "message": "최종 결정 적용 전에 사용자 확인이 필요합니다.",
        "decision": final_decision.get("decision"),
        "branch": final_decision.get("branch"),
        "options": [
            {"decision": "APPROVE", "label": "승인"},
            {"decision": "REJECT", "label": "거절"},
            {"decision": "REVISE", "label": "수정 요청"},
        ],
    }
    response = interrupt(approval_request)
    approval_response = _normalize_approval_response(response)

    return {
        "approval_request": approval_request,
        "approval_response": approval_response,
        "run_status": "completed_after_resume",
    }


def _normalize_approval_response(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return {
            "approved": bool(value.get("approved", False)),
            "decision": value.get("decision"),
            "option_index": value.get("option_index"),
            "override_plan": value.get("override_plan"),
            "note": value.get("note"),
        }
    return {"approved": bool(value), "decision": None, "option_index": None}


def build_graph(checkpointer=None):
    builder: StateGraph = StateGraph(GraphState)

    builder.add_node("compliance_before", _node_compliance_before)
    builder.add_node("round1", _node_round1)
    builder.add_node("mediator", _node_mediator)
    builder.add_node("final_judge", _node_final_judge)
    builder.add_node("human_review", _node_human_review)

    builder.add_edge(START, "compliance_before")
    builder.add_edge("compliance_before", "round1")
    builder.add_edge("round1", "mediator")
    builder.add_edge("mediator", "final_judge")
    builder.add_edge("final_judge", "human_review")
    builder.add_edge("human_review", END)

    return builder.compile(checkpointer=checkpointer or get_checkpointer())
