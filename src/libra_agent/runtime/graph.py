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

import asyncio
import os
from collections.abc import Mapping
from contextlib import ExitStack
from datetime import UTC, datetime
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from libra_agent.common.logging import get_logger
from libra_agent.knowledge import KnowledgeReader, build_domain_inputs
from libra_agent.libra.config import backend_config_from_env
from libra_agent.libra.direct_indexing import PortfolioDefinition
from libra_agent.libra.llm_clients import open_chat_client
from libra_agent.libra_models import PortfolioSnapshot, TriggerEvent
from libra_agent.libra_runtime import JudgeOrchestrator, LocalKnowledgeBase
from libra_agent.runtime.checkpointer import get_checkpointer
from libra_agent.runtime.debate_events import publish_llm_skipped

log = get_logger(__name__)


class GraphState(TypedDict, total=False):
    """그래프 state. 다음 단계에서 schemas/ 들로 강타입화."""

    thread_id: str
    trigger: str
    query: str
    portfolio: dict[str, Any]
    knowledge_sources: dict[str, Any] | None
    knowledge_base: dict[str, Any] | None
    portfolio_definition: dict[str, Any] | None
    trigger_event: dict[str, Any] | None
    governance_v1: dict[str, Any] | None
    depth: str
    deadline_seconds: int | None
    approval_required: bool
    enable_human_interrupts: bool
    knowledge_snapshot: dict[str, Any]
    domain_inputs: dict[str, Any]
    agent_result: dict[str, Any]
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
    knowledge_snapshot = _knowledge_snapshot_for_state(state)
    domain_inputs = build_domain_inputs(knowledge_snapshot)
    log.info(
        "node.round1.knowledge_loaded",
        thread_id=state.get("thread_id"),
        available=knowledge_snapshot["summary"]["available"],
        source=knowledge_snapshot["summary"]["source"],
        domain_counts=domain_inputs["summary"]["domain_counts"],
    )
    # 다음 단계: Send API 로 11개 에이전트 병렬 호출
    return {
        "knowledge_snapshot": knowledge_snapshot,
        "domain_inputs": domain_inputs,
        "round1_opinions": [],
    }


def _knowledge_snapshot_for_state(state: GraphState) -> dict[str, Any]:
    if isinstance(state.get("knowledge_base"), Mapping) or isinstance(
        state.get("knowledge_sources"),
        Mapping,
    ):
        knowledge_base = _knowledge_base_from_state(state)
        state_payload = knowledge_base.to_state_payload()
        loaded_at = datetime.now(UTC).isoformat()
        return {
            "summary": {
                "available": bool(knowledge_base.events or knowledge_base.documents),
                "source": "request",
                "loaded_at": loaded_at,
                "generated_at": loaded_at,
                "counts": {
                    "events": len(knowledge_base.events),
                    "normalized_documents": len(knowledge_base.documents),
                },
                "available_payloads": ["events", "normalized_documents"],
                "missing_files": [],
                "error": None,
            },
            "file_locations": dict(knowledge_base.source_paths),
            "payloads": {
                "events": {"events": state_payload["events"]},
                "normalized_documents": {"documents": state_payload["documents"]},
            },
        }

    return KnowledgeReader.from_settings().load_current().to_dict(include_payloads=True)


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
    approval_required = _human_review_enabled(state)
    if _should_run_agent_core(state):
        result = await asyncio.to_thread(_run_agent_core, state)
        decision = _final_decision_from_agent_result(result, approval_required=approval_required)
        return {
            "agent_result": result,
            "compliance_after": {"can_proceed": True, "violations": [], "state": "AFTER"},
            "final_decision": decision,
        }

    portfolio_payload = state.get("portfolio")
    holdings = (
        portfolio_payload.get("holdings", [])
        if isinstance(portfolio_payload, Mapping)
        and isinstance(portfolio_payload.get("holdings"), list)
        else []
    )
    publish_llm_skipped(
        actor="judge",
        phase="final_judge",
        reason=(
            "보유 종목 또는 포트폴리오 정의가 없고 현재 실행 환경에서 LLM 백엔드가 구성되지 않아 "
            "agent core를 실행하지 않았습니다."
        ),
        context={
            "has_portfolio": isinstance(portfolio_payload, Mapping),
            "holdings_count": len(holdings),
            "has_portfolio_definition": isinstance(state.get("portfolio_definition"), Mapping),
        },
    )
    return {
        "compliance_after": {"can_proceed": True, "violations": [], "state": "AFTER"},
        "final_decision": {
            "decision": "HOLD",
            "branch": "CONSENSUS",
            "requires_approval": False,
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
            "interrupt_id": value.get("interrupt_id"),
            "option_index": value.get("option_index"),
            "override_decision": value.get("override_decision"),
            "override_plan": value.get("override_plan"),
            "note": value.get("note"),
            "effective_at": value.get("effective_at"),
            "responder": value.get("responder"),
            "metadata": value.get("metadata"),
        }
    return {"approved": bool(value), "decision": None, "option_index": None}


def _has_portfolio_holdings(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    holdings = value.get("holdings")
    return isinstance(holdings, list) and any(isinstance(item, Mapping) for item in holdings)


def _should_run_agent_core(state: GraphState) -> bool:
    if _has_portfolio_holdings(state.get("portfolio")) or isinstance(
        state.get("portfolio_definition"), Mapping
    ):
        return True
    return isinstance(state.get("portfolio"), Mapping) and _llm_backend_configured()


def _llm_backend_configured() -> bool:
    return bool(
        os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("LIBRA_LLM_PROVIDER") in {"ollama", "llama_cpp"}
    )


def _human_review_enabled(state: GraphState) -> bool:
    return bool(state.get("enable_human_interrupts") or state.get("approval_required"))


def _run_agent_core(state: GraphState) -> dict[str, Any]:
    portfolio = _portfolio_from_state(state)
    knowledge_base = _knowledge_base_from_state(state)
    portfolio_definition = _portfolio_definition_from_state(state)
    trigger_event = _trigger_event_from_state(state)
    human_review_enabled = _human_review_enabled(state)

    with ExitStack() as stack:
        client = open_chat_client(
            backend_config_from_env(default_backend="anthropic"),
            stack=stack,
        )
        client.ensure_available()
        orchestrator = JudgeOrchestrator(client=client, checkpoint_path=None)
        if _governance_v1_execution_mode(state) == "primary":
            return orchestrator.run_v1_committee(
                query=str(state.get("query") or ""),
                portfolio=portfolio,
                knowledge_base=knowledge_base,
                portfolio_definition=portfolio_definition,
                depth=str(state.get("depth") or "medium"),
                trigger=str(state.get("trigger") or "pull"),
                trigger_event=trigger_event,
                deadline_seconds=state.get("deadline_seconds"),
                thread_id=str(state.get("thread_id") or "") or None,
                enable_human_interrupts=human_review_enabled,
            )
        return orchestrator.run(
            query=str(state.get("query") or ""),
            portfolio=portfolio,
            knowledge_base=knowledge_base,
            portfolio_definition=portfolio_definition,
            depth=str(state.get("depth") or "medium"),
            trigger=str(state.get("trigger") or "pull"),
            trigger_event=trigger_event,
            deadline_seconds=state.get("deadline_seconds"),
            thread_id=str(state.get("thread_id") or "") or None,
            enable_human_interrupts=human_review_enabled,
        )


def _governance_v1_execution_mode(state: GraphState) -> str:
    payload = state.get("governance_v1")
    if not isinstance(payload, Mapping):
        return ""
    return str(payload.get("execution_mode") or "").strip().casefold()


def _portfolio_from_state(state: GraphState) -> PortfolioSnapshot:
    raw = dict(state.get("portfolio") or {})
    raw.setdefault("generated_at", datetime.now(UTC).isoformat())
    return PortfolioSnapshot.from_dict(raw)


def _portfolio_definition_from_state(state: GraphState) -> PortfolioDefinition | None:
    payload = state.get("portfolio_definition")
    if isinstance(payload, Mapping):
        return PortfolioDefinition.from_dict(payload)
    return None


def _trigger_event_from_state(state: GraphState) -> TriggerEvent | None:
    payload = state.get("trigger_event")
    if isinstance(payload, Mapping):
        return TriggerEvent.from_dict(payload)
    return None


def _knowledge_base_from_state(state: GraphState) -> LocalKnowledgeBase:
    inline = state.get("knowledge_base")
    if isinstance(inline, Mapping):
        return LocalKnowledgeBase.from_state_payload(_without_ingest_refresh(inline))

    knowledge_sources = state.get("knowledge_sources")
    if isinstance(knowledge_sources, Mapping):
        knowledge_base = LocalKnowledgeBase.from_files(
            events_path=knowledge_sources.get("events"),
            normalized_documents_path=knowledge_sources.get("normalized_documents"),
            enriched_documents_path=knowledge_sources.get("enriched_documents"),
        )
        knowledge_base.source_paths.update(
            {str(key): str(value) for key, value in knowledge_sources.items() if value is not None}
        )
        knowledge_base.source_paths["ingest_refresh_enabled"] = "false"
        return knowledge_base

    raw_snapshot = state.get("knowledge_snapshot")
    snapshot = raw_snapshot if isinstance(raw_snapshot, Mapping) else {}
    payloads = snapshot.get("payloads") if isinstance(snapshot, Mapping) else {}
    payloads = payloads if isinstance(payloads, Mapping) else {}
    file_locations = snapshot.get("file_locations") if isinstance(snapshot, Mapping) else {}
    source_paths = dict(file_locations) if isinstance(file_locations, Mapping) else {}
    source_paths.setdefault("ingest_refresh_enabled", "false")

    events_payload = payloads.get("events")
    documents_payload = payloads.get("normalized_documents")
    events = _list_payload(events_payload, key="events")
    documents = _list_payload(documents_payload, key="documents")
    return LocalKnowledgeBase.from_state_payload(
        {
            "events": events,
            "documents": documents,
            "source_paths": source_paths,
        }
    )


def _without_ingest_refresh(payload: Mapping[str, Any]) -> dict[str, Any]:
    sanitized = dict(payload)
    source_paths = (
        dict(sanitized.get("source_paths"))
        if isinstance(sanitized.get("source_paths"), Mapping)
        else {}
    )
    source_paths["ingest_refresh_enabled"] = "false"
    sanitized["source_paths"] = source_paths
    return sanitized


def _list_payload(payload: Any, *, key: str) -> list[dict[str, Any]]:
    if isinstance(payload, Mapping):
        value = payload.get(key)
        if isinstance(value, list):
            return [dict(item) for item in value if isinstance(item, Mapping)]
    if isinstance(payload, list):
        return [dict(item) for item in payload if isinstance(item, Mapping)]
    return []


def _final_decision_from_agent_result(
    result: Mapping[str, Any],
    *,
    approval_required: bool,
) -> dict[str, Any]:
    decision = result.get("decision") if isinstance(result.get("decision"), Mapping) else {}
    notification = (
        decision.get("user_notification")
        if isinstance(decision.get("user_notification"), Mapping)
        else {}
    )
    decision_value = str(decision.get("decision") or "").upper()
    trades = decision.get("candidate_rebalance_plan") or {}
    has_trade_plan = isinstance(trades, Mapping) and bool(trades)
    explicit_action_required = (
        bool(notification.get("action_required")) or decision_value == "USER_DECISION_REQUIRED"
    )
    requires_approval = (
        explicit_action_required
        or (approval_required and has_trade_plan and decision_value in {"REBALANCE"})
    )
    return {
        "decision": decision.get("decision") or "HOLD",
        "branch": (
            "USER_APPROVAL_REQUIRED"
            if requires_approval
            else decision.get("branch") or decision.get("urgency") or "AGENT_RUNTIME"
        ),
        "requires_approval": requires_approval,
        "trades": trades,
        "reasoning": decision.get("reasoning") or decision.get("summary") or "",
        "summary": decision.get("summary") or "",
        "confidence": decision.get("confidence"),
        "urgency": decision.get("urgency"),
    }


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
