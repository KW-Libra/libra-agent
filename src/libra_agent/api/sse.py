"""SSE event generator.

graph.astream_events() 결과를 spec §2 RunEvent (다음 단계) 로 매핑.
현재는 노드 진입/완료 만 흘리는 stub.

이벤트 종류 (계획):
  run_started
  compliance_check (BEFORE / AFTER)
  agent_started / agent_completed / agent_silent  (R1 ×11, R2 ×N)
  consensus_computed
  mediator_started / mediator_completed
  tentative_trades_computed
  branch_determined
  final_decision_completed
  interrupt_required  XOR  report_uploaded
  run_completed | run_failed
"""
from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from langgraph.types import Command

from libra_agent.common.logging import get_logger
from libra_agent.runtime.graph import build_graph

log = get_logger(__name__)

NODE_NAMES = {"compliance_before", "round1", "mediator", "final_judge", "human_review"}


def _event(event_type: str, payload: dict[str, Any]) -> dict[str, str]:
    """sse_starlette 가 받는 dict 형식: {event, data}.

    SSE wire format:
      event: <event_type>
      data: <json string>
      \n
    """
    return {"event": event_type, "data": json.dumps(payload, ensure_ascii=False)}


async def _stream_node_events(graph, graph_input: Any, config: dict[str, Any]):
    async for event in graph.astream_events(graph_input, config=config, version="v2"):
        kind = event.get("event")
        name = event.get("name")
        if kind == "on_chain_start" and name in NODE_NAMES:
            yield _event("node_started", {"node": name})
        elif kind == "on_chain_end" and name in NODE_NAMES:
            yield _event("node_completed", {"node": name})


def _serialize_interrupts(interrupts) -> list[dict[str, Any]]:
    return [
        {
            "id": getattr(item, "id", None),
            "value": getattr(item, "value", None),
        }
        for item in interrupts
    ]


def _interrupt_required_event(thread_id: str, interrupts) -> dict[str, str]:
    serialized = _serialize_interrupts(interrupts)
    payload: dict[str, Any] = {
        "thread_id": thread_id,
        "interrupts": serialized,
    }
    if serialized:
        payload["interrupt_id"] = serialized[0].get("id")
        value = serialized[0].get("value")
        if isinstance(value, dict):
            payload.update(value)
    return _event("interrupt_required", payload)


def _run_completed_event(thread_id: str, values: dict[str, Any]) -> dict[str, str]:
    final = values.get("final_decision") or {}
    payload: dict[str, Any] = {
        "thread_id": thread_id,
        "decision": final.get("decision"),
        "branch": final.get("branch"),
        "run_status": values.get("run_status", "completed"),
    }
    if values.get("approval_response") is not None:
        payload["approval_response"] = values.get("approval_response")
    return _event("run_completed", payload)


def _resume_payload(request) -> dict[str, Any]:
    return {
        "approved": request.approved,
        "decision": request.decision,
        "option_index": request.option_index,
        "override_plan": request.override_plan,
        "note": request.note,
    }


async def run_and_stream(thread_id: str, request) -> AsyncIterator[dict[str, str]]:
    graph = build_graph()
    config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}

    yield _event("run_started", {
        "thread_id": thread_id,
        "trigger": request.trigger,
        "query": request.query,
        "approval_required": getattr(request, "approval_required", False),
    })

    initial_state = {
        "thread_id": thread_id,
        "trigger": request.trigger,
        "query": request.query,
        "portfolio": request.portfolio or {},
        "approval_required": getattr(request, "approval_required", False),
    }

    try:
        async for event in _stream_node_events(graph, initial_state, config):
            yield event
    except Exception as e:
        log.exception("run_failed", thread_id=thread_id)
        yield _event("run_failed", {"thread_id": thread_id, "error": str(e)})
        return

    snapshot = await graph.aget_state(config)
    if snapshot.interrupts:
        yield _interrupt_required_event(thread_id, snapshot.interrupts)
        return

    yield _run_completed_event(thread_id, snapshot.values or {})


async def resume_and_stream(thread_id: str, request) -> AsyncIterator[dict[str, str]]:
    """interrupt() 후 사용자 응답을 받아 graph 재개."""
    graph = build_graph()
    config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}

    snapshot = await graph.aget_state(config)
    if not snapshot.interrupts:
        yield _event("resume_ignored", {
            "thread_id": thread_id,
            "reason": "no_pending_interrupt",
        })
        return

    yield _event("resume_received", {
        "thread_id": thread_id,
        "approved": request.approved,
        "option_index": request.option_index,
    })

    try:
        async for event in _stream_node_events(
            graph,
            Command(resume=_resume_payload(request)),
            config,
        ):
            yield event
    except Exception as e:
        log.exception("resume_failed", thread_id=thread_id)
        yield _event("run_failed", {"thread_id": thread_id, "error": str(e)})
        return

    snapshot = await graph.aget_state(config)
    if snapshot.interrupts:
        yield _interrupt_required_event(thread_id, snapshot.interrupts)
        return

    yield _run_completed_event(thread_id, snapshot.values or {})
