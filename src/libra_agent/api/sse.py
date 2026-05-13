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
from typing import Any, AsyncIterator

from libra_agent.common.logging import get_logger
from libra_agent.runtime.graph import build_graph

log = get_logger(__name__)


def _event(event_type: str, payload: dict[str, Any]) -> dict[str, str]:
    """sse_starlette 가 받는 dict 형식: {event, data}.

    SSE wire format:
      event: <event_type>
      data: <json string>
      \n
    """
    return {"event": event_type, "data": json.dumps(payload, ensure_ascii=False)}


async def run_and_stream(thread_id: str, request) -> AsyncIterator[dict[str, str]]:
    graph = build_graph()
    config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}

    yield _event("run_started", {
        "thread_id": thread_id,
        "trigger": request.trigger,
        "query": request.query,
    })

    initial_state = {
        "thread_id": thread_id,
        "trigger": request.trigger,
        "query": request.query,
        "portfolio": request.portfolio or {},
    }

    try:
        async for event in graph.astream_events(initial_state, config=config, version="v2"):
            kind = event.get("event")
            name = event.get("name")
            # 일단 노드 진입/완료 만 흘림. 다음 단계에서 정교화.
            if kind == "on_chain_start" and name in {
                "compliance_before", "round1", "mediator", "final_judge"
            }:
                yield _event("node_started", {"node": name})
            elif kind == "on_chain_end" and name in {
                "compliance_before", "round1", "mediator", "final_judge"
            }:
                yield _event("node_completed", {"node": name})
    except Exception as e:
        log.exception("run_failed", thread_id=thread_id)
        yield _event("run_failed", {"thread_id": thread_id, "error": str(e)})
        return

    # 최종 state
    snapshot = await graph.aget_state(config)
    final = (snapshot.values or {}).get("final_decision") or {}
    yield _event("run_completed", {
        "thread_id": thread_id,
        "decision": final.get("decision"),
        "branch": final.get("branch"),
    })


async def resume_and_stream(thread_id: str, request) -> AsyncIterator[dict[str, str]]:
    """interrupt() 후 사용자 응답을 받아 graph 재개.

    다음 단계에서:
      from langgraph.types import Command
      graph.ainvoke(Command(resume={"option_index": req.option_index, ...}), config)
    """
    yield _event("resume_received", {
        "thread_id": thread_id,
        "approved": request.approved,
        "option_index": request.option_index,
    })
    yield _event("resume_not_implemented", {
        "thread_id": thread_id,
        "note": "다음 단계에서 LangGraph Command(resume=...) 패턴으로 채움",
    })
