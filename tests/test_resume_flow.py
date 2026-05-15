from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from libra_agent.runtime.graph import (
    _knowledge_base_from_state,
    _knowledge_snapshot_for_state,
    build_graph,
)


async def test_approval_required_run_interrupts_and_resumes():
    graph = build_graph(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "test-thread"}}

    result = await graph.ainvoke(
        {
            "thread_id": "test-thread",
            "trigger": "test",
            "query": "smoke",
            "portfolio": {},
            "approval_required": True,
        },
        config=config,
    )
    snapshot = await graph.aget_state(config)

    assert "__interrupt__" in result
    assert snapshot.interrupts
    assert snapshot.interrupts[0].value["type"] == "human_approval"
    assert snapshot.interrupts[0].value["decision"] == "HOLD"

    resumed = await graph.ainvoke(
        Command(
            resume={
                "approved": True,
                "decision": "APPROVE",
                "option_index": 0,
                "note": "ok",
            }
        ),
        config=config,
    )

    assert resumed["run_status"] == "completed_after_resume"
    assert resumed["approval_response"]["approved"] is True
    assert resumed["approval_response"]["decision"] == "APPROVE"


async def test_enable_human_interrupts_run_interrupts():
    graph = build_graph(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "test-enable-human-interrupts"}}

    result = await graph.ainvoke(
        {
            "thread_id": "test-enable-human-interrupts",
            "trigger": "test",
            "query": "smoke",
            "portfolio": {},
            "enable_human_interrupts": True,
        },
        config=config,
    )
    snapshot = await graph.aget_state(config)

    assert "__interrupt__" in result
    assert snapshot.interrupts
    assert snapshot.interrupts[0].value["type"] == "human_approval"


def test_inline_knowledge_base_disables_ingest_refresh():
    knowledge_base = _knowledge_base_from_state(
        {
            "knowledge_base": {
                "events": [],
                "documents": [],
                "source_paths": {
                    "ingest_refresh_enabled": "true",
                    "events": "events.json",
                },
            }
        }
    )

    assert knowledge_base.source_paths["ingest_refresh_enabled"] == "false"


def test_inline_knowledge_base_drives_round1_snapshot():
    snapshot = _knowledge_snapshot_for_state(
        {
            "knowledge_base": {
                "events": [
                    {
                        "event_id": "evt-1",
                        "event_type": "NEWS",
                        "headline": "삼성전자 뉴스",
                        "summary": "테스트 이벤트",
                        "occurred_at": "2026-05-15T00:00:00+09:00",
                        "confidence": 0.8,
                    }
                ],
                "documents": [],
                "source_paths": {"events": "inline"},
            }
        }
    )

    assert snapshot["summary"]["source"] == "request"
    assert snapshot["summary"]["counts"]["events"] == 1
    assert snapshot["payloads"]["events"]["events"][0]["event_id"] == "evt-1"
