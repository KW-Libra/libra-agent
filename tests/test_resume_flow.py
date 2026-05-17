from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END

from libra_agent.runtime.graph import (
    _final_decision_from_agent_result,
    _knowledge_base_from_state,
    _knowledge_snapshot_for_state,
    _route_after_final_judge,
    build_graph,
)


async def test_approval_required_noop_run_completes_without_interrupt():
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

    assert "__interrupt__" not in result
    assert not snapshot.interrupts
    assert result["run_status"] == "completed"
    assert result["final_decision"]["requires_approval"] is False


async def test_enable_human_interrupts_noop_run_completes_without_interrupt():
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

    assert "__interrupt__" not in result
    assert not snapshot.interrupts
    assert result["run_status"] == "completed"
    assert result["final_decision"]["requires_approval"] is False


def test_actionable_rebalance_requires_approval_when_human_review_enabled():
    final = _final_decision_from_agent_result(
        {
            "decision": {
                "decision": "REBALANCE",
                "candidate_rebalance_plan": {"005930": 0.05},
                "summary": "리밸런싱 초안",
                "reasoning": "거래 초안이 있습니다.",
                "user_notification": {"action_required": False},
            }
        },
        approval_required=True,
    )

    assert final["requires_approval"] is True
    assert final["branch"] == "USER_APPROVAL_REQUIRED"


def test_route_after_final_judge_skips_human_review_when_no_approval():
    assert _route_after_final_judge({"final_decision": {"requires_approval": False}}) == END


def test_route_after_final_judge_enters_human_review_when_required():
    assert _route_after_final_judge({"final_decision": {"requires_approval": True}}) == "human_review"


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
