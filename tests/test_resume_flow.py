from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END

from libra_agent.runtime.graph import (
    _empty_portfolio_no_trade_decision_payload,
    _final_decision_from_agent_result,
    _knowledge_base_from_state,
    _knowledge_snapshot_for_state,
    _route_after_compliance_before,
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


async def test_empty_portfolio_check_fast_path_skips_round1_and_mediator():
    graph = build_graph(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "test-empty-fast-path"}}

    result = await graph.ainvoke(
        {
            "thread_id": "test-empty-fast-path",
            "trigger": "pull",
            "query": "현재 포트폴리오를 점검하고 유지/조정 필요성을 판단해줘.",
            "portfolio": {"holdings": []},
            "enable_human_interrupts": True,
        },
        config=config,
    )
    snapshot = await graph.aget_state(config)

    assert "__interrupt__" not in result
    assert not snapshot.interrupts
    assert result["run_status"] == "completed"
    assert result["final_decision"]["decision"] == "DEFER"
    assert result["final_decision"]["requires_approval"] is False
    assert result["agent_result"]["decision"]["follow_up_at"] is None
    assert result["agent_result"]["decision"]["needs_trade_evaluation"] is False
    assert "round1_opinions" not in result
    assert "mediator_decision" not in result


async def test_empty_portfolio_fast_path_stream_skips_empty_workflow_nodes():
    graph = build_graph(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "test-empty-fast-path-events"}}
    starts: list[str] = []

    async for event in graph.astream_events(
        {
            "thread_id": "test-empty-fast-path-events",
            "trigger": "pull",
            "query": "현재 포트폴리오를 점검하고 유지/조정 필요성을 판단해줘.",
            "portfolio": {"holdings": []},
            "enable_human_interrupts": True,
        },
        config=config,
        version="v2",
    ):
        if event.get("event") == "on_chain_start":
            starts.append(str(event.get("name")))

    assert "compliance_before" in starts
    assert "empty_portfolio_finalize" in starts
    assert "round1" not in starts
    assert "mediator" not in starts
    assert "final_judge" not in starts
    assert "human_review" not in starts


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


def test_route_after_compliance_before_fast_paths_empty_portfolio_check():
    assert (
        _route_after_compliance_before(
            {
                "trigger": "pull",
                "query": "현재 포트폴리오를 점검하고 유지/조정 필요성을 판단해줘.",
                "portfolio": {"holdings": []},
            }
        )
        == "empty_portfolio_finalize"
    )


def test_route_after_compliance_before_keeps_generation_request_on_normal_path():
    assert (
        _route_after_compliance_before(
            {
                "trigger": "pull",
                "query": "3천만 원으로 리스크 우선 초기 포트폴리오 후보를 만들어줘.",
                "portfolio": {"holdings": []},
            }
        )
        == "round1"
    )


def test_empty_portfolio_no_trade_decision_payload_is_stable():
    payload = _empty_portfolio_no_trade_decision_payload()

    assert payload == {
        "decision": "DEFER",
        "summary": "포트폴리오가 비어 있고 후보 리밸런싱 초안도 없어 지금 실행할 매수·매도 조정은 없습니다.",
        "confidence": 0.95,
        "urgency": "defer",
        "reasoning": "현재는 실행할 매매가 없으므로 사용자 승인은 필요하지 않습니다. 투자 검토를 시작하려면 초기 포트폴리오 후보 구성이 먼저 필요합니다.",
        "candidate_rebalance_plan": {},
        "needs_trade_evaluation": False,
        "follow_up_at": None,
        "feedback_checkpoint": None,
        "user_notification": {
            "level": "info",
            "body": "포트폴리오가 비어 있고 후보 리밸런싱 초안도 없어 지금 실행할 매수·매도 조정은 없습니다.",
            "action_required": False,
            "kind": "final_decision",
            "estimated_followup": None,
            "sent_at": None,
        },
        "options": [],
        "auto_safeguards": {},
    }


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
