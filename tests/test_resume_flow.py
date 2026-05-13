from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from libra_agent.runtime.graph import build_graph


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
