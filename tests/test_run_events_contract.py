from __future__ import annotations

import json
from types import SimpleNamespace

from libra_agent.api.sse import _resume_payload, _run_completed_event
from libra_agent.contracts.run_events import RUN_EVENT_TYPES, RUN_NODE_NAMES, TERMINAL_EVENT_TYPES


def test_run_event_names_are_stable():
    assert RUN_EVENT_TYPES == (
        "run_started",
        "node_started",
        "node_completed",
        "judge_action",
        "agent_started",
        "agent_completed",
        "agent_failed",
        "tool_observation",
        "llm_prompt",
        "llm_response",
        "llm_error",
        "llm_skipped",
        "mediator_decision",
        "consensus_updated",
        "final_decision_draft",
        "interrupt_required",
        "resume_received",
        "resume_ignored",
        "run_completed",
        "run_failed",
    )


def test_run_node_names_are_workflow_checkpoints():
    assert RUN_NODE_NAMES == (
        "compliance_before",
        "round1",
        "mediator",
        "final_judge",
        "human_review",
    )


def test_terminal_events_are_explicit():
    assert TERMINAL_EVENT_TYPES == (
        "interrupt_required",
        "resume_ignored",
        "run_completed",
        "run_failed",
    )


def test_resume_ignored_is_terminal():
    assert "resume_ignored" in TERMINAL_EVENT_TYPES


def test_resume_payload_preserves_contract_fields():
    request = SimpleNamespace(
        approved=True,
        decision="APPROVE",
        interrupt_id="interrupt-1",
        option_index=0,
        override_decision="HOLD",
        override_plan={"005930": 0.1},
        note="ok",
        effective_at="2026-05-15T00:00:00+09:00",
        responder="user-1",
        metadata={"source": "ui"},
    )

    assert _resume_payload(request) == {
        "approved": True,
        "decision": "APPROVE",
        "interrupt_id": "interrupt-1",
        "option_index": 0,
        "override_decision": "HOLD",
        "override_plan": {"005930": 0.1},
        "note": "ok",
        "effective_at": "2026-05-15T00:00:00+09:00",
        "responder": "user-1",
        "metadata": {"source": "ui"},
    }


def test_run_completed_event_includes_runtime_state_payloads():
    event = _run_completed_event(
        "thread-1",
        {
            "final_decision": {
                "decision": "HOLD",
                "branch": "CONSENSUS",
                "requires_approval": False,
            },
            "agent_result": {"decision": {"decision": "HOLD"}},
            "approval_response": {"approved": True, "metadata": {"source": "ui"}},
            "run_status": "completed_after_resume",
        },
    )

    assert event["event"] == "run_completed"
    payload = json.loads(event["data"])
    assert payload["thread_id"] == "thread-1"
    assert payload["decision"] == "HOLD"
    assert payload["branch"] == "CONSENSUS"
    assert payload["run_status"] == "completed_after_resume"
    assert payload["final_decision"]["requires_approval"] is False
    assert payload["agent_result"] == {"decision": {"decision": "HOLD"}}
    assert payload["approval_response"] == {"approved": True, "metadata": {"source": "ui"}}
