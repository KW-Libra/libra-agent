from __future__ import annotations

from libra_agent.contracts.run_events import RUN_EVENT_TYPES, RUN_NODE_NAMES, TERMINAL_EVENT_TYPES


def test_run_event_names_are_stable():
    assert RUN_EVENT_TYPES == (
        "run_started",
        "node_started",
        "node_completed",
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
        "run_completed",
        "run_failed",
    )
