from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from fastapi.testclient import TestClient

from libra_agent.api import sse
from libra_agent.main import app
from libra_agent.runtime.debate_events import publish_debate_event


class FakeSseGraph:
    def __init__(self, *, emit_debate: bool = False) -> None:
        self.emit_debate = emit_debate
        self.initial_states: list[dict[str, Any]] = []
        self.resume_payloads: list[dict[str, Any]] = []
        self.thread_states: dict[str, SimpleNamespace] = {}

    async def astream_events(self, graph_input: Any, *, config: dict[str, Any], version: str):
        assert version == "v2"
        thread_id = config["configurable"]["thread_id"]

        if isinstance(graph_input, dict):
            self.initial_states.append(graph_input)
            if graph_input.get("enable_human_interrupts"):
                self.thread_states[thread_id] = SimpleNamespace(
                    interrupts=[
                        SimpleNamespace(
                            id="interrupt-1",
                            value={
                                "type": "human_approval",
                                "reason": "approval_required",
                                "decision": "HOLD",
                                "branch": "USER_APPROVAL_REQUIRED",
                            },
                        )
                    ],
                    values=dict(graph_input),
                )
            else:
                self.thread_states[thread_id] = SimpleNamespace(
                    interrupts=[],
                    values={
                        "run_status": "completed",
                        "final_decision": {
                            "decision": "HOLD",
                            "branch": "CONSENSUS",
                            "requires_approval": False,
                            "trades": [],
                            "reasoning": "e2e smoke completed",
                        },
                    },
                )
            nodes = ("compliance_before", "round1", "mediator", "final_judge", "human_review")
        else:
            resume_payload = dict(graph_input.resume or {})
            self.resume_payloads.append(resume_payload)
            self.thread_states[thread_id] = SimpleNamespace(
                interrupts=[],
                values={
                    "run_status": "completed_after_resume",
                    "final_decision": {
                        "decision": "HOLD",
                        "branch": "USER_APPROVAL_REQUIRED",
                        "requires_approval": True,
                        "trades": [],
                        "reasoning": "e2e resume completed",
                    },
                    "approval_response": resume_payload,
                },
            )
            nodes = ("human_review",)

        for node in nodes:
            yield {"event": "on_chain_start", "name": node}
            if self.emit_debate and node == "final_judge":
                publish_debate_event(
                    "judge_action",
                    {
                        "layer": "core",
                        "action": "CALL_AGENT",
                        "agent_id": "news",
                        "reason": "e2e debate event",
                    },
                )
                publish_debate_event(
                    "agent_started",
                    {"agent_id": "news", "layer": "core", "turn_number": 1},
                )
                publish_debate_event(
                    "agent_completed",
                    {
                        "agent_id": "news",
                        "layer": "core",
                        "turn_number": 1,
                        "confidence": 0.7,
                        "reasoning": "뉴스 에이전트가 근거를 점검했습니다.",
                    },
                )
            yield {"event": "on_chain_end", "name": node}

    async def aget_state(self, config: dict[str, Any]) -> SimpleNamespace:
        thread_id = config["configurable"]["thread_id"]
        return self.thread_states.get(thread_id, SimpleNamespace(interrupts=[], values={}))


def _parse_sse(text: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    current: dict[str, str] = {}
    for line in text.splitlines():
        if not line.strip():
            if current:
                events.append(
                    {
                        "event": current.get("event"),
                        "data": json.loads(current.get("data", "{}")),
                    }
                )
                current = {}
            continue
        if line.startswith("event: "):
            current["event"] = line.removeprefix("event: ")
        elif line.startswith("data: "):
            current["data"] = line.removeprefix("data: ")
    if current:
        events.append(
            {
                "event": current.get("event"),
                "data": json.loads(current.get("data", "{}")),
            }
        )
    return events


def _portfolio() -> dict[str, Any]:
    return {
        "generated_at": "2026-05-15T00:00:00+09:00",
        "holdings": [],
    }


def test_api_runs_e2e_streams_completed_run(monkeypatch):
    graph = FakeSseGraph()
    monkeypatch.setattr(sse, "build_graph", lambda: graph)
    client = TestClient(app)

    response = client.post(
        "/api/runs",
        json={
            "query": "e2e smoke",
            "portfolio": _portfolio(),
            "knowledge_base": {"events": [], "documents": [], "source_paths": {}},
            "thread_id": "e2e-complete",
        },
    )

    assert response.status_code == 200
    events = _parse_sse(response.text)
    assert [event["event"] for event in events] == [
        "run_started",
        "node_started",
        "node_completed",
        "node_started",
        "node_completed",
        "node_started",
        "node_completed",
        "node_started",
        "node_completed",
        "node_started",
        "node_completed",
        "run_completed",
    ]
    assert events[0]["data"]["thread_id"] == "e2e-complete"
    assert events[0]["data"]["enable_human_interrupts"] is False
    assert events[-1]["data"]["decision"] == "HOLD"
    assert graph.initial_states[0]["knowledge_base"]["events"] == []


def test_api_runs_e2e_interrupts_and_resumes(monkeypatch):
    graph = FakeSseGraph()
    monkeypatch.setattr(sse, "build_graph", lambda: graph)
    client = TestClient(app)

    start = client.post(
        "/api/runs",
        json={
            "query": "e2e approval",
            "portfolio": _portfolio(),
            "thread_id": "e2e-resume",
            "enable_human_interrupts": True,
        },
    )
    start_events = _parse_sse(start.text)

    assert start.status_code == 200
    assert start_events[-1]["event"] == "interrupt_required"
    assert start_events[-1]["data"]["interrupt_id"] == "interrupt-1"
    assert graph.initial_states[0]["enable_human_interrupts"] is True

    resumed = client.post(
        "/api/runs/e2e-resume/resume",
        json={
            "approved": True,
            "decision": "APPROVE",
            "interrupt_id": "interrupt-1",
            "option_index": 0,
            "override_decision": "HOLD",
            "metadata": {"source": "e2e"},
        },
    )
    resume_events = _parse_sse(resumed.text)

    assert resumed.status_code == 200
    assert resume_events[0]["event"] == "resume_received"
    assert resume_events[-1]["event"] == "run_completed"
    assert resume_events[-1]["data"]["run_status"] == "completed_after_resume"
    assert resume_events[-1]["data"]["approval_response"]["metadata"] == {"source": "e2e"}
    assert graph.resume_payloads[0]["override_decision"] == "HOLD"


def test_api_runs_e2e_streams_debate_events(monkeypatch):
    graph = FakeSseGraph(emit_debate=True)
    monkeypatch.setattr(sse, "build_graph", lambda: graph)
    client = TestClient(app)

    response = client.post(
        "/api/runs",
        json={
            "query": "e2e debate",
            "portfolio": _portfolio(),
            "thread_id": "e2e-debate",
        },
    )

    assert response.status_code == 200
    events = _parse_sse(response.text)
    event_names = [event["event"] for event in events]
    assert "judge_action" in event_names
    assert "agent_started" in event_names
    assert "agent_completed" in event_names
    completed = next(event for event in events if event["event"] == "agent_completed")
    assert completed["data"]["agent_id"] == "news"
    assert completed["data"]["confidence"] == 0.7
