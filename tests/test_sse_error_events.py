from __future__ import annotations

import json
from types import SimpleNamespace

from libra_agent.api import sse
from libra_agent.common.correlation import trace_id_var
from libra_agent.common.errors import ApiError, ErrorCode


class FailingGraph:
    async def astream_events(self, *_args, **_kwargs):
        raise RuntimeError("secret-token-should-not-leak")
        yield


class ApiFailingGraph:
    async def astream_events(self, *_args, **_kwargs):
        raise ApiError(ErrorCode.VALIDATION_FAILED, "bad request")
        yield


def _request() -> SimpleNamespace:
    return SimpleNamespace(
        trigger="pull",
        query="점검",
        portfolio={},
        knowledge_sources=None,
        knowledge_base=None,
        portfolio_definition=None,
        trigger_event=None,
        governance_v1=None,
        depth="medium",
        deadline_seconds=None,
        approval_required=False,
        enable_human_interrupts=False,
    )


async def test_run_failed_event_does_not_leak_raw_exception(monkeypatch):
    monkeypatch.setattr(sse, "build_graph", lambda: FailingGraph())
    token = trace_id_var.set("trace-test")
    try:
        events = [event async for event in sse.run_and_stream("thread-1", _request())]
    finally:
        trace_id_var.reset(token)

    payload = json.loads(events[-1]["data"])
    assert events[-1]["event"] == "run_failed"
    assert payload["code"] == "AGENT_UPSTREAM_ERROR"
    assert payload["error"] == "agent run failed"
    assert payload["traceId"] == "trace-test"
    assert "secret-token" not in events[-1]["data"]


async def test_run_failed_event_preserves_api_error_code(monkeypatch):
    monkeypatch.setattr(sse, "build_graph", lambda: ApiFailingGraph())

    events = [event async for event in sse.run_and_stream("thread-1", _request())]
    payload = json.loads(events[-1]["data"])

    assert payload["code"] == "VALIDATION_FAILED"
    assert payload["error"] == "bad request"
