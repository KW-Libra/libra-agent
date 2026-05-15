from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient


def _is_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "y", "on"}


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


def _live_database_url() -> str:
    return (
        os.environ.get("LIBRA_LIVE_DATABASE_URL")
        or os.environ.get("LIBRA_INTEGRATION_DATABASE_URL")
        or os.environ.get("DATABASE_URL")
        or ""
    ).strip()


def _require_live_agent_env() -> str:
    if not _is_enabled("LIBRA_LIVE_E2E"):
        pytest.skip("set LIBRA_LIVE_E2E=1 to run live agent E2E")

    database_url = _live_database_url()
    if not database_url:
        pytest.skip("set LIBRA_LIVE_DATABASE_URL, LIBRA_INTEGRATION_DATABASE_URL, or DATABASE_URL")

    provider = os.environ.get("LIBRA_LLM_PROVIDER", "anthropic").strip().lower()
    if provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY", "").strip():
        pytest.skip("set ANTHROPIC_API_KEY for live Anthropic E2E")
    if provider == "gemini" and not os.environ.get("GEMINI_API_KEY", "").strip():
        pytest.skip("set GEMINI_API_KEY for live Gemini E2E")
    return database_url


def _sample_portfolio() -> dict[str, Any]:
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "holdings": [
            {
                "ticker": "005930",
                "company_name": "삼성전자",
                "weight": 0.62,
                "aliases": ["005930.KS", "삼성전자", "Samsung Electronics"],
                "sector": "semiconductor",
                "shares": 10,
                "last_price": 75000,
                "market_value_krw": 750000,
            },
            {
                "ticker": "035420",
                "company_name": "NAVER",
                "weight": 0.28,
                "aliases": ["035420.KS", "네이버", "NAVER"],
                "sector": "internet",
                "shares": 3,
                "last_price": 210000,
                "market_value_krw": 630000,
            },
        ],
        "total_value_krw": 1_500_000,
        "cash_weight": 0.10,
        "user_preferences": ["모의투자 계좌 기준", "과도한 회전율 회피", "리스크 우선"],
    }


def _sample_knowledge_base() -> dict[str, Any]:
    now = datetime.now(UTC).isoformat()
    entity_samsung = {
        "entity_id": "entity-005930",
        "entity_type": "company",
        "entity_name": "삼성전자",
        "ticker": "005930",
        "confidence": 0.95,
    }
    entity_naver = {
        "entity_id": "entity-035420",
        "entity_type": "company",
        "entity_name": "NAVER",
        "ticker": "035420",
        "confidence": 0.92,
    }
    return {
        "events": [
            {
                "event_id": "live-e2e-news-005930",
                "event_type": "DISCLOSURE",
                "event_time": now,
                "headline": "삼성전자 실적 전망 점검",
                "summary": "반도체 업황 회복 기대와 단기 변동성 요인이 동시에 관찰된다.",
                "confidence": 0.72,
                "source_documents": ["live-e2e-doc-005930"],
                "matched_holdings": ["005930"],
                "entities": [entity_samsung],
                "metadata": {"fixture": "live-e2e"},
            },
            {
                "event_id": "live-e2e-news-035420",
                "event_type": "PRODUCT",
                "event_time": now,
                "headline": "NAVER AI 서비스 투자 확대",
                "summary": "신규 AI 서비스와 비용 증가 가능성이 함께 언급되었다.",
                "confidence": 0.68,
                "source_documents": ["live-e2e-doc-035420"],
                "matched_holdings": ["035420"],
                "entities": [entity_naver],
                "metadata": {"fixture": "live-e2e"},
            },
        ],
        "documents": [
            {
                "doc_id": "live-e2e-doc-005930",
                "doc_type": "report",
                "title": "삼성전자 포트폴리오 점검 리포트",
                "body": "메모리 가격 회복과 설비투자 부담을 함께 고려해야 한다. 현재 비중은 높아 추가 매수보다 유지 판단을 우선 검토한다.",
                "publisher": "Libra Live E2E",
                "source_name": "local-live-fixture",
                "source_url": "https://example.invalid/live-e2e/005930",
                "region": "KR",
                "published_at": now,
                "relevance_score": 0.85,
                "event_type": "REPORT",
                "event_type_score": 0.7,
                "entities": [entity_samsung],
                "matched_holdings": ["005930"],
                "metadata": {"fixture": "live-e2e"},
            },
            {
                "doc_id": "live-e2e-doc-035420",
                "doc_type": "news",
                "title": "NAVER 비용 구조 점검",
                "body": "AI 투자 확대는 장기 성장 기대를 만들지만 단기 비용 증가와 마진 압박 가능성을 동반한다.",
                "publisher": "Libra Live E2E",
                "source_name": "local-live-fixture",
                "source_url": "https://example.invalid/live-e2e/035420",
                "region": "KR",
                "published_at": now,
                "relevance_score": 0.78,
                "event_type": "NEWS",
                "event_type_score": 0.65,
                "entities": [entity_naver],
                "matched_holdings": ["035420"],
                "metadata": {"fixture": "live-e2e"},
            },
        ],
        "source_paths": {"fixture": "tests/test_live_e2e.py", "ingest_refresh_enabled": "false"},
    }


@pytest.mark.live_e2e
def test_live_api_runs_real_llm_postgres_interrupt_and_resume(monkeypatch: pytest.MonkeyPatch):
    database_url = _require_live_agent_env()
    os.environ.setdefault("LIBRA_LLM_PROVIDER", "anthropic")
    os.environ.setdefault("LIBRA_DOMAIN_AGENTS_ENABLED", "false")

    from libra_agent import main as main_module
    from libra_agent.config import settings

    monkeypatch.setattr(settings, "database_url", database_url)

    thread_id = f"live-e2e-{uuid4()}"
    with TestClient(main_module.app, raise_server_exceptions=False) as client:
        response = client.post(
            "/api/runs",
            headers={"X-Trace-Id": thread_id},
            json={
                "thread_id": thread_id,
                "query": (
                    "live e2e smoke: 제공된 포트폴리오와 근거만 보고, "
                    "하위 에이전트가 필요하면 호출하되 최종 판단은 간결하게 내려줘."
                ),
                "portfolio": _sample_portfolio(),
                "knowledge_base": _sample_knowledge_base(),
                "trigger": "pull",
                "depth": os.environ.get("LIBRA_LIVE_E2E_DEPTH", "shallow"),
                "deadline_seconds": int(os.environ.get("LIBRA_LIVE_E2E_DEADLINE_SECONDS", "180")),
                "enable_human_interrupts": True,
            },
        )

        assert response.status_code == 200, response.text
        start_events = _parse_sse(response.text)
        start_event_names = {event["event"] for event in start_events}
        assert start_events[0]["event"] == "run_started"
        assert not any(event["event"] == "run_failed" for event in start_events), start_events[-1]
        assert "final_decision_draft" in start_event_names
        assert start_event_names.intersection({"judge_action", "agent_started", "agent_completed"})
        assert start_events[-1]["event"] == "interrupt_required", start_events[-1]
        assert start_events[-1]["data"]["thread_id"] == thread_id

        resume = client.post(
            f"/api/runs/{thread_id}/resume",
            headers={"X-Trace-Id": f"{thread_id}-resume"},
            json={
                "approved": True,
                "decision": "APPROVE",
                "interrupt_id": start_events[-1]["data"].get("interrupt_id"),
                "option_index": 0,
                "note": "live e2e approval",
                "metadata": {"source": "test_live_e2e"},
            },
        )

        assert resume.status_code == 200, resume.text
        resume_events = _parse_sse(resume.text)
        assert not any(event["event"] == "run_failed" for event in resume_events), resume_events[-1]
        assert resume_events[0]["event"] == "resume_received"
        assert resume_events[-1]["event"] == "run_completed", resume_events[-1]
        assert resume_events[-1]["data"]["run_status"] == "completed_after_resume"
        assert resume_events[-1]["data"]["approval_response"]["metadata"] == {
            "source": "test_live_e2e"
        }


def _kis_value(*names: str) -> str:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""


@pytest.mark.live_e2e
def test_live_kis_portfolio_read_only_smoke():
    if not _is_enabled("LIBRA_LIVE_KIS_E2E"):
        pytest.skip("set LIBRA_LIVE_KIS_E2E=1 to run live KIS portfolio read-only smoke")

    from libra_agent.libra.portfolio_sources.kis import build_kis_portfolio_snapshot

    args = SimpleNamespace(
        kis_env=os.environ.get("LIBRA_KIS_ENV", "demo"),
        kis_config=os.environ.get("LIBRA_KIS_CONFIG", ""),
        kis_app_key=_kis_value(
            "LIBRA_KIS_APP_KEY",
            "LIBRA_KIS_REAL_APP_KEY",
            "LIBRA_KIS_PAPER_APP_KEY",
        ),
        kis_app_secret=_kis_value(
            "LIBRA_KIS_APP_SECRET",
            "LIBRA_KIS_REAL_APP_SECRET",
            "LIBRA_KIS_PAPER_APP_SECRET",
        ),
        kis_account_no=_kis_value(
            "LIBRA_KIS_ACCOUNT_NO",
            "LIBRA_KIS_REAL_ACCOUNT_NO",
            "LIBRA_KIS_PAPER_ACCOUNT_NO",
        ),
        kis_product_code=_kis_value(
            "LIBRA_KIS_PRODUCT_CODE",
            "LIBRA_KIS_REAL_PRODUCT_CODE",
            "LIBRA_KIS_PAPER_PRODUCT_CODE",
        )
        or "01",
        kis_user_agent=os.environ.get("LIBRA_KIS_USER_AGENT", "LIBRA-LIVE-E2E/1.0"),
    )

    snapshot = build_kis_portfolio_snapshot(args)
    payload = snapshot.to_dict()
    assert payload["generated_at"]
    assert isinstance(payload["holdings"], list)
