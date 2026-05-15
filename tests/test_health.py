"""health 엔드포인트 스모크 테스트.

주의: TestClient 를 context manager 로 쓰면 lifespan 이 호출되며
AsyncPostgresSaver.setup() 이 진짜 Postgres 를 요구. 골격 테스트는
context manager *없이* 사용해 lifespan 우회 → routes 만 검증.

본격 통합 테스트는 다음 단계에서 Testcontainers 도입 시 별도 클래스로.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from libra_agent.main import app


def test_health_endpoint():
    client = TestClient(app)  # context manager X → lifespan 미실행
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "UP"
    assert body["service"] == "libra-agent"
    assert "now" in body


def test_internal_knowledge_current_endpoint_returns_summary():
    client = TestClient(app)
    response = client.get("/internal/knowledge/current")
    assert response.status_code == 200
    body = response.json()
    assert "summary" in body
    assert "file_locations" in body
    assert "payloads" not in body


def test_internal_knowledge_domain_inputs_endpoint_returns_agent_slices():
    client = TestClient(app)
    response = client.get("/internal/knowledge/domain-inputs")
    assert response.status_code == 200
    body = response.json()
    assert "summary" in body
    assert "common_events" in body
    assert {"news", "disclosure", "report", "profit"} <= set(body)
