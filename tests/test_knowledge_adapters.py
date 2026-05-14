from __future__ import annotations

from libra_agent.knowledge import build_domain_inputs


def test_build_domain_inputs_splits_documents_and_domain_payloads():
    knowledge_snapshot = {
        "summary": {
            "available": True,
            "source": "local",
            "generated_at": "2026-05-14T00:00:00Z",
        },
        "payloads": {
            "normalized_documents": {
                "documents": [
                    {"doc_id": "news-1", "doc_type": "NEWS", "title": "뉴스"},
                    {"doc_id": "disc-1", "doc_type": "DISCLOSURE", "title": "공시"},
                    {"doc_id": "report-1", "doc_type": "REPORT", "title": "리포트"},
                ]
            },
            "events": {
                "events": [
                    {
                        "event_id": "event-1",
                        "entities": [{"ticker": "005930"}],
                        "source_documents": [
                            {"doc_id": "news-1"},
                            {"doc_id": "disc-1"},
                            {"doc_id": "report-1"},
                        ],
                    },
                    {
                        "event_id": "event-2",
                        "entities": [{"ticker": "000660"}],
                        "source_documents": [{"doc_id": "news-1"}],
                    },
                ]
            },
            "consensus_snapshot": {
                "snapshots": [
                    {"ticker": "005930", "target_price": 95000},
                    {"ticker": "000660", "target_price": 250000},
                ]
            },
            "financial_statement": {
                "statements": [
                    {"ticker": "005930", "fiscal_period": "2026-Q1", "revenue": 100},
                ]
            },
        },
    }

    domain_inputs = build_domain_inputs(knowledge_snapshot)

    assert domain_inputs["news"]["documents"][0]["doc_id"] == "news-1"
    assert domain_inputs["disclosure"]["documents"][0]["doc_id"] == "disc-1"
    assert domain_inputs["report"]["documents"][0]["doc_id"] == "report-1"
    assert [item["ticker"] for item in domain_inputs["report"]["consensus_snapshots"]] == [
        "005930"
    ]
    assert domain_inputs["profit"]["financial_statements"][0]["ticker"] == "005930"
    assert [event["event_id"] for event in domain_inputs["profit"]["events"]] == ["event-1"]
    assert domain_inputs["summary"]["domain_counts"]["news"]["documents"] == 1
    assert domain_inputs["summary"]["domain_counts"]["profit"]["financial_statements"] == 1


def test_build_domain_inputs_handles_missing_payloads():
    domain_inputs = build_domain_inputs({"summary": {"available": False, "source": "missing"}})

    assert domain_inputs["summary"]["available"] is False
    assert domain_inputs["common_events"] == []
    assert domain_inputs["news"]["documents"] == []
    assert domain_inputs["profit"]["financial_statements"] == []
