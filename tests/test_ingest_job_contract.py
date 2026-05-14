from __future__ import annotations

from libra_agent.contracts.ingest_jobs import (
    DEFAULT_DOMAIN_TTL_SECONDS,
    DEFAULT_KNOWLEDGE_ARTIFACTS,
    IngestDomain,
    IngestJobRecord,
    IngestJobStatus,
)


def test_ingest_job_contract_serializes_status_and_domains():
    job = IngestJobRecord(
        id="ing_20260514_005930_001",
        requested_by_thread_id="thread-123",
        ticker="005930",
        domains=[IngestDomain.NEWS, IngestDomain.CONSENSUS, IngestDomain.FINANCIAL],
    )

    payload = job.model_dump(mode="json")

    assert payload["status"] == "QUEUED"
    assert payload["domains"] == ["NEWS", "CONSENSUS", "FINANCIAL"]
    assert payload["freshness_policy"]["allow_stale_on_failure"] is True
    assert job.is_terminal() is False


def test_terminal_status_and_default_artifacts_are_explicit():
    job = IngestJobRecord(id="ing_done", ticker="005930", status=IngestJobStatus.SUCCEEDED)

    artifact_names = {item.file_name for item in DEFAULT_KNOWLEDGE_ARTIFACTS}

    assert job.is_terminal() is True
    assert "manifest.json" in artifact_names
    assert "events.json" in artifact_names
    assert "consensus_snapshot.json" in artifact_names
    assert "financial_statement.json" in artifact_names
    assert DEFAULT_DOMAIN_TTL_SECONDS[IngestDomain.DISCLOSURE] == 10 * 60
