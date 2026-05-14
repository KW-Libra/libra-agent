from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class IngestDomain(StrEnum):
    NEWS = "NEWS"
    DISCLOSURE = "DISCLOSURE"
    REPORT = "REPORT"
    CONSENSUS = "CONSENSUS"
    FINANCIAL = "FINANCIAL"


class IngestJobStatus(StrEnum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELED = "CANCELED"


class FreshnessPolicy(BaseModel):
    max_age_seconds: int = Field(default=900, ge=0)
    force_refresh: bool = False
    allow_stale_on_failure: bool = True


class KnowledgeArtifact(BaseModel):
    logical_name: Literal[
        "manifest",
        "normalized_documents",
        "events",
        "push_candidates",
        "consensus_snapshot",
        "financial_statement",
    ]
    file_name: str
    required: bool = True


class IngestJobRequest(BaseModel):
    requested_by_thread_id: str | None = None
    ticker: str = Field(..., min_length=1)
    domains: list[IngestDomain] = Field(default_factory=list)
    freshness_policy: FreshnessPolicy = Field(default_factory=FreshnessPolicy)
    priority: int = Field(default=100, ge=0, le=1000)
    reason: str | None = None


class IngestJobRecord(IngestJobRequest):
    id: str = Field(..., min_length=1)
    status: IngestJobStatus = IngestJobStatus.QUEUED
    output_s3_prefix: str | None = None
    output_local_path: str | None = None
    error_message: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    finished_at: datetime | None = None

    def is_terminal(self) -> bool:
        return self.status in {
            IngestJobStatus.SUCCEEDED,
            IngestJobStatus.FAILED,
            IngestJobStatus.CANCELED,
        }


DEFAULT_KNOWLEDGE_ARTIFACTS = [
    KnowledgeArtifact(logical_name="manifest", file_name="manifest.json"),
    KnowledgeArtifact(logical_name="normalized_documents", file_name="normalized_documents.json"),
    KnowledgeArtifact(logical_name="events", file_name="events.json"),
    KnowledgeArtifact(logical_name="push_candidates", file_name="push_candidates.json", required=False),
    KnowledgeArtifact(logical_name="consensus_snapshot", file_name="consensus_snapshot.json", required=False),
    KnowledgeArtifact(logical_name="financial_statement", file_name="financial_statement.json", required=False),
]


DEFAULT_DOMAIN_TTL_SECONDS: dict[IngestDomain, int] = {
    IngestDomain.NEWS: 15 * 60,
    IngestDomain.DISCLOSURE: 10 * 60,
    IngestDomain.REPORT: 24 * 60 * 60,
    IngestDomain.CONSENSUS: 24 * 60 * 60,
    IngestDomain.FINANCIAL: 24 * 60 * 60,
}
