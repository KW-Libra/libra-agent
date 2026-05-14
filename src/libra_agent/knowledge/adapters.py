from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, Field

KnowledgeDomain = Literal["news", "disclosure", "report", "profit"]


class DomainKnowledgeInput(BaseModel):
    """Stable input slice for one downstream domain agent."""

    domain: KnowledgeDomain
    documents: list[dict[str, Any]] = Field(default_factory=list)
    events: list[dict[str, Any]] = Field(default_factory=list)
    consensus_snapshots: list[dict[str, Any]] = Field(default_factory=list)
    financial_statements: list[dict[str, Any]] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)


class KnowledgeDomainInputs(BaseModel):
    """Knowledge payloads reshaped for domain-agent fan-out."""

    summary: dict[str, Any]
    common_events: list[dict[str, Any]] = Field(default_factory=list)
    news: DomainKnowledgeInput
    disclosure: DomainKnowledgeInput
    report: DomainKnowledgeInput
    profit: DomainKnowledgeInput


def build_domain_inputs(knowledge_snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Convert a KnowledgeSnapshot dict into stable domain-agent inputs.

    This intentionally performs only deterministic routing and counting. Domain
    scoring, prompt shaping, and trade judgment remain separate agent concerns.
    """

    summary = _mapping(knowledge_snapshot.get("summary"))
    payloads = _mapping(knowledge_snapshot.get("payloads"))

    documents = _list_from_payload(payloads.get("normalized_documents"), "documents")
    events = _list_from_payload(payloads.get("events"), "events")
    consensus_snapshots = _list_from_payload(payloads.get("consensus_snapshot"), "snapshots")
    financial_statements = _list_from_payload(payloads.get("financial_statement"), "statements")

    news_documents = _documents_by_type(documents, "NEWS")
    disclosure_documents = _documents_by_type(documents, "DISCLOSURE")
    report_documents = _documents_by_type(documents, "REPORT")

    news_events = _events_for_documents(events, _document_ids(news_documents))
    disclosure_events = _events_for_documents(events, _document_ids(disclosure_documents))
    report_events = _events_for_documents(events, _document_ids(report_documents))

    statement_tickers = _tickers_from_records(financial_statements)
    profit_events = _events_for_tickers(events, statement_tickers)

    report_tickers = _tickers_from_events(report_events) or _tickers_from_records(
        consensus_snapshots
    )
    filtered_snapshots = _records_for_tickers(consensus_snapshots, report_tickers)

    domain_inputs = KnowledgeDomainInputs(
        summary={
            **summary,
            "domain_counts": {
                "news": _domain_counts(news_documents, news_events),
                "disclosure": _domain_counts(disclosure_documents, disclosure_events),
                "report": _domain_counts(
                    report_documents,
                    report_events,
                    consensus_snapshots=filtered_snapshots,
                ),
                "profit": _domain_counts(
                    [],
                    profit_events,
                    financial_statements=financial_statements,
                ),
            },
        },
        common_events=events,
        news=DomainKnowledgeInput(
            domain="news",
            documents=news_documents,
            events=news_events,
            summary=_slice_summary(summary, news_documents, news_events),
        ),
        disclosure=DomainKnowledgeInput(
            domain="disclosure",
            documents=disclosure_documents,
            events=disclosure_events,
            summary=_slice_summary(summary, disclosure_documents, disclosure_events),
        ),
        report=DomainKnowledgeInput(
            domain="report",
            documents=report_documents,
            events=report_events,
            consensus_snapshots=filtered_snapshots,
            summary=_slice_summary(
                summary,
                report_documents,
                report_events,
                consensus_snapshots=filtered_snapshots,
            ),
        ),
        profit=DomainKnowledgeInput(
            domain="profit",
            events=profit_events,
            financial_statements=financial_statements,
            summary=_slice_summary(
                summary,
                [],
                profit_events,
                financial_statements=financial_statements,
            ),
        ),
    )
    return domain_inputs.model_dump(mode="json")


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _list_from_payload(payload: Any, key: str) -> list[dict[str, Any]]:
    if not isinstance(payload, Mapping):
        return []
    value = payload.get(key)
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _documents_by_type(documents: Sequence[dict[str, Any]], doc_type: str) -> list[dict[str, Any]]:
    return [
        document
        for document in documents
        if str(document.get("doc_type", "")).upper() == doc_type
    ]


def _document_ids(documents: Iterable[Mapping[str, Any]]) -> set[str]:
    return {
        str(document["doc_id"])
        for document in documents
        if document.get("doc_id") is not None
    }


def _events_for_documents(
    events: Sequence[dict[str, Any]],
    document_ids: set[str],
) -> list[dict[str, Any]]:
    if not document_ids:
        return []
    matched: list[dict[str, Any]] = []
    for event in events:
        source_documents = event.get("source_documents")
        if not isinstance(source_documents, list):
            continue
        if any(_source_document_id(source) in document_ids for source in source_documents):
            matched.append(event)
    return matched


def _source_document_id(source: Any) -> str | None:
    if isinstance(source, str):
        return source
    if isinstance(source, Mapping):
        document_id = source.get("doc_id") or source.get("id")
        return str(document_id) if document_id is not None else None
    return None


def _tickers_from_records(records: Sequence[Mapping[str, Any]]) -> set[str]:
    tickers: set[str] = set()
    for record in records:
        ticker = record.get("ticker") or record.get("symbol")
        if ticker is not None:
            tickers.add(str(ticker))
    return tickers


def _tickers_from_events(events: Sequence[Mapping[str, Any]]) -> set[str]:
    tickers: set[str] = set()
    for event in events:
        entities = event.get("entities")
        if not isinstance(entities, list):
            continue
        for entity in entities:
            if isinstance(entity, Mapping) and entity.get("ticker") is not None:
                tickers.add(str(entity["ticker"]))
    return tickers


def _events_for_tickers(
    events: Sequence[dict[str, Any]],
    tickers: set[str],
) -> list[dict[str, Any]]:
    if not tickers:
        return []
    return [event for event in events if _tickers_from_events([event]) & tickers]


def _records_for_tickers(
    records: Sequence[dict[str, Any]],
    tickers: set[str],
) -> list[dict[str, Any]]:
    if not tickers:
        return list(records)
    return [
        record
        for record in records
        if str(record.get("ticker") or record.get("symbol")) in tickers
    ]


def _domain_counts(
    documents: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    *,
    consensus_snapshots: Sequence[Mapping[str, Any]] | None = None,
    financial_statements: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, int]:
    return {
        "documents": len(documents),
        "events": len(events),
        "consensus_snapshots": len(consensus_snapshots or []),
        "financial_statements": len(financial_statements or []),
    }


def _slice_summary(
    snapshot_summary: Mapping[str, Any],
    documents: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    *,
    consensus_snapshots: Sequence[Mapping[str, Any]] | None = None,
    financial_statements: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "available": bool(snapshot_summary.get("available")),
        "source": snapshot_summary.get("source"),
        "generated_at": snapshot_summary.get("generated_at"),
        "counts": _domain_counts(
            documents,
            events,
            consensus_snapshots=consensus_snapshots,
            financial_statements=financial_statements,
        ),
    }
