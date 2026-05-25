from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from libra_agent.utils import parse_datetime_or_none

REQUIRED_BUNDLE_FIELDS = (
    "bundle_id",
    "as_of",
    "portfolio_id",
    "source_policy",
    "prices_until",
    "observed_count",
    "portfolio_relevant_count",
    "usable_for_trade_decision",
    "items",
)


class IngestBundleError(ValueError):
    """Raised when a point-in-time ingest bundle is not safe to use."""


def validate_ingest_bundle(bundle: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_BUNDLE_FIELDS:
        if field not in bundle:
            errors.append(f"missing required field: {field}")

    items = bundle.get("items", [])
    documents = bundle.get("documents", [])
    if not isinstance(items, list):
        errors.append("items must be a list")
        items = []
    if documents is None:
        documents = []
    if not isinstance(documents, list):
        errors.append("documents must be a list")
        documents = []

    observed_count = _as_int(bundle.get("observed_count"))
    if observed_count is not None and observed_count != len(items):
        errors.append(f"observed_count mismatch: {observed_count} != {len(items)}")
    document_count = _as_int(bundle.get("document_count"))
    if document_count is not None and document_count != len(documents):
        errors.append(f"document_count mismatch: {document_count} != {len(documents)}")
    portfolio_relevant_count = _as_int(bundle.get("portfolio_relevant_count"))
    if portfolio_relevant_count is not None and portfolio_relevant_count > len(items):
        errors.append("portfolio_relevant_count cannot exceed items length")

    as_of = parse_datetime_or_none(str(bundle.get("as_of") or ""))
    if as_of is None and "as_of" in bundle:
        errors.append("as_of must be a parseable datetime")

    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            errors.append(f"items[{index}] must be an object")
            continue
        published_at = parse_datetime_or_none(
            str(item.get("event_time") or item.get("published_at") or item.get("date") or "")
        )
        if published_at is None:
            errors.append(f"items[{index}] is missing a parseable event time")
        elif as_of is not None and published_at > as_of:
            errors.append(f"items[{index}] is after bundle as_of")
        if not str(item.get("headline") or "").strip():
            errors.append(f"items[{index}] is missing headline")

    news_documents = 0
    for index, document in enumerate(documents):
        if not isinstance(document, Mapping):
            errors.append(f"documents[{index}] must be an object")
            continue
        doc_type = str(document.get("doc_type") or "").upper()
        published_at = parse_datetime_or_none(_document_published_at(document))
        if published_at is None:
            errors.append(f"documents[{index}] is missing a parseable published_at")
        elif as_of is not None and published_at > as_of:
            errors.append(f"documents[{index}] is after bundle as_of")
        if doc_type == "NEWS":
            news_documents += 1
            if not _document_body(document):
                errors.append(f"documents[{index}] NEWS document must include article body")
            if _document_uses_title_fallback(document):
                errors.append(f"documents[{index}] NEWS document must not use title/summary fallback")

    has_news_item = any(
        isinstance(item, Mapping)
        and (
            "news" in {str(tag).casefold() for tag in _as_list(item.get("tags"))}
            or "news" in str(item.get("event_type") or "").casefold()
            or "news" in str(item.get("source") or "").casefold()
        )
        for item in items
    )
    if has_news_item and news_documents == 0:
        errors.append("news events require at least one NEWS document with article body")
    return errors


def knowledge_payload_from_ingest_bundle(
    bundle: Mapping[str, Any],
    *,
    source_path: str | None = None,
) -> dict[str, Any]:
    errors = validate_ingest_bundle(bundle)
    if errors:
        raise IngestBundleError("; ".join(errors))

    source = source_path or str(bundle.get("bundle_id") or "inline:ingest_bundle")
    documents = [
        _document_for_local_knowledge(document)
        for document in bundle.get("documents", [])
        if isinstance(document, Mapping)
    ]
    return {
        "events": [dict(item) for item in bundle.get("items", []) if isinstance(item, Mapping)],
        "documents": documents,
        "source_paths": {
            "ingest_bundle": source,
            "events": source,
            "documents": source,
            "ingest_refresh_enabled": "false",
        },
    }


def _document_for_local_knowledge(document: Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(document.get("normalized_content"), Mapping):
        source_info = document.get("source_info")
        source_info = source_info if isinstance(source_info, Mapping) else {}
        return {
            "doc_id": str(document.get("doc_id") or ""),
            "doc_type": str(document.get("doc_type") or ""),
            "title": _document_title(document),
            "body": _document_body(document),
            "publisher": str(source_info.get("publisher") or ""),
            "source_name": str(source_info.get("source_name") or ""),
            "source_url": str(source_info.get("source_url") or ""),
            "region": str(source_info.get("region") or ""),
            "published_at": _document_published_at(document),
            "metadata": {"source_format": "normalized_document", "ingest_bundle": True},
        }
    return dict(document)


def _document_title(document: Mapping[str, Any]) -> str:
    normalized = document.get("normalized_content")
    if isinstance(normalized, Mapping):
        return str(normalized.get("title") or "")
    return str(document.get("title") or "")


def _document_body(document: Mapping[str, Any]) -> str:
    normalized = document.get("normalized_content")
    if isinstance(normalized, Mapping):
        return str(normalized.get("body") or "").strip()
    return str(document.get("body") or "").strip()


def _document_published_at(document: Mapping[str, Any]) -> str:
    timing = document.get("timing_info")
    if isinstance(timing, Mapping):
        return str(timing.get("published_at") or "")
    return str(document.get("published_at") or "")


def _document_uses_title_fallback(document: Mapping[str, Any]) -> bool:
    metadata = document.get("metadata")
    if isinstance(metadata, Mapping) and metadata.get("body_source") == "article_text":
        return False
    if isinstance(metadata, Mapping) and metadata.get("body_source"):
        return True

    source_info = document.get("source_info")
    source_info = source_info if isinstance(source_info, Mapping) else {}
    if str(source_info.get("source_type") or "").upper() == "CRAWLER":
        return False
    raw_content = document.get("raw_content")
    raw_content = raw_content if isinstance(raw_content, Mapping) else {}
    body = _document_body(document)
    summary = str(raw_content.get("summary_raw") or "").strip()
    return bool(str(document.get("doc_type") or "").upper() == "NEWS" and body and body == summary)


def _as_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []
