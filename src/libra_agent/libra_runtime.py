from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Mapping

from .libra.prompts import (
    InformationAgentPromptProfile,
    JUDGE_ACTION_RULES,
    JUDGE_ACTION_SYSTEM_PROMPT,
    JUDGE_NOTIFICATION_LEVELS,
    JUDGE_PHASE_REQUIRED_KEYS,
    JUDGE_PHASE_SYSTEM_PROMPT,
    default_agent_fallback,
    default_agent_note,
    default_agent_query,
    get_information_prompt_profile,
)
from .libra.llm_clients.base import ChatClientError, ChatClientProtocol
from .libra_validation import (
    sanitize_agent_evidence,
    sanitize_agent_response_payload,
    sanitize_judge_payload,
)
from .libra_models import (
    AgentResponse,
    AgentVerdict,
    DecisionPhase,
    DecisionTraceNode,
    DecisionType,
    JudgeDecision,
    KnowledgeDocument,
    KnowledgeEntity,
    KnowledgeEvent,
    PortfolioSnapshot,
    TriggerEvent,
    ToolCall,
    Urgency,
    UserNotification,
)
from .utils import coerce_datetime, collapse_whitespace, stable_hash


ChatClient = ChatClientProtocol


POSITIVE_KEYWORDS = (
    "상회",
    "서프라이즈",
    "개선",
    "상향",
    "회복",
    "증가",
    "호조",
    "수주",
    "확대",
    "강세",
    "흑자",
    "improving",
    "beat",
    "upgrade",
    "recovery",
    "growth",
)

NEGATIVE_KEYWORDS = (
    "하회",
    "미스",
    "악화",
    "하향",
    "감소",
    "우려",
    "약세",
    "조사",
    "리콜",
    "소송",
    "규제",
    "화재",
    "적자",
    "miss",
    "downgrade",
    "probe",
    "investigation",
    "recall",
    "lawsuit",
    "weakness",
)

PUSH_RISK_KEYWORDS = (
    "조사",
    "규제",
    "리콜",
    "화재",
    "소송",
    "probe",
    "investigation",
    "lawsuit",
    "recall",
    "fire",
)

EVENT_TYPE_BIAS = {
    "EARNINGS": 0.35,
    "CAPEX": 0.15,
    "RESEARCH": 0.22,
    "DISCLOSURE": 0.08,
    "PRODUCT": 0.12,
    "FUNDING": 0.1,
    "GOVERNANCE": -0.08,
    "REGULATION": -0.28,
    "LEGAL": -0.35,
    "GEOPOLITICAL": -0.22,
    "MACRO": 0.0,
    "MNA": 0.18,
    "OTHER": 0.0,
}


def normalize_ticker(value: str) -> str:
    return "".join(char for char in value.upper() if char.isalnum())


def canonical_agent_id(value: str) -> str:
    normalized = value.strip().casefold()
    if normalized == "dart":
        return "disclosure"
    return normalized


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def truncate(value: str, limit: int = 700) -> str:
    text = collapse_whitespace(value)
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def event_direction_score(event: KnowledgeEvent) -> float:
    text = f"{event.headline}\n{event.summary}".casefold()
    score = EVENT_TYPE_BIAS.get(event.event_type, 0.0)
    positive_hits = sum(1 for token in POSITIVE_KEYWORDS if token in text)
    negative_hits = sum(1 for token in NEGATIVE_KEYWORDS if token in text)
    score += positive_hits * 0.16
    score -= negative_hits * 0.2
    score *= 0.5 + (event.confidence * 0.5)
    return clamp(score, -1.0, 1.0)


@dataclass(slots=True, frozen=True)
class KnowledgeSlice:
    events: list[KnowledgeEvent]
    documents: list[KnowledgeDocument]
    tools_called: list[ToolCall]


@dataclass(slots=True, frozen=True)
class PlannedAgentCall:
    agent_id: str
    query: str
    context: str
    depth: str
    fallback: str | None = None
    note: str | None = None


@dataclass(slots=True)
class RunState:
    trigger: str
    trigger_event: TriggerEvent | None
    started_at: datetime
    deadline_at: datetime | None
    notification_log: list[UserNotification]


class LocalKnowledgeBase:
    def __init__(
        self,
        *,
        events: list[KnowledgeEvent],
        documents: list[KnowledgeDocument],
        source_paths: dict[str, str],
    ) -> None:
        self.events = events
        self.documents = documents
        self.source_paths = source_paths
        self.documents_by_id = {document.doc_id: document for document in documents}

    @classmethod
    def from_files(
        cls,
        *,
        events_path: str | Path | None = None,
        enriched_documents_path: str | Path | None = None,
        normalized_documents_path: str | Path | None = None,
    ) -> LocalKnowledgeBase:
        events: list[KnowledgeEvent] = []
        source_paths: dict[str, str] = {}
        if events_path and Path(events_path).exists():
            events = [cls._event_from_payload(item) for item in cls._read_records(events_path)]
            source_paths["events"] = str(Path(events_path))

        documents: list[KnowledgeDocument] = []
        if normalized_documents_path and Path(normalized_documents_path).exists():
            source_paths["normalized_documents"] = str(Path(normalized_documents_path))
            documents = [
                cls._document_from_normalized_payload(item)
                for item in cls._read_records(normalized_documents_path)
            ]
        elif enriched_documents_path and Path(enriched_documents_path).exists():
            source_paths["enriched_documents"] = str(Path(enriched_documents_path))
            documents = [
                cls._document_from_enriched_payload(item)
                for item in cls._read_records(enriched_documents_path)
            ]
        return cls(events=events, documents=documents, source_paths=source_paths)

    @classmethod
    def from_state_payload(cls, payload: Mapping[str, Any]) -> LocalKnowledgeBase:
        events = [
            KnowledgeEvent.from_dict(item)
            for item in payload.get("events", [])
            if isinstance(item, Mapping)
        ]
        documents = [
            KnowledgeDocument.from_dict(item)
            for item in payload.get("documents", [])
            if isinstance(item, Mapping)
        ]
        source_paths = dict(payload.get("source_paths", {})) if isinstance(payload.get("source_paths"), Mapping) else {}
        return cls(events=events, documents=documents, source_paths=source_paths)

    def to_state_payload(self) -> dict[str, Any]:
        return {
            "events": [event.to_dict() for event in self.events],
            "documents": [document.to_dict() for document in self.documents],
            "source_paths": dict(self.source_paths),
        }

    @staticmethod
    def _read_json(path: str | Path) -> Any:
        return json.loads(Path(path).read_text(encoding="utf-8"))

    @classmethod
    def _read_records(cls, path: str | Path) -> list[Mapping[str, Any]]:
        file_path = Path(path)
        if file_path.suffix.lower() == ".jsonl":
            records: list[Mapping[str, Any]] = []
            with file_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    payload = json.loads(stripped)
                    if isinstance(payload, Mapping):
                        records.append(payload)
            return records
        return cls._as_records(cls._read_json(file_path))

    @staticmethod
    def _as_records(payload: Any) -> list[Mapping[str, Any]]:
        if isinstance(payload, Mapping):
            documents = payload.get("documents")
            if isinstance(documents, list):
                return [item for item in documents if isinstance(item, Mapping)]
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, Mapping)]
        return []

    @classmethod
    def _entities_from_payload(cls, payload: Any) -> tuple[KnowledgeEntity, ...]:
        if not isinstance(payload, list):
            return ()
        entities: list[KnowledgeEntity] = []
        for item in payload:
            if isinstance(item, Mapping):
                entities.append(KnowledgeEntity.from_dict(item))
        return tuple(entities)

    @classmethod
    def _document_from_enriched_payload(cls, payload: Mapping[str, Any]) -> KnowledgeDocument:
        document_payload = payload.get("document", {})
        source_info = document_payload.get("source_info", {}) if isinstance(document_payload, Mapping) else {}
        normalized_content = document_payload.get("normalized_content", {}) if isinstance(document_payload, Mapping) else {}
        timing_info = document_payload.get("timing_info", {}) if isinstance(document_payload, Mapping) else {}
        return KnowledgeDocument(
            doc_id=str(document_payload.get("doc_id", "")),
            doc_type=str(document_payload.get("doc_type", "")),
            title=str(normalized_content.get("title", "")),
            body=str(normalized_content.get("body", "")),
            publisher=str(source_info.get("publisher", "")),
            source_name=str(source_info.get("source_name", "")),
            source_url=str(source_info.get("source_url", "")),
            region=str(source_info.get("region", "")),
            published_at=coerce_datetime(str(timing_info.get("published_at", "")) or None),
            relevance_score=float(payload.get("relevance_score", 0.0)) if payload.get("relevance_score") is not None else None,
            event_type=str(payload.get("event_type", "")) or None,
            event_type_score=float(payload.get("event_type_score", 0.0)) if payload.get("event_type_score") is not None else None,
            entities=cls._entities_from_payload(payload.get("entities")),
            metadata=dict(payload.get("cluster_metadata", {})) if isinstance(payload.get("cluster_metadata"), Mapping) else {},
        )

    @classmethod
    def _document_from_normalized_payload(cls, payload: Mapping[str, Any]) -> KnowledgeDocument:
        source_info = payload.get("source_info", {}) if isinstance(payload, Mapping) else {}
        normalized_content = payload.get("normalized_content", {}) if isinstance(payload, Mapping) else {}
        timing_info = payload.get("timing_info", {}) if isinstance(payload, Mapping) else {}
        return KnowledgeDocument(
            doc_id=str(payload.get("doc_id", "")),
            doc_type=str(payload.get("doc_type", "")),
            title=str(normalized_content.get("title", "")),
            body=str(normalized_content.get("body", "")),
            publisher=str(source_info.get("publisher", "")),
            source_name=str(source_info.get("source_name", "")),
            source_url=str(source_info.get("source_url", "")),
            region=str(source_info.get("region", "")),
            published_at=coerce_datetime(str(timing_info.get("published_at", "")) or None),
            entities=(),
            metadata={},
        )

    @classmethod
    def _event_from_payload(cls, payload: Mapping[str, Any]) -> KnowledgeEvent:
        return KnowledgeEvent(
            event_id=str(payload.get("event_id", "")),
            event_type=str(payload.get("event_type", "")),
            event_time=coerce_datetime(str(payload.get("event_time", "")) or None),
            headline=str(payload.get("headline", "")),
            summary=str(payload.get("summary", "")),
            confidence=clamp(float(payload.get("confidence", 0.0)), 0.0, 1.0),
            source_documents=tuple(str(item) for item in payload.get("source_documents", []) if str(item)),
            entities=cls._entities_from_payload(payload.get("entities")),
            metadata=dict(payload.get("metadata", {})) if isinstance(payload.get("metadata"), Mapping) else {},
        )

    def slice_for_agent(
        self,
        *,
        agent_id: str,
        portfolio: PortfolioSnapshot,
        query: str,
        depth: str = "medium",
    ) -> KnowledgeSlice:
        agent_id = canonical_agent_id(agent_id)
        alias_map = self._alias_map(portfolio)
        relevant_events = self._filter_events(agent_id=agent_id, alias_map=alias_map, query=query)
        relevant_documents = self._filter_documents(agent_id=agent_id, alias_map=alias_map, query=query)

        event_limit = {"shallow": 4, "medium": 8, "deep": 12}.get(depth, 8)
        document_limit = {"shallow": 3, "medium": 6, "deep": 9}.get(depth, 6)
        relevant_events = relevant_events[:event_limit]
        relevant_documents = relevant_documents[:document_limit]

        tools_called = [
            ToolCall(
                tool_name="local_knowledge.load_events",
                purpose=f"{agent_id} agent event context",
                summary=f"Loaded {len(relevant_events)} relevant local events from {self.source_paths.get('events', 'events')}.",
            ),
            ToolCall(
                tool_name="local_knowledge.load_documents",
                purpose=f"{agent_id} agent document context",
                summary=f"Loaded {len(relevant_documents)} relevant local documents from the normalized cache.",
            ),
        ]
        return KnowledgeSlice(events=relevant_events, documents=relevant_documents, tools_called=tools_called)

    def ticker_signal(self, ticker: str, portfolio: PortfolioSnapshot) -> float:
        alias_map = self._alias_map(portfolio)
        normalized_ticker = normalize_ticker(ticker)
        matches = []
        for event in self.events:
            matched = self._match_tickers(
                headline=event.headline,
                body=event.summary,
                entities=event.entities,
                alias_map=alias_map,
            )
            if normalized_ticker in matched:
                matches.append(self._event_direction(event))
        if not matches:
            return 0.0
        return clamp(sum(matches) / len(matches), -1.0, 1.0)

    def _filter_events(
        self,
        *,
        agent_id: str,
        alias_map: dict[str, set[str]],
        query: str,
    ) -> list[KnowledgeEvent]:
        results: list[KnowledgeEvent] = []
        agent_id = canonical_agent_id(agent_id)
        wants_macro = any(token in query.casefold() for token in ("거시", "macro", "환율", "금리", "지수"))
        for event in sorted(self.events, key=lambda item: item.event_time, reverse=True):
            matched = self._match_tickers(
                headline=event.headline,
                body=event.summary,
                entities=event.entities,
                alias_map=alias_map,
            )
            if agent_id == "disclosure" and event.event_type not in {"DISCLOSURE", "EARNINGS"}:
                continue
            if agent_id == "report" and event.event_type not in {"RESEARCH", "EARNINGS", "DISCLOSURE"}:
                continue
            if agent_id == "news" and event.event_type == "RESEARCH":
                continue
            if matched or event.event_type == "MACRO" or wants_macro:
                results.append(
                    KnowledgeEvent(
                        event_id=event.event_id,
                        event_type=event.event_type,
                        event_time=event.event_time,
                        headline=event.headline,
                        summary=event.summary,
                        confidence=event.confidence,
                        source_documents=event.source_documents,
                        matched_holdings=tuple(sorted(matched)),
                        entities=event.entities,
                        metadata=event.metadata,
                    )
                )
        return results

    def _filter_documents(
        self,
        *,
        agent_id: str,
        alias_map: dict[str, set[str]],
        query: str,
    ) -> list[KnowledgeDocument]:
        agent_id = canonical_agent_id(agent_id)
        doc_type_filter = {
            "disclosure": {"DISCLOSURE"},
            "news": {"NEWS"},
            "report": {"REPORT"},
        }.get(agent_id, set())
        results: list[KnowledgeDocument] = []
        wants_macro = any(token in query.casefold() for token in ("거시", "macro", "환율", "금리", "지수"))
        for document in sorted(self.documents, key=lambda item: item.published_at, reverse=True):
            if doc_type_filter and document.doc_type not in doc_type_filter:
                continue
            matched = self._match_tickers(
                headline=document.title,
                body=document.body,
                entities=document.entities,
                alias_map=alias_map,
            )
            if matched or (agent_id == "news" and wants_macro and document.doc_type == "NEWS"):
                results.append(
                    KnowledgeDocument(
                        doc_id=document.doc_id,
                        doc_type=document.doc_type,
                        title=document.title,
                        body=document.body,
                        publisher=document.publisher,
                        source_name=document.source_name,
                        source_url=document.source_url,
                        region=document.region,
                        published_at=document.published_at,
                        relevance_score=document.relevance_score,
                        event_type=document.event_type,
                        event_type_score=document.event_type_score,
                        entities=document.entities,
                        matched_holdings=tuple(sorted(matched)),
                        metadata=document.metadata,
                    )
                )
        return results

    def _alias_map(self, portfolio: PortfolioSnapshot) -> dict[str, set[str]]:
        alias_map: dict[str, set[str]] = {}
        for holding in portfolio.holdings:
            normalized = normalize_ticker(holding.ticker)
            aliases = {
                holding.company_name.casefold(),
                holding.ticker.casefold(),
                normalized.casefold(),
            }
            short_numeric = "".join(char for char in holding.ticker if char.isdigit())
            if short_numeric:
                aliases.add(short_numeric.casefold())
            for alias in holding.aliases:
                aliases.add(alias.casefold())
            alias_map[normalized] = {alias for alias in aliases if alias}
        return alias_map

    def _match_tickers(
        self,
        *,
        headline: str,
        body: str,
        entities: tuple[KnowledgeEntity, ...],
        alias_map: dict[str, set[str]],
    ) -> set[str]:
        matched: set[str] = set()
        haystack = f"{headline}\n{body}".casefold()
        entity_tickers = {
            normalize_ticker(entity.ticker)
            for entity in entities
            if entity.ticker
        }
        for ticker, aliases in alias_map.items():
            if ticker in entity_tickers:
                matched.add(ticker)
                continue
            for alias in aliases:
                if alias and alias in haystack:
                    matched.add(ticker)
                    break
        return matched

    def _event_direction(self, event: KnowledgeEvent) -> float:
        return event_direction_score(event)


class LLMAgent:
    def __init__(
        self,
        *,
        agent_id: str,
        client: ChatClient,
        prompt_profile: InformationAgentPromptProfile | None = None,
    ) -> None:
        self.agent_id = canonical_agent_id(agent_id)
        self.client = client
        self.prompt_profile = prompt_profile or get_information_prompt_profile(self.agent_id)

    def run(
        self,
        *,
        query: str,
        context: str | None = None,
        fallback: str | None = None,
        note: str | None = None,
        turn_number: int,
        portfolio: PortfolioSnapshot,
        knowledge_base: LocalKnowledgeBase,
        depth: str = "medium",
    ) -> AgentResponse:
        knowledge_slice = knowledge_base.slice_for_agent(
            agent_id=self.agent_id,
            portfolio=portfolio,
            query=query,
            depth=depth,
        )
        opinion_id = f"{self.agent_id}_{stable_hash({'agent': self.agent_id, 'turn': turn_number, 'query': query})[:12]}"

        if not knowledge_slice.events and not knowledge_slice.documents:
            verdict = AgentVerdict.QUIET if self.agent_id == "news" else AgentVerdict.DIRECT_ANSWER_UNAVAILABLE
            response = AgentResponse(
                agent_id=self.agent_id,
                opinion_id=opinion_id,
                turn_number=turn_number,
                query_understood=query,
                verdict=verdict,
                evidence=sanitize_agent_evidence(agent_id=self.agent_id, evidence={}, portfolio=portfolio),
                direction=0.0,
                strength=0.0,
                urgency=Urgency.DEFER,
                confidence=0.2,
                reasoning_for_judge_agent="Relevant local context was not available for this agent.",
                limits_acknowledged="Current local cache does not contain matching items for the portfolio holdings.",
                tools_called=knowledge_slice.tools_called,
                depth_used=depth,
            )
            return sanitize_agent_response_payload(
                response.to_dict(),
                agent_id=self.agent_id,
                portfolio=portfolio,
                query=query,
                turn_number=turn_number,
                opinion_id=opinion_id,
                depth=depth,
            )

        system_prompt = self._system_prompt()
        user_prompt = self._user_prompt(
            query=query,
            context=context,
            fallback=fallback,
            note=note,
            portfolio=portfolio,
            knowledge_slice=knowledge_slice,
            depth=depth,
            turn_number=turn_number,
        )
        try:
            raw_response = self.client.chat_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.0,
            )
            response = sanitize_agent_response_payload(
                raw_response,
                agent_id=self.agent_id,
                portfolio=portfolio,
                query=query,
                turn_number=turn_number,
                opinion_id=opinion_id,
                depth=depth,
            )
        except (ChatClientError, ValueError, TypeError):
            response = self._fallback_response(
                query=query,
                turn_number=turn_number,
                knowledge_slice=knowledge_slice,
                depth=depth,
                opinion_id=opinion_id,
                failure_reason="Structured agent generation failed for this turn.",
            )

        response.agent_id = self.agent_id
        response.opinion_id = response.opinion_id or opinion_id
        response.turn_number = turn_number
        response.query_understood = response.query_understood or query
        response.direction = clamp(response.direction, -1.0, 1.0)
        response.strength = clamp(response.strength, 0.0, 1.0)
        response.confidence = clamp(response.confidence, 0.0, 1.0)
        response.depth_used = depth
        response.tools_called = knowledge_slice.tools_called
        if not response.evidence:
            response.evidence = {
                "events": [event.to_dict() for event in knowledge_slice.events[:4]],
                "documents": [document.to_dict() for document in knowledge_slice.documents[:3]],
            }
        if not response.focus_tickers:
            focus_tickers = set()
            for event in knowledge_slice.events:
                focus_tickers.update(event.matched_holdings)
            for document in knowledge_slice.documents:
                focus_tickers.update(document.matched_holdings)
            response.focus_tickers = sorted(focus_tickers)
        if self._is_low_signal_response(response):
            response = self._fallback_response(
                query=query,
                turn_number=turn_number,
                knowledge_slice=knowledge_slice,
                depth=depth,
                opinion_id=response.opinion_id,
                failure_reason="Local LLM response was too sparse to trust.",
            )
        response = sanitize_agent_response_payload(
            response.to_dict(),
            agent_id=self.agent_id,
            portfolio=portfolio,
            query=query,
            turn_number=turn_number,
            opinion_id=opinion_id,
            depth=depth,
        )
        return response

    def _is_low_signal_response(self, response: AgentResponse) -> bool:
        if response.confidence > 0 and response.reasoning_for_judge_agent.strip():
            return False
        if response.direction != 0 or response.strength != 0:
            return False
        if response.verdict not in {AgentVerdict.PARTIAL_ANSWER, AgentVerdict.DIRECT_ANSWER_UNAVAILABLE, AgentVerdict.QUIET}:
            return False
        return True

    def _fallback_response(
        self,
        *,
        query: str,
        turn_number: int,
        knowledge_slice: KnowledgeSlice,
        depth: str,
        opinion_id: str,
        failure_reason: str,
    ) -> AgentResponse:
        event_scores = [event_direction_score(event) for event in knowledge_slice.events]
        avg_score = sum(event_scores) / len(event_scores) if event_scores else 0.0
        confidence = clamp(0.35 + (0.05 * min(len(knowledge_slice.events), 5)) + (0.03 * min(len(knowledge_slice.documents), 5)), 0.0, 0.72)
        strength = clamp(abs(avg_score) + (0.04 * min(len(knowledge_slice.events), 3)), 0.0, 1.0)
        urgency = Urgency.WATCH if abs(avg_score) >= 0.25 else Urgency.DEFER
        focus_tickers = sorted({ticker for event in knowledge_slice.events for ticker in event.matched_holdings} | {ticker for document in knowledge_slice.documents for ticker in document.matched_holdings})
        evidence = self._fallback_evidence(knowledge_slice)
        reasoning = self._fallback_reasoning(knowledge_slice, avg_score)
        return AgentResponse(
            agent_id=self.agent_id,
            opinion_id=opinion_id,
            turn_number=turn_number,
            query_understood=query,
            verdict=AgentVerdict.PARTIAL_ANSWER,
            evidence=evidence,
            direction=clamp(avg_score, -1.0, 1.0),
            strength=strength,
            urgency=urgency,
            confidence=confidence,
            reasoning_for_judge_agent=reasoning,
            limits_acknowledged=failure_reason,
            tools_called=knowledge_slice.tools_called,
            depth_used=depth,
            focus_tickers=focus_tickers,
        )

    def _fallback_reasoning(self, knowledge_slice: KnowledgeSlice, avg_score: float) -> str:
        if self.agent_id == "disclosure":
            return f"Local disclosure fallback found {len(knowledge_slice.documents)} disclosure docs and {len(knowledge_slice.events)} related events; directional score {avg_score:.2f}."
        if self.agent_id == "report":
            return f"Local report fallback summarized {len(knowledge_slice.documents)} report docs; directional score {avg_score:.2f}."
        return f"Local news fallback summarized {len(knowledge_slice.events)} events and {len(knowledge_slice.documents)} news docs; directional score {avg_score:.2f}."

    def _fallback_evidence(self, knowledge_slice: KnowledgeSlice) -> dict[str, Any]:
        if self.agent_id == "disclosure":
            items = []
            for document in knowledge_slice.documents[:4]:
                items.append(
                    {
                        "ticker": next(iter(document.matched_holdings), None),
                        "company_name": None,
                        "disclosure_type": document.title,
                        "headline": document.title,
                        "timestamp": document.published_at.isoformat(),
                        "summary": truncate(document.body, 240),
                    }
                )
            return {
                "found_count": len(knowledge_slice.documents),
                "items": items,
                "upcoming_disclosures": [],
            }
        if self.agent_id == "report":
            items = []
            for document in knowledge_slice.documents[:4]:
                items.append(
                    {
                        "broker": document.publisher,
                        "published_at": document.published_at.isoformat(),
                        "report_type": "coverage",
                        "key_thesis": truncate(document.body, 220),
                        "matched_holdings": list(document.matched_holdings),
                    }
                )
            return {
                "coverage_reports_count": len(knowledge_slice.documents),
                "preview_reports_count": 0,
                "items": items,
                "consensus": None,
            }
        sub_role = "company_specific"
        if any(event.event_type == "MACRO" for event in knowledge_slice.events):
            sub_role = "mixed" if any(event.matched_holdings for event in knowledge_slice.events) else "macro"
        unique_sources = {
            document.source_name for document in knowledge_slice.documents
        } | {
            source_name
            for event in knowledge_slice.events
            for source_name in event.metadata.get("source_names", [])
            if isinstance(event.metadata.get("source_names"), list)
        }
        company_findings: dict[str, Any] = {}
        for event in knowledge_slice.events[:6]:
            for ticker in event.matched_holdings or ("portfolio",):
                company_findings.setdefault(
                    ticker,
                    {
                        "sentiment": "neutral",
                        "key_headlines": [],
                        "market_reaction": None,
                        "sector_comparison": None,
                    },
                )
                company_findings[ticker]["key_headlines"].append(event.headline)
        for ticker, payload in company_findings.items():
            joined = " ".join(payload["key_headlines"]).casefold()
            positive_hits = sum(1 for token in POSITIVE_KEYWORDS if token in joined)
            negative_hits = sum(1 for token in NEGATIVE_KEYWORDS if token in joined)
            if positive_hits > negative_hits:
                payload["sentiment"] = "positive"
            elif negative_hits > positive_hits:
                payload["sentiment"] = "negative"
            elif positive_hits or negative_hits:
                payload["sentiment"] = "mixed"
        return {
            "sub_role": sub_role,
            "company_findings": company_findings,
            "macro_findings": None,
            "source_reliability": "medium",
            "cross_check_count": len(unique_sources) or len(knowledge_slice.events),
        }

    def _system_prompt(self) -> str:
        return self.prompt_profile.system_prompt

    def _user_prompt(
        self,
        *,
        query: str,
        context: str | None,
        fallback: str | None,
        note: str | None,
        portfolio: PortfolioSnapshot,
        knowledge_slice: KnowledgeSlice,
        depth: str,
        turn_number: int,
    ) -> str:
        title_limit, body_limit, event_summary_limit = self._snippet_limits(depth)
        sections = [
            f"agent={self.agent_id}",
            f"turn={turn_number}",
            f"depth={depth}",
            f"query={query}",
        ]
        if context:
            sections.append(f"context={context}")
        if fallback:
            sections.append(f"fallback={fallback}")
        if note:
            sections.append(f"note={note}")
        sections.append("portfolio:")
        sections.extend(
            f"- {holding['ticker']} {holding['company_name']} weight={holding['weight']}"
            for holding in self._compact_portfolio(portfolio)["holdings"]
        )
        sections.append("preferences:")
        sections.extend(f"- {item}" for item in self._compact_portfolio(portfolio)["user_preferences"])
        sections.append("events:")
        for event in knowledge_slice.events:
            sections.append(
                "- "
                f"{event.event_type} "
                f"tickers={','.join(event.matched_holdings) or 'none'} "
                f"headline={truncate(event.headline, title_limit)} "
                f"summary={truncate(event.summary, event_summary_limit)}"
            )
        sections.append("documents:")
        for document in knowledge_slice.documents:
            sections.append(
                "- "
                f"{document.doc_type} "
                f"tickers={','.join(document.matched_holdings) or 'none'} "
                f"title={truncate(document.title, title_limit)} "
                f"excerpt={truncate(document.body, body_limit)}"
            )
        guidance = self._agent_guidance()
        sections.append(f"focus={guidance['focus']}")
        sections.append(f"evidence_hint={json.dumps(guidance['evidence_shape_hint'], ensure_ascii=False, separators=(',', ':'))}")
        sections.append(
            "return_json="
            + json.dumps(self._response_template(), ensure_ascii=False, separators=(",", ":"))
        )
        return "\n".join(sections)

    def _snippet_limits(self, depth: str) -> tuple[int, int, int]:
        if depth == "deep":
            return (90, 160, 120)
        if depth == "medium":
            return (80, 120, 90)
        return (72, 84, 68)

    def _compact_portfolio(self, portfolio: PortfolioSnapshot) -> dict[str, Any]:
        return {
            "generated_at": portfolio.generated_at.isoformat(),
            "holdings": [
                {
                    "ticker": holding.ticker,
                    "company_name": holding.company_name,
                    "weight": round(float(holding.weight), 4),
                }
                for holding in portfolio.holdings
            ],
            "cash_weight": round(float(portfolio.cash_weight), 4),
            "user_preferences": list(portfolio.user_preferences[:4]),
        }

    def _agent_guidance(self) -> dict[str, Any]:
        return {
            "focus": self.prompt_profile.focus,
            "evidence_shape_hint": dict(self.prompt_profile.evidence_shape_hint),
        }

    def _response_template(self) -> dict[str, Any]:
        return {
            **dict(self.prompt_profile.response_template),
            "evidence": dict(self.prompt_profile.evidence_shape_hint),
        }


class DeterministicProfitAgent:
    agent_id = "profit"

    def run(
        self,
        *,
        query: str,
        turn_number: int,
        portfolio: PortfolioSnapshot,
        knowledge_base: LocalKnowledgeBase,
        rebalance_plan: dict[str, float],
    ) -> AgentResponse:
        del query
        signals: dict[str, float] = {}
        gross_change = 0.0
        plan_score = 0.0
        for ticker, delta in rebalance_plan.items():
            signal = knowledge_base.ticker_signal(ticker, portfolio)
            signals[ticker] = signal
            gross_change += abs(delta)
            plan_score += delta * signal

        expected_return_1m = plan_score * 40.0
        expected_return_3m = plan_score * 65.0
        sharpe_ratio = plan_score / max(0.05, gross_change * 0.75)
        max_drawdown = -1.0 * ((gross_change * 10.0) + max(0.0, -plan_score * 35.0))
        confidence = clamp(0.42 + (len(signals) * 0.08), 0.0, 0.75)
        recommendation = (
            "Heuristic v1 simulation suggests modest upside relative to trade size."
            if plan_score >= 0
            else "Heuristic v1 simulation suggests the proposed trade is paying for weak expected follow-through."
        )
        opinion_id = f"profit_{stable_hash({'turn': turn_number, 'plan': rebalance_plan})[:12]}"
        return AgentResponse(
            agent_id=self.agent_id,
            opinion_id=opinion_id,
            turn_number=turn_number,
            query_understood="Evaluate the proposed rebalance plan with a heuristic local simulator.",
            verdict=AgentVerdict.DIRECT_ANSWER,
            evidence={
                "mode": "plan_simulation",
                "plan_simulation": {
                    "rebalance_plan": rebalance_plan,
                    "ticker_signals": signals,
                    "expected_return_1m": round(expected_return_1m, 3),
                    "expected_return_3m": round(expected_return_3m, 3),
                    "sharpe_ratio": round(sharpe_ratio, 3),
                    "max_drawdown": round(max_drawdown, 3),
                    "recommendation_text": recommendation,
                },
            },
            direction=clamp(plan_score * 8.0, -1.0, 1.0),
            strength=clamp(gross_change * 4.0, 0.0, 1.0),
            urgency=Urgency.SCHEDULED,
            confidence=confidence,
            reasoning_for_judge_agent="This is a static-rule simulator. Use it as a relative check on whether the candidate plan improves expected follow-through.",
            limits_acknowledged="Profit uses a heuristic local signal model, not a calibrated Monte Carlo engine.",
            tools_called=[
                ToolCall(
                    tool_name="local_profit.heuristic_plan_simulation",
                    purpose="Estimate directional payoff of the candidate rebalance plan",
                    summary=f"Computed heuristic returns for {len(rebalance_plan)} planned position changes.",
                )
            ],
            depth_used="medium",
            focus_tickers=sorted(rebalance_plan),
        )


class DeterministicCostAgent:
    agent_id = "cost"

    def run(
        self,
        *,
        query: str,
        turn_number: int,
        portfolio: PortfolioSnapshot,
        rebalance_plan: dict[str, float],
    ) -> AgentResponse:
        del query
        gross_change = sum(abs(delta) for delta in rebalance_plan.values())
        reference_value = portfolio.total_value_krw or 100000000.0
        traded_notional = reference_value * gross_change
        commission_bp = 1.5
        sell_tax_bp = 18.0
        slippage_bp = 6.0 + (gross_change * 100.0)
        spread_bp = 3.0 + (gross_change * 40.0)
        sells_notional = reference_value * sum(abs(delta) for delta in rebalance_plan.values() if delta < 0)
        commission_krw = traded_notional * (commission_bp / 10000.0)
        tax_krw = sells_notional * (sell_tax_bp / 10000.0)
        total_friction_bp = commission_bp + slippage_bp + spread_bp + (sell_tax_bp if sells_notional else 0.0)
        opinion_id = f"cost_{stable_hash({'turn': turn_number, 'plan': rebalance_plan})[:12]}"
        return AgentResponse(
            agent_id=self.agent_id,
            opinion_id=opinion_id,
            turn_number=turn_number,
            query_understood="Estimate the execution friction of the candidate rebalance plan.",
            verdict=AgentVerdict.DIRECT_ANSWER,
            evidence={
                "mode": "trade_cost",
                "trade_cost": {
                    "rebalance_plan": rebalance_plan,
                    "commission_krw": round(commission_krw, 0),
                    "tax_krw": round(tax_krw, 0),
                    "estimated_slippage_bp": round(slippage_bp, 3),
                    "spread_state_bp": round(spread_bp, 3),
                    "total_friction_bp": round(total_friction_bp, 3),
                },
            },
            direction=0.0,
            strength=0.0,
            urgency=Urgency.WATCH if total_friction_bp >= 30.0 else Urgency.SCHEDULED,
            confidence=0.58,
            reasoning_for_judge_agent="Execution friction is manageable for small reallocations but rises quickly with gross turnover.",
            limits_acknowledged="Cost uses configurable heuristic basis-point defaults, not live broker fees or orderbook snapshots.",
            tools_called=[
                ToolCall(
                    tool_name="local_cost.heuristic_trade_cost",
                    purpose="Estimate commission, tax, spread, and slippage",
                    summary=f"Estimated costs for gross turnover {gross_change:.3f} using local heuristic defaults.",
                )
            ],
            depth_used="medium",
            focus_tickers=sorted(rebalance_plan),
        )


class JudgeOrchestrator:
    def __init__(self, *, client: ChatClient, checkpoint_path: str | Path | None = None) -> None:
        self.client = client
        from .libra.agents import build_default_agent_bundle

        agent_bundle = build_default_agent_bundle(client=client)
        self.disclosure_agent = agent_bundle.disclosure
        self.news_agent = agent_bundle.news
        self.report_agent = agent_bundle.report
        self.profit_agent = agent_bundle.profit
        self.cost_agent = agent_bundle.cost
        self.checkpoint_path = Path(checkpoint_path).expanduser() if checkpoint_path else None
        from .libra_graph import LibraLangGraphRuntime
        self._graph_runtime = LibraLangGraphRuntime(self)

    def run(
        self,
        *,
        query: str,
        portfolio: PortfolioSnapshot,
        knowledge_base: LocalKnowledgeBase,
        depth: str = "medium",
        trigger: str = "pull",
        trigger_event: TriggerEvent | None = None,
        deadline_seconds: int | None = None,
        thread_id: str | None = None,
        enable_human_interrupts: bool = False,
    ) -> dict[str, Any]:
        return self._graph_runtime.run(
            query=query,
            portfolio=portfolio,
            knowledge_base=knowledge_base,
            depth=depth,
            trigger=trigger,
            trigger_event=trigger_event,
            deadline_seconds=deadline_seconds,
            thread_id=thread_id,
            enable_human_interrupts=enable_human_interrupts,
        )

    def resume(
        self,
        *,
        thread_id: str,
        resume_payload: Any,
    ) -> dict[str, Any]:
        return self._graph_runtime.resume(
            thread_id=thread_id,
            resume_payload=resume_payload,
        )

    def _initialize_run_state(
        self,
        *,
        trigger: str,
        trigger_event: TriggerEvent | None,
        deadline_seconds: int | None,
    ) -> RunState:
        started_at = datetime.now().astimezone()
        if deadline_seconds is None and trigger == "push":
            deadline_seconds = 300
        deadline_at = (started_at + timedelta(seconds=deadline_seconds)) if deadline_seconds else None
        notification_log: list[UserNotification] = []
        if trigger == "push" and trigger_event is not None:
            notification_log.append(
                self._acknowledgement_notification(
                    trigger_event=trigger_event,
                    deadline_at=deadline_at,
                    sent_at=started_at,
                )
            )
        return RunState(
            trigger=trigger,
            trigger_event=trigger_event,
            started_at=started_at,
            deadline_at=deadline_at,
            notification_log=notification_log,
        )

    def _agent_by_id(self, agent_id: str) -> LLMAgent:
        agent_id = canonical_agent_id(agent_id)
        return {
            "disclosure": self.disclosure_agent,
            "news": self.news_agent,
            "report": self.report_agent,
        }[agent_id]

    def _build_information_plan(
        self,
        *,
        query: str,
        depth: str,
        trigger: str = "pull",
        trigger_event: TriggerEvent | None = None,
    ) -> list[PlannedAgentCall]:
        lowered = query.casefold()
        urgent_keywords = ("속보", "breaking", "장중", "리콜", "화재", "소송", "규제", "조사")
        report_keywords = ("리포트", "report", "컨센서스", "목표주가", "사업부")
        disclosure_keywords = ("공시", "dart", "잠정", "실적", "감사보고서")
        trigger_context = self._trigger_context_text(trigger_event)

        if trigger == "push":
            order = ("news", "disclosure", "report")
        elif any(token in lowered for token in urgent_keywords):
            order = ("news", "disclosure", "report")
        elif any(token in lowered for token in report_keywords):
            order = ("report", "news", "disclosure")
        elif any(token in lowered for token in disclosure_keywords):
            order = ("disclosure", "news", "report")
        else:
            order = ("disclosure", "news", "report")

        plan: list[PlannedAgentCall] = []
        for agent_id in order:
            planned_depth = depth
            context_parts: list[str] = []
            fallback: str | None = None
            note: str | None = None
            if trigger_context:
                context_parts.append(trigger_context)
            if agent_id == "news":
                agent_query = "포트폴리오 관련 뉴스, 시장 반응, 필요시 매크로 배경을 요약해줘."
                if trigger == "push":
                    planned_depth = "deep"
                    note = "Judge wants to know whether the signal is structural or temporary."
                fallback = "Focus on market reaction, cross-checks, and whether the headline changes the thesis."
            elif agent_id == "disclosure":
                agent_query = "포트폴리오 관련 신규 공시와 실적 신호를 요약해줘."
                fallback = "Focus on filing relevance, earnings signals, and upcoming scheduled disclosures."
            else:
                agent_query = "포트폴리오 관련 증권사 리포트와 컨센서스 변화, 사업부 단서를 요약해줘."
                fallback = "Focus on target-price revisions, consensus drift, and business-unit commentary."
            plan.append(
                PlannedAgentCall(
                    agent_id=agent_id,
                    query=agent_query,
                    context=" | ".join(part for part in context_parts if part) or query,
                    depth=planned_depth,
                    fallback=fallback,
                    note=note,
                )
            )
        return plan

    def _should_skip_agent(
        self,
        *,
        planned_call: PlannedAgentCall,
        original_query: str,
        depth: str,
        trigger: str,
        responses: list[AgentResponse],
    ) -> bool:
        del original_query
        if trigger == "push":
            return False
        if canonical_agent_id(planned_call.agent_id) != "report":
            return False
        if depth != "shallow":
            return False
        if not responses:
            return False
        low_signal = all(
            abs(response.direction) < 0.08 and response.urgency in {Urgency.DEFER, Urgency.SCHEDULED}
            for response in responses
        )
        return low_signal

    def _candidate_plan_from_trigger(
        self,
        *,
        portfolio: PortfolioSnapshot,
        trigger_event: TriggerEvent | None,
    ) -> dict[str, float]:
        if trigger_event is None or not trigger_event.ticker:
            return {}
        holdings = {normalize_ticker(item.ticker): item for item in portfolio.holdings}
        target = holdings.get(normalize_ticker(trigger_event.ticker))
        if target is None:
            return {}
        text = "\n".join(
            part for part in (trigger_event.headline, trigger_event.summary or "", trigger_event.market_reaction or "") if part
        ).casefold()
        is_risk_event = any(token in text for token in PUSH_RISK_KEYWORDS)
        if not is_risk_event:
            return {target.ticker: 0.05}

        plan: dict[str, float] = {target.ticker: -0.1}
        other_holdings = sorted(
            (holding for holding in portfolio.holdings if normalize_ticker(holding.ticker) != normalize_ticker(target.ticker)),
            key=lambda item: float(item.weight),
            reverse=True,
        )
        for holding in other_holdings[:2]:
            plan[holding.ticker] = -0.05
        return plan

    def _trade_agent_order(
        self,
        *,
        query: str,
        planning: Mapping[str, Any],
        trigger: str,
    ) -> tuple[str, ...]:
        lowered = query.casefold()
        if trigger == "push" or planning.get("decision") == DecisionType.USER_DECISION_REQUIRED.value:
            return ("cost", "profit")
        if any(token in lowered for token in ("비용", "세금", "슬리피지", "실행")):
            return ("cost", "profit")
        return ("profit", "cost")

    def _judge_next_action(
        self,
        *,
        query: str,
        portfolio: PortfolioSnapshot,
        responses: list[AgentResponse],
        called_agents: list[str],
        depth: str,
        trigger: str,
        trigger_event: TriggerEvent | None,
        candidate_plan: Mapping[str, float] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "query": query,
            "trigger": trigger,
            "trigger_event": trigger_event.to_dict() if trigger_event else None,
            "depth": depth,
            "called_agents": list(called_agents),
            "candidate_rebalance_plan": dict(candidate_plan or {}),
            "portfolio": {
                "holdings": [
                    {
                        "ticker": item.ticker,
                        "company_name": item.company_name,
                        "weight": round(float(item.weight), 4),
                    }
                    for item in portfolio.holdings
                ],
                "user_preferences": list(portfolio.user_preferences[:4]),
            },
            "agent_responses": [self._compact_agent_response(response) for response in responses],
            "instructions": {
                "action_values": ["CALL_AGENT", "FINALIZE"],
                "agent_values": ["disclosure", "news", "report", "profit", "cost"],
                "depth_values": ["shallow", "medium", "deep"],
                "rules": JUDGE_ACTION_RULES,
            },
        }
        system_prompt = JUDGE_ACTION_SYSTEM_PROMPT
        try:
            raw = self.client.chat_json(
                system_prompt=system_prompt,
                user_prompt=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                temperature=0.0,
            )
        except ChatClientError:
            return self._fallback_judge_action(
                query=query,
                portfolio=portfolio,
                responses=responses,
                called_agents=called_agents,
                depth=depth,
                trigger=trigger,
                trigger_event=trigger_event,
                candidate_plan=candidate_plan,
            )
        normalized = self._normalize_judge_action(
            raw,
            query=query,
            portfolio=portfolio,
            responses=responses,
            called_agents=called_agents,
            depth=depth,
            trigger=trigger,
            trigger_event=trigger_event,
            candidate_plan=candidate_plan,
        )
        if normalized is None:
            return self._fallback_judge_action(
                query=query,
                portfolio=portfolio,
                responses=responses,
                called_agents=called_agents,
                depth=depth,
                trigger=trigger,
                trigger_event=trigger_event,
                candidate_plan=candidate_plan,
            )
        return normalized

    def _normalize_judge_action(
        self,
        payload: Mapping[str, Any],
        *,
        query: str,
        portfolio: PortfolioSnapshot,
        responses: list[AgentResponse],
        called_agents: list[str],
        depth: str,
        trigger: str,
        trigger_event: TriggerEvent | None,
        candidate_plan: Mapping[str, float] | None,
    ) -> dict[str, Any] | None:
        action = str(payload.get("action", "")).strip().upper()
        if action not in {"CALL_AGENT", "FINALIZE"}:
            return None
        response_map = {canonical_agent_id(item.agent_id): item for item in responses}
        normalized_plan = self._draft_candidate_plan(
            query=query,
            portfolio=portfolio,
            responses=responses,
            trigger=trigger,
            trigger_event=trigger_event,
            candidate_plan=payload.get("candidate_rebalance_plan") if isinstance(payload.get("candidate_rebalance_plan"), Mapping) else candidate_plan,
            allow_inference=False,
        )
        result: dict[str, Any] = {
            "action": action,
            "reason": str(payload.get("reason", "")).strip(),
            "candidate_rebalance_plan": normalized_plan,
        }
        if action == "FINALIZE":
            return result

        agent_id = canonical_agent_id(str(payload.get("agent_id", "")))
        if agent_id not in {"disclosure", "news", "report", "profit", "cost"}:
            return None
        called_set = {canonical_agent_id(item) for item in called_agents}
        if agent_id in called_set:
            return None
        original_agent_id = agent_id
        action_depth = str(payload.get("depth", depth)).strip().lower()
        if action_depth not in {"shallow", "medium", "deep"}:
            action_depth = depth
        if trigger == "push" and self._push_prescreen_is_sufficient(trigger_event):
            if agent_id in {"disclosure", "news", "report"}:
                preferred_push_agent = self._next_push_agent(
                    called_agents=called_set,
                    trigger_event=trigger_event,
                    candidate_plan=normalized_plan,
                )
                if preferred_push_agent is None:
                    return {
                        "action": "FINALIZE",
                        "reason": "Push pre-screen already gives Judge enough context for a final decision.",
                        "candidate_rebalance_plan": normalized_plan,
                    }
                agent_id = preferred_push_agent
                action_depth = "medium"
        if trigger == "pull" and not called_set:
            query_lower = query.casefold()
            if agent_id != "disclosure" and not any(
                token in query_lower for token in ("속보", "장중", "breaking", "급락", "급등")
            ):
                agent_id = "disclosure"
                action_depth = depth
        elif trigger == "pull" and "disclosure" in called_set and "news" not in called_set and agent_id == "report":
            query_lower = query.casefold()
            if not any(token in query_lower for token in ("리포트", "report", "컨센서스")):
                agent_id = "news"
                action_depth = self._recommended_news_depth(
                    default_depth=depth,
                    disclosure_response=next(
                        (item for item in responses if canonical_agent_id(item.agent_id) == "disclosure"),
                        None,
                    ),
                )
        if trigger == "pull" and {"disclosure", "news"}.issubset(called_set):
            disclosure_response = response_map.get("disclosure")
            news_response = response_map.get("news")
            if self._should_finalize_after_basic_scan(
                query=query,
                depth=depth,
                disclosure_response=disclosure_response,
                news_response=news_response,
                candidate_plan=normalized_plan,
            ):
                return {
                    "action": "FINALIZE",
                    "reason": "Disclosure and News were sufficient for this turn, so extra collection is not justified.",
                    "candidate_rebalance_plan": normalized_plan,
                }
        agent_overridden = agent_id != original_agent_id

        if agent_id in {"profit", "cost"} and not normalized_plan:
            normalized_plan = self._draft_candidate_plan(
                query=query,
                portfolio=portfolio,
                responses=responses,
                trigger=trigger,
                trigger_event=trigger_event,
                candidate_plan=candidate_plan,
                allow_inference=False,
            )
            if not normalized_plan:
                return None
            result["candidate_rebalance_plan"] = normalized_plan
        disclosure_response = next(
            (item for item in responses if canonical_agent_id(item.agent_id) == "disclosure"),
            None,
        )
        normalized_call_depth = action_depth if agent_id in {"disclosure", "news", "report"} else "medium"

        result.update(
            {
                "agent_id": agent_id,
                "query": self._default_agent_query(
                    agent_id=agent_id,
                    trigger=trigger,
                    disclosure_response=disclosure_response,
                    responses=responses,
                ),
                "context": self._default_agent_context(
                    agent_id=agent_id,
                    query=query,
                    responses=responses,
                    trigger_event=trigger_event,
                    candidate_plan=normalized_plan,
                ),
                "depth": normalized_call_depth,
                "fallback": self._default_agent_fallback(agent_id=agent_id, trigger=trigger),
                "note": self._default_agent_note(
                    agent_id=agent_id,
                    query=query,
                    responses=responses,
                    trigger=trigger,
                    trigger_event=trigger_event,
                    candidate_plan=normalized_plan,
                ),
            }
        )
        return result

    def _fallback_judge_action(
        self,
        *,
        query: str,
        portfolio: PortfolioSnapshot,
        responses: list[AgentResponse],
        called_agents: list[str],
        depth: str,
        trigger: str,
        trigger_event: TriggerEvent | None,
        candidate_plan: Mapping[str, float] | None,
    ) -> dict[str, Any]:
        called = {canonical_agent_id(item) for item in called_agents}
        normalized_plan = self._draft_candidate_plan(
            query=query,
            portfolio=portfolio,
            responses=responses,
            trigger=trigger,
            trigger_event=trigger_event,
            candidate_plan=candidate_plan,
            allow_inference=self._should_attempt_plan_inference(
                query=query,
                responses=responses,
                called_agents=called_agents,
                trigger=trigger,
            ),
        )
        response_map = {canonical_agent_id(item.agent_id): item for item in responses}

        if trigger == "push":
            push_next = self._next_push_agent(
                called_agents=called,
                trigger_event=trigger_event,
                candidate_plan=normalized_plan,
            )
            if push_next == "cost":
                return {
                    "action": "CALL_AGENT",
                    "reason": "Push wake-up already has pre-screening, so Judge checks execution friction first.",
                    "agent_id": "cost",
                    "query": self._default_agent_query(agent_id="cost", trigger=trigger, responses=responses),
                    "context": self._default_agent_context(
                        agent_id="cost",
                        query=query,
                        responses=responses,
                        trigger_event=trigger_event,
                        candidate_plan=normalized_plan,
                    ),
                    "depth": "medium",
                    "fallback": self._default_agent_fallback(agent_id="cost", trigger=trigger),
                    "note": self._default_agent_note(
                        agent_id="cost",
                        query=query,
                        responses=responses,
                        trigger=trigger,
                        trigger_event=trigger_event,
                        candidate_plan=normalized_plan,
                    ),
                    "candidate_rebalance_plan": normalized_plan,
                }
            if push_next == "profit":
                return {
                    "action": "CALL_AGENT",
                    "reason": "Push wake-up starts with evaluating whether the draft defensive rebalance is worth doing.",
                    "agent_id": "profit",
                    "query": self._default_agent_query(agent_id="profit", trigger=trigger, responses=responses),
                    "context": self._default_agent_context(
                        agent_id="profit",
                        query=query,
                        responses=responses,
                        trigger_event=trigger_event,
                        candidate_plan=normalized_plan,
                    ),
                    "depth": "medium",
                    "fallback": self._default_agent_fallback(agent_id="profit", trigger=trigger),
                    "note": self._default_agent_note(
                        agent_id="profit",
                        query=query,
                        responses=responses,
                        trigger=trigger,
                        trigger_event=trigger_event,
                        candidate_plan=normalized_plan,
                    ),
                    "candidate_rebalance_plan": normalized_plan,
                }
            return {
                "action": "FINALIZE",
                "reason": "Push-triggered sequence already has enough pre-screening and execution context.",
                "candidate_rebalance_plan": normalized_plan,
            }

        if "disclosure" not in called:
            return {
                "action": "CALL_AGENT",
                "reason": "Regular checks begin with a disclosure and earnings scan.",
                "agent_id": "disclosure",
                "query": self._default_agent_query(agent_id="disclosure", trigger=trigger, responses=responses),
                "context": self._default_agent_context(
                    agent_id="disclosure",
                    query=query,
                    responses=responses,
                    trigger_event=trigger_event,
                    candidate_plan=normalized_plan,
                ),
                "depth": depth,
                "fallback": self._default_agent_fallback(agent_id="disclosure", trigger=trigger),
                "note": self._default_agent_note(
                    agent_id="disclosure",
                    query=query,
                    responses=responses,
                    trigger=trigger,
                    trigger_event=trigger_event,
                    candidate_plan=normalized_plan,
                ),
                "candidate_rebalance_plan": normalized_plan,
            }

        if "news" not in called:
            disclosure_response = response_map.get("disclosure")
            news_depth = self._recommended_news_depth(
                default_depth=depth,
                disclosure_response=disclosure_response,
            )
            return {
                "action": "CALL_AGENT",
                "reason": "After disclosure, Judge checks whether the signal is structural, transient, or already priced.",
                "agent_id": "news",
                "query": self._default_agent_query(
                    agent_id="news",
                    trigger=trigger,
                    disclosure_response=disclosure_response,
                    responses=responses,
                ),
                "context": self._default_agent_context(
                    agent_id="news",
                    query=query,
                    responses=responses,
                    trigger_event=trigger_event,
                    candidate_plan=normalized_plan,
                ),
                "depth": news_depth,
                "fallback": self._default_agent_fallback(agent_id="news", trigger=trigger),
                "note": self._default_agent_note(
                    agent_id="news",
                    query=query,
                    responses=responses,
                    trigger=trigger,
                    trigger_event=trigger_event,
                    candidate_plan=normalized_plan,
                ),
                "candidate_rebalance_plan": normalized_plan,
            }

        disclosure_response = response_map.get("disclosure")
        news_response = response_map.get("news")
        if self._should_finalize_after_basic_scan(
            query=query,
            depth=depth,
            disclosure_response=disclosure_response,
            news_response=news_response,
            candidate_plan=normalized_plan,
        ):
            return {
                "action": "FINALIZE",
                "reason": "Disclosure and News were enough for this turn, so Judge stops here.",
                "candidate_rebalance_plan": normalized_plan,
            }

        if "report" not in called and self._should_call_report(
            query=query,
            depth=depth,
            disclosure_response=disclosure_response,
            news_response=news_response,
        ):
            return {
                "action": "CALL_AGENT",
                "reason": "Need sell-side interpretation or business-segment decomposition before deciding.",
                "agent_id": "report",
                "query": self._default_agent_query(agent_id="report", trigger=trigger, responses=responses),
                "context": self._default_agent_context(
                    agent_id="report",
                    query=query,
                    responses=responses,
                    trigger_event=trigger_event,
                    candidate_plan=normalized_plan,
                ),
                "depth": depth if depth != "shallow" else "medium",
                "fallback": self._default_agent_fallback(agent_id="report", trigger=trigger),
                "note": self._default_agent_note(
                    agent_id="report",
                    query=query,
                    responses=responses,
                    trigger=trigger,
                    trigger_event=trigger_event,
                    candidate_plan=normalized_plan,
                ),
                "candidate_rebalance_plan": normalized_plan,
            }

        if normalized_plan and "profit" not in called:
            return {
                "action": "CALL_AGENT",
                "reason": "A draft rebalance exists, so Judge asks Profit to test expected payoff and risk.",
                "agent_id": "profit",
                "query": self._default_agent_query(agent_id="profit", trigger=trigger, responses=responses),
                "context": self._default_agent_context(
                    agent_id="profit",
                    query=query,
                    responses=responses,
                    trigger_event=trigger_event,
                    candidate_plan=normalized_plan,
                ),
                "depth": "medium",
                "fallback": self._default_agent_fallback(agent_id="profit", trigger=trigger),
                "note": self._default_agent_note(
                    agent_id="profit",
                    query=query,
                    responses=responses,
                    trigger=trigger,
                    trigger_event=trigger_event,
                    candidate_plan=normalized_plan,
                ),
                "candidate_rebalance_plan": normalized_plan,
            }

        if normalized_plan and "cost" not in called:
            return {
                "action": "CALL_AGENT",
                "reason": "Judge still needs cost and liquidity before committing to the draft plan.",
                "agent_id": "cost",
                "query": self._default_agent_query(agent_id="cost", trigger=trigger, responses=responses),
                "context": self._default_agent_context(
                    agent_id="cost",
                    query=query,
                    responses=responses,
                    trigger_event=trigger_event,
                    candidate_plan=normalized_plan,
                ),
                "depth": "medium",
                "fallback": self._default_agent_fallback(agent_id="cost", trigger=trigger),
                "note": self._default_agent_note(
                    agent_id="cost",
                    query=query,
                    responses=responses,
                    trigger=trigger,
                    trigger_event=trigger_event,
                    candidate_plan=normalized_plan,
                ),
                "candidate_rebalance_plan": normalized_plan,
            }

        return {
            "action": "FINALIZE",
            "reason": "Current observations are sufficient for a final Judge decision.",
            "candidate_rebalance_plan": normalized_plan,
        }

    def _draft_candidate_plan(
        self,
        *,
        query: str,
        portfolio: PortfolioSnapshot,
        responses: list[AgentResponse],
        trigger: str,
        trigger_event: TriggerEvent | None,
        candidate_plan: Mapping[str, Any] | None,
        allow_inference: bool,
    ) -> dict[str, float]:
        if isinstance(candidate_plan, Mapping):
            sanitized = self._sanitize_plan(candidate_plan, portfolio)
            if sanitized:
                return sanitized
        if trigger == "push":
            push_plan = self._candidate_plan_from_trigger(
                portfolio=portfolio,
                trigger_event=trigger_event,
            )
            if push_plan:
                return push_plan
        if not allow_inference:
            return {}
        payload = self._fallback_judge_payload(
            query=query,
            portfolio=portfolio,
            responses=responses,
            stage="planning",
        )
        raw_plan = payload.get("candidate_rebalance_plan", {})
        return self._sanitize_plan(raw_plan if isinstance(raw_plan, Mapping) else {}, portfolio)

    def _default_agent_query(
        self,
        *,
        agent_id: str,
        trigger: str,
        disclosure_response: AgentResponse | None = None,
        responses: list[AgentResponse] | None = None,
    ) -> str:
        del responses
        return default_agent_query(
            agent_id=agent_id,
            trigger=trigger,
            has_disclosure_context=disclosure_response is not None,
        )

    def _default_agent_context(
        self,
        *,
        agent_id: str,
        query: str,
        responses: list[AgentResponse],
        trigger_event: TriggerEvent | None,
        candidate_plan: Mapping[str, float] | None = None,
    ) -> str:
        context_parts: list[str] = []
        trigger_context = self._trigger_context_text(trigger_event)
        if trigger_context:
            context_parts.append(trigger_context)
        if responses:
            latest = responses[-1]
            context_parts.append(
                f"Latest observation from {latest.agent_id}: {truncate(latest.reasoning_for_judge_agent or latest.query_understood, 180)}"
            )
        if agent_id in {"profit", "cost"}:
            context_parts.append("Judge is evaluating whether the current draft rebalance should be executed.")
            if candidate_plan:
                context_parts.append(f"Candidate rebalance plan: {dict(candidate_plan)}")
        else:
            context_parts.append(f"Original user request: {query}")
        return " | ".join(part for part in context_parts if part)

    def _default_agent_fallback(self, *, agent_id: str, trigger: str) -> str | None:
        return default_agent_fallback(agent_id=agent_id, trigger=trigger)

    def _default_agent_note(
        self,
        *,
        agent_id: str,
        query: str,
        responses: list[AgentResponse],
        trigger: str,
        trigger_event: TriggerEvent | None,
        candidate_plan: Mapping[str, float] | None = None,
    ) -> str | None:
        del query, trigger_event
        latest = responses[-1] if responses else None
        return default_agent_note(
            agent_id=agent_id,
            latest_agent_id=canonical_agent_id(latest.agent_id) if latest is not None else None,
            trigger=trigger,
            has_candidate_plan=bool(candidate_plan),
        )

    def _recommended_news_depth(
        self,
        *,
        default_depth: str,
        disclosure_response: AgentResponse | None,
    ) -> str:
        if disclosure_response is None:
            return default_depth
        text = f"{disclosure_response.reasoning_for_judge_agent}\n{disclosure_response.limits_acknowledged or ''}".casefold()
        if any(token in text for token in ("어닝", "earnings", "실적", "화재", "규제", "조사", "리콜")):
            return "deep"
        return default_depth

    def _has_rebalance_intent(self, query: str) -> bool:
        lowered = query.casefold()
        return any(token in lowered for token in ("리밸런싱", "리밸런스", "rebalance", "초안", "비중", "매수", "매도"))

    def _is_explicit_report_request(self, query: str) -> bool:
        lowered = query.casefold()
        return any(token in lowered for token in ("리포트", "report", "컨센서스", "증권사", "목표주가"))

    def _should_attempt_plan_inference(
        self,
        *,
        query: str,
        responses: list[AgentResponse],
        called_agents: list[str],
        trigger: str,
    ) -> bool:
        if trigger == "push":
            return True
        if not self._has_rebalance_intent(query):
            return False
        called_set = {canonical_agent_id(item) for item in called_agents}
        if "report" not in called_set:
            return False
        info_directions = [
            response.direction
            for response in responses
            if canonical_agent_id(response.agent_id) in {"disclosure", "news", "report"} and response.confidence >= 0.45
        ]
        if not info_directions:
            return False
        return any(value >= 0.18 for value in info_directions) and any(value <= -0.08 for value in info_directions)

    def _push_prescreen_is_sufficient(self, trigger_event: TriggerEvent | None) -> bool:
        if trigger_event is None:
            return False
        if not (trigger_event.ticker or trigger_event.company_name or trigger_event.headline):
            return False
        text = f"{trigger_event.headline} {trigger_event.summary or ''}".casefold()
        cross_check_count = int(trigger_event.cross_check_count or 0)
        return (
            cross_check_count >= 2
            or bool(trigger_event.market_reaction)
            or any(token in text for token in ("조사", "규제", "리콜", "화재", "안전", "investigation", "recall", "probe", "fire"))
        )

    def _prefer_cost_first_for_push(self, trigger_event: TriggerEvent | None) -> bool:
        if trigger_event is None:
            return False
        text = f"{trigger_event.headline} {trigger_event.summary or ''}".casefold()
        return any(token in text for token in ("조사", "규제", "리콜", "화재", "안전", "investigation", "recall", "probe", "fire"))

    def _next_push_agent(
        self,
        *,
        called_agents: set[str],
        trigger_event: TriggerEvent | None,
        candidate_plan: Mapping[str, float],
    ) -> str | None:
        if not candidate_plan:
            return None
        ordered = ["cost", "profit"] if self._prefer_cost_first_for_push(trigger_event) else ["profit", "cost"]
        for agent_id in ordered:
            if agent_id not in called_agents:
                return agent_id
        return None

    def _should_finalize_after_basic_scan(
        self,
        *,
        query: str,
        depth: str,
        disclosure_response: AgentResponse | None,
        news_response: AgentResponse | None,
        candidate_plan: Mapping[str, float],
    ) -> bool:
        if disclosure_response is None or news_response is None:
            return False
        if candidate_plan:
            return False
        if self._is_explicit_report_request(query):
            return False
        if self._should_call_report(
            query=query,
            depth=depth,
            disclosure_response=disclosure_response,
            news_response=news_response,
        ):
            return False
        responses = (disclosure_response, news_response)
        if any(response.verdict == AgentVerdict.DIRECT_ANSWER_UNAVAILABLE for response in responses):
            return False
        reasoning = "\n".join(
            collapse_whitespace(f"{response.reasoning_for_judge_agent} {response.limits_acknowledged or ''}").casefold()
            for response in responses
        )
        if any(
            token in reasoning
            for token in ("리포트 필요", "리포트 확인", "report needed", "call report", "컨센서스", "preview", "추가 확인", "추가 정보")
        ):
            return False
        max_signal = max(abs(response.direction) for response in responses)
        min_confidence = min(response.confidence for response in responses)
        if max_signal < 0.14 and min_confidence >= 0.5:
            return True
        if depth == "shallow" and max_signal < 0.18 and min_confidence >= 0.45:
            return True
        return False

    def _should_call_report(
        self,
        *,
        query: str,
        depth: str,
        disclosure_response: AgentResponse | None,
        news_response: AgentResponse | None,
    ) -> bool:
        if self._is_explicit_report_request(query):
            return True
        if disclosure_response is None or news_response is None:
            return False
        responses = (disclosure_response, news_response)
        for response in responses:
            if response is None:
                continue
            reasoning = f"{response.reasoning_for_judge_agent}\n{response.limits_acknowledged or ''}".casefold()
            if any(
                token in reasoning
                for token in ("리포트 필요", "리포트 확인", "report needed", "call report", "사업부", "컨센서스", "추가 정보", "preview")
            ):
                return True
            if response.verdict == AgentVerdict.DIRECT_ANSWER_UNAVAILABLE:
                return True
        directions = [response.direction for response in responses]
        confidences = [response.confidence for response in responses]
        if directions[0] * directions[1] < -0.01 and abs(directions[0] - directions[1]) >= 0.14:
            return True
        if max(abs(value) for value in directions) >= 0.25 and min(confidences) < 0.55:
            return True
        if depth == "deep" and max(abs(value) for value in directions) >= 0.2 and min(confidences) < 0.7:
            return True
        return False

    def _infer_skip_rationale(
        self,
        *,
        query: str,
        trigger: str,
        trigger_event: TriggerEvent | None,
        called_agents: list[str],
        responses: list[AgentResponse],
        candidate_plan: Mapping[str, float],
        depth: str,
    ) -> dict[str, str]:
        called_set = {canonical_agent_id(item) for item in called_agents}
        rationale: dict[str, str] = {}
        response_map = {canonical_agent_id(item.agent_id): item for item in responses}

        if trigger == "push" and self._push_prescreen_is_sufficient(trigger_event):
            for agent_id in ("disclosure", "news", "report"):
                if agent_id not in called_set:
                    rationale[agent_id] = "Push trigger already carried pre-screening, so Judge skipped extra information collection."

        if trigger == "pull":
            disclosure_response = response_map.get("disclosure")
            news_response = response_map.get("news")
            if "report" not in called_set and disclosure_response is not None and news_response is not None:
                if self._should_finalize_after_basic_scan(
                    query=query,
                    depth=depth,
                    disclosure_response=disclosure_response,
                    news_response=news_response,
                    candidate_plan=candidate_plan,
                ):
                    rationale["report"] = "Disclosure and News were already sufficient for this turn."
                elif not self._should_call_report(
                    query=query,
                    depth=depth,
                    disclosure_response=disclosure_response,
                    news_response=news_response,
                ):
                    rationale["report"] = "Judge did not see a justified need for sell-side interpretation on this turn."

        if not candidate_plan:
            if "profit" not in called_set:
                rationale["profit"] = "Judge had no concrete rebalance draft to evaluate on return and risk."
            if "cost" not in called_set:
                rationale["cost"] = "No execution draft existed, so cost and liquidity checks were unnecessary."

        return rationale

    def _notification_timestamp(self) -> str:
        return datetime.now().astimezone().isoformat(timespec="seconds")

    def _coerce_datetime(self, value: Any) -> datetime | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        try:
            return coerce_datetime(text)
        except (TypeError, ValueError):
            return None

    def _coerce_datetime_or_now(self, value: Any) -> datetime:
        return self._coerce_datetime(value) or datetime.now().astimezone()

    def _deadline_exceeded(self, run_state: RunState) -> bool:
        if run_state.deadline_at is None:
            return False
        return datetime.now().astimezone(run_state.deadline_at.tzinfo) > run_state.deadline_at

    def _elapsed_seconds(self, run_state: RunState) -> float:
        return max(0.0, (datetime.now().astimezone() - run_state.started_at).total_seconds())

    def _deadline_result(
        self,
        *,
        query: str,
        portfolio: PortfolioSnapshot,
        knowledge_base: LocalKnowledgeBase,
        called_agents: list[str],
        skipped_agents: list[str],
        skip_rationale: dict[str, str],
        executed_calls: list[PlannedAgentCall],
        responses: list[AgentResponse],
        run_state: RunState,
        candidate_plan: Mapping[str, float] | None = None,
    ) -> dict[str, Any]:
        normalized_plan = dict(candidate_plan or {})
        summary = "분석 deadline을 넘겨 자동 판단보다 사용자 확인이 우선인 상태로 전환합니다."
        decision = JudgeDecision(
            decision=DecisionType.USER_DECISION_REQUIRED,
            summary=summary,
            confidence=0.55,
            urgency=Urgency.WATCH,
            called_agents=list(called_agents),
            skipped_agents=list(skipped_agents),
            skip_rationale=dict(skip_rationale),
            candidate_rebalance_plan=normalized_plan,
            reasoning="Self-imposed deadline expired before the orchestration finished.",
            follow_up_at=self._default_follow_up_at(query=query, responses=responses),
            consensus_score=0.0,
            divergence_score=0.0,
            needs_trade_evaluation=bool(normalized_plan),
            trigger=run_state.trigger,
            trigger_event=run_state.trigger_event,
            deadline_at=run_state.deadline_at.isoformat(timespec="seconds") if run_state.deadline_at else None,
            elapsed_seconds=self._elapsed_seconds(run_state),
            options=["권고안 승인", "전량 매도", "유지", "직접 비율 설정"] if run_state.trigger == "push" else [],
            auto_safeguards=self._build_auto_safeguards(
                portfolio=portfolio,
                trigger_event=run_state.trigger_event,
                candidate_plan=normalized_plan,
            ),
        )
        self._apply_push_guardrails(
            decision=decision,
            portfolio=portfolio,
            run_state=run_state,
        )
        final_notification = UserNotification(
            level="push",
            body=decision.summary,
            action_required=True,
            kind="deadline_expired",
            estimated_followup=decision.follow_up_at,
            sent_at=self._notification_timestamp(),
        )
        decision.user_notification = final_notification
        decision.notification_log = list(run_state.notification_log) + [final_notification]
        decision.decision_trace = self._decision_trace(
            query=query,
            executed_calls=executed_calls,
            responses=responses,
            decision=decision,
        )
        return {
            "model": self.client.model,
            "query": query,
            "portfolio": portfolio.to_dict(),
            "agent_responses": [response.to_dict() for response in responses],
            "decision": decision.to_dict(),
            "knowledge_sources": dict(knowledge_base.source_paths),
        }

    def _apply_push_guardrails(
        self,
        *,
        decision: JudgeDecision,
        portfolio: PortfolioSnapshot,
        run_state: RunState,
    ) -> None:
        if run_state.trigger != "push" or run_state.trigger_event is None:
            return
        trigger_event = run_state.trigger_event
        if not trigger_event.ticker:
            return
        normalized_ticker = normalize_ticker(trigger_event.ticker)
        if normalized_ticker not in {normalize_ticker(item.ticker) for item in portfolio.holdings}:
            return
        text = "\n".join(
            part for part in (trigger_event.headline, trigger_event.summary or "", trigger_event.market_reaction or "") if part
        ).casefold()
        is_risk_event = any(token in text for token in PUSH_RISK_KEYWORDS) or trigger_event.cross_check_count >= 3
        if not is_risk_event:
            return
        label = trigger_event.company_name or trigger_event.ticker or "해당 종목"
        decision.decision = DecisionType.USER_DECISION_REQUIRED
        decision.urgency = Urgency.WATCH
        decision.summary = f"{label} 관련 규제·안전 리스크 이벤트는 사전 위임 범위를 벗어나므로 사용자 확인이 먼저 필요합니다."
        decision.reasoning = "Push-triggered risk event crossed LIBRA's autonomy boundary, so the Judge is handing authority back to the user."
        decision.options = ["권고안 승인", "전량 매도", "유지", "직접 비율 설정"]
        decision.auto_safeguards = self._build_auto_safeguards(
            portfolio=portfolio,
            trigger_event=trigger_event,
            candidate_plan=decision.candidate_rebalance_plan,
        )

    def _trigger_context_text(self, trigger_event: TriggerEvent | None) -> str:
        if trigger_event is None:
            return ""
        parts = [
            f"Triggered by {trigger_event.source or trigger_event.trigger_type}: {trigger_event.headline}",
        ]
        if trigger_event.company_name:
            parts.append(f"company={trigger_event.company_name}")
        if trigger_event.ticker:
            parts.append(f"ticker={trigger_event.ticker}")
        if trigger_event.summary:
            parts.append(f"summary={trigger_event.summary}")
        if trigger_event.cross_check_count:
            parts.append(f"cross_check_count={trigger_event.cross_check_count}")
        if trigger_event.market_reaction:
            parts.append(f"market_reaction={trigger_event.market_reaction}")
        return " | ".join(parts)

    def _build_auto_safeguards(
        self,
        *,
        portfolio: PortfolioSnapshot,
        trigger_event: TriggerEvent | None,
        candidate_plan: Mapping[str, float],
    ) -> dict[str, Any]:
        if trigger_event is None or not trigger_event.ticker:
            return dict(candidate_plan=candidate_plan)
        ticker = normalize_ticker(trigger_event.ticker)
        if ticker not in {normalize_ticker(item.ticker) for item in portfolio.holdings}:
            return dict(candidate_plan=candidate_plan)
        return {
            "tripwire_1": f"{trigger_event.ticker} 장중 -5% 시 보유 수량의 절반 자동 축소",
            "tripwire_2": f"{trigger_event.ticker} 장중 -8% 시 잔여 보유분 추가 축소",
            "candidate_plan": dict(candidate_plan),
        }

    def _acknowledgement_notification(
        self,
        *,
        trigger_event: TriggerEvent,
        deadline_at: datetime | None,
        sent_at: datetime,
    ) -> UserNotification:
        label = trigger_event.company_name or trigger_event.ticker or "이벤트"
        if deadline_at is not None:
            local_deadline = deadline_at.astimezone().strftime("%H:%M")
            body = f"{label} 관련 이벤트 감지. {local_deadline} 전후로 분석 결과를 알려드립니다. 액션은 잠시 보류해 주세요."
        else:
            body = f"{label} 관련 이벤트 감지. 분석 중이며, 결과가 준비되면 바로 알려드립니다."
        return UserNotification(
            level="watch",
            body=body,
            action_required=False,
            kind="event_acknowledged",
            estimated_followup=deadline_at.isoformat(timespec="seconds") if deadline_at else None,
            sent_at=sent_at.isoformat(timespec="seconds"),
        )

    def _judge_phase(
        self,
        *,
        query: str,
        portfolio: PortfolioSnapshot,
        responses: list[AgentResponse],
        stage: str,
    ) -> dict[str, Any]:
        response_payloads = [self._compact_agent_response(response) for response in responses]
        prompt = json.dumps(
            {
                "stage": stage,
                "query": query,
                "portfolio": {
                    "holdings": [
                        {
                            "ticker": holding.ticker,
                            "company_name": holding.company_name,
                            "weight": round(float(holding.weight), 4),
                        }
                        for holding in portfolio.holdings
                    ],
                    "user_preferences": list(portfolio.user_preferences[:4]),
                },
                "agent_responses": response_payloads,
                "instructions": {
                    "required_keys": JUDGE_PHASE_REQUIRED_KEYS,
                    "decision_values": [item.value for item in DecisionType],
                    "urgency_values": [item.value for item in Urgency],
                    "notification_levels": JUDGE_NOTIFICATION_LEVELS,
                    "plan_format": {"ticker": "weight_delta"},
                    "notes": [
                        "Use only supplied agent responses.",
                        "Keep summary concise and in Korean when possible.",
                        "If no trade is justified, return an empty candidate_rebalance_plan object.",
                    ],
                },
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        system_prompt = JUDGE_PHASE_SYSTEM_PROMPT
        try:
            payload = self.client.chat_json(
                system_prompt=system_prompt,
                user_prompt=prompt,
                temperature=0.0,
            )
        except ChatClientError:
            return self._fallback_judge_payload(query=query, portfolio=portfolio, responses=responses, stage=stage)
        payload = sanitize_judge_payload(payload, portfolio=portfolio, stage=stage)
        if self._is_low_signal_judge_payload(payload):
            return self._fallback_judge_payload(query=query, portfolio=portfolio, responses=responses, stage=stage)
        return payload

    def _compact_agent_response(self, response: AgentResponse) -> dict[str, Any]:
        evidence = response.evidence
        evidence_summary: dict[str, Any]
        if response.agent_id == "disclosure":
            evidence_summary = {
                "found_count": evidence.get("found_count", 0),
                "upcoming_disclosures": evidence.get("upcoming_disclosures", []),
                "items": [
                    {
                        "ticker": item.get("ticker"),
                        "disclosure_type": item.get("disclosure_type"),
                        "timestamp": item.get("timestamp"),
                    }
                    for item in evidence.get("items", [])[:3]
                    if isinstance(item, Mapping)
                ],
            }
        elif response.agent_id == "report":
            evidence_summary = {
                "coverage_reports_count": evidence.get("coverage_reports_count", 0),
                "preview_reports_count": evidence.get("preview_reports_count", 0),
                "items": [
                    {
                        "broker": item.get("broker"),
                        "report_type": item.get("report_type"),
                        "matched_holdings": item.get("matched_holdings"),
                    }
                    for item in evidence.get("items", [])[:3]
                    if isinstance(item, Mapping)
                ],
            }
        elif response.agent_id == "news":
            company_findings = evidence.get("company_findings")
            evidence_summary = {
                "sub_role": evidence.get("sub_role"),
                "company_findings": list(company_findings.keys())[:4] if isinstance(company_findings, Mapping) else [],
                "cross_check_count": evidence.get("cross_check_count", 0),
            }
        else:
            evidence_summary = {
                "mode": evidence.get("mode"),
                "keys": list(evidence.keys())[:6],
            }
        return {
            "agent_id": response.agent_id,
            "verdict": response.verdict.value,
            "direction": response.direction,
            "strength": response.strength,
            "urgency": response.urgency.value,
            "confidence": response.confidence,
            "reasoning_for_judge_agent": truncate(response.reasoning_for_judge_agent, 220),
            "limits_acknowledged": truncate(response.limits_acknowledged or "", 140) or None,
            "focus_tickers": list(response.focus_tickers),
            "evidence_summary": evidence_summary,
        }

    def _sanitize_plan(
        self,
        candidate_plan: Mapping[str, Any],
        portfolio: PortfolioSnapshot,
    ) -> dict[str, float]:
        allowed = {normalize_ticker(holding.ticker): holding.ticker for holding in portfolio.holdings}
        sanitized: dict[str, float] = {}
        for raw_ticker, raw_delta in candidate_plan.items():
            normalized = normalize_ticker(str(raw_ticker))
            if normalized not in allowed:
                continue
            try:
                delta = clamp(float(raw_delta), -0.1, 0.1)
            except (TypeError, ValueError):
                continue
            if abs(delta) < 0.005:
                continue
            sanitized[allowed[normalized]] = round(delta, 4)
        return sanitized

    def _consensus_metrics(self, responses: list[AgentResponse]) -> tuple[float, float]:
        weighted_total = 0.0
        weight_sum = 0.0
        directional_values: list[float] = []
        for response in responses:
            if response.agent_id == "cost":
                continue
            weight = max(0.05, response.confidence * max(response.strength, 0.2))
            weighted_total += response.direction * weight
            weight_sum += weight
            directional_values.append(response.direction)
        consensus = weighted_total / weight_sum if weight_sum else 0.0
        if len(directional_values) <= 1:
            divergence = 0.0
        else:
            mean_value = sum(directional_values) / len(directional_values)
            variance = sum((value - mean_value) ** 2 for value in directional_values) / len(directional_values)
            divergence = variance ** 0.5
        return round(clamp(consensus, -1.0, 1.0), 4), round(clamp(divergence, 0.0, 1.0), 4)

    def _decision_trace(
        self,
        *,
        query: str,
        executed_calls: list[PlannedAgentCall],
        responses: list[AgentResponse],
        decision: JudgeDecision,
    ) -> list[DecisionTraceNode]:
        trace: list[DecisionTraceNode] = []
        for index, response in enumerate(responses):
            planned_call = executed_calls[index] if index < len(executed_calls) else None
            trace.append(
                DecisionTraceNode(
                    turn_number=response.turn_number,
                    phase=DecisionPhase.INFORMATION_GATHERING if response.agent_id in {"disclosure", "news", "report"} else DecisionPhase.DELIBERATION,
                    actor=response.agent_id,
                    query=planned_call.query if planned_call else query,
                    summary=response.reasoning_for_judge_agent or response.query_understood,
                    context=planned_call.context if planned_call else None,
                    note=planned_call.note if planned_call else None,
                    references=tuple(response.references),
                    tools_called=tuple(response.tools_called),
                )
            )
        trace.append(
            DecisionTraceNode(
                turn_number=len(trace) + 1,
                phase=DecisionPhase.CONSENSUS,
                actor="judge",
                query="합의 형성",
                summary=(
                    f"Consensus {decision.consensus_score:.2f}, divergence {decision.divergence_score:.2f}, "
                    f"candidate plan {decision.candidate_rebalance_plan or '{}'}."
                ),
            )
        )
        trace.append(
            DecisionTraceNode(
                turn_number=len(trace) + 1,
                phase=DecisionPhase.DECISION,
                actor="judge",
                query=query,
                summary=decision.summary,
                references=(),
                tools_called=(),
            )
        )
        return trace

    def _is_low_signal_judge_payload(self, payload: Mapping[str, Any]) -> bool:
        decision = str(payload.get("decision", "")).strip()
        summary = str(payload.get("summary", "")).strip()
        confidence = payload.get("confidence", 0)
        try:
            confidence_value = float(confidence)
        except (TypeError, ValueError):
            confidence_value = 0.0
        return not decision or not summary or confidence_value <= 0.0

    def _fallback_judge_payload(
        self,
        *,
        query: str,
        portfolio: PortfolioSnapshot,
        responses: list[AgentResponse],
        stage: str,
    ) -> dict[str, Any]:
        del portfolio, stage
        consensus, divergence = self._consensus_metrics(responses)
        ticker_votes: dict[str, float] = {}
        for response in responses:
            if not response.focus_tickers or response.agent_id == "cost":
                continue
            contribution = response.direction * max(response.confidence, 0.2) / len(response.focus_tickers)
            for ticker in response.focus_tickers:
                ticker_votes[ticker] = ticker_votes.get(ticker, 0.0) + contribution

        best_ticker = None
        best_score = 0.0
        worst_ticker = None
        worst_score = 0.0
        for ticker, score in ticker_votes.items():
            if score > best_score:
                best_ticker = ticker
                best_score = score
            if score < worst_score:
                worst_ticker = ticker
                worst_score = score

        candidate_plan: dict[str, float] = {}
        if best_ticker and worst_ticker and best_score >= 0.18 and worst_score <= -0.08 and best_ticker != worst_ticker:
            candidate_plan = {
                best_ticker: 0.05,
                worst_ticker: -0.05,
            }

        if divergence >= 0.28:
            decision = DecisionType.DEFER.value
            urgency = Urgency.DEFER.value
            summary = "신호가 엇갈려 지금은 결정을 미루는 편이 낫습니다."
        elif candidate_plan and consensus >= 0.18:
            decision = DecisionType.REBALANCE.value
            urgency = Urgency.SCHEDULED.value
            summary = f"{best_ticker} 비중 확대와 {worst_ticker} 비중 축소 초안을 검토할 만합니다."
        elif consensus <= -0.25:
            decision = DecisionType.USER_DECISION_REQUIRED.value
            urgency = Urgency.WATCH.value
            summary = "부정 신호가 우세하지만 자동 매매보다 사용자 확인이 먼저 필요합니다."
        else:
            decision = DecisionType.HOLD.value
            urgency = Urgency.DEFER.value
            summary = "현재 로컬 근거만으로는 포트폴리오를 바로 바꿀 이유가 크지 않습니다."

        follow_up_at = self._default_follow_up_at(query=query, responses=responses) if decision == DecisionType.DEFER.value else None
        feedback_checkpoint = self._default_feedback_checkpoint() if decision == DecisionType.REBALANCE.value else None
        return {
            "decision": decision,
            "summary": summary,
            "confidence": clamp(0.35 + (0.25 * abs(consensus)), 0.0, 0.78),
            "urgency": urgency,
            "reasoning": "Heuristic judge fallback derived from weighted agent directions and disagreement.",
            "candidate_rebalance_plan": candidate_plan,
            "needs_trade_evaluation": bool(candidate_plan),
            "follow_up_at": follow_up_at,
            "feedback_checkpoint": feedback_checkpoint,
            "user_notification": {
                "level": self._notification_level(DecisionType(decision), Urgency(urgency)),
                "body": summary,
                "action_required": decision == DecisionType.USER_DECISION_REQUIRED.value,
            },
        }

    def _notification_level(self, decision: DecisionType, urgency: Urgency) -> str:
        if decision == DecisionType.USER_DECISION_REQUIRED:
            return "push"
        if decision == DecisionType.DEFER:
            return "info"
        if urgency in {Urgency.IMMEDIATE, Urgency.WATCH}:
            return "watch"
        if decision == DecisionType.HOLD:
            return "silent"
        return "info"

    def _default_follow_up_at(self, *, query: str, responses: list[AgentResponse]) -> str:
        now = datetime.now().astimezone()
        lowered = query.casefold()
        if any(response.agent_id == "report" and response.verdict == AgentVerdict.DIRECT_ANSWER_UNAVAILABLE for response in responses):
            delta = timedelta(hours=1)
        elif any(token in lowered for token in ("장중", "속보", "breaking")):
            delta = timedelta(minutes=30)
        else:
            delta = timedelta(hours=4)
        return (now + delta).isoformat(timespec="seconds")

    def _default_feedback_checkpoint(self) -> str:
        return (datetime.now().astimezone() + timedelta(days=7)).isoformat(timespec="seconds")

    def _sanitize_future_timestamp(self, value: str | None, *, default: str | None) -> str | None:
        if not value:
            return default
        candidate = coerce_datetime(value)
        now = datetime.now().astimezone(candidate.tzinfo)
        if candidate < now - timedelta(minutes=5):
            return default
        return candidate.isoformat(timespec="seconds")

