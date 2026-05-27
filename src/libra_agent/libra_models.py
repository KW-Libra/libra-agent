from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from .utils import coerce_datetime


class AgentVerdict(StrEnum):
    DIRECT_ANSWER = "DIRECT_ANSWER"
    PARTIAL_ANSWER = "PARTIAL_ANSWER"
    DIRECT_ANSWER_UNAVAILABLE = "DIRECT_ANSWER_UNAVAILABLE"
    QUIET = "QUIET"


class Urgency(StrEnum):
    IMMEDIATE = "immediate"
    SCHEDULED = "scheduled"
    WATCH = "watch"
    DEFER = "defer"


class DecisionType(StrEnum):
    HOLD = "HOLD"
    DEFER = "DEFER"
    USER_DECISION_REQUIRED = "USER_DECISION_REQUIRED"
    REBALANCE = "REBALANCE"


class DecisionPhase(StrEnum):
    INFORMATION_GATHERING = "information_gathering"
    DELIBERATION = "deliberation"
    CONSENSUS = "consensus"
    DECISION = "decision"


def _as_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    return default


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    return []


def _normalize_urgency(value: Any, default: Urgency = Urgency.DEFER) -> Urgency:
    try:
        return Urgency(str(value))
    except ValueError:
        return default


def _normalize_verdict(
    value: Any, default: AgentVerdict = AgentVerdict.PARTIAL_ANSWER
) -> AgentVerdict:
    try:
        return AgentVerdict(str(value))
    except ValueError:
        return default


def _normalize_decision(value: Any, default: DecisionType = DecisionType.HOLD) -> DecisionType:
    try:
        return DecisionType(str(value))
    except ValueError:
        return default


def _normalize_phase(
    value: Any, default: DecisionPhase = DecisionPhase.INFORMATION_GATHERING
) -> DecisionPhase:
    try:
        return DecisionPhase(str(value))
    except ValueError:
        return default


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


@dataclass(slots=True, frozen=True)
class Reference:
    agent_id: str
    opinion_id: str
    relation: str
    note: str | None = None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Reference:
        return cls(
            agent_id=_as_str(payload.get("agent_id")),
            opinion_id=_as_str(payload.get("opinion_id")),
            relation=_as_str(payload.get("relation")),
            note=_as_str(payload.get("note")) or None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "opinion_id": self.opinion_id,
            "relation": self.relation,
            "note": self.note,
        }


@dataclass(slots=True, frozen=True)
class ToolCall:
    tool_name: str
    purpose: str
    summary: str

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ToolCall:
        return cls(
            tool_name=_as_str(payload.get("tool_name")),
            purpose=_as_str(payload.get("purpose")),
            summary=_as_str(payload.get("summary")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "purpose": self.purpose,
            "summary": self.summary,
        }


@dataclass(slots=True)
class AgentResponse:
    agent_id: str
    opinion_id: str
    turn_number: int
    query_understood: str
    verdict: AgentVerdict
    evidence: dict[str, Any]
    direction: float
    strength: float
    urgency: Urgency
    confidence: float
    reasoning_for_judge_agent: str
    signal_score: float = 0.0
    source_trust: float = 0.5
    event_type: str | None = None
    horizon: str | None = None
    risk_level: str = "low"
    opinion: str = "NEUTRAL"
    limits_acknowledged: str | None = None
    references: list[Reference] = field(default_factory=list)
    tools_called: list[ToolCall] = field(default_factory=list)
    depth_used: str = "medium"
    focus_tickers: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> AgentResponse:
        references = [
            Reference.from_dict(item)
            for item in _as_list(payload.get("references"))
            if isinstance(item, Mapping)
        ]
        tools_called = [
            ToolCall.from_dict(item)
            for item in _as_list(payload.get("tools_called"))
            if isinstance(item, Mapping)
        ]
        return cls(
            agent_id=_as_str(payload.get("agent_id")),
            opinion_id=_as_str(payload.get("opinion_id")),
            turn_number=int(_as_float(payload.get("turn_number"), 0)),
            query_understood=_as_str(payload.get("query_understood")),
            verdict=_normalize_verdict(payload.get("verdict")),
            evidence=_as_dict(payload.get("evidence")),
            direction=_clamp(_as_float(payload.get("direction"), 0.0), -1.0, 1.0),
            strength=_clamp(_as_float(payload.get("strength"), 0.0), 0.0, 1.0),
            urgency=_normalize_urgency(payload.get("urgency")),
            confidence=_clamp(_as_float(payload.get("confidence"), 0.0), 0.0, 1.0),
            signal_score=_clamp(_as_float(payload.get("signal_score"), 0.0), -1.0, 1.0),
            source_trust=_clamp(_as_float(payload.get("source_trust"), 0.5), 0.0, 1.0),
            event_type=_as_str(payload.get("event_type")) or None,
            horizon=_as_str(payload.get("horizon")) or None,
            risk_level=_as_str(payload.get("risk_level"), "low") or "low",
            opinion=_as_str(payload.get("opinion"), "NEUTRAL") or "NEUTRAL",
            reasoning_for_judge_agent=_as_str(payload.get("reasoning_for_judge_agent")),
            limits_acknowledged=_as_str(payload.get("limits_acknowledged")) or None,
            references=references,
            tools_called=tools_called,
            depth_used=_as_str(payload.get("depth_used"), "medium") or "medium",
            focus_tickers=[
                _as_str(item) for item in _as_list(payload.get("focus_tickers")) if _as_str(item)
            ],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "opinion_id": self.opinion_id,
            "turn_number": self.turn_number,
            "query_understood": self.query_understood,
            "verdict": self.verdict.value,
            "evidence": self.evidence,
            "direction": round(float(self.direction), 4),
            "strength": round(float(self.strength), 4),
            "urgency": self.urgency.value,
            "confidence": round(float(self.confidence), 4),
            "signal_score": round(float(self.signal_score), 4),
            "source_trust": round(float(self.source_trust), 4),
            "event_type": self.event_type,
            "horizon": self.horizon,
            "risk_level": self.risk_level,
            "opinion": self.opinion,
            "reasoning_for_judge_agent": self.reasoning_for_judge_agent,
            "limits_acknowledged": self.limits_acknowledged,
            "references": [item.to_dict() for item in self.references],
            "tools_called": [item.to_dict() for item in self.tools_called],
            "depth_used": self.depth_used,
            "focus_tickers": list(self.focus_tickers),
        }


@dataclass(slots=True, frozen=True)
class PortfolioHolding:
    ticker: str
    company_name: str
    weight: float
    aliases: tuple[str, ...] = ()
    sector: str | None = None
    esg_score: float | None = None
    carbon_intensity: float | None = None
    shares: float | None = None
    last_price: float | None = None
    average_price: float | None = None
    market_value_krw: float | None = None
    unrealized_pnl_krw: float | None = None
    avg_daily_volume: float | None = None
    avg_daily_turnover_krw: float | None = None
    bid_ask_spread_bps: float | None = None
    free_float_ratio_pct: float | None = None
    ohlcv: tuple[dict[str, Any], ...] = ()
    daily_returns: tuple[float, ...] = ()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PortfolioHolding:
        return cls(
            ticker=_as_str(payload.get("ticker")),
            company_name=_as_str(payload.get("company_name") or payload.get("name")),
            weight=_clamp(_as_float(payload.get("weight"), 0.0), 0.0, 1.0),
            aliases=tuple(
                _as_str(item) for item in _as_list(payload.get("aliases")) if _as_str(item)
            ),
            sector=_as_str(payload.get("sector")) or None,
            esg_score=_as_float(payload.get("esg_score"))
            if payload.get("esg_score") is not None
            else None,
            carbon_intensity=_as_float(payload.get("carbon_intensity"))
            if payload.get("carbon_intensity") is not None
            else None,
            shares=_as_float(payload.get("shares")) if payload.get("shares") is not None else None,
            last_price=_as_float(payload.get("last_price"))
            if payload.get("last_price") is not None
            else None,
            average_price=_as_float(payload.get("average_price"))
            if payload.get("average_price") is not None
            else None,
            market_value_krw=_as_float(payload.get("market_value_krw"))
            if payload.get("market_value_krw") is not None
            else None,
            unrealized_pnl_krw=_as_float(payload.get("unrealized_pnl_krw"))
            if payload.get("unrealized_pnl_krw") is not None
            else None,
            avg_daily_volume=_as_float(
                payload.get("avg_daily_volume") or payload.get("adv_volume")
            )
            if payload.get("avg_daily_volume") is not None
            or payload.get("adv_volume") is not None
            else None,
            avg_daily_turnover_krw=_as_float(
                payload.get("avg_daily_turnover_krw")
                or payload.get("average_daily_turnover_krw")
                or payload.get("adv_krw")
            )
            if payload.get("avg_daily_turnover_krw") is not None
            or payload.get("average_daily_turnover_krw") is not None
            or payload.get("adv_krw") is not None
            else None,
            bid_ask_spread_bps=_as_float(payload.get("bid_ask_spread_bps"))
            if payload.get("bid_ask_spread_bps") is not None
            else None,
            free_float_ratio_pct=_as_float(payload.get("free_float_ratio_pct"))
            if payload.get("free_float_ratio_pct") is not None
            else None,
            ohlcv=tuple(
                dict(item)
                for item in _as_list(payload.get("ohlcv"))
                if isinstance(item, Mapping)
            ),
            daily_returns=tuple(
                _as_float(item)
                for item in _as_list(payload.get("daily_returns") or payload.get("returns"))
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "company_name": self.company_name,
            "weight": round(float(self.weight), 6),
            "aliases": list(self.aliases),
            "sector": self.sector,
            "esg_score": self.esg_score,
            "carbon_intensity": self.carbon_intensity,
            "shares": self.shares,
            "last_price": self.last_price,
            "average_price": self.average_price,
            "market_value_krw": self.market_value_krw,
            "unrealized_pnl_krw": self.unrealized_pnl_krw,
            "avg_daily_volume": self.avg_daily_volume,
            "avg_daily_turnover_krw": self.avg_daily_turnover_krw,
            "bid_ask_spread_bps": self.bid_ask_spread_bps,
            "free_float_ratio_pct": self.free_float_ratio_pct,
            "ohlcv": [dict(item) for item in self.ohlcv],
            "daily_returns": list(self.daily_returns),
        }


@dataclass(slots=True, frozen=True)
class PortfolioSnapshot:
    generated_at: datetime
    holdings: tuple[PortfolioHolding, ...]
    total_value_krw: float | None = None
    cash_weight: float = 0.0
    user_preferences: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PortfolioSnapshot:
        holdings = tuple(
            PortfolioHolding.from_dict(item)
            for item in _as_list(payload.get("holdings"))
            if isinstance(item, Mapping)
        )
        return cls(
            generated_at=coerce_datetime(_as_str(payload.get("generated_at")) or None),
            holdings=holdings,
            total_value_krw=_as_float(payload.get("total_value_krw"))
            if payload.get("total_value_krw") is not None
            else None,
            cash_weight=_clamp(_as_float(payload.get("cash_weight"), 0.0), 0.0, 1.0),
            user_preferences=tuple(
                _as_str(item) for item in _as_list(payload.get("user_preferences")) if _as_str(item)
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at.isoformat(),
            "holdings": [holding.to_dict() for holding in self.holdings],
            "total_value_krw": self.total_value_krw,
            "cash_weight": round(float(self.cash_weight), 6),
            "user_preferences": list(self.user_preferences),
        }


@dataclass(slots=True, frozen=True)
class KnowledgeEntity:
    entity_id: str
    entity_type: str
    entity_name: str
    ticker: str | None = None
    confidence: float = 0.0

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> KnowledgeEntity:
        return cls(
            entity_id=_as_str(payload.get("entity_id")),
            entity_type=_as_str(payload.get("entity_type")),
            entity_name=_as_str(payload.get("entity_name")),
            ticker=_as_str(payload.get("ticker")) or None,
            confidence=_clamp(_as_float(payload.get("confidence"), 0.0), 0.0, 1.0),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "entity_name": self.entity_name,
            "ticker": self.ticker,
            "confidence": round(float(self.confidence), 4),
        }


@dataclass(slots=True, frozen=True)
class KnowledgeDocument:
    doc_id: str
    doc_type: str
    title: str
    body: str
    publisher: str
    source_name: str
    source_url: str
    region: str
    published_at: datetime
    relevance_score: float | None = None
    event_type: str | None = None
    event_type_score: float | None = None
    entities: tuple[KnowledgeEntity, ...] = ()
    matched_holdings: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> KnowledgeDocument:
        return cls(
            doc_id=_as_str(payload.get("doc_id")),
            doc_type=_as_str(payload.get("doc_type")),
            title=_as_str(payload.get("title")),
            body=_as_str(payload.get("body")),
            publisher=_as_str(payload.get("publisher")),
            source_name=_as_str(payload.get("source_name")),
            source_url=_as_str(payload.get("source_url")),
            region=_as_str(payload.get("region")),
            published_at=coerce_datetime(_as_str(payload.get("published_at")) or None),
            relevance_score=_as_float(payload.get("relevance_score"))
            if payload.get("relevance_score") is not None
            else None,
            event_type=_as_str(payload.get("event_type")) or None,
            event_type_score=_as_float(payload.get("event_type_score"))
            if payload.get("event_type_score") is not None
            else None,
            entities=tuple(
                KnowledgeEntity.from_dict(item)
                for item in _as_list(payload.get("entities"))
                if isinstance(item, Mapping)
            ),
            matched_holdings=tuple(
                _as_str(item) for item in _as_list(payload.get("matched_holdings")) if _as_str(item)
            ),
            metadata=dict(payload.get("metadata", {}))
            if isinstance(payload.get("metadata"), Mapping)
            else {},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "doc_type": self.doc_type,
            "title": self.title,
            "body": self.body,
            "publisher": self.publisher,
            "source_name": self.source_name,
            "source_url": self.source_url,
            "region": self.region,
            "published_at": self.published_at.isoformat(),
            "relevance_score": round(float(self.relevance_score), 4)
            if self.relevance_score is not None
            else None,
            "event_type": self.event_type,
            "event_type_score": round(float(self.event_type_score), 4)
            if self.event_type_score is not None
            else None,
            "entities": [entity.to_dict() for entity in self.entities],
            "matched_holdings": list(self.matched_holdings),
            "metadata": self.metadata,
        }


@dataclass(slots=True, frozen=True)
class KnowledgeEvent:
    event_id: str
    event_type: str
    event_time: datetime
    headline: str
    summary: str
    confidence: float
    source_documents: tuple[str, ...]
    matched_holdings: tuple[str, ...] = ()
    entities: tuple[KnowledgeEntity, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> KnowledgeEvent:
        return cls(
            event_id=_as_str(payload.get("event_id")),
            event_type=_as_str(payload.get("event_type")),
            event_time=coerce_datetime(_as_str(payload.get("event_time")) or None),
            headline=_as_str(payload.get("headline")),
            summary=_as_str(payload.get("summary")),
            confidence=_clamp(_as_float(payload.get("confidence"), 0.0), 0.0, 1.0),
            source_documents=tuple(
                _as_str(item) for item in _as_list(payload.get("source_documents")) if _as_str(item)
            ),
            matched_holdings=tuple(
                _as_str(item) for item in _as_list(payload.get("matched_holdings")) if _as_str(item)
            ),
            entities=tuple(
                KnowledgeEntity.from_dict(item)
                for item in _as_list(payload.get("entities"))
                if isinstance(item, Mapping)
            ),
            metadata=dict(payload.get("metadata", {}))
            if isinstance(payload.get("metadata"), Mapping)
            else {},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "event_time": self.event_time.isoformat(),
            "headline": self.headline,
            "summary": self.summary,
            "confidence": round(float(self.confidence), 4),
            "source_documents": list(self.source_documents),
            "matched_holdings": list(self.matched_holdings),
            "entities": [entity.to_dict() for entity in self.entities],
            "metadata": self.metadata,
        }


@dataclass(slots=True, frozen=True)
class TriggerEvent:
    trigger_type: str = "news_push"
    headline: str = ""
    summary: str | None = None
    ticker: str | None = None
    company_name: str | None = None
    source: str | None = None
    event_time: str | None = None
    cross_check_count: int = 1
    market_reaction: str | None = None
    severity: str = "watch"

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TriggerEvent:
        return cls(
            trigger_type=_as_str(payload.get("trigger_type"), "news_push"),
            headline=_as_str(payload.get("headline")),
            summary=_as_str(payload.get("summary")) or None,
            ticker=_as_str(payload.get("ticker")) or None,
            company_name=_as_str(payload.get("company_name")) or None,
            source=_as_str(payload.get("source")) or None,
            event_time=_as_str(payload.get("event_time")) or None,
            cross_check_count=int(_as_float(payload.get("cross_check_count"), 1)),
            market_reaction=_as_str(payload.get("market_reaction")) or None,
            severity=_as_str(payload.get("severity"), "watch") or "watch",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "trigger_type": self.trigger_type,
            "headline": self.headline,
            "summary": self.summary,
            "ticker": self.ticker,
            "company_name": self.company_name,
            "source": self.source,
            "event_time": self.event_time,
            "cross_check_count": self.cross_check_count,
            "market_reaction": self.market_reaction,
            "severity": self.severity,
        }


@dataclass(slots=True, frozen=True)
class UserNotification:
    level: str
    body: str
    action_required: bool = False
    kind: str | None = None
    estimated_followup: str | None = None
    sent_at: str | None = None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> UserNotification:
        return cls(
            level=_as_str(payload.get("level") or payload.get("urgency"), "info"),
            body=_as_str(payload.get("body")),
            action_required=_as_bool(payload.get("action_required"), False),
            kind=_as_str(payload.get("kind") or payload.get("type")) or None,
            estimated_followup=_as_str(payload.get("estimated_followup")) or None,
            sent_at=_as_str(payload.get("sent_at")) or None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "body": self.body,
            "action_required": self.action_required,
            "kind": self.kind,
            "estimated_followup": self.estimated_followup,
            "sent_at": self.sent_at,
        }


@dataclass(slots=True, frozen=True)
class DecisionTraceNode:
    turn_number: int
    phase: DecisionPhase
    actor: str
    query: str
    summary: str
    context: str | None = None
    note: str | None = None
    references: tuple[Reference, ...] = ()
    tools_called: tuple[ToolCall, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_number": self.turn_number,
            "phase": self.phase.value,
            "actor": self.actor,
            "query": self.query,
            "summary": self.summary,
            "context": self.context,
            "note": self.note,
            "references": [item.to_dict() for item in self.references],
            "tools_called": [item.to_dict() for item in self.tools_called],
        }


@dataclass(slots=True)
class JudgeDecision:
    decision: DecisionType
    summary: str
    confidence: float
    urgency: Urgency
    called_agents: list[str] = field(default_factory=list)
    skipped_agents: list[str] = field(default_factory=list)
    skip_rationale: dict[str, str] = field(default_factory=dict)
    candidate_rebalance_plan: dict[str, float] = field(default_factory=dict)
    decision_trace: list[DecisionTraceNode] = field(default_factory=list)
    reasoning: str = ""
    user_notification: UserNotification | None = None
    follow_up_at: str | None = None
    feedback_checkpoint: str | None = None
    consensus_score: float = 0.0
    divergence_score: float = 0.0
    needs_trade_evaluation: bool = False
    trigger: str = "pull"
    trigger_event: TriggerEvent | None = None
    deadline_at: str | None = None
    elapsed_seconds: float | None = None
    options: list[str] = field(default_factory=list)
    auto_safeguards: dict[str, Any] = field(default_factory=dict)
    notification_log: list[UserNotification] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> JudgeDecision:
        trace = []
        for item in _as_list(payload.get("decision_trace")):
            if not isinstance(item, Mapping):
                continue
            trace.append(
                DecisionTraceNode(
                    turn_number=int(_as_float(item.get("turn_number"), 0)),
                    phase=_normalize_phase(item.get("phase")),
                    actor=_as_str(item.get("actor")),
                    query=_as_str(item.get("query")),
                    summary=_as_str(item.get("summary")),
                    context=_as_str(item.get("context")) or None,
                    note=_as_str(item.get("note")) or None,
                    references=tuple(
                        Reference.from_dict(ref_item)
                        for ref_item in _as_list(item.get("references"))
                        if isinstance(ref_item, Mapping)
                    ),
                    tools_called=tuple(
                        ToolCall.from_dict(tool_item)
                        for tool_item in _as_list(item.get("tools_called"))
                        if isinstance(tool_item, Mapping)
                    ),
                )
            )
        notification_payload = payload.get("user_notification")
        trigger_event_payload = payload.get("trigger_event")
        notification_log = [
            UserNotification.from_dict(item)
            for item in _as_list(payload.get("notification_log"))
            if isinstance(item, Mapping)
        ]
        return cls(
            decision=_normalize_decision(payload.get("decision")),
            summary=_as_str(payload.get("summary")),
            confidence=_clamp(_as_float(payload.get("confidence"), 0.0), 0.0, 1.0),
            urgency=_normalize_urgency(payload.get("urgency")),
            called_agents=[
                _as_str(item) for item in _as_list(payload.get("called_agents")) if _as_str(item)
            ],
            skipped_agents=[
                _as_str(item) for item in _as_list(payload.get("skipped_agents")) if _as_str(item)
            ],
            skip_rationale={
                _as_str(key): _as_str(value)
                for key, value in _as_dict(payload.get("skip_rationale")).items()
            },
            candidate_rebalance_plan={
                _as_str(key): _clamp(_as_float(value), -1.0, 1.0)
                for key, value in _as_dict(payload.get("candidate_rebalance_plan")).items()
                if _as_str(key)
            },
            decision_trace=trace,
            reasoning=_as_str(payload.get("reasoning")),
            user_notification=UserNotification.from_dict(notification_payload)
            if isinstance(notification_payload, Mapping)
            else None,
            follow_up_at=_as_str(payload.get("follow_up_at")) or None,
            feedback_checkpoint=_as_str(payload.get("feedback_checkpoint")) or None,
            consensus_score=_clamp(_as_float(payload.get("consensus_score"), 0.0), -1.0, 1.0),
            divergence_score=_clamp(_as_float(payload.get("divergence_score"), 0.0), 0.0, 1.0),
            needs_trade_evaluation=_as_bool(payload.get("needs_trade_evaluation"), False),
            trigger=_as_str(payload.get("trigger"), "pull") or "pull",
            trigger_event=TriggerEvent.from_dict(trigger_event_payload)
            if isinstance(trigger_event_payload, Mapping)
            else None,
            deadline_at=_as_str(payload.get("deadline_at")) or None,
            elapsed_seconds=_as_float(payload.get("elapsed_seconds"))
            if payload.get("elapsed_seconds") is not None
            else None,
            options=[_as_str(item) for item in _as_list(payload.get("options")) if _as_str(item)],
            auto_safeguards=_as_dict(payload.get("auto_safeguards")),
            notification_log=notification_log,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision.value,
            "summary": self.summary,
            "confidence": round(float(self.confidence), 4),
            "urgency": self.urgency.value,
            "called_agents": list(self.called_agents),
            "skipped_agents": list(self.skipped_agents),
            "skip_rationale": dict(self.skip_rationale),
            "candidate_rebalance_plan": {
                key: round(float(value), 6) for key, value in self.candidate_rebalance_plan.items()
            },
            "decision_trace": [node.to_dict() for node in self.decision_trace],
            "reasoning": self.reasoning,
            "user_notification": self.user_notification.to_dict()
            if self.user_notification
            else None,
            "follow_up_at": self.follow_up_at,
            "feedback_checkpoint": self.feedback_checkpoint,
            "consensus_score": round(float(self.consensus_score), 4),
            "divergence_score": round(float(self.divergence_score), 4),
            "needs_trade_evaluation": self.needs_trade_evaluation,
            "trigger": self.trigger,
            "trigger_event": self.trigger_event.to_dict() if self.trigger_event else None,
            "deadline_at": self.deadline_at,
            "elapsed_seconds": round(float(self.elapsed_seconds), 3)
            if self.elapsed_seconds is not None
            else None,
            "options": list(self.options),
            "auto_safeguards": dict(self.auto_safeguards),
            "notification_log": [item.to_dict() for item in self.notification_log],
        }
