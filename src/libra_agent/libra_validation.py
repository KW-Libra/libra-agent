from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .libra.signals import infer_signal_profile
from .libra_models import (
    AgentResponse,
    AgentVerdict,
    DecisionType,
    PortfolioSnapshot,
    Reference,
    Urgency,
)
from .utils import collapse_whitespace, parse_datetime_or_none

ALLOWED_NEWS_SUB_ROLES = {"macro", "company_specific", "mixed"}
ALLOWED_SENTIMENTS = {"positive", "negative", "neutral", "mixed"}
ALLOWED_SOURCE_RELIABILITY = {"low", "medium", "high"}
ALLOWED_NOTIFICATION_LEVELS = {"silent", "info", "watch", "push"}


def _normalize_key(value: str) -> str:
    return "".join(char for char in value.upper() if char.isalnum())


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    return []


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


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _clean_text(value: Any, *, default: str = "", limit: int | None = None) -> str:
    if value is None:
        return default
    text = collapse_whitespace(str(value))
    if not text:
        return default
    if limit is not None and len(text) > limit:
        return text[: limit - 3].rstrip() + "..."
    return text


def _clean_optional_string(value: Any, *, limit: int | None = None) -> str | None:
    if value is None or isinstance(value, bool):
        return None
    cleaned = _clean_text(value, limit=limit)
    return cleaned or None


def _datetime_string(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    parsed = parse_datetime_or_none(value)
    if parsed is None:
        return None
    return parsed.isoformat(timespec="seconds")


def _build_ticker_alias_map(portfolio: PortfolioSnapshot) -> dict[str, str]:
    alias_map: dict[str, str] = {}
    for holding in portfolio.holdings:
        aliases = {
            holding.ticker,
            holding.company_name,
            *holding.aliases,
        }
        for alias in aliases:
            cleaned = _clean_text(alias)
            if not cleaned:
                continue
            alias_map[_normalize_key(cleaned)] = holding.ticker
    return alias_map


def _canonical_ticker(value: Any, alias_map: Mapping[str, str]) -> str | None:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            canonical = _canonical_ticker(item, alias_map)
            if canonical:
                return canonical
        return None
    text = _clean_text(value)
    if not text:
        return None
    return alias_map.get(_normalize_key(text))


def _canonical_ticker_list(value: Any, alias_map: Mapping[str, str]) -> list[str]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        items = list(value)
    else:
        items = [value]
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        canonical = _canonical_ticker(item, alias_map)
        if canonical and canonical not in seen:
            seen.add(canonical)
            result.append(canonical)
    return result


def _sanitize_references(value: Any) -> list[Reference]:
    references: list[Reference] = []
    for item in _as_list(value):
        if not isinstance(item, Mapping):
            continue
        reference = Reference.from_dict(item)
        if not (reference.agent_id and reference.opinion_id and reference.relation):
            continue
        references.append(reference)
    return references


def _sanitize_jsonish(value: Any, *, depth: int = 0, max_depth: int = 2) -> Any:
    if depth > max_depth:
        return None
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        cleaned = _clean_text(value)
        return cleaned or None
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            key = _clean_text(raw_key)
            if not key:
                continue
            sanitized = _sanitize_jsonish(raw_value, depth=depth + 1, max_depth=max_depth)
            if sanitized is not None:
                result[key] = sanitized
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        result = []
        for item in value:
            sanitized = _sanitize_jsonish(item, depth=depth + 1, max_depth=max_depth)
            if sanitized is not None:
                result.append(sanitized)
        return result
    return _clean_text(value) or None


def _sanitize_disclosure_evidence(value: Any, alias_map: Mapping[str, str]) -> dict[str, Any]:
    payload = _as_mapping(value)
    items: list[dict[str, Any]] = []
    for item in _as_list(payload.get("items")):
        if not isinstance(item, Mapping):
            continue
        ticker = _canonical_ticker(
            item.get("ticker") or item.get("matched_holdings") or item.get("company_name"),
            alias_map,
        )
        headline = _clean_text(
            item.get("headline")
            or item.get("title")
            or item.get("disclosure_type")
            or item.get("type"),
            limit=180,
        )
        disclosure_type = _clean_text(
            item.get("disclosure_type") or item.get("type") or headline,
            limit=120,
        )
        summary = _clean_text(item.get("summary") or item.get("body"), limit=320)
        company_name = _clean_text(item.get("company_name"), limit=120) or None
        timestamp = _datetime_string(
            item.get("timestamp") or item.get("published_at") or item.get("date")
        )
        if not any((ticker, headline, summary)):
            continue
        items.append(
            {
                "ticker": ticker,
                "company_name": company_name,
                "disclosure_type": disclosure_type or None,
                "headline": headline or None,
                "summary": summary or None,
                "timestamp": timestamp,
            }
        )

    upcoming: list[dict[str, Any]] = []
    for item in _as_list(payload.get("upcoming_disclosures")):
        if not isinstance(item, Mapping):
            continue
        ticker = _canonical_ticker(item.get("ticker") or item.get("company_name"), alias_map)
        headline = _clean_text(item.get("headline") or item.get("title"), limit=180) or None
        date = _datetime_string(
            item.get("date") or item.get("timestamp") or item.get("scheduled_at")
        )
        if not any((ticker, headline, date)):
            continue
        upcoming.append(
            {
                "ticker": ticker,
                "headline": headline,
                "date": date,
            }
        )

    found_count = _as_int(payload.get("found_count"), len(items))
    if found_count < len(items):
        found_count = len(items)
    return {
        "found_count": found_count,
        "items": items,
        "upcoming_disclosures": upcoming,
    }


def _sanitize_report_evidence(value: Any, alias_map: Mapping[str, str]) -> dict[str, Any]:
    payload = _as_mapping(value)
    items: list[dict[str, Any]] = []
    for item in _as_list(payload.get("items")):
        if not isinstance(item, Mapping):
            continue
        matched_holdings = _canonical_ticker_list(
            item.get("matched_holdings") or item.get("tickers") or item.get("ticker"),
            alias_map,
        )
        broker = _clean_text(item.get("broker") or item.get("publisher"), limit=120)
        published_at = _datetime_string(item.get("published_at") or item.get("date"))
        report_type = _clean_text(item.get("report_type"), default="other", limit=40).lower()
        key_thesis = _clean_text(
            item.get("key_thesis") or item.get("summary") or item.get("headline"), limit=320
        )
        rating = _clean_text(item.get("rating"), limit=40) or None
        target_price = item.get("target_price")
        try:
            target_price_value = float(target_price) if target_price is not None else None
        except (TypeError, ValueError):
            target_price_value = None
        if not any((broker, key_thesis, matched_holdings)):
            continue
        items.append(
            {
                "broker": broker or None,
                "published_at": published_at,
                "report_type": report_type or "other",
                "key_thesis": key_thesis or None,
                "matched_holdings": matched_holdings,
                "rating": rating,
                "target_price": target_price_value,
            }
        )

    coverage_reports_count = _as_int(payload.get("coverage_reports_count"), len(items))
    preview_reports_count = _as_int(payload.get("preview_reports_count"), 0)
    if coverage_reports_count < len(items):
        coverage_reports_count = len(items)
    consensus = _sanitize_jsonish(payload.get("consensus"), max_depth=2)
    return {
        "coverage_reports_count": coverage_reports_count,
        "preview_reports_count": max(0, preview_reports_count),
        "items": items,
        "consensus": consensus,
    }


def _sanitize_news_evidence(value: Any, alias_map: Mapping[str, str]) -> dict[str, Any]:
    payload = _as_mapping(value)
    company_findings_raw = payload.get("company_findings")
    company_findings: dict[str, Any] = {}
    if isinstance(company_findings_raw, Mapping):
        for raw_key, raw_value in company_findings_raw.items():
            if not isinstance(raw_value, Mapping):
                continue
            cleaned_key = _clean_text(raw_key)
            canonical_key = (
                "portfolio"
                if cleaned_key.casefold() == "portfolio"
                else _canonical_ticker(cleaned_key, alias_map)
            )
            if not canonical_key:
                continue
            sentiment = _clean_text(raw_value.get("sentiment"), default="neutral", limit=20).lower()
            if sentiment not in ALLOWED_SENTIMENTS:
                sentiment = "neutral"
            key_headlines = []
            for item in _as_list(raw_value.get("key_headlines")):
                headline = _clean_text(item, limit=180)
                if headline:
                    key_headlines.append(headline)
            company_findings[canonical_key] = {
                "sentiment": sentiment,
                "key_headlines": key_headlines,
                "market_reaction": _clean_text(raw_value.get("market_reaction"), limit=180) or None,
                "sector_comparison": _clean_text(raw_value.get("sector_comparison"), limit=180)
                or None,
            }

    sub_role = _clean_text(payload.get("sub_role"), default="company_specific", limit=40).lower()
    if sub_role not in ALLOWED_NEWS_SUB_ROLES:
        sub_role = "mixed" if company_findings else "company_specific"

    source_reliability = _clean_text(
        payload.get("source_reliability"), default="medium", limit=20
    ).lower()
    if source_reliability not in ALLOWED_SOURCE_RELIABILITY:
        source_reliability = "medium"

    cross_check_count = max(0, _as_int(payload.get("cross_check_count"), len(company_findings)))
    macro_findings = _sanitize_jsonish(payload.get("macro_findings"), max_depth=2)
    return {
        "sub_role": sub_role,
        "company_findings": company_findings,
        "macro_findings": macro_findings,
        "source_reliability": source_reliability,
        "cross_check_count": cross_check_count,
    }


def _sanitize_profit_evidence(value: Any, alias_map: Mapping[str, str]) -> dict[str, Any]:
    payload = _as_mapping(value)
    plan_simulation = _as_mapping(payload.get("plan_simulation"))
    rebalance_plan = {
        ticker: _clamp(_as_float(delta), -1.0, 1.0)
        for ticker, delta in (
            (_canonical_ticker(raw_ticker, alias_map), raw_delta)
            for raw_ticker, raw_delta in _as_mapping(plan_simulation.get("rebalance_plan")).items()
        )
        if ticker
    }
    ticker_signals = {
        ticker: _clamp(_as_float(signal), -1.0, 1.0)
        for ticker, signal in (
            (_canonical_ticker(raw_ticker, alias_map), raw_signal)
            for raw_ticker, raw_signal in _as_mapping(plan_simulation.get("ticker_signals")).items()
        )
        if ticker
    }
    return {
        "mode": "plan_simulation",
        "plan_simulation": {
            "rebalance_plan": rebalance_plan,
            "ticker_signals": ticker_signals,
            "expected_return_1m": round(_as_float(plan_simulation.get("expected_return_1m")), 4),
            "expected_return_3m": round(_as_float(plan_simulation.get("expected_return_3m")), 4),
            "sharpe_ratio": round(_as_float(plan_simulation.get("sharpe_ratio")), 4),
            "max_drawdown": round(_as_float(plan_simulation.get("max_drawdown")), 4),
            "recommendation_text": _clean_text(
                plan_simulation.get("recommendation_text"), limit=240
            )
            or None,
        },
    }


def _sanitize_cost_evidence(value: Any, alias_map: Mapping[str, str]) -> dict[str, Any]:
    payload = _as_mapping(value)
    trade_cost = _as_mapping(payload.get("trade_cost"))
    rebalance_plan = {
        ticker: _clamp(_as_float(delta), -1.0, 1.0)
        for ticker, delta in (
            (_canonical_ticker(raw_ticker, alias_map), raw_delta)
            for raw_ticker, raw_delta in _as_mapping(trade_cost.get("rebalance_plan")).items()
        )
        if ticker
    }
    return {
        "mode": "trade_cost",
        "trade_cost": {
            "rebalance_plan": rebalance_plan,
            "commission_krw": round(max(0.0, _as_float(trade_cost.get("commission_krw"))), 2),
            "tax_krw": round(max(0.0, _as_float(trade_cost.get("tax_krw"))), 2),
            "estimated_slippage_bp": round(
                max(0.0, _as_float(trade_cost.get("estimated_slippage_bp"))), 4
            ),
            "spread_state_bp": round(max(0.0, _as_float(trade_cost.get("spread_state_bp"))), 4),
            "total_friction_bp": round(max(0.0, _as_float(trade_cost.get("total_friction_bp"))), 4),
        },
    }


def sanitize_agent_evidence(
    *,
    agent_id: str,
    evidence: Any,
    portfolio: PortfolioSnapshot,
) -> dict[str, Any]:
    alias_map = _build_ticker_alias_map(portfolio)
    if agent_id == "disclosure":
        return _sanitize_disclosure_evidence(evidence, alias_map)
    if agent_id == "report":
        return _sanitize_report_evidence(evidence, alias_map)
    if agent_id == "news":
        return _sanitize_news_evidence(evidence, alias_map)
    if agent_id == "profit":
        return _sanitize_profit_evidence(evidence, alias_map)
    if agent_id == "cost":
        return _sanitize_cost_evidence(evidence, alias_map)
    return _sanitize_jsonish(evidence, max_depth=2) or {}


def sanitize_agent_response_payload(
    payload: Mapping[str, Any] | None,
    *,
    agent_id: str,
    portfolio: PortfolioSnapshot,
    query: str,
    turn_number: int,
    opinion_id: str,
    depth: str,
) -> AgentResponse:
    raw = dict(payload) if isinstance(payload, Mapping) else {}
    response = AgentResponse.from_dict(raw)
    evidence = sanitize_agent_evidence(
        agent_id=agent_id, evidence=raw.get("evidence"), portfolio=portfolio
    )
    alias_map = _build_ticker_alias_map(portfolio)

    if raw.get("verdict") in {item.value for item in AgentVerdict}:
        verdict = AgentVerdict(str(raw.get("verdict")))
    else:
        verdict = AgentVerdict.PARTIAL_ANSWER

    if raw.get("urgency") in {item.value for item in Urgency}:
        urgency = Urgency(str(raw.get("urgency")))
    else:
        urgency = Urgency.DEFER

    focus_tickers = _canonical_ticker_list(raw.get("focus_tickers"), alias_map)
    if not focus_tickers:
        if agent_id == "disclosure":
            focus_tickers = [
                item["ticker"]
                for item in evidence.get("items", [])
                if isinstance(item, Mapping) and item.get("ticker")
            ]
        elif agent_id == "report":
            collected: list[str] = []
            for item in evidence.get("items", []):
                if not isinstance(item, Mapping):
                    continue
                for ticker in _as_list(item.get("matched_holdings")):
                    if isinstance(ticker, str) and ticker not in collected:
                        collected.append(ticker)
            focus_tickers = collected
        elif agent_id == "news":
            focus_tickers = [
                ticker
                for ticker in evidence.get("company_findings", {}).keys()
                if ticker != "portfolio"
            ]
        else:
            plan = _as_mapping(evidence.get("plan_simulation") or evidence.get("trade_cost")).get(
                "rebalance_plan"
            )
            focus_tickers = _canonical_ticker_list(list(_as_mapping(plan).keys()), alias_map)

    response.agent_id = agent_id
    response.opinion_id = (
        _clean_text(raw.get("opinion_id"), default=opinion_id, limit=120) or opinion_id
    )
    response.turn_number = turn_number
    response.query_understood = (
        _clean_text(raw.get("query_understood"), default=query, limit=240) or query
    )
    response.verdict = verdict
    response.evidence = evidence
    response.direction = _clamp(_as_float(raw.get("direction"), response.direction), -1.0, 1.0)
    response.strength = _clamp(_as_float(raw.get("strength"), response.strength), 0.0, 1.0)
    response.urgency = urgency
    response.confidence = _clamp(_as_float(raw.get("confidence"), response.confidence), 0.0, 1.0)
    profile = infer_signal_profile(
        agent_id=agent_id,
        direction=response.direction,
        strength=response.strength,
        confidence=response.confidence,
        evidence=evidence,
        explicit_signal_score=raw.get("signal_score"),
        explicit_source_trust=raw.get("source_trust"),
        explicit_event_type=raw.get("event_type"),
        explicit_horizon=raw.get("horizon"),
    )
    response.signal_score = profile.signal_score
    response.source_trust = profile.source_trust
    response.event_type = profile.event_type
    response.horizon = profile.horizon
    response.risk_level = profile.risk_level
    response.opinion = profile.opinion
    response.reasoning_for_judge_agent = _clean_text(
        raw.get("reasoning_for_judge_agent"),
        default=response.reasoning_for_judge_agent or "",
        limit=420,
    )
    response.limits_acknowledged = _clean_optional_string(raw.get("limits_acknowledged"), limit=240)
    response.references = _sanitize_references(raw.get("references"))
    response.depth_used = _clean_text(raw.get("depth_used"), default=depth, limit=20) or depth
    response.focus_tickers = focus_tickers
    return response


def _default_summary_for_decision(decision: DecisionType) -> str:
    if decision == DecisionType.REBALANCE:
        return "후보 리밸런싱 초안을 추가 검토할 수 있습니다."
    if decision == DecisionType.USER_DECISION_REQUIRED:
        return "자동 판단 범위를 넘어 사용자 확인이 필요합니다."
    if decision == DecisionType.HOLD:
        return "현재 근거 기준으로는 보유 유지 쪽이 합리적입니다."
    return "지금 단계에서는 결정을 보류합니다."


def _default_urgency_for_decision(decision: DecisionType) -> Urgency:
    if decision == DecisionType.REBALANCE:
        return Urgency.SCHEDULED
    if decision == DecisionType.USER_DECISION_REQUIRED:
        return Urgency.WATCH
    return Urgency.DEFER


def _default_notification_level(decision: DecisionType, urgency: Urgency) -> str:
    if decision == DecisionType.USER_DECISION_REQUIRED:
        return "push"
    if urgency in {Urgency.IMMEDIATE, Urgency.WATCH}:
        return "watch"
    if decision == DecisionType.HOLD:
        return "silent"
    return "info"


def sanitize_judge_payload(
    payload: Mapping[str, Any] | None,
    *,
    portfolio: PortfolioSnapshot,
    stage: str,
) -> dict[str, Any]:
    raw = dict(payload) if isinstance(payload, Mapping) else {}
    alias_map = _build_ticker_alias_map(portfolio)
    candidate_plan = {
        ticker: _clamp(_as_float(delta), -1.0, 1.0)
        for ticker, delta in (
            (_canonical_ticker(raw_ticker, alias_map), raw_delta)
            for raw_ticker, raw_delta in _as_mapping(raw.get("candidate_rebalance_plan")).items()
        )
        if ticker
    }

    raw_decision = _clean_text(raw.get("decision"), limit=40).upper()
    if raw_decision in {item.value for item in DecisionType}:
        decision = DecisionType(raw_decision)
    else:
        decision = DecisionType.REBALANCE if candidate_plan else DecisionType.DEFER
    empty_portfolio_no_trade = stage == "final" and not portfolio.holdings and not candidate_plan
    if empty_portfolio_no_trade:
        decision = DecisionType.DEFER

    raw_urgency = _clean_text(raw.get("urgency"), limit=20)
    if raw_urgency in {item.value for item in Urgency}:
        urgency = Urgency(raw_urgency)
    else:
        urgency = _default_urgency_for_decision(decision)
    if empty_portfolio_no_trade:
        urgency = Urgency.DEFER

    summary = _clean_text(
        raw.get("summary"), default=_default_summary_for_decision(decision), limit=320
    )
    reasoning = _clean_text(raw.get("reasoning"), default=summary, limit=420)
    confidence = _clamp(_as_float(raw.get("confidence"), 0.0), 0.0, 1.0)
    needs_trade_evaluation = (
        bool(candidate_plan)
        or decision == DecisionType.REBALANCE
        or _as_bool(raw.get("needs_trade_evaluation"), default=False)
    )
    if empty_portfolio_no_trade:
        needs_trade_evaluation = False

    notification = _as_mapping(raw.get("user_notification"))
    level = _clean_text(notification.get("level"), limit=20).lower()
    if level not in ALLOWED_NOTIFICATION_LEVELS:
        level = _default_notification_level(decision, urgency)
    action_required = _as_bool(
        notification.get("action_required"),
        default=decision == DecisionType.USER_DECISION_REQUIRED,
    )
    if empty_portfolio_no_trade:
        level = "info"
        action_required = False
    user_notification = {
        "level": level,
        "body": summary,
        "action_required": action_required,
        "kind": _clean_text(notification.get("kind"), default="final_decision", limit=40)
        or "final_decision",
        "estimated_followup": _datetime_string(notification.get("estimated_followup")),
        "sent_at": _datetime_string(notification.get("sent_at")),
    }

    options = []
    for item in _as_list(raw.get("options")):
        cleaned = _clean_text(item, limit=80)
        if cleaned and cleaned not in options:
            options.append(cleaned)
    if empty_portfolio_no_trade:
        options = []
    if decision == DecisionType.USER_DECISION_REQUIRED and not options:
        options = ["권고안 승인", "유지", "직접 판단"]

    sanitized = {
        "decision": decision.value,
        "summary": summary,
        "confidence": confidence,
        "urgency": urgency.value,
        "reasoning": reasoning,
        "candidate_rebalance_plan": candidate_plan,
        "needs_trade_evaluation": needs_trade_evaluation,
        "follow_up_at": _datetime_string(raw.get("follow_up_at")),
        "feedback_checkpoint": None
        if empty_portfolio_no_trade
        else _datetime_string(raw.get("feedback_checkpoint")),
        "user_notification": user_notification,
        "options": options,
        "auto_safeguards": _sanitize_jsonish(raw.get("auto_safeguards"), max_depth=3) or {},
    }
    if stage == "planning":
        sanitized["user_notification"]["kind"] = "planning"
    return sanitized
