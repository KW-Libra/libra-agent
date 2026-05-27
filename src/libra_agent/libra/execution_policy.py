from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum

from libra_agent.libra_models import PortfolioSnapshot

from .schemas.decision import Trade


class ExecutionMode(StrEnum):
    DELTA_ONLY = "DELTA_ONLY"
    POLICY_TARGET = "POLICY_TARGET"
    PARTIAL_POLICY_TARGET = "PARTIAL_POLICY_TARGET"
    RISK_TRIM_AND_REDISTRIBUTE = "RISK_TRIM_AND_REDISTRIBUTE"
    CASH_RAISE = "CASH_RAISE"
    USER_HANDOFF = "USER_HANDOFF"


class ExecutionReasonCode(StrEnum):
    NO_SIGNAL = "NO_SIGNAL"
    SIGNAL_TOO_WEAK = "SIGNAL_TOO_WEAK"
    CONFLICT_UNRESOLVED = "CONFLICT_UNRESOLVED"
    ONE_SIDED_PLAN_REPAIRED = "ONE_SIDED_PLAN_REPAIRED"
    ONE_SIDED_PLAN_UNREPAIRABLE = "ONE_SIDED_PLAN_UNREPAIRABLE"
    MIN_TRADE_SIZE_NOT_MET = "MIN_TRADE_SIZE_NOT_MET"
    TURNOVER_CAP_EXCEEDED = "TURNOVER_CAP_EXCEEDED"
    POLICY_TARGET_DELTA_TOO_SMALL = "POLICY_TARGET_DELTA_TOO_SMALL"
    PENDING_USER_DECISION_SUPPRESSED = "PENDING_USER_DECISION_SUPPRESSED"


@dataclass(slots=True)
class ExecutionPlan:
    mode: ExecutionMode
    target_weights: dict[str, float]
    trade_deltas: dict[str, float]
    trades: list[Trade]
    validation_status: str
    reason_codes: list[ExecutionReasonCode] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode.value,
            "target_weights": dict(self.target_weights),
            "trade_deltas": dict(self.trade_deltas),
            "trades": [trade.to_dict() for trade in self.trades],
            "validation_status": self.validation_status,
            "reason_codes": [code.value for code in self.reason_codes],
        }


@dataclass(slots=True)
class IssueState:
    issue_key: str
    first_seen: str
    last_seen: str
    count: int = 1
    status: str = "PENDING_USER_DECISION"

    def to_dict(self) -> dict[str, object]:
        return {
            "issue_key": self.issue_key,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "count": self.count,
            "status": self.status,
        }


class IssueStateManager:
    def __init__(self, *, cooldown_observations: int = 20) -> None:
        self.cooldown_observations = max(1, int(cooldown_observations))
        self._issues: dict[str, IssueState] = {}

    def observe(
        self,
        *,
        branch: str,
        candidate_plan: Mapping[str, float] | None,
        seen_at: str,
    ) -> tuple[str, IssueState]:
        key = issue_key(branch=branch, candidate_plan=candidate_plan)
        current = self._issues.get(key)
        if current is None:
            state = IssueState(issue_key=key, first_seen=seen_at, last_seen=seen_at)
            self._issues[key] = state
            return "NEW_ISSUE", state
        current.count += 1
        current.last_seen = seen_at
        if current.count <= self.cooldown_observations + 1:
            current.status = "SUPPRESSED_BY_COOLDOWN"
            return "SUPPRESSED_BY_COOLDOWN", current
        current.status = "PENDING_USER_DECISION"
        return "PENDING_USER_DECISION", current


def issue_key(*, branch: str, candidate_plan: Mapping[str, float] | None) -> str:
    deltas = _normalize_delta_map(candidate_plan)
    if deltas:
        parts = [
            f"{ticker}:{'reduce' if delta < 0 else 'increase'}"
            for ticker, delta in sorted(deltas.items())
        ]
        return f"{branch}|plan={','.join(parts)}"
    return f"{branch}|plan=none"


def normalize_ticker(value: object) -> str:
    return "".join(char for char in str(value).upper() if char.isalnum())


def current_weights_from_portfolio(portfolio: PortfolioSnapshot) -> dict[str, float]:
    return {
        normalize_ticker(holding.ticker): float(holding.weight)
        for holding in portfolio.holdings
        if normalize_ticker(holding.ticker)
    }


def normalize_weight_map(weights: Mapping[str, float] | None) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for ticker, weight in dict(weights or {}).items():
        key = normalize_ticker(ticker)
        if not key:
            continue
        try:
            value = float(weight)
        except (TypeError, ValueError):
            continue
        if value <= 0:
            continue
        normalized[key] = value
    total = sum(normalized.values())
    if total <= 0:
        return {}
    return {ticker: value / total for ticker, value in normalized.items()}


def build_execution_plan(
    *,
    portfolio: PortfolioSnapshot,
    candidate_plan: Mapping[str, float] | None,
    target_weights: Mapping[str, float] | None,
    mode: ExecutionMode | str,
    participation_rate: float = 1.0,
    max_abs_delta_pct: float | None = None,
    min_abs_delta_pct: float = 0.1,
) -> ExecutionPlan:
    execution_mode = ExecutionMode(str(mode).upper())
    current_weights = current_weights_from_portfolio(portfolio)
    policy_target = normalize_weight_map(target_weights)
    candidate_deltas = _normalize_delta_map(candidate_plan)
    reason_codes: list[ExecutionReasonCode] = []

    if execution_mode == ExecutionMode.DELTA_ONLY:
        trade_deltas, repair_codes = cash_neutralize_deltas(
            candidate_deltas,
            current_weights=current_weights,
            target_weights=policy_target,
            max_abs_delta_pct=max_abs_delta_pct,
            min_abs_delta_pct=min_abs_delta_pct,
        )
        reason_codes.extend(repair_codes)
    elif execution_mode == ExecutionMode.RISK_TRIM_AND_REDISTRIBUTE:
        sells_only = {ticker: delta for ticker, delta in candidate_deltas.items() if delta < 0}
        trade_deltas, repair_codes = cash_neutralize_deltas(
            sells_only,
            current_weights=current_weights,
            target_weights=policy_target,
            max_abs_delta_pct=max_abs_delta_pct,
            min_abs_delta_pct=min_abs_delta_pct,
        )
        reason_codes.extend(repair_codes)
    elif execution_mode in {ExecutionMode.POLICY_TARGET, ExecutionMode.PARTIAL_POLICY_TARGET}:
        if not policy_target:
            return ExecutionPlan(
                mode=execution_mode,
                target_weights={},
                trade_deltas={},
                trades=[],
                validation_status="INVALID",
                reason_codes=[ExecutionReasonCode.CONFLICT_UNRESOLVED],
            )
        rate = _clamp(float(participation_rate), 0.0, 1.0)
        trade_deltas = _target_diff(
            current_weights=current_weights,
            target_weights=policy_target,
            participation_rate=rate,
            max_abs_delta_pct=max_abs_delta_pct,
        )
        trade_deltas, repair_codes = cash_neutralize_deltas(
            trade_deltas,
            current_weights=current_weights,
            target_weights=policy_target,
            max_abs_delta_pct=max_abs_delta_pct,
            min_abs_delta_pct=min_abs_delta_pct,
        )
        reason_codes.extend(repair_codes)
    else:
        return ExecutionPlan(
            mode=execution_mode,
            target_weights=policy_target,
            trade_deltas={},
            trades=[],
            validation_status="INVALID",
            reason_codes=[ExecutionReasonCode.CONFLICT_UNRESOLVED],
        )

    trades = _deltas_to_trades(
        trade_deltas,
        min_abs_delta_pct=min_abs_delta_pct,
        rationale=_rationale_for_mode(execution_mode, reason_codes),
    )
    if not trades:
        if not reason_codes:
            reason_codes.append(ExecutionReasonCode.POLICY_TARGET_DELTA_TOO_SMALL)
        return ExecutionPlan(
            mode=execution_mode,
            target_weights=policy_target,
            trade_deltas={},
            trades=[],
            validation_status="INVALID",
            reason_codes=reason_codes,
        )
    return ExecutionPlan(
        mode=execution_mode,
        target_weights=policy_target,
        trade_deltas=trade_deltas,
        trades=trades,
        validation_status="VALID",
        reason_codes=reason_codes or [ExecutionReasonCode.ONE_SIDED_PLAN_REPAIRED]
        if _is_repaired(reason_codes)
        else reason_codes,
    )


def cash_neutralize_deltas(
    deltas: Mapping[str, float],
    *,
    current_weights: Mapping[str, float],
    target_weights: Mapping[str, float],
    max_abs_delta_pct: float | None,
    min_abs_delta_pct: float,
) -> tuple[dict[str, float], list[ExecutionReasonCode]]:
    raw = _cap_and_filter_deltas(
        deltas,
        max_abs_delta_pct=max_abs_delta_pct,
        min_abs_delta_pct=min_abs_delta_pct,
    )
    if not raw:
        return {}, [ExecutionReasonCode.MIN_TRADE_SIZE_NOT_MET]

    positive_total = sum(delta for delta in raw.values() if delta > 0)
    negative_total = -sum(delta for delta in raw.values() if delta < 0)
    if positive_total > 0 and negative_total > 0:
        executable_total = min(positive_total, negative_total)
        return (
            {
                ticker: delta
                * (executable_total / positive_total if delta > 0 else executable_total / negative_total)
                for ticker, delta in raw.items()
            },
            [],
        )
    if negative_total > 0 and positive_total <= 0:
        repaired = repair_one_sided_sell(
            raw,
            current_weights=current_weights,
            target_weights=target_weights,
            min_abs_delta_pct=min_abs_delta_pct,
        )
        if repaired:
            return repaired, [ExecutionReasonCode.ONE_SIDED_PLAN_REPAIRED]
        return {}, [ExecutionReasonCode.ONE_SIDED_PLAN_UNREPAIRABLE]
    return {}, [ExecutionReasonCode.ONE_SIDED_PLAN_UNREPAIRABLE]


def repair_one_sided_sell(
    sell_deltas: Mapping[str, float],
    *,
    current_weights: Mapping[str, float],
    target_weights: Mapping[str, float],
    min_abs_delta_pct: float = 0.1,
    blocked_buys: set[str] | None = None,
) -> dict[str, float]:
    blocked = {normalize_ticker(ticker) for ticker in (blocked_buys or set())}
    sells = {ticker: float(delta) for ticker, delta in sell_deltas.items() if float(delta) < 0}
    proceeds = -sum(sells.values())
    if proceeds * 100.0 < min_abs_delta_pct:
        return {}

    sell_tickers = set(sells)
    eligible_gaps: dict[str, float] = {}
    for ticker in sorted(set(target_weights) | set(current_weights)):
        if ticker in sell_tickers or ticker in blocked:
            continue
        gap = float(target_weights.get(ticker, 0.0)) - float(current_weights.get(ticker, 0.0))
        if gap * 100.0 >= min_abs_delta_pct:
            eligible_gaps[ticker] = gap
    total_gap = sum(eligible_gaps.values())
    if total_gap <= 0:
        return {}

    repaired: dict[str, float] = dict(sells)
    for ticker, gap in eligible_gaps.items():
        repaired[ticker] = proceeds * gap / total_gap
    return repaired


def _target_diff(
    *,
    current_weights: Mapping[str, float],
    target_weights: Mapping[str, float],
    participation_rate: float,
    max_abs_delta_pct: float | None,
) -> dict[str, float]:
    deltas: dict[str, float] = {}
    for ticker in sorted(set(current_weights) | set(target_weights)):
        delta = (float(target_weights.get(ticker, 0.0)) - float(current_weights.get(ticker, 0.0))) * participation_rate
        if max_abs_delta_pct is not None:
            cap = abs(float(max_abs_delta_pct)) / 100.0
            delta = max(-cap, min(cap, delta))
        deltas[ticker] = delta
    return deltas


def _normalize_delta_map(deltas: Mapping[str, float] | None) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for ticker, delta in dict(deltas or {}).items():
        key = normalize_ticker(ticker)
        if not key:
            continue
        try:
            value = float(delta)
        except (TypeError, ValueError):
            continue
        if value != 0:
            normalized[key] = value
    return normalized


def _cap_and_filter_deltas(
    deltas: Mapping[str, float],
    *,
    max_abs_delta_pct: float | None,
    min_abs_delta_pct: float,
) -> dict[str, float]:
    raw: dict[str, float] = {}
    for ticker, delta in deltas.items():
        value = float(delta)
        if max_abs_delta_pct is not None:
            cap = abs(float(max_abs_delta_pct)) / 100.0
            value = max(-cap, min(cap, value))
        if abs(value) * 100.0 >= min_abs_delta_pct:
            raw[ticker] = value
    return raw


def _deltas_to_trades(
    deltas: Mapping[str, float],
    *,
    min_abs_delta_pct: float,
    rationale: str,
) -> list[Trade]:
    trades: list[Trade] = []
    for ticker in sorted(deltas):
        delta_pct = round(float(deltas[ticker]) * 100.0, 1)
        if abs(delta_pct) < min_abs_delta_pct:
            continue
        trades.append(Trade(subject=ticker, delta_pct=delta_pct, rationale=rationale))
    return trades


def _rationale_for_mode(
    mode: ExecutionMode,
    reason_codes: list[ExecutionReasonCode],
) -> str:
    if ExecutionReasonCode.ONE_SIDED_PLAN_REPAIRED in reason_codes:
        return "LLM 리밸런싱 신호의 한쪽 거래를 policy target 기준 현금중립 주문으로 보정"
    if mode == ExecutionMode.POLICY_TARGET:
        return "LLM 리밸런싱 신호를 사전 정의 policy target으로 번역"
    if mode == ExecutionMode.PARTIAL_POLICY_TARGET:
        return "LLM 리밸런싱 신호를 부분 policy target 이동으로 번역"
    if mode == ExecutionMode.RISK_TRIM_AND_REDISTRIBUTE:
        return "위험/집중도 축소 신호를 underweight 종목 재배분 주문으로 번역"
    return "candidate delta 기반 현금중립 주문"


def _is_repaired(reason_codes: list[ExecutionReasonCode]) -> bool:
    return ExecutionReasonCode.ONE_SIDED_PLAN_REPAIRED in reason_codes


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))
