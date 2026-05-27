from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping

from ..schemas.agent import Direction, Vote
from ..schemas.compliance import ComplianceCheck
from ..schemas.decision import (
    ConsensusBranch,
    ConsensusScore,
    DecisionBranch,
    DecisionType,
    FinalDecision,
    Trade,
    UserOption,
)

WEAK_CONSERVATIVE_COEF = 0.5
MAX_CONFLICT_RESOLUTION_DELTA_PCT = 10.0
MIN_EXECUTABLE_DELTA_PCT = 0.1


def determine_branch(
    consensus_per_subject: dict[str, ConsensusScore],
    compliance_after: ComplianceCheck,
) -> tuple[DecisionType, DecisionBranch]:
    if not compliance_after.can_proceed:
        return DecisionType.USER_DECISION_REQUIRED, DecisionBranch.COMPLIANCE_VETO
    if any(score.branch == ConsensusBranch.CONFLICT for score in consensus_per_subject.values()):
        return DecisionType.USER_DECISION_REQUIRED, DecisionBranch.STRONG_CONFLICT
    if any(
        score.branch == ConsensusBranch.WEAK_CONSENSUS for score in consensus_per_subject.values()
    ):
        return DecisionType.REBALANCE, DecisionBranch.WEAK_CONSENSUS_CONSERVATIVE
    if consensus_per_subject and all(
        score.branch == ConsensusBranch.STRONG_HOLD for score in consensus_per_subject.values()
    ):
        return DecisionType.HOLD, DecisionBranch.HOLD
    if not consensus_per_subject:
        return DecisionType.DEFER, DecisionBranch.STRONG_CONFLICT
    return DecisionType.REBALANCE, DecisionBranch.CONSENSUS


def compute_trade_consensus(
    subject: str, votes: list[Vote], *, conservative: bool = False
) -> Trade:
    eligible = [
        vote for vote in votes if not vote.informational and vote.direction != Direction.HOLD
    ]
    if not eligible:
        return Trade(
            subject=subject, delta_pct=0.0, rationale="실행 가능한 방향성 투표가 없습니다."
        )
    weighted = sum(vote.magnitude_pct * vote.confidence for vote in eligible)
    denominator = sum(vote.confidence for vote in eligible)
    delta = weighted / denominator if denominator else 0.0
    if conservative:
        delta *= WEAK_CONSERVATIVE_COEF
    return Trade(
        subject=subject,
        delta_pct=round(delta, 1),
        rationale="위원회 발화의 confidence 가중 평균으로 산정",
    )


def compute_tentative_trades(
    consensus_per_subject: dict[str, ConsensusScore],
    votes_by_subject: dict[str, list[Vote]],
) -> list[Trade]:
    trades: list[Trade] = []
    for subject, score in consensus_per_subject.items():
        if subject == "PORTFOLIO":
            continue
        if score.branch == ConsensusBranch.STRONG_CONSENSUS:
            trade = compute_trade_consensus(subject, votes_by_subject.get(subject, []))
        elif score.branch == ConsensusBranch.WEAK_CONSENSUS:
            trade = compute_trade_consensus(
                subject, votes_by_subject.get(subject, []), conservative=True
            )
        else:
            continue
        if abs(trade.delta_pct) >= 0.1:
            trades.append(trade)
    return trades


def candidate_plan_to_trades(
    candidate_plan: Mapping[str, float] | None,
    *,
    max_abs_delta_pct: float = MAX_CONFLICT_RESOLUTION_DELTA_PCT,
    min_abs_delta_pct: float = MIN_EXECUTABLE_DELTA_PCT,
) -> list[Trade]:
    """Convert a service candidate plan into executable percentage-point trades.

    ``candidate_rebalance_plan`` uses fractional weights, while v1 governance
    trades use percentage points. Conflict resolution keeps the plan cash-neutral
    and caps each leg so a single oversized drift does not become a handoff-only
    outcome.
    """
    raw: dict[str, float] = {}
    for ticker, delta in dict(candidate_plan or {}).items():
        subject = str(ticker).strip()
        if not subject:
            continue
        try:
            value = float(delta)
        except (TypeError, ValueError):
            continue
        if abs(value) * 100.0 < min_abs_delta_pct:
            continue
        capped = max(-max_abs_delta_pct / 100.0, min(max_abs_delta_pct / 100.0, value))
        raw[subject] = capped
    if not raw:
        return []

    positive_total = sum(delta for delta in raw.values() if delta > 0)
    negative_total = -sum(delta for delta in raw.values() if delta < 0)
    if positive_total <= 0 or negative_total <= 0:
        return []

    executable_total = min(positive_total, negative_total)
    pos_scale = executable_total / positive_total
    neg_scale = executable_total / negative_total

    trades: list[Trade] = []
    for subject in sorted(raw):
        delta = raw[subject]
        scaled = delta * (pos_scale if delta > 0 else neg_scale)
        delta_pct = round(scaled * 100.0, 1)
        if abs(delta_pct) < min_abs_delta_pct:
            continue
        trades.append(
            Trade(
                subject=subject,
                delta_pct=delta_pct,
                rationale=(
                    "포트폴리오 레벨 충돌을 direct-indexing drift 초안으로 해소한 "
                    "10%p cap 현금중립 부분 리밸런싱"
                ),
            )
        )
    return trades


def cash_neutral_trades(
    trades: list[Trade],
    *,
    min_abs_delta_pct: float = MIN_EXECUTABLE_DELTA_PCT,
) -> list[Trade]:
    executable = [trade for trade in trades if abs(float(trade.delta_pct)) >= min_abs_delta_pct]
    if not executable:
        return []
    positive_total = sum(float(trade.delta_pct) for trade in executable if trade.delta_pct > 0)
    negative_total = -sum(float(trade.delta_pct) for trade in executable if trade.delta_pct < 0)
    if positive_total <= 0 or negative_total <= 0:
        return []

    executable_total = min(positive_total, negative_total)
    pos_scale = executable_total / positive_total
    neg_scale = executable_total / negative_total
    balanced: list[Trade] = []
    for trade in executable:
        scale = pos_scale if trade.delta_pct > 0 else neg_scale
        delta_pct = round(float(trade.delta_pct) * scale, 1)
        if abs(delta_pct) < min_abs_delta_pct:
            continue
        balanced.append(
            Trade(subject=trade.subject, delta_pct=delta_pct, rationale=trade.rationale)
        )
    return balanced


def can_auto_resolve_conflict(
    consensus_per_subject: dict[str, ConsensusScore],
    candidate_trades: list[Trade],
    *,
    allow_ticker_conflicts: bool = False,
) -> bool:
    if not candidate_trades:
        return False
    conflict_subjects = {
        subject
        for subject, score in consensus_per_subject.items()
        if score.branch == ConsensusBranch.CONFLICT
    }
    if conflict_subjects == {"PORTFOLIO"}:
        return True
    if not allow_ticker_conflicts or not conflict_subjects:
        return False
    trade_subjects = {trade.subject for trade in candidate_trades}
    return conflict_subjects <= trade_subjects | {"PORTFOLIO"}


def render_rule_based_final_decision(
    *,
    consensus_per_subject: dict[str, ConsensusScore],
    votes: list[Vote],
    compliance_after: ComplianceCheck,
    candidate_trades: list[Trade] | None = None,
    allow_ticker_conflict_resolution: bool = False,
) -> FinalDecision:
    """Deterministic final branch renderer used before Final Judge LLM wiring.

    The v1 design reserves Korean report wording and option copy for the Final
    Judge LLM. This function keeps the branch semantics executable in tests and
    benchmark scripts while that prompt path is wired in.
    """
    decision, branch = determine_branch(consensus_per_subject, compliance_after)
    grouped: dict[str, list[Vote]] = defaultdict(list)
    for vote in votes:
        grouped[vote.subject].append(vote)
    trades = cash_neutral_trades(
        list(candidate_trades or compute_tentative_trades(consensus_per_subject, grouped))
    )
    auto_resolved_conflict = can_auto_resolve_conflict(
        consensus_per_subject,
        trades,
        allow_ticker_conflicts=allow_ticker_conflict_resolution,
    )
    if (
        decision == DecisionType.USER_DECISION_REQUIRED
        and branch == DecisionBranch.STRONG_CONFLICT
        and auto_resolved_conflict
        and compliance_after.can_proceed
    ):
        decision = DecisionType.REBALANCE
        branch = DecisionBranch.CONFLICT_RESOLUTION
    if decision == DecisionType.REBALANCE and not trades:
        decision = DecisionType.DEFER
        branch = DecisionBranch.NO_EXECUTABLE_TRADE

    if branch == DecisionBranch.COMPLIANCE_VETO:
        question = "사용자 정책 또는 IPS 위반이 감지되었습니다. 다음 중 선택해주세요."
        options = [
            UserOption(
                "거래 취소", ["Compliance"], "위반 거래를 실행하지 않고 현재 포트폴리오 유지"
            ),
            UserOption(
                "IPS 한도 일시 완화",
                ["Final Judge"],
                "이번 거래 한정으로 사용자 명시 동의 후 재검토",
            ),
            UserOption(
                "다른 자산으로 대체", ["Risk", "Profit"], "위반하지 않는 대체 리밸런싱 경로 탐색"
            ),
        ]
        return FinalDecision(
            decision=decision,
            branch=branch,
            trades=[],
            compliance_check=compliance_after,
            reasoning="Compliance Rule Engine의 BLOCKING 위반으로 자동 실행을 중단하고 사용자 선택을 요청합니다.",
            user_question=question,
            user_options=options,
        )
    if branch == DecisionBranch.STRONG_CONFLICT:
        return FinalDecision(
            decision=decision,
            branch=branch,
            trades=[],
            compliance_check=compliance_after,
            reasoning="위원회 의견 충돌이 커 자동 결정을 강제하지 않습니다.",
            user_question="위원회 의견이 갈렸습니다. 어떤 방향을 선택하시겠습니까?",
            user_options=[
                UserOption("위험축소", ["Risk", "Sentiment"], "하방 위험을 줄이는 보수적 선택"),
                UserOption("현상유지", ["Macro", "Tax"], "추가 정보가 나올 때까지 현재 비중 유지"),
                UserOption("적극행동", ["Profit"], "기회 신호를 반영해 제한적으로 비중 조정"),
            ],
        )
    if decision == DecisionType.HOLD:
        trades = []
    if branch == DecisionBranch.CONFLICT_RESOLUTION:
        if allow_ticker_conflict_resolution:
            reasoning = (
                "의견 충돌은 있었지만, 리밸런싱 신호를 사전 정의된 포트폴리오 "
                "execution policy로 번역한 결과 현금중립 실행 계획이 만들어져 자동 해소합니다."
            )
        else:
            reasoning = (
                "포트폴리오 레벨 의견 충돌은 있었지만, direct-indexing drift 초안이 "
                "10%p cap과 현금중립 조건을 만족해 부분 리밸런싱으로 자동 해소합니다."
            )
    elif decision == DecisionType.DEFER and not trades:
        reasoning = "리밸런싱 합의는 있었지만 실행 가능한 종목별 거래가 없어 자동 체결을 보류합니다."
    else:
        reasoning = f"{branch.value} 분기에 따라 결정했습니다."
    return FinalDecision(
        decision=decision,
        branch=branch,
        trades=trades,
        compliance_check=compliance_after,
        reasoning=reasoning,
    )
