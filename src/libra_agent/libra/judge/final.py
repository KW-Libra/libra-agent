from __future__ import annotations

from collections import defaultdict

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


def render_rule_based_final_decision(
    *,
    consensus_per_subject: dict[str, ConsensusScore],
    votes: list[Vote],
    compliance_after: ComplianceCheck,
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
    trades = compute_tentative_trades(consensus_per_subject, grouped)
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
    return FinalDecision(
        decision=decision,
        branch=branch,
        trades=trades,
        compliance_check=compliance_after,
        reasoning=(
            "리밸런싱 합의는 있었지만 실행 가능한 종목별 거래가 없어 자동 체결을 보류합니다."
            if decision == DecisionType.DEFER and not trades
            else f"{branch.value} 분기에 따라 결정했습니다."
        ),
    )
