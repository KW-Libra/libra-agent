from __future__ import annotations

from typing import Any

from libra_agent.libra_models import AgentResponse, DecisionType, JudgeDecision, Urgency, UserNotification


DOMAIN_AGENT_IDS = {"risk", "tax", "compliance", "macro", "sentiment", "execution", "esg"}


def _vote_from_response(response: AgentResponse) -> str:
    vote = str(response.evidence.get("vote") or "").strip().lower()
    if vote in {"approve", "reject", "abstain"}:
        return vote
    opinion = response.opinion.strip().upper()
    if opinion == "POSITIVE" or response.direction > 0.05:
        return "approve"
    if opinion == "NEGATIVE" or response.direction < -0.05:
        return "reject"
    return "abstain"


def compute_domain_consensus(responses: list[AgentResponse]) -> dict[str, Any]:
    """Compute 7-domain-agent vote counts and confidence-weighted score."""
    n_approve = 0
    n_reject = 0
    n_abstain = 0
    weighted_sum = 0.0
    weight_sum = 0.0
    rejecting_agents: list[str] = []
    compliance_veto = False

    for response in responses:
        if response.agent_id not in DOMAIN_AGENT_IDS:
            continue
        vote = _vote_from_response(response)
        if vote == "approve":
            n_approve += 1
        elif vote == "reject":
            n_reject += 1
            rejecting_agents.append(response.agent_id)
            if response.agent_id == "compliance":
                compliance_veto = True
        else:
            n_abstain += 1
        weight = max(0.05, float(response.confidence or 0.0))
        weighted_sum += float(response.direction or 0.0) * weight
        weight_sum += weight

    score = weighted_sum / weight_sum if weight_sum else 0.0
    return {
        "n_approve": n_approve,
        "n_reject": n_reject,
        "n_abstain": n_abstain,
        "score": round(max(-1.0, min(1.0, score)), 4),
        "compliance_veto": compliance_veto,
        "rejecting_agents": rejecting_agents,
    }


def apply_compliance_veto(
    judge_decision: JudgeDecision,
    domain_responses: list[AgentResponse],
) -> JudgeDecision:
    """Force USER_DECISION_REQUIRED when Compliance rejects a proposed action."""
    compliance = next((item for item in domain_responses if item.agent_id == "compliance"), None)
    if compliance is None:
        return judge_decision
    if _vote_from_response(compliance) != "reject":
        return judge_decision

    reason = compliance.reasoning_for_judge_agent.strip() or "Compliance Agent가 사용자 정책 위반 가능성을 감지했습니다."
    prefix = f"Compliance 거부권: {reason}"
    judge_decision.decision = DecisionType.USER_DECISION_REQUIRED
    judge_decision.urgency = Urgency.WATCH
    judge_decision.reasoning = f"{prefix}\n\n{judge_decision.reasoning}".strip()
    judge_decision.summary = "Compliance Agent가 사용자 정책 위반 가능성을 감지해 자동 리밸런싱 결정을 사용자 확인으로 전환했습니다."
    judge_decision.options = ["사용자 정책을 확인한 뒤 진행", "해당 주문 후보 제외", "이번 리밸런싱 보류"]
    judge_decision.user_notification = UserNotification(
        level="push",
        body=judge_decision.summary,
        action_required=True,
        kind="compliance_veto",
        estimated_followup=judge_decision.follow_up_at,
    )
    return judge_decision
