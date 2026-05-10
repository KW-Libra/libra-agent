from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from libra_agent.libra_models import AgentResponse, PortfolioSnapshot

from .compliance import build_compliance_context_from_portfolio, default_compliance_engine
from .judge.final import compute_tentative_trades, render_rule_based_final_decision
from .mediator import consensus_by_subject, select_targets
from .schemas.agent import AgentOpinion, Direction, Vote
from .schemas.compliance import ComplianceCheck, MarketSnapshot
from .schemas.decision import ConsensusScore, FinalDecision, Trade
from .schemas.ips import IPSConfig, KYCProfile

CORE_COMMITTEE_AGENTS = ("disclosure", "news", "report", "profit", "cost")
DOMAIN_COMMITTEE_AGENTS = ("risk", "tax", "macro", "sentiment", "execution", "esg")
INFO_AGENTS = {"disclosure", "cost", "execution"}


@dataclass(slots=True)
class CommitteeRunResult:
    round1_opinions: list[AgentOpinion]
    round2_opinions: list[AgentOpinion]
    consensus_per_subject: dict[str, ConsensusScore]
    targets_to_recall: list[str]
    compliance_before: ComplianceCheck
    compliance_after: ComplianceCheck
    final_decision: FinalDecision
    tentative_trades: list[Trade] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "round1_opinions": [opinion.to_dict() for opinion in self.round1_opinions],
            "round2_opinions": [opinion.to_dict() for opinion in self.round2_opinions],
            "consensus_per_subject": {
                subject: score.to_dict() for subject, score in self.consensus_per_subject.items()
            },
            "targets_to_recall": list(self.targets_to_recall),
            "compliance_before": self.compliance_before.to_dict(),
            "compliance_after": self.compliance_after.to_dict(),
            "tentative_trades": [trade.to_dict() for trade in self.tentative_trades],
            "final_decision": self.final_decision.to_dict(),
        }


def agent_response_to_opinion(response: AgentResponse, *, round_number: int = 1) -> AgentOpinion:
    agent = response.agent_id
    subjects = list(response.focus_tickers) or _subjects_from_evidence(response) or ["PORTFOLIO"]
    informational = _is_informational(response)
    direction = _direction_from_response(response)
    magnitude = _magnitude_from_response(response, direction=direction)
    votes = [
        Vote(
            subject=subject,
            direction=direction,
            magnitude_pct=magnitude,
            confidence=response.confidence,
            concerns=_concerns_from_response(response),
            informational=informational,
        )
        for subject in subjects
    ]
    if direction == Direction.HOLD and response.confidence <= 0.05 and agent in {"news", "report", "sentiment"}:
        votes = []
    return AgentOpinion(
        agent=_display_agent_name(agent),
        round=round_number,  # type: ignore[arg-type]
        votes=votes,
        silence_reason=None if votes else response.limits_acknowledged or "판단 가능한 신호가 없습니다.",
        reasoning=response.reasoning_for_judge_agent or response.query_understood,
        evidence_refs=_evidence_refs_from_response(response),
        metadata={
            "source_agent_id": agent,
            "opinion": response.opinion,
            "signal_score": response.signal_score,
            "risk_level": response.risk_level,
        },
    )


def responses_to_opinions(responses: list[AgentResponse], *, round_number: int = 1) -> list[AgentOpinion]:
    return [
        agent_response_to_opinion(response, round_number=round_number)
        for response in responses
        if response.agent_id != "compliance"
    ]


class CommitteeRuntime:
    """v1 governance runtime over AgentOpinion schema.

    Existing LIBRA agents can keep returning AgentResponse while this runtime
    evaluates the new committee/governance semantics. Compliance is handled by
    code, not by the legacy ComplianceAgent.
    """

    def __init__(self, *, compliance_engine=None) -> None:
        self.compliance_engine = compliance_engine or default_compliance_engine()

    def run_from_agent_responses(
        self,
        *,
        portfolio: PortfolioSnapshot,
        responses: list[AgentResponse],
        ips: IPSConfig | None = None,
        kyc: KYCProfile | None = None,
        market_data: MarketSnapshot | None = None,
        round2_opinions: list[AgentOpinion] | None = None,
    ) -> CommitteeRunResult:
        round1 = responses_to_opinions(responses, round_number=1)
        round2 = list(round2_opinions or [])
        all_opinions = [*round1, *round2]
        consensus = consensus_by_subject(all_opinions)
        targets = select_targets(consensus, round1)

        before_ctx = build_compliance_context_from_portfolio(
            portfolio,
            proposed_trades=[],
            ips=ips,
            kyc=kyc,
            market_data=market_data,
        )
        compliance_before = self.compliance_engine.check(before_ctx, "BEFORE")

        votes_by_subject: dict[str, list[Vote]] = defaultdict(list)
        for opinion in all_opinions:
            for vote in opinion.votes:
                votes_by_subject[vote.subject].append(vote)
        tentative_trades = compute_tentative_trades(consensus, votes_by_subject)

        after_ctx = build_compliance_context_from_portfolio(
            portfolio,
            proposed_trades=tentative_trades,
            ips=ips,
            kyc=kyc,
            market_data=market_data,
        )
        compliance_after = self.compliance_engine.check(after_ctx, "AFTER")
        all_votes = [vote for opinion in all_opinions for vote in opinion.votes]
        final_decision = render_rule_based_final_decision(
            consensus_per_subject=consensus,
            votes=all_votes,
            compliance_after=compliance_after,
        )
        return CommitteeRunResult(
            round1_opinions=round1,
            round2_opinions=round2,
            consensus_per_subject=consensus,
            targets_to_recall=targets,
            compliance_before=compliance_before,
            compliance_after=compliance_after,
            tentative_trades=tentative_trades,
            final_decision=final_decision,
        )


def run_agent_callables_parallel(
    agent_calls: dict[str, Callable[[], AgentResponse]],
    *,
    max_workers: int = 11,
) -> list[AgentResponse]:
    """Run Round 1 committee calls concurrently.

    Exceptions are intentionally not swallowed. v1 design avoids deterministic
    fallback: if an agent LLM/tool call fails, the benchmark/runtime should show
    the failure rather than inventing an opinion.
    """
    if not agent_calls:
        return []
    worker_count = max(1, min(max_workers, len(agent_calls)))
    results: dict[str, AgentResponse] = {}
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        future_to_agent = {pool.submit(call): agent_id for agent_id, call in agent_calls.items()}
        for future in as_completed(future_to_agent):
            agent_id = future_to_agent[future]
            results[agent_id] = future.result()
    return [results[agent_id] for agent_id in agent_calls.keys()]


def _display_agent_name(agent_id: str) -> str:
    return {
        "disclosure": "Disclosure",
        "news": "News",
        "report": "Report",
        "profit": "Profit",
        "cost": "Cost",
        "risk": "Risk",
        "tax": "Tax",
        "macro": "Macro",
        "sentiment": "Sentiment",
        "execution": "Execution",
        "esg": "ESG",
    }.get(agent_id, agent_id.title())


def _is_informational(response: AgentResponse) -> bool:
    if response.agent_id in INFO_AGENTS:
        return True
    if response.agent_id == "tax" and abs(response.direction) < 0.5:
        return True
    return False


def _direction_from_response(response: AgentResponse) -> Direction:
    if response.direction > 0.1:
        return Direction.INCREASE
    if response.direction < -0.1:
        return Direction.DECREASE
    return Direction.HOLD


def _magnitude_from_response(response: AgentResponse, *, direction: Direction) -> float:
    if direction == Direction.HOLD:
        return 0.0
    sign = 1.0 if direction == Direction.INCREASE else -1.0
    scale = max(abs(response.signal_score), response.strength, 0.1)
    return round(sign * min(10.0, scale * 10.0), 1)


def _concerns_from_response(response: AgentResponse) -> list[str]:
    concerns = [
        item
        for item in (response.event_type, response.risk_level, response.opinion)
        if item and item != "NEUTRAL"
    ]
    if response.agent_id == "cost":
        concerns.append("거래비용")
    if response.agent_id == "execution":
        concerns.append("체결가능성")
    return concerns[:5]


def _evidence_refs_from_response(response: AgentResponse) -> list[str]:
    refs = [reference.url or reference.title for reference in response.references if reference.url or reference.title]
    for key in ("source_documents", "documents", "events", "items"):
        value = response.evidence.get(key)
        if isinstance(value, list) and value:
            refs.append(str(key))
    return refs[:6]


def _subjects_from_evidence(response: AgentResponse) -> list[str]:
    subjects: list[str] = []
    for key in ("items", "events", "documents"):
        value = response.evidence.get(key)
        if not isinstance(value, list):
            continue
        for item in value:
            if not isinstance(item, dict):
                continue
            ticker = item.get("ticker")
            if isinstance(ticker, str) and ticker not in subjects:
                subjects.append(ticker)
            for matched in item.get("matched_holdings") or []:
                if isinstance(matched, str) and matched not in subjects:
                    subjects.append(matched)
    return subjects
