from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context
from dataclasses import dataclass, field
from typing import Any

from libra_agent.errors import ChatClientError
from libra_agent.libra_models import AgentResponse, PortfolioSnapshot
from libra_agent.runtime.debate_events import (
    publish_debate_event,
    publish_llm_error,
    publish_llm_prompt,
    publish_llm_response,
)

from .compliance import build_compliance_context_from_portfolio, default_compliance_engine
from .judge.final import compute_tentative_trades, render_rule_based_final_decision
from .mediator import consensus_by_subject, select_targets
from .schemas.agent import AgentOpinion, Direction, Vote
from .schemas.compliance import ComplianceCheck, MarketSnapshot
from .schemas.decision import ConsensusScore, FinalDecision, MediatorDecision, Trade
from .schemas.ips import IPSConfig, KYCProfile

CORE_COMMITTEE_AGENTS = ("disclosure", "news", "report", "profit", "cost")
DOMAIN_COMMITTEE_AGENTS = (
    "risk",
    "tax",
    "macro",
    "sentiment",
    "execution",
    "esg",
    "liquidity",
    "technical",
)
INFO_AGENTS = {"disclosure", "cost", "execution"}
DISPLAY_TO_AGENT_ID = {
    "disclosure": "disclosure",
    "news": "news",
    "report": "report",
    "profit": "profit",
    "cost": "cost",
    "risk": "risk",
    "tax": "tax",
    "macro": "macro",
    "sentiment": "sentiment",
    "execution": "execution",
    "esg": "esg",
    "liquidity": "liquidity",
    "technical": "technical",
}
DISPLAY_TO_AGENT_ID.update(
    {
        _display: _agent
        for _agent, _display in {
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
            "liquidity": "Liquidity",
            "technical": "Technical",
        }.items()
    }
)


@dataclass(slots=True)
class CommitteeRunResult:
    round1_opinions: list[AgentOpinion]
    round2_opinions: list[AgentOpinion]
    consensus_per_subject: dict[str, ConsensusScore]
    targets_to_recall: list[str]
    mediator_decision: MediatorDecision
    compliance_before: ComplianceCheck
    compliance_after: ComplianceCheck
    final_decision: FinalDecision
    round1_responses: list[AgentResponse] = field(default_factory=list)
    round2_responses: list[AgentResponse] = field(default_factory=list)
    tentative_trades: list[Trade] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "round1_opinions": [opinion.to_dict() for opinion in self.round1_opinions],
            "round2_opinions": [opinion.to_dict() for opinion in self.round2_opinions],
            "consensus_per_subject": {
                subject: score.to_dict() for subject, score in self.consensus_per_subject.items()
            },
            "targets_to_recall": list(self.targets_to_recall),
            "mediator_decision": self.mediator_decision.to_dict(),
            "compliance_before": self.compliance_before.to_dict(),
            "compliance_after": self.compliance_after.to_dict(),
            "round1_responses": [response.to_dict() for response in self.round1_responses],
            "round2_responses": [response.to_dict() for response in self.round2_responses],
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
    if (
        direction == Direction.HOLD
        and response.confidence <= 0.05
        and agent in {"news", "report", "sentiment"}
    ):
        votes = []
    return AgentOpinion(
        agent=_display_agent_name(agent),
        round=round_number,  # type: ignore[arg-type]
        votes=votes,
        silence_reason=None
        if votes
        else response.limits_acknowledged or "판단 가능한 신호가 없습니다.",
        reasoning=response.reasoning_for_judge_agent or response.query_understood,
        evidence_refs=_evidence_refs_from_response(response),
        metadata={
            "source_agent_id": agent,
            "opinion": response.opinion,
            "signal_score": response.signal_score,
            "risk_level": response.risk_level,
        },
    )


def responses_to_opinions(
    responses: list[AgentResponse], *, round_number: int = 1
) -> list[AgentOpinion]:
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
        mediator_client: Any | None = None,
        final_judge_client: Any | None = None,
    ) -> CommitteeRunResult:
        before_ctx = build_compliance_context_from_portfolio(
            portfolio,
            proposed_trades=[],
            ips=ips,
            kyc=kyc,
            market_data=market_data,
        )
        compliance_before = self.compliance_engine.check(before_ctx, "BEFORE")
        round1 = responses_to_opinions(responses, round_number=1)
        preliminary_consensus = consensus_by_subject(round1)
        candidate_targets = select_targets(preliminary_consensus, round1)
        mediator_decision = self._mediate(
            round1_opinions=round1,
            consensus_per_subject=preliminary_consensus,
            candidate_targets=candidate_targets,
            compliance_before=compliance_before,
            client=mediator_client,
        )
        _publish_mediator_decision(mediator_decision, round1_count=len(round1), round2_count=0)
        return self._complete_from_opinions(
            portfolio=portfolio,
            round1_opinions=round1,
            round2_opinions=list(round2_opinions or []),
            mediator_decision=mediator_decision,
            compliance_before=compliance_before,
            ips=ips,
            kyc=kyc,
            market_data=market_data,
            final_judge_client=final_judge_client,
            round1_responses=responses,
        )

    def run_from_agent_rounds(
        self,
        *,
        portfolio: PortfolioSnapshot,
        round1_agent_calls: dict[str, Callable[[], AgentResponse]],
        round2_agent_call_factory: Callable[[str, str], Callable[[], AgentResponse]],
        ips: IPSConfig | None = None,
        kyc: KYCProfile | None = None,
        market_data: MarketSnapshot | None = None,
        mediator_client: Any | None = None,
        final_judge_client: Any | None = None,
    ) -> CommitteeRunResult:
        before_ctx = build_compliance_context_from_portfolio(
            portfolio,
            proposed_trades=[],
            ips=ips,
            kyc=kyc,
            market_data=market_data,
        )
        compliance_before = self.compliance_engine.check(before_ctx, "BEFORE")
        round1_responses = run_agent_callables_parallel(round1_agent_calls, max_workers=11)
        round1 = responses_to_opinions(round1_responses, round_number=1)
        preliminary_consensus = consensus_by_subject(round1)
        candidate_targets = select_targets(preliminary_consensus, round1)
        mediator_decision = self._mediate(
            round1_opinions=round1,
            consensus_per_subject=preliminary_consensus,
            candidate_targets=candidate_targets,
            compliance_before=compliance_before,
            client=mediator_client,
        )
        _publish_mediator_decision(mediator_decision, round1_count=len(round1), round2_count=0)
        round2_responses: list[AgentResponse] = []
        round2: list[AgentOpinion] = []
        if not mediator_decision.skip_round_2 and mediator_decision.targets_to_recall:
            round2_calls: dict[str, Callable[[], AgentResponse]] = {}
            round2_context = _round2_context(
                round1_opinions=round1,
                consensus_per_subject=preliminary_consensus,
                compliance_before=compliance_before,
            )
            for target in mediator_decision.targets_to_recall:
                agent_id = _agent_id_from_display(target)
                round2_calls[agent_id] = round2_agent_call_factory(agent_id, round2_context)
            round2_responses = run_agent_callables_parallel(round2_calls, max_workers=4)
            round2 = responses_to_opinions(round2_responses, round_number=2)
            _publish_mediator_decision(
                mediator_decision,
                round1_count=len(round1),
                round2_count=len(round2),
            )
            _annotate_round2_opinions(
                round2,
                previous_round_summary=_round1_summary(round1),
                exposed_signals=_exposed_signals(
                    round1, exclude_agents={op.agent for op in round2}
                ),
            )

        return self._complete_from_opinions(
            portfolio=portfolio,
            round1_opinions=round1,
            round2_opinions=round2,
            mediator_decision=mediator_decision,
            compliance_before=compliance_before,
            ips=ips,
            kyc=kyc,
            market_data=market_data,
            final_judge_client=final_judge_client,
            round1_responses=round1_responses,
            round2_responses=round2_responses,
        )

    def _complete_from_opinions(
        self,
        *,
        portfolio: PortfolioSnapshot,
        round1_opinions: list[AgentOpinion],
        round2_opinions: list[AgentOpinion],
        mediator_decision: MediatorDecision,
        compliance_before: ComplianceCheck,
        ips: IPSConfig | None,
        kyc: KYCProfile | None,
        market_data: MarketSnapshot | None,
        final_judge_client: Any | None,
        round1_responses: list[AgentResponse] | None = None,
        round2_responses: list[AgentResponse] | None = None,
    ) -> CommitteeRunResult:
        all_opinions = [*round1_opinions, *round2_opinions]
        consensus = consensus_by_subject(all_opinions)
        targets = list(mediator_decision.targets_to_recall)
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
        if final_judge_client is not None:
            final_decision = _fill_final_decision_with_llm(
                client=final_judge_client,
                final_decision=final_decision,
                consensus_per_subject=consensus,
                opinions=all_opinions,
                compliance_after=compliance_after,
                tentative_trades=tentative_trades,
            )
        publish_debate_event(
            "consensus_updated",
            {
                "consensus_per_subject": {
                    subject: score.to_dict() for subject, score in consensus.items()
                },
                "tentative_trades": [trade.to_dict() for trade in tentative_trades],
            },
        )
        publish_debate_event(
            "final_decision_draft",
            {
                "decision": final_decision.decision.value,
                "branch": final_decision.branch.value,
                "reasoning": final_decision.reasoning,
                "requires_approval": final_decision.user_question is not None,
                "trades": [trade.to_dict() for trade in final_decision.trades],
            },
        )
        return CommitteeRunResult(
            round1_opinions=round1_opinions,
            round2_opinions=round2_opinions,
            consensus_per_subject=consensus,
            targets_to_recall=targets,
            mediator_decision=mediator_decision,
            compliance_before=compliance_before,
            compliance_after=compliance_after,
            round1_responses=list(round1_responses or []),
            round2_responses=list(round2_responses or []),
            tentative_trades=tentative_trades,
            final_decision=final_decision,
        )

    def _mediate(
        self,
        *,
        round1_opinions: list[AgentOpinion],
        consensus_per_subject: dict[str, ConsensusScore],
        candidate_targets: list[str],
        compliance_before: ComplianceCheck,
        client: Any | None,
    ) -> MediatorDecision:
        if client is None:
            return MediatorDecision(
                consensus_per_subject=consensus_per_subject,
                targets_to_recall=list(candidate_targets),
                skip_round_2=not bool(candidate_targets),
                rationale=_deterministic_mediator_rationale(candidate_targets, compliance_before),
            )
        payload = {
            "round1_opinions": [_compact_opinion(opinion) for opinion in round1_opinions],
            "compliance_before": compliance_before.to_dict(),
            "consensus_per_subject": {
                subject: score.to_dict() for subject, score in consensus_per_subject.items()
            },
            "candidate_targets": list(candidate_targets),
            "schema": {
                "required_keys": ["targets_to_recall", "skip_round_2", "rationale"],
                "targets_to_recall": "list of agent display names from candidate_targets only, max 4",
                "skip_round_2": "boolean",
                "rationale": "Korean 1-3 sentence explanation",
            },
        }
        try:
            raw = _chat_json_strict(
                client,
                system_prompt=_MEDIATOR_SYSTEM_PROMPT,
                user_prompt=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                tool_name="submit_mediator_decision",
                tool_description="Round 1 consensus를 검토하고 Round 2 표적 재호출 여부를 제출합니다.",
                input_schema=_mediator_decision_schema(candidate_targets),
                temperature=0.0,
                actor="mediator",
                phase="round2_target_selection",
            )
        except ChatClientError:
            raise
        except Exception as exc:
            raise ChatClientError("Mediator Judge LLM failed.") from exc
        return _parse_mediator_decision(
            raw,
            consensus_per_subject=consensus_per_subject,
            candidate_targets=candidate_targets,
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
        future_to_agent = {
            pool.submit(copy_context().run, _run_agent_call_with_events, agent_id, call): agent_id
            for agent_id, call in agent_calls.items()
        }
        for future in as_completed(future_to_agent):
            agent_id = future_to_agent[future]
            results[agent_id] = future.result()
    return [results[agent_id] for agent_id in agent_calls.keys()]


def _run_agent_call_with_events(agent_id: str, call: Callable[[], AgentResponse]) -> AgentResponse:
    publish_debate_event(
        "agent_started",
        {"agent_id": agent_id, "layer": "committee", "turn_number": None},
    )
    try:
        response = call()
    except Exception as exc:
        publish_debate_event(
            "agent_failed",
            {"agent_id": agent_id, "layer": "committee", "error": str(exc)[:500]},
        )
        raise
    publish_debate_event(
        "agent_completed",
        {
            "agent_id": response.agent_id,
            "layer": "committee",
            "turn_number": response.turn_number,
            "verdict": response.verdict.value,
            "opinion": response.opinion,
            "direction": response.direction,
            "strength": response.strength,
            "confidence": response.confidence,
            "urgency": response.urgency.value,
            "risk_level": response.risk_level,
            "focus_tickers": list(response.focus_tickers),
            "reasoning": response.reasoning_for_judge_agent[:900],
        },
    )
    return response


def _publish_mediator_decision(
    decision: MediatorDecision,
    *,
    round1_count: int,
    round2_count: int,
) -> None:
    publish_debate_event(
        "mediator_decision",
        {
            "targets_to_recall": list(decision.targets_to_recall),
            "skip_round_2": decision.skip_round_2,
            "rationale": decision.rationale,
            "round1_count": round1_count,
            "round2_count": round2_count,
        },
    )


_MEDIATOR_SYSTEM_PROMPT = """당신은 Libra 시스템의 Mediator Judge 입니다.

[역할]
- Round 1 의견과 코드가 계산한 합의 점수, 후보 표적을 검토합니다.
- 충돌이 있는 subject에 대해 Round 2에서 다시 부를 에이전트를 고릅니다.
- 합의 점수는 재계산하지 않습니다.

[제약]
- 거래 산정 금지.
- 최종 decision 생성 금지.
- targets_to_recall은 candidate_targets 안에서만 선택합니다.
- targets_to_recall은 최대 4명입니다.
- 출력은 JSON object만 반환합니다.
- 한국어만 사용하고 Japanese kana는 사용하지 않습니다.
"""


_FINAL_JUDGE_SYSTEM_PROMPT = """당신은 Libra 시스템의 Final Judge 입니다.

[역할]
- 코드가 이미 결정한 branch와 decision을 바꾸지 않고 사용자에게 보여줄 reasoning을 채웁니다.
- Compliance BLOCKING은 절대 우회하지 않습니다.
- tentative_trades의 magnitude를 바꾸지 않습니다.

[출력]
- JSON object만 반환합니다.
- reasoning: 한국어 한 문단.
- user_question: USER_DECISION_REQUIRED일 때만 한국어 질문, 아니면 null.
- user_options: USER_DECISION_REQUIRED일 때 정확히 3개. 각 항목은 label, supporting_agents, expected_effect를 포함.
- 한국어만 사용하고 Japanese kana는 사용하지 않습니다.
"""


def _parse_mediator_decision(
    payload: Mapping[str, Any],
    *,
    consensus_per_subject: dict[str, ConsensusScore],
    candidate_targets: list[str],
) -> MediatorDecision:
    raw_targets = payload.get("targets_to_recall", [])
    if not isinstance(raw_targets, list):
        raise ChatClientError("Mediator Judge returned invalid targets_to_recall.")
    candidate_ids = {_agent_id_from_display(item) for item in candidate_targets}
    targets: list[str] = []
    for item in raw_targets:
        agent_id = _agent_id_from_display(str(item))
        if agent_id not in candidate_ids:
            raise ChatClientError(f"Mediator Judge selected non-candidate target: {item}")
        display = _display_agent_name(agent_id)
        if display not in targets:
            targets.append(display)
        if len(targets) > 4:
            raise ChatClientError("Mediator Judge selected more than 4 targets.")
    skip = bool(payload.get("skip_round_2"))
    if skip:
        targets = []
    if not skip and not targets:
        skip = True
    rationale = str(payload.get("rationale") or "").strip()
    if not rationale:
        raise ChatClientError("Mediator Judge returned empty rationale.")
    return MediatorDecision(
        consensus_per_subject=consensus_per_subject,
        targets_to_recall=targets,
        skip_round_2=skip,
        rationale=rationale,
    )


def _fill_final_decision_with_llm(
    *,
    client: Any,
    final_decision: FinalDecision,
    consensus_per_subject: dict[str, ConsensusScore],
    opinions: list[AgentOpinion],
    compliance_after: ComplianceCheck,
    tentative_trades: list[Trade],
) -> FinalDecision:
    payload = {
        "locked_decision": final_decision.decision.value,
        "locked_branch": final_decision.branch.value,
        "locked_trades": [trade.to_dict() for trade in final_decision.trades],
        "tentative_trades": [trade.to_dict() for trade in tentative_trades],
        "compliance_after": compliance_after.to_dict(),
        "consensus_per_subject": {
            subject: score.to_dict() for subject, score in consensus_per_subject.items()
        },
        "opinions": [_compact_opinion(opinion) for opinion in opinions],
        "schema": {
            "required_keys": ["reasoning", "user_question", "user_options"],
            "user_options": "null or list of exactly 3 {label, supporting_agents, expected_effect}",
        },
    }
    try:
        raw = _chat_json_strict(
            client,
            system_prompt=_FINAL_JUDGE_SYSTEM_PROMPT,
            user_prompt=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            tool_name="submit_final_judge_fill",
            tool_description="코드가 잠근 최종 branch에 맞춰 사용자 설명과 선택지를 제출합니다.",
            input_schema=_final_judge_fill_schema(),
            temperature=0.0,
            actor="final_judge",
            phase="final_explanation",
        )
    except ChatClientError:
        raise
    except Exception as exc:
        raise ChatClientError("Final Judge LLM failed.") from exc

    reasoning = str(raw.get("reasoning") or "").strip()
    if not reasoning:
        raise ChatClientError("Final Judge returned empty reasoning.")
    question = raw.get("user_question")
    raw_options = raw.get("user_options")
    user_options = final_decision.user_options
    if final_decision.user_question is not None:
        if not isinstance(question, str) or not question.strip():
            raise ChatClientError(
                "Final Judge must return user_question for USER_DECISION_REQUIRED."
            )
        if not isinstance(raw_options, list) or len(raw_options) != 3:
            raise ChatClientError("Final Judge must return exactly 3 user_options.")
        from .schemas.decision import UserOption

        user_options = []
        for item in raw_options:
            if not isinstance(item, Mapping):
                raise ChatClientError("Final Judge user_options must be objects.")
            user_options.append(
                UserOption(
                    label=str(item.get("label") or "").strip(),
                    supporting_agents=[
                        str(agent)
                        for agent in item.get("supporting_agents", [])
                        if str(agent).strip()
                    ]
                    if isinstance(item.get("supporting_agents"), list)
                    else [],
                    expected_effect=str(item.get("expected_effect") or "").strip(),
                )
            )
        if any(not option.label or not option.expected_effect for option in user_options):
            raise ChatClientError("Final Judge user_options contain empty fields.")
        final_decision.user_question = question.strip()
    else:
        final_decision.user_question = None
        user_options = None
    final_decision.reasoning = reasoning
    final_decision.user_options = user_options
    return final_decision


def _compact_opinion(opinion: AgentOpinion) -> dict[str, Any]:
    return {
        "agent": opinion.agent,
        "round": opinion.round,
        "votes": [vote.to_dict() for vote in opinion.votes],
        "silence_reason": opinion.silence_reason,
        "reasoning": opinion.reasoning[:700],
        "metadata": dict(opinion.metadata),
        "delta_from_round1": opinion.delta_from_round1,
    }


def _chat_json_strict(
    client: Any,
    *,
    system_prompt: str,
    user_prompt: str,
    tool_name: str,
    tool_description: str,
    input_schema: dict[str, Any],
    temperature: float,
    actor: str = "judge",
    phase: str = "tool_call",
) -> dict[str, Any]:
    model = str(getattr(client, "model", "unknown"))
    publish_llm_prompt(
        actor=actor,
        phase=phase,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=temperature,
        tool_name=tool_name,
        tool_description=tool_description,
        input_schema=input_schema,
    )
    if hasattr(client, "chat_json_tool"):
        try:
            response = client.chat_json_tool(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                tool_name=tool_name,
                tool_description=tool_description,
                input_schema=input_schema,
                temperature=temperature,
            )
        except Exception as exc:
            publish_llm_error(
                actor=actor,
                phase=phase,
                model=model,
                error=exc,
                tool_name=tool_name,
            )
            raise
    else:
        try:
            response = client.chat_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
            )
        except Exception as exc:
            publish_llm_error(
                actor=actor,
                phase=phase,
                model=model,
                error=exc,
                tool_name=tool_name,
            )
            raise
    publish_llm_response(
        actor=actor,
        phase=phase,
        model=model,
        output=response,
        tool_name=tool_name,
    )
    return response


def _mediator_decision_schema(candidate_targets: list[str]) -> dict[str, Any]:
    target_values = [
        _display_agent_name(_agent_id_from_display(item)) for item in candidate_targets
    ]
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "targets_to_recall": {
                "type": "array",
                "items": {"type": "string", "enum": target_values or [""]},
                "maxItems": 4,
            },
            "skip_round_2": {"type": "boolean"},
            "rationale": {"type": "string", "minLength": 1},
        },
        "required": ["targets_to_recall", "skip_round_2", "rationale"],
    }


def _final_judge_fill_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "reasoning": {"type": "string", "minLength": 1},
            "user_question": {
                "anyOf": [
                    {"type": "string"},
                    {"type": "null"},
                ]
            },
            "user_options": {
                "anyOf": [
                    {"type": "null"},
                    {
                        "type": "array",
                        "minItems": 3,
                        "maxItems": 3,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "label": {"type": "string", "minLength": 1},
                                "supporting_agents": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "expected_effect": {"type": "string", "minLength": 1},
                            },
                            "required": ["label", "supporting_agents", "expected_effect"],
                        },
                    },
                ]
            },
        },
        "required": ["reasoning", "user_question", "user_options"],
    }


def _deterministic_mediator_rationale(
    targets: list[str], compliance_before: ComplianceCheck
) -> str:
    if not targets:
        return "코드 합의 점수 기준으로 Round 2가 필요한 충돌 subject가 없습니다."
    suffix = ""
    if not compliance_before.can_proceed:
        suffix = " Compliance BEFORE 단계의 BLOCKING 위반도 함께 확인되었습니다."
    return f"충돌 subject에 직접 발화한 에이전트 {', '.join(targets)}를 Round 2 표적으로 선정했습니다.{suffix}"


def _round2_context(
    *,
    round1_opinions: list[AgentOpinion],
    consensus_per_subject: dict[str, ConsensusScore],
    compliance_before: ComplianceCheck,
) -> str:
    return json.dumps(
        {
            "round1_summary": _round1_summary(round1_opinions),
            "other_signals": _exposed_signals(round1_opinions),
            "consensus_per_subject": {
                subject: score.to_dict() for subject, score in consensus_per_subject.items()
            },
            "compliance_before": compliance_before.to_dict(),
            "instruction": (
                "Round 2입니다. 다른 에이전트 신호와 Compliance 결과를 고려한 뒤 "
                "기존 의견을 유지/강화/약화/반전할지 판단하세요."
            ),
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _round1_summary(opinions: list[AgentOpinion]) -> str:
    parts: list[str] = []
    for opinion in opinions:
        vote_text = (
            ", ".join(
                f"{vote.subject}:{vote.direction.value}:{vote.magnitude_pct:+.1f}%:{vote.confidence:.2f}"
                for vote in opinion.votes[:4]
            )
            or f"silence={opinion.silence_reason or ''}"
        )
        parts.append(f"{opinion.agent}({vote_text})")
    return " | ".join(parts)


def _exposed_signals(
    opinions: list[AgentOpinion], *, exclude_agents: set[str] | None = None
) -> list[str]:
    excluded = {item.casefold() for item in (exclude_agents or set())}
    signals: list[str] = []
    for opinion in opinions:
        if opinion.agent.casefold() in excluded:
            continue
        for vote in opinion.votes:
            signals.append(
                f"{opinion.agent}: {vote.subject} {vote.direction.value} {vote.magnitude_pct:+.1f}% conf={vote.confidence:.2f}"
            )
    return signals[:20]


def _annotate_round2_opinions(
    opinions: list[AgentOpinion],
    *,
    previous_round_summary: str,
    exposed_signals: list[str],
) -> None:
    for opinion in opinions:
        opinion.previous_round_summary = previous_round_summary
        opinion.exposed_signals = list(exposed_signals)


def _agent_id_from_display(value: str) -> str:
    text = str(value).strip()
    if not text:
        raise ChatClientError("empty agent target")
    return DISPLAY_TO_AGENT_ID.get(text, DISPLAY_TO_AGENT_ID.get(text.casefold(), text.casefold()))


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
    refs = [
        reference.url or reference.title
        for reference in response.references
        if reference.url or reference.title
    ]
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
