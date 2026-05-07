from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ["LIBRA_DOMAIN_AGENTS_ENABLED"] = "true"

from libra_agent.errors import ChatClientError
from libra_agent.domain_agents._consensus import apply_compliance_veto, compute_domain_consensus
from libra_agent.libra.direct_indexing import PortfolioDefinition
from libra_agent.libra.agents import build_default_agent_bundle
from libra_agent.libra_models import AgentResponse, AgentVerdict, DecisionType, JudgeDecision, PortfolioSnapshot, Urgency
from libra_agent.libra_runtime import JudgeOrchestrator, LocalKnowledgeBase


class OfflineChatClient:
    model = "offline-script"

    def chat_json(self, **_: object) -> dict[str, object]:
        raise ChatClientError("offline")

    def ensure_available(self) -> None:
        return None


def _portfolio() -> PortfolioSnapshot:
    return PortfolioSnapshot.from_dict(
        {
            "generated_at": "2026-05-07T09:00:00+09:00",
            "holdings": [
                {"ticker": "005930", "company_name": "삼성전자", "weight": 0.0},
                {"ticker": "000660", "company_name": "SK하이닉스", "weight": 0.0},
            ],
            "total_value_krw": 30000000,
            "cash_weight": 1.0,
        }
    )


def _definition() -> PortfolioDefinition:
    return PortfolioDefinition.from_dict(
        {
            "name": "13-agent integration smoke",
            "drift_threshold": 0.05,
            "target_weights": [
                {"ticker": "005930", "company_name": "삼성전자", "weight": 0.5},
                {"ticker": "000660", "company_name": "SK하이닉스", "weight": 0.5},
            ],
        }
    )


def _rejecting_compliance_response() -> AgentResponse:
    return AgentResponse.from_dict(
        {
            "agent_id": "compliance",
            "opinion_id": "compliance-test",
            "turn_number": 1,
            "query_understood": "compliance check",
            "verdict": AgentVerdict.DIRECT_ANSWER.value,
            "evidence": {
                "vote": "reject",
                "domain_signals": [{"label": "IPS violations", "value": "1"}],
                "llm_used": "rule-based",
            },
            "direction": -0.7,
            "strength": 1.0,
            "urgency": Urgency.SCHEDULED.value,
            "confidence": 1.0,
            "reasoning_for_judge_agent": "사용자 IPS 제외 룰 위반",
            "opinion": "NEGATIVE",
        }
    )


def main() -> None:
    client = OfflineChatClient()
    bundle = build_default_agent_bundle(client=client)
    domain_agents = bundle.domain_agents()
    assert set(domain_agents) == {"risk", "tax", "compliance", "macro", "sentiment", "execution", "esg"}
    print("Step 1 PASS: 7 domain agents registered")

    orchestrator = JudgeOrchestrator(client=client)
    result = orchestrator.run(
        query="초기 목표비중 기준으로 13개 에이전트 통합 판단을 검증해줘",
        portfolio=_portfolio(),
        knowledge_base=LocalKnowledgeBase(events=[], documents=[], source_paths={}),
        portfolio_definition=_definition(),
        depth="medium",
        trigger="pull",
    )
    agent_ids = {item["agent_id"] for item in result["agent_responses"]}
    assert {"disclosure", "news", "profit", "cost"}.issubset(agent_ids)
    assert set(domain_agents).issubset(agent_ids)
    trace_actors = {item["actor"] for item in result["decision"]["decision_trace"]}
    assert set(domain_agents).issubset(trace_actors)
    print("Step 2 PASS: Judge run exposes 7 domain verdicts in Decision Trace")

    domain_responses = [
        AgentResponse.from_dict(item)
        for item in result["agent_responses"]
        if item["agent_id"] in domain_agents
    ]
    consensus = compute_domain_consensus(domain_responses)
    assert consensus["n_approve"] + consensus["n_reject"] + consensus["n_abstain"] == 7
    print(f"Step 3 PASS: domain consensus computed {consensus}")

    decision = JudgeDecision.from_dict(
        {
            "decision": DecisionType.REBALANCE.value,
            "summary": "리밸런싱 초안",
            "confidence": 0.8,
            "urgency": Urgency.SCHEDULED.value,
            "reasoning": "초기 판단",
            "candidate_rebalance_plan": {"005930": 0.1},
        }
    )
    vetoed = apply_compliance_veto(decision, [_rejecting_compliance_response()])
    assert vetoed.decision == DecisionType.USER_DECISION_REQUIRED
    assert vetoed.user_notification is not None and vetoed.user_notification.action_required
    print("Step 4 PASS: Compliance reject forces USER_DECISION_REQUIRED")


if __name__ == "__main__":
    main()
