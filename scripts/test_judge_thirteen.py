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


class ScriptedJudgeClient:
    model = "scripted-agentic-script"

    def __init__(self) -> None:
        self.core_actions = [
            {"action": "CALL_AGENT", "agent_id": "disclosure", "reason": "목표비중 판단 전 공시를 확인합니다."},
            {"action": "CALL_AGENT", "agent_id": "news", "reason": "시장 반응을 확인합니다."},
            {"action": "CALL_AGENT", "agent_id": "profit", "reason": "후보 리밸런싱의 기대수익을 검토합니다."},
            {"action": "CALL_AGENT", "agent_id": "cost", "reason": "후보 리밸런싱의 실행비용을 검토합니다."},
            {"action": "FINALIZE", "reason": "Core 검토가 끝나 도메인 심의로 이동합니다."},
        ]
        self.domain_actions = [
            {"action": "CALL_AGENT", "agent_id": agent_id, "reason": "도메인 심의 대상을 선택합니다."}
            for agent_id in ("risk", "tax", "compliance", "macro", "sentiment", "execution", "esg")
        ] + [{"action": "FINALIZE_DOMAIN_REVIEW", "reason": "도메인 심의를 종료합니다."}]

    def chat_json(self, **kwargs: object) -> dict[str, object]:
        system_prompt = str(kwargs.get("system_prompt") or "")
        user_prompt = str(kwargs.get("user_prompt") or "")
        if "FINALIZE_DOMAIN_REVIEW" in user_prompt or system_prompt.startswith("You are the LIBRA Judge orchestrating the domain council layer"):
            if not self.domain_actions:
                raise ChatClientError("domain routing script exhausted")
            return dict(self.domain_actions.pop(0))
        if '"action_values":["CALL_AGENT","FINALIZE"]' in user_prompt:
            if not self.core_actions:
                raise ChatClientError("core routing script exhausted")
            return dict(self.core_actions.pop(0))
        if '"required_keys"' in user_prompt or "Decide among HOLD, DEFER, USER_DECISION_REQUIRED, REBALANCE" in system_prompt:
            return {
                "decision": DecisionType.REBALANCE.value,
                "summary": "목표비중 편차가 커 후보 리밸런싱 초안을 승인 단계로 올립니다.",
                "confidence": 0.78,
                "urgency": Urgency.SCHEDULED.value,
                "reasoning": "Judge LLM이 Core 검토와 Domain Council 심의를 종합했습니다.",
                "candidate_rebalance_plan": {"005930": 0.1, "000660": 0.1},
                "needs_trade_evaluation": True,
                "follow_up_at": None,
                "feedback_checkpoint": None,
                "user_notification": {"level": "info", "body": "리밸런싱 초안을 확인하세요.", "action_required": False},
            }
        raise ChatClientError("subagent LLM intentionally offline in script")

    def ensure_available(self) -> None:
        return None


class DomainLLMRouter:
    def ask(self, *, agent_id: str, **_: object) -> str:
        if agent_id == "sentiment":
            return '{"sentiment_score": 0.15, "vote": "approve", "rationale": "테스트 도메인 LLM 응답"}'
        return "테스트 도메인 LLM 응답. 판단: approve."

    def model_name_for(self, agent_id: str) -> str:
        return f"test-domain-{agent_id}"


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
    client = ScriptedJudgeClient()
    bundle = build_default_agent_bundle(client=client)
    domain_agents = bundle.domain_agents()
    assert set(domain_agents) == {"risk", "tax", "compliance", "macro", "sentiment", "execution", "esg"}
    print("Step 1 PASS: 7 domain agents registered")

    orchestrator = JudgeOrchestrator(client=client)
    orchestrator.domain_router = DomainLLMRouter()
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
