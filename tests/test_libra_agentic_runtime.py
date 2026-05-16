from __future__ import annotations

import asyncio
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("LIBRA_DOMAIN_AGENTS_ENABLED", "true")

from libra_agent.domain_agents._adapter import _run_async
from libra_agent.errors import ChatClientError
from libra_agent.libra.constraints import validate_rebalance_plan
from libra_agent.libra.direct_indexing import PortfolioDefinition
from libra_agent.libra.evaluation import evaluate_decision_outcome
from libra_agent.libra.signals import infer_signal_profile
from libra_agent.libra_models import AgentResponse, AgentVerdict, PortfolioSnapshot, Urgency
from libra_agent.libra_runtime import (
    CoreChatDomainRouter,
    JudgeOrchestrator,
    LLMAgent,
    LocalKnowledgeBase,
)


class FakeChatClient:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.model = "fake-model"

    def chat_json(self, **_: object) -> dict[str, object]:
        return dict(self.payload)


class FailingChatClient:
    model = "offline-test"

    def chat_json(self, **_: object) -> dict[str, object]:
        raise ChatClientError("offline")

    def ensure_available(self) -> None:
        return None


class DomainAdapterChatClient:
    model = "domain-adapter-test"

    def chat_json(self, **kwargs: object) -> dict[str, object]:
        user_prompt = str(kwargs.get("user_prompt") or "")
        self.last_user_prompt = user_prompt
        return {"rationale": "도메인 라우터 어댑터 응답"}

    def ensure_available(self) -> None:
        return None


class RoutingScriptChatClient:
    model = "scripted-agentic-test"

    def __init__(
        self,
        *,
        core_actions: list[dict[str, object]],
        domain_actions: list[dict[str, object]] | None = None,
        final_payload: dict[str, object] | None = None,
    ) -> None:
        self.core_actions = list(core_actions)
        self.domain_actions = list(domain_actions or [])
        self.final_payload = dict(final_payload or _hold_final_payload())

    def chat_json(self, **kwargs: object) -> dict[str, object]:
        system_prompt = str(kwargs.get("system_prompt") or "")
        user_prompt = str(kwargs.get("user_prompt") or "")
        if "FINALIZE_DOMAIN_REVIEW" in user_prompt or system_prompt.startswith(
            "You are the LIBRA Judge orchestrating the domain council layer"
        ):
            if not self.domain_actions:
                raise ChatClientError("domain routing script exhausted")
            return dict(self.domain_actions.pop(0))
        if '"action_values":["CALL_AGENT","FINALIZE"]' in user_prompt:
            if not self.core_actions:
                raise ChatClientError("core routing script exhausted")
            return dict(self.core_actions.pop(0))
        if (
            '"required_keys"' in user_prompt
            or "Decide among HOLD, DEFER, USER_DECISION_REQUIRED, REBALANCE" in system_prompt
        ):
            return dict(self.final_payload)
        raise ChatClientError("subagent LLM intentionally offline in orchestration test")

    def ensure_available(self) -> None:
        return None


class RepairingRoutingChatClient:
    model = "repairing-routing-test"

    def __init__(self) -> None:
        self.calls = 0

    def chat_json(self, **kwargs: object) -> dict[str, object]:
        self.calls += 1
        user_prompt = str(kwargs.get("user_prompt") or "")
        if '"invalid_response"' in user_prompt:
            return {
                "action": "CALL_AGENT",
                "agent_id": "news",
                "reason": "거래 초안 없이 profit 호출은 부적절하므로 시장 반응을 먼저 확인합니다.",
            }
        return {
            "action": "CALL_AGENT",
            "agent_id": "profit",
            "reason": "거래 초안 없이 수익성을 검토하려 했습니다.",
            "candidate_rebalance_plan": {},
        }

    def ensure_available(self) -> None:
        return None


class DomainLLMRouter:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def ask(self, *, agent_id: str, **_: object) -> str:
        self.calls.append(agent_id)
        if agent_id == "sentiment":
            return '{"sentiment_score": 0.15, "vote": "approve", "rationale": "테스트 도메인 LLM 응답"}'
        return "테스트 도메인 LLM 응답. 판단: approve."

    def model_name_for(self, agent_id: str) -> str:
        return f"test-domain-{agent_id}"


class CapturingDomainAgent:
    def __init__(self, agent_id: str = "risk") -> None:
        self.agent_id = agent_id
        self.contexts: list[object] = []

    async def deliberate(self, ctx: object) -> SimpleNamespace:
        self.contexts.append(ctx)
        return SimpleNamespace(
            agent_id=self.agent_id,
            vote="approve",
            confidence=0.7,
            rationale="테스트 도메인 에이전트가 컨텍스트를 캡처했습니다.",
            signals=[],
            llm_used="test-domain-agent",
        )


def _call(agent_id: str, reason: str = "LLM이 다음 호출 대상을 선택했습니다.") -> dict[str, object]:
    return {"action": "CALL_AGENT", "agent_id": agent_id, "reason": reason}


def _finalize(reason: str = "LLM이 현재 관찰로 충분하다고 판단했습니다.") -> dict[str, object]:
    return {"action": "FINALIZE", "reason": reason}


def _domain_call(
    agent_id: str, reason: str = "LLM이 도메인 심의 대상을 선택했습니다."
) -> dict[str, object]:
    return {"action": "CALL_AGENT", "agent_id": agent_id, "reason": reason}


def _domain_done(reason: str = "LLM이 도메인 심의를 마치기로 했습니다.") -> dict[str, object]:
    return {"action": "FINALIZE_DOMAIN_REVIEW", "reason": reason}


def _domain_script() -> list[dict[str, object]]:
    return [
        _domain_call(agent_id)
        for agent_id in ("risk", "tax", "compliance", "macro", "sentiment", "execution", "esg")
    ] + [_domain_done()]


def _hold_final_payload() -> dict[str, object]:
    return {
        "decision": "HOLD",
        "summary": "현재 관찰만으로는 포트폴리오를 변경하지 않습니다.",
        "confidence": 0.71,
        "urgency": "defer",
        "reasoning": "Judge LLM이 Core 관찰과 Domain Council 심의를 종합해 유지 결정을 내렸습니다.",
        "candidate_rebalance_plan": {},
        "needs_trade_evaluation": False,
        "follow_up_at": None,
        "feedback_checkpoint": None,
        "user_notification": {
            "level": "silent",
            "body": "현재 포트폴리오를 유지합니다.",
            "action_required": False,
        },
    }


def _rebalance_final_payload() -> dict[str, object]:
    return {
        "decision": "REBALANCE",
        "summary": "목표비중 편차가 커 후보 리밸런싱 초안을 승인 단계로 올립니다.",
        "confidence": 0.78,
        "urgency": "scheduled",
        "reasoning": "Judge LLM이 목표비중 편차, Profit 검토, Cost 검토, Domain Council 심의를 종합했습니다.",
        "candidate_rebalance_plan": {"005930": 0.1, "000660": 0.1},
        "needs_trade_evaluation": True,
        "follow_up_at": None,
        "feedback_checkpoint": None,
        "user_notification": {
            "level": "info",
            "body": "리밸런싱 초안을 확인하세요.",
            "action_required": False,
        },
    }


def _portfolio() -> PortfolioSnapshot:
    return PortfolioSnapshot.from_dict(
        {
            "generated_at": "2026-04-14T09:00:00+09:00",
            "holdings": [
                {
                    "ticker": "005930",
                    "company_name": "삼성전자",
                    "weight": 0.6,
                },
                {
                    "ticker": "000660",
                    "company_name": "SK하이닉스",
                    "weight": 0.4,
                },
            ],
            "cash_weight": 0.0,
        }
    )


def _empty_direct_indexing_portfolio() -> PortfolioSnapshot:
    return PortfolioSnapshot.from_dict(
        {
            "generated_at": "2026-05-07T09:00:00+09:00",
            "holdings": [
                {
                    "ticker": "005930",
                    "company_name": "삼성전자",
                    "weight": 0.0,
                },
                {
                    "ticker": "000660",
                    "company_name": "SK하이닉스",
                    "weight": 0.0,
                },
            ],
            "total_value_krw": 30000000,
            "cash_weight": 1.0,
        }
    )


def _empty_cash_portfolio() -> PortfolioSnapshot:
    return PortfolioSnapshot.from_dict(
        {
            "generated_at": "2026-05-07T09:00:00+09:00",
            "holdings": [],
            "total_value_krw": 30000000,
            "cash_weight": 1.0,
            "user_preferences": ["모의투자 기준", "리스크 우선", "무리한 회전율 회피"],
        }
    )


def _portfolio_definition() -> PortfolioDefinition:
    return PortfolioDefinition.from_dict(
        {
            "name": "반도체 테스트 포트폴리오",
            "risk_profile": "위험중립형",
            "drift_threshold": 0.05,
            "target_weights": [
                {"ticker": "005930", "company_name": "삼성전자", "weight": 0.5},
                {"ticker": "000660", "company_name": "SK하이닉스", "weight": 0.5},
            ],
        }
    )


class LibraAgenticRuntimeTests(unittest.TestCase):
    def test_signal_profile_applies_source_trust_and_opinion(self) -> None:
        profile = infer_signal_profile(
            agent_id="news",
            direction=-0.8,
            strength=0.9,
            confidence=0.85,
            evidence={"event_type": "regulation", "horizon": "short"},
        )

        self.assertEqual(profile.signal_score, -0.4284)
        self.assertEqual(profile.source_trust, 0.7)
        self.assertEqual(profile.event_type, "regulation")
        self.assertEqual(profile.horizon, "short")
        self.assertEqual(profile.risk_level, "mid")
        self.assertEqual(profile.opinion, "SELL_BIAS")

    def test_rebalance_constraint_blocks_daily_trade_over_limit(self) -> None:
        result = validate_rebalance_plan(
            portfolio=_portfolio(),
            plan={"005930": -0.15, "000660": 0.15},
        )

        self.assertFalse(result.passed)
        self.assertIn("exceeds max", result.reason)

    def test_rebalance_constraint_allows_small_valid_plan(self) -> None:
        result = validate_rebalance_plan(
            portfolio=_portfolio(),
            plan={"005930": -0.05, "000660": 0.05},
        )

        self.assertTrue(result.passed)
        self.assertEqual(result.adjusted_plan, {"005930": -0.05, "000660": 0.05})

    def test_evaluation_marks_user_rejection_wrong_when_signal_was_right(self) -> None:
        result = evaluate_decision_outcome(
            decision="REBALANCE",
            rebalance_plan={"005930": -0.05},
            signal_score=-0.5,
            user_feedback="rejected: 단기 노이즈",
            realized_return_pct=-7.0,
            cost_pct=0.1,
        )

        self.assertEqual(result.verdict, "USER_WRONG")
        self.assertTrue(result.direction_accuracy)

    def test_evaluation_keeps_deferred_decision_informational(self) -> None:
        result = evaluate_decision_outcome(
            decision="DEFER",
            rebalance_plan={},
            signal_score=-0.2,
            user_feedback="rejected: 테스트",
            realized_return_pct=-2.4,
            cost_pct=0.1,
        )

        self.assertEqual(result.verdict, "BLOCKED")
        self.assertIsNone(result.cost_efficiency)

    def test_core_normalization_preserves_llm_report_choice(self) -> None:
        orchestrator = JudgeOrchestrator(
            client=FakeChatClient(
                {
                    "action": "CALL_AGENT",
                    "agent_id": "report",
                    "reason": "자료가 부족하므로 리포트 확인",
                }
            )
        )
        disclosure = AgentResponse.from_dict(
            {
                "agent_id": "disclosure",
                "opinion_id": "disclosure_1",
                "turn_number": 1,
                "query_understood": "공시 확인",
                "verdict": AgentVerdict.DIRECT_ANSWER_UNAVAILABLE.value,
                "evidence": {"found_count": 0, "items": [], "upcoming_disclosures": []},
                "direction": 0.0,
                "strength": 0.0,
                "urgency": Urgency.DEFER.value,
                "confidence": 0.2,
                "reasoning_for_judge_agent": "관련 로컬 공시가 없습니다.",
            }
        )
        news = AgentResponse.from_dict(
            {
                "agent_id": "news",
                "opinion_id": "news_1",
                "turn_number": 2,
                "query_understood": "뉴스 확인",
                "verdict": AgentVerdict.QUIET.value,
                "evidence": {"company_findings": {}, "cross_check_count": 0},
                "direction": 0.0,
                "strength": 0.0,
                "urgency": Urgency.DEFER.value,
                "confidence": 0.2,
                "reasoning_for_judge_agent": "관련 로컬 뉴스가 없습니다.",
            }
        )

        action = orchestrator._normalize_judge_action(
            {
                "action": "CALL_AGENT",
                "agent_id": "report",
                "reason": "자료가 부족하므로 리포트 확인",
            },
            query="포트폴리오 점검",
            portfolio=_portfolio(),
            responses=[disclosure, news],
            called_agents=["disclosure", "news"],
            depth="medium",
            trigger="pull",
            trigger_event=None,
            candidate_plan=None,
        )

        self.assertEqual(action["action"], "CALL_AGENT")
        self.assertEqual(action["agent_id"], "report")
        self.assertEqual(action["candidate_rebalance_plan"], {})

    def test_langgraph_trace_starts_with_judge_and_interleaves_routing(self) -> None:
        orchestrator = JudgeOrchestrator(
            client=RoutingScriptChatClient(
                core_actions=[
                    _call("disclosure", "정기 점검의 1차 근거로 공시를 확인합니다."),
                    _call("news", "공시 이후 시장 반응을 확인합니다."),
                    _finalize("Core 관찰이 충분해 도메인 심의로 이동합니다."),
                ],
                domain_actions=_domain_script(),
                final_payload=_hold_final_payload(),
            )
        )
        orchestrator.domain_router = DomainLLMRouter()

        result = orchestrator.run(
            query="포트폴리오 정기 점검",
            portfolio=_portfolio(),
            knowledge_base=LocalKnowledgeBase(events=[], documents=[], source_paths={}),
            depth="medium",
            trigger="pull",
        )

        decision = result["decision"]
        trace = decision["decision_trace"]

        self.assertEqual(
            decision["called_agents"],
            [
                "disclosure",
                "news",
                "risk",
                "tax",
                "compliance",
                "macro",
                "sentiment",
                "execution",
                "esg",
            ],
        )
        self.assertIn("report", decision["skipped_agents"])
        self.assertIn("profit", decision["skipped_agents"])
        self.assertIn("cost", decision["skipped_agents"])
        self.assertEqual(
            [node["actor"] for node in trace[:5]],
            ["judge", "disclosure", "judge", "news", "judge"],
        )
        self.assertEqual(trace[0]["query"], "다음 호출 결정: disclosure")
        self.assertIn("공시", trace[0]["summary"])
        self.assertEqual(trace[2]["query"], "다음 호출 결정: news")
        self.assertEqual(trace[4]["query"], "추가 호출 여부 판단")
        self.assertEqual(trace[5]["query"], "도메인 심의 호출: risk")
        self.assertIn("Core 판단안", trace[5]["summary"])

    def test_domain_council_action_can_route_domain_agent_directly(self) -> None:
        orchestrator = JudgeOrchestrator(
            client=FakeChatClient(
                {
                    "action": "CALL_AGENT",
                    "agent_id": "risk",
                    "reason": "집중도 리스크를 먼저 확인",
                }
            )
        )

        action = orchestrator._domain_next_action(
            query="반도체 포트폴리오 집중도 위험부터 판단해줘",
            portfolio=_portfolio(),
            responses=[],
            called_agents=[],
            depth="medium",
            trigger="pull",
            trigger_event=None,
            candidate_plan=None,
        )

        self.assertEqual(action["action"], "CALL_AGENT")
        self.assertEqual(action["agent_id"], "risk")
        self.assertEqual(action["layer"], "domain")

    def test_core_chat_domain_router_uses_active_chat_client(self) -> None:
        router = CoreChatDomainRouter(DomainAdapterChatClient())

        answer = router.ask(agent_id="risk", system="리스크를 검토해줘.", user="포트폴리오")

        self.assertEqual(answer, "도메인 라우터 어댑터 응답")
        self.assertEqual(router.model_name_for("risk"), "domain-adapter-test")

    def test_v1_domain_agent_receives_backward_compatible_proposed_trades(self) -> None:
        domain_agent = CapturingDomainAgent()
        orchestrator = JudgeOrchestrator(client=FailingChatClient())
        orchestrator.domain_agents = {"risk": domain_agent}

        orchestrator._execute_v1_domain_agent(
            agent_id="risk",
            query="후보 리밸런싱 리스크 검토",
            context="후보 리밸런싱 초안 검토",
            turn_number=1,
            portfolio=_portfolio(),
            candidate_plan={"005930": 0.08, "000660": -0.03},
        )

        self.assertEqual(len(domain_agent.contexts), 1)
        ctx = domain_agent.contexts[0]
        self.assertEqual(
            ctx.proposed_trades,
            [
                {
                    "symbol": "005930",
                    "weight_delta": 0.08,
                    "side": "buy",
                    "delta": 0.08,
                    "action": "BUY",
                },
                {
                    "symbol": "000660",
                    "weight_delta": -0.03,
                    "side": "sell",
                    "delta": -0.03,
                    "action": "SELL",
                },
            ],
        )

    def test_langgraph_domain_agent_receives_backward_compatible_proposed_trades(self) -> None:
        domain_agent = CapturingDomainAgent()
        orchestrator = JudgeOrchestrator(client=FailingChatClient())
        orchestrator.domain_agents = {"risk": domain_agent}

        orchestrator._graph_runtime._execute_agent(
            {
                "query": "후보 리밸런싱 리스크 검토",
                "portfolio": _portfolio().to_dict(),
                "knowledge_base": LocalKnowledgeBase(
                    events=[], documents=[], source_paths={}
                ).to_state_payload(),
                "responses": [],
                "executed_calls": [],
                "called_agents": [],
                "pending_call": {
                    "agent_id": "risk",
                    "query": "후보 리밸런싱 리스크 검토",
                    "context": "도메인 리스크 검토",
                    "depth": "medium",
                },
                "pending_call_layer": "domain",
                "candidate_plan": {"005930": 0.08, "000660": -0.03},
                "turn_number": 1,
            }
        )

        self.assertEqual(len(domain_agent.contexts), 1)
        ctx = domain_agent.contexts[0]
        self.assertEqual(
            ctx.proposed_trades,
            [
                {
                    "symbol": "005930",
                    "weight_delta": 0.08,
                    "side": "buy",
                    "delta": 0.08,
                    "action": "BUY",
                },
                {
                    "symbol": "000660",
                    "weight_delta": -0.03,
                    "side": "sell",
                    "delta": -0.03,
                    "action": "SELL",
                },
            ],
        )

    def test_run_async_handles_existing_event_loop_without_reusing_coroutine(self) -> None:
        async def inner() -> int:
            async def coro() -> int:
                return 7

            return _run_async(coro())

        self.assertEqual(asyncio.run(inner()), 7)

    def test_judge_routing_repairs_invalid_trade_review_without_plan(self) -> None:
        client = RepairingRoutingChatClient()
        orchestrator = JudgeOrchestrator(client=client)

        action = orchestrator._judge_next_action(
            query="포트폴리오 점검",
            portfolio=_portfolio(),
            responses=[],
            called_agents=[],
            depth="medium",
            trigger="pull",
            trigger_event=None,
            candidate_plan=None,
        )

        self.assertEqual(action["action"], "CALL_AGENT")
        self.assertEqual(action["agent_id"], "news")
        self.assertEqual(client.calls, 2)

    def test_judge_routing_stages_large_candidate_plan_for_trade_review(self) -> None:
        orchestrator = JudgeOrchestrator(client=FakeChatClient(_call("profit")))

        action = orchestrator._judge_next_action(
            query="목표비중 이탈 초안의 수익성을 먼저 검토해줘",
            portfolio=_portfolio(),
            responses=[],
            called_agents=[],
            depth="medium",
            trigger="pull",
            trigger_event=None,
            candidate_plan={"005930": -0.15, "000660": 0.15},
        )

        self.assertEqual(action["action"], "CALL_AGENT")
        self.assertEqual(action["agent_id"], "profit")
        self.assertLessEqual(
            sum(abs(delta) for delta in action["candidate_rebalance_plan"].values()), 0.2
        )
        self.assertEqual(action["candidate_rebalance_plan"], {"005930": -0.1, "000660": 0.1})

    def test_direct_indexing_definition_creates_plan_and_trade_reviews(self) -> None:
        orchestrator = JudgeOrchestrator(
            client=RoutingScriptChatClient(
                core_actions=[
                    _call("disclosure", "목표비중 판단 전 공시를 확인합니다."),
                    _call("news", "시장 반응을 확인합니다."),
                    _call("profit", "후보 리밸런싱의 기대수익을 검토합니다."),
                    _call("cost", "후보 리밸런싱의 실행비용을 검토합니다."),
                    _finalize("Core 검토가 끝나 도메인 심의로 이동합니다."),
                ],
                domain_actions=_domain_script(),
                final_payload=_rebalance_final_payload(),
            )
        )
        orchestrator.domain_router = DomainLLMRouter()

        result = orchestrator.run(
            query="초기 설정 목표비중 기준으로 리밸런싱 판단해줘",
            portfolio=_empty_direct_indexing_portfolio(),
            knowledge_base=LocalKnowledgeBase(events=[], documents=[], source_paths={}),
            portfolio_definition=_portfolio_definition(),
            depth="medium",
            trigger="pull",
        )

        decision = result["decision"]

        self.assertEqual(decision["decision"], "REBALANCE")
        self.assertEqual(decision["candidate_rebalance_plan"], {"005930": 0.1, "000660": 0.1})
        self.assertEqual(
            decision["called_agents"],
            [
                "disclosure",
                "news",
                "profit",
                "cost",
                "risk",
                "tax",
                "compliance",
                "macro",
                "sentiment",
                "execution",
                "esg",
            ],
        )
        self.assertIn("direct_indexing", result)
        self.assertEqual(
            result["direct_indexing"]["portfolio_definition"]["name"], "반도체 테스트 포트폴리오"
        )
        self.assertGreater(result["direct_indexing"]["drift_report"]["portfolio_drift_max"], 0.49)

    def test_initial_pull_call_can_start_with_explicit_report_request(self) -> None:
        orchestrator = JudgeOrchestrator(
            client=FakeChatClient(
                {
                    "action": "CALL_AGENT",
                    "agent_id": "report",
                    "reason": "리포트 요청",
                }
            )
        )

        action = orchestrator._normalize_judge_action(
            {
                "action": "CALL_AGENT",
                "agent_id": "report",
                "reason": "리포트 요청",
            },
            query="포트폴리오 보유 종목의 리포트와 컨센서스만 먼저 확인해줘",
            portfolio=_portfolio(),
            responses=[],
            called_agents=[],
            depth="medium",
            trigger="pull",
            trigger_event=None,
            candidate_plan=None,
        )

        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(action["action"], "CALL_AGENT")
        self.assertEqual(action["agent_id"], "report")

    def test_information_subagent_runs_observe_act_observe_tool_loop(self) -> None:
        agent = LLMAgent(agent_id="news", client=FailingChatClient())

        response = agent.run(
            query="삼성전자 관련 뉴스 반응 확인",
            turn_number=1,
            portfolio=_portfolio(),
            knowledge_base=LocalKnowledgeBase(events=[], documents=[], source_paths={}),
            depth="shallow",
        )

        tool_names = [tool.tool_name for tool in response.tools_called]

        self.assertEqual(tool_names[0], "news.observe_request")
        self.assertIn("news.replan_search", tool_names)
        self.assertIn("ingest.refresh_news", tool_names)
        self.assertIn("local_knowledge.portfolio_signal_scan", tool_names)
        self.assertEqual(tool_names[-1], "news.stop")
        self.assertIn("stop_reason=", response.tools_called[-1].summary)

    def test_ingest_refresh_no_live_documents_is_graceful_empty_observation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ingest_root = Path(tmp) / "libra-ingest"
            (ingest_root / "src" / "libra_ingest").mkdir(parents=True)
            (ingest_root / "src" / "libra_ingest" / "ingest_cli.py").write_text(
                "", encoding="utf-8"
            )
            knowledge_base = LocalKnowledgeBase(
                events=[],
                documents=[],
                source_paths={
                    "ingest_refresh_enabled": "true",
                    "ingest_refresh_mode": "live",
                    "ingest_root": str(ingest_root),
                    "ingest_out_dir": str(Path(tmp) / "out"),
                },
            )
            completed = subprocess.CompletedProcess(
                args=["python", "-m", "libra_ingest.ingest_cli"],
                returncode=1,
                stderr="RuntimeError: Live baseline fetch returned no documents.",
            )

            with patch("libra_agent.libra_runtime.subprocess.run", return_value=completed):
                result = knowledge_base.refresh_from_ingest(agent_id="disclosure")

        self.assertFalse(result.changed)
        self.assertIn("새 문서가 없습니다", result.tool_call.summary)
        self.assertNotIn("Traceback", result.tool_call.summary)

    def test_judge_routing_fails_when_llm_is_unavailable(self) -> None:
        orchestrator = JudgeOrchestrator(client=FailingChatClient())

        with self.assertRaises(ChatClientError):
            orchestrator._judge_next_action(
                query="포트폴리오 점검",
                portfolio=_portfolio(),
                responses=[],
                called_agents=[],
                depth="medium",
                trigger="pull",
                trigger_event=None,
                candidate_plan=None,
            )

    def test_domain_routing_finalizes_empty_portfolio_without_llm(self) -> None:
        orchestrator = JudgeOrchestrator(client=FailingChatClient())

        action = orchestrator._domain_next_action(
            query="현재 포트폴리오를 점검하고 유지/조정 필요성을 판단해줘.",
            portfolio=_empty_cash_portfolio(),
            responses=[],
            called_agents=["disclosure", "news"],
            depth="shallow",
            trigger="pull",
            trigger_event=None,
            candidate_plan={},
        )

        self.assertEqual(action["action"], "FINALIZE_DOMAIN_REVIEW")
        self.assertEqual(action["candidate_rebalance_plan"], {})
        self.assertIn("도메인 심의 대상이 없습니다", action["reason"])

    def test_judge_phase_rejects_japanese_kana_payload(self) -> None:
        orchestrator = JudgeOrchestrator(
            client=FakeChatClient(
                {
                    "decision": "DEFER",
                    "summary": "중요 신호が検出されませんでした。",
                    "confidence": 0.8,
                    "urgency": "defer",
                    "reasoning": "추가 분석を行います。",
                    "candidate_rebalance_plan": {},
                    "needs_trade_evaluation": False,
                    "user_notification": {
                        "level": "info",
                        "body": "중요 신호が検出されませんでした。",
                    },
                }
            )
        )

        with self.assertRaises(ChatClientError):
            orchestrator._judge_phase(
                query="포트폴리오 점검",
                portfolio=_portfolio(),
                responses=[],
                stage="final",
            )

    def test_local_knowledge_base_loads_ingest_event_wrapper(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            events_path = Path(tmp_dir) / "events.json"
            events_path.write_text(
                json.dumps(
                    {
                        "events": [
                            {
                                "event_id": "evt_1",
                                "event_type": "EARNINGS",
                                "event_time": "2026-04-26T09:00:00+09:00",
                                "cluster_key": "005930|EARNINGS|2026-04-26",
                                "confidence": 0.8,
                                "headline": "삼성전자 실적 개선",
                                "summary": "삼성전자 메모리 가격 개선으로 실적이 상향됐습니다.",
                                "entities": [
                                    {
                                        "entity_type": "STOCK",
                                        "entity_id": "005930",
                                        "entity_name": "삼성전자",
                                        "ticker": "005930",
                                    }
                                ],
                                "source_documents": ["doc_1"],
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            knowledge_base = LocalKnowledgeBase.from_files(events_path=events_path)

        self.assertEqual(len(knowledge_base.events), 1)
        self.assertEqual(knowledge_base.events[0].event_id, "evt_1")

    def test_local_knowledge_base_matches_normalized_document_entities(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            documents_path = Path(tmp_dir) / "normalized_documents.json"
            documents_path.write_text(
                json.dumps(
                    {
                        "documents": [
                            {
                                "doc_id": "report_1",
                                "doc_type": "REPORT",
                                "source_info": {
                                    "publisher": "증권사",
                                    "source_name": "Hankyung Consensus",
                                    "source_url": "https://example.test/report.pdf",
                                    "region": "KR",
                                },
                                "normalized_content": {
                                    "title": "상상인 Macro Daily",
                                    "body": "반도체 업황과 수급을 점검합니다.",
                                },
                                "timing_info": {
                                    "published_at": "2026-05-15T00:00:00+09:00",
                                },
                                "entities": [
                                    {
                                        "entity_type": "STOCK",
                                        "entity_id": "005930",
                                        "entity_name": "삼성전자",
                                        "ticker": "005930",
                                    }
                                ],
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            knowledge_base = LocalKnowledgeBase.from_files(
                normalized_documents_path=documents_path
            )

        slice_ = knowledge_base.slice_for_agent(
            agent_id="report",
            portfolio=_portfolio(),
            query="삼성전자 리포트 확인",
            depth="shallow",
        )
        self.assertEqual(len(slice_.documents), 1)
        self.assertEqual(slice_.documents[0].matched_holdings, ("005930",))


if __name__ == "__main__":
    unittest.main()
