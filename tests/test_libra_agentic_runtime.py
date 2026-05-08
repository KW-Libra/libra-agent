from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("LIBRA_DOMAIN_AGENTS_ENABLED", "true")

from libra_agent.errors import ChatClientError
from libra_agent.libra.constraints import validate_rebalance_plan
from libra_agent.libra.direct_indexing import PortfolioDefinition
from libra_agent.libra.evaluation import evaluate_decision_outcome
from libra_agent.libra.signals import infer_signal_profile
from libra_agent.libra_runtime import JudgeOrchestrator, LLMAgent, LocalKnowledgeBase
from libra_agent.libra_models import AgentResponse, AgentVerdict, PortfolioSnapshot, Urgency
from libra_agent.utils import contains_japanese_kana


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

    def test_empty_regular_scan_finalizes_without_report_collection(self) -> None:
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

        self.assertEqual(action["action"], "FINALIZE")
        self.assertEqual(action["candidate_rebalance_plan"], {})

    def test_langgraph_trace_starts_with_judge_and_interleaves_routing(self) -> None:
        orchestrator = JudgeOrchestrator(client=FailingChatClient())

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

    def test_direct_indexing_definition_creates_plan_and_trade_reviews(self) -> None:
        orchestrator = JudgeOrchestrator(client=FailingChatClient())

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
        self.assertEqual(result["direct_indexing"]["portfolio_definition"]["name"], "반도체 테스트 포트폴리오")
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
            (ingest_root / "src" / "libra_ingest" / "ingest_cli.py").write_text("", encoding="utf-8")
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

    def test_judge_phase_falls_back_on_japanese_kana(self) -> None:
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
                    "user_notification": {"level": "info", "body": "중요 신호が検出されませんでした。"},
                }
            )
        )

        payload = orchestrator._judge_phase(
            query="포트폴리오 점검",
            portfolio=_portfolio(),
            responses=[],
            stage="final",
        )

        self.assertFalse(contains_japanese_kana(payload["summary"]))
        self.assertFalse(contains_japanese_kana(payload["reasoning"]))

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


if __name__ == "__main__":
    unittest.main()
