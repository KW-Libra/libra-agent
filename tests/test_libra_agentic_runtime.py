from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from libra_agent.libra.constraints import validate_rebalance_plan
from libra_agent.libra.evaluation import evaluate_decision_outcome
from libra_agent.libra.signals import infer_signal_profile
from libra_agent.libra_runtime import JudgeOrchestrator, LocalKnowledgeBase
from libra_agent.libra_models import AgentResponse, AgentVerdict, PortfolioSnapshot, Urgency
from libra_agent.utils import contains_japanese_kana


class FakeChatClient:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.model = "fake-model"

    def chat_json(self, **_: object) -> dict[str, object]:
        return dict(self.payload)


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
