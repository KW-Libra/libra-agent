from __future__ import annotations

import unittest

from libra_agent.errors import ChatClientError
from libra_agent.libra.agents.disclosure_agent import DisclosureAgent
from libra_agent.libra.prompts import DISCLOSURE_PROMPT_PROFILE
from libra_agent.libra_models import AgentVerdict, PortfolioSnapshot, Urgency
from libra_agent.libra_runtime import JudgeOrchestrator, LLMAgent, LocalKnowledgeBase
from libra_agent.libra_validation import sanitize_judge_payload
from libra_agent.runtime.debate_events import debate_event_publisher


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


def _portfolio() -> PortfolioSnapshot:
    return PortfolioSnapshot.from_dict(
        {
            "generated_at": "2026-04-14T09:00:00+09:00",
            "holdings": [
                {
                    "ticker": "005930",
                    "company_name": "삼성전자",
                    "weight": 0.4,
                    "aliases": ["Samsung Electronics", "005930.KS"],
                }
            ],
            "cash_weight": 0.0,
            "user_preferences": ["국내 대형주 중심"],
        }
    )


def _empty_portfolio() -> PortfolioSnapshot:
    return PortfolioSnapshot.from_dict(
        {
            "generated_at": "2026-05-16T12:00:00+09:00",
            "holdings": [],
            "total_value_krw": 30_000_000,
            "cash_weight": 1.0,
            "user_preferences": ["모의투자 기준", "리스크 우선"],
        }
    )


def _knowledge_base_with_disclosure() -> LocalKnowledgeBase:
    return LocalKnowledgeBase.from_state_payload(
        {
            "events": [],
            "documents": [
                {
                    "doc_id": "doc-1",
                    "doc_type": "DISCLOSURE",
                    "title": "분기보고서",
                    "body": "삼성전자 분기 실적 공시",
                    "publisher": "DART",
                    "source_name": "dart",
                    "source_url": "https://example.test/doc-1",
                    "region": "KR",
                    "published_at": "2026-04-02T09:00:00+09:00",
                    "matched_holdings": ["005930"],
                }
            ],
            "source_paths": {
                "normalized_documents": "examples/normalized-documents.sample.json",
            },
        }
    )


class LibraSchemaValidationTests(unittest.TestCase):
    def test_disclosure_agent_exposes_prompt_profile(self) -> None:
        self.assertIs(DisclosureAgent.prompt_profile, DISCLOSURE_PROMPT_PROFILE)
        self.assertEqual(DisclosureAgent.prompt_profile.agent_id, "disclosure")

    def test_llm_agent_sanitizes_disclosure_payload(self) -> None:
        agent = LLMAgent(
            agent_id="disclosure",
            client=FakeChatClient(
                {
                    "verdict": "PARTIAL_ANSWER",
                    "evidence": {
                        "found_count": "1",
                        "items": [
                            {
                                "ticker": ["Samsung Electronics"],
                                "type": "분기보고서",
                                "summary": "실적 공시 확인",
                                "date": "20260402",
                            }
                        ],
                        "upcoming_disclosures": [
                            {
                                "ticker": "005930.KS",
                                "title": "실적 발표 예정",
                                "date": "202 a.m. KST",
                            }
                        ],
                    },
                    "direction": "0.35",
                    "strength": "1.7",
                    "urgency": "scheduled",
                    "confidence": "0.91",
                    "reasoning_for_judge_agent": "  삼성전자 실적 공시가 확인되었습니다.  ",
                    "limits_acknowledged": True,
                    "references": [
                        {
                            "agent_id": "news",
                            "opinion_id": "news_1",
                            "relation": "supports",
                        }
                    ],
                    "focus_tickers": ["Samsung Electronics", "UNKNOWN"],
                }
            ),
        )

        response = agent.run(
            query="삼성전자 공시를 요약해줘.",
            turn_number=1,
            portfolio=_portfolio(),
            knowledge_base=_knowledge_base_with_disclosure(),
            depth="shallow",
        )

        self.assertEqual(response.verdict, AgentVerdict.PARTIAL_ANSWER)
        self.assertEqual(response.evidence["items"][0]["ticker"], "005930")
        self.assertEqual(response.evidence["upcoming_disclosures"][0]["ticker"], "005930")
        self.assertIsNone(response.evidence["upcoming_disclosures"][0]["date"])
        self.assertEqual(response.focus_tickers, ["005930"])
        self.assertEqual(response.strength, 1.0)
        self.assertIsNone(response.limits_acknowledged)

    def test_llm_agent_repairs_zero_confidence_when_evidence_is_cited(self) -> None:
        agent = LLMAgent(
            agent_id="disclosure",
            client=FakeChatClient(
                {
                    "verdict": "PARTIAL_ANSWER",
                    "evidence": {
                        "found_count": 1,
                        "items": [
                            {
                                "ticker": "005930",
                                "type": "분기보고서",
                                "summary": "삼성전자 분기보고서 공시가 확인되었습니다.",
                                "date": "20260402",
                            }
                        ],
                    },
                    "direction": 0.0,
                    "strength": 0.0,
                    "urgency": "defer",
                    "confidence": 0.0,
                    "reasoning_for_judge_agent": "삼성전자 분기보고서 공시가 확인되었지만 방향성 판단은 보류합니다.",
                    "limits_acknowledged": "공시 본문만 확인되어 투자 방향성은 제한적입니다.",
                    "references": [],
                    "focus_tickers": ["005930"],
                }
            ),
        )

        response = agent.run(
            query="삼성전자 공시를 요약해줘.",
            turn_number=1,
            portfolio=_portfolio(),
            knowledge_base=_knowledge_base_with_disclosure(),
            depth="shallow",
        )

        self.assertEqual(response.verdict, AgentVerdict.PARTIAL_ANSWER)
        self.assertEqual(response.confidence, 0.35)

    def test_llm_agent_falls_back_to_local_evidence_when_llm_fails(self) -> None:
        events: list[tuple[str, dict[str, object]]] = []
        token = debate_event_publisher.set(lambda event, payload: events.append((event, payload)))
        try:
            response = LLMAgent(agent_id="disclosure", client=FailingChatClient()).run(
                query="삼성전자 공시를 요약해줘.",
                turn_number=1,
                portfolio=_portfolio(),
                knowledge_base=_knowledge_base_with_disclosure(),
                depth="shallow",
            )
        finally:
            debate_event_publisher.reset(token)

        self.assertEqual(response.verdict, AgentVerdict.PARTIAL_ANSWER)
        self.assertGreaterEqual(response.confidence, 0.2)
        self.assertEqual(response.evidence["found_count"], 1)
        self.assertIn("자동 요약", response.reasoning_for_judge_agent)
        self.assertTrue(
            any(
                event == "llm_error" and payload.get("phase") == "agent_response"
                for event, payload in events
            )
        )
        self.assertTrue(
            any(
                event == "llm_skipped" and payload.get("phase") == "agent_response_fallback"
                for event, payload in events
            )
        )

    def test_judge_payload_sanitizer_normalizes_defaults(self) -> None:
        payload = sanitize_judge_payload(
            {
                "decision": "rebalance",
                "summary": "",
                "confidence": "1.7",
                "urgency": "later",
                "reasoning": "",
                "candidate_rebalance_plan": {
                    "Samsung Electronics": "0.05",
                    "BAD": "0.9",
                },
                "needs_trade_evaluation": "false",
                "follow_up_at": "not a date",
                "feedback_checkpoint": "2026-04-21T09:00:00+09:00",
                "user_notification": {
                    "level": "popup",
                    "body": "",
                    "action_required": "0",
                    "sent_at": "bad",
                },
                "options": [1, "권고안 승인", "권고안 승인"],
                "auto_safeguards": {
                    "tripwire_1": "  test  ",
                },
            },
            portfolio=_portfolio(),
            stage="final",
        )

        self.assertEqual(payload["decision"], "REBALANCE")
        self.assertEqual(payload["candidate_rebalance_plan"], {"005930": 0.05})
        self.assertEqual(payload["urgency"], Urgency.SCHEDULED.value)
        self.assertEqual(payload["confidence"], 1.0)
        self.assertTrue(payload["needs_trade_evaluation"])
        self.assertIsNone(payload["follow_up_at"])
        self.assertIsNotNone(payload["feedback_checkpoint"])
        self.assertEqual(payload["user_notification"]["level"], "info")
        self.assertEqual(payload["user_notification"]["body"], payload["summary"])
        self.assertEqual(payload["options"], ["1", "권고안 승인"])

    def test_judge_payload_sanitizer_demotes_empty_portfolio_user_decision(self) -> None:
        payload = sanitize_judge_payload(
            {
                "decision": "USER_DECISION_REQUIRED",
                "summary": "초기 포트폴리오 후보가 필요합니다.",
                "confidence": 0.95,
                "urgency": "watch",
                "reasoning": "보유 종목과 후보 리밸런싱 초안이 없습니다.",
                "candidate_rebalance_plan": {},
                "needs_trade_evaluation": True,
                "feedback_checkpoint": "2026-05-17T09:00:00+09:00",
                "user_notification": {
                    "level": "push",
                    "body": "승인이 필요합니다.",
                    "action_required": True,
                },
                "options": ["승인", "거절"],
            },
            portfolio=_empty_portfolio(),
            stage="final",
        )

        self.assertEqual(payload["decision"], "DEFER")
        self.assertEqual(payload["urgency"], "defer")
        self.assertEqual(payload["candidate_rebalance_plan"], {})
        self.assertFalse(payload["needs_trade_evaluation"])
        self.assertIsNone(payload["feedback_checkpoint"])
        self.assertEqual(payload["user_notification"]["level"], "info")
        self.assertFalse(payload["user_notification"]["action_required"])
        self.assertEqual(payload["options"], [])

    def test_judge_phase_publishes_sanitized_empty_portfolio_decision(self) -> None:
        events: list[tuple[str, dict[str, object]]] = []
        token = debate_event_publisher.set(lambda event, payload: events.append((event, payload)))
        try:
            payload = JudgeOrchestrator(
                client=FakeChatClient(
                    {
                        "decision": "USER_DECISION_REQUIRED",
                        "summary": "초기 포트폴리오 후보가 필요합니다.",
                        "confidence": 0.95,
                        "urgency": "watch",
                        "reasoning": "보유 종목과 후보 리밸런싱 초안이 없습니다.",
                        "candidate_rebalance_plan": {},
                        "needs_trade_evaluation": False,
                        "user_notification": {"level": "push", "action_required": True},
                    }
                )
            )._judge_phase(
                query="현재 포트폴리오를 점검해줘.",
                portfolio=_empty_portfolio(),
                responses=[],
                stage="final",
                candidate_plan={},
            )
        finally:
            debate_event_publisher.reset(token)

        self.assertEqual(payload["decision"], "DEFER")
        response_events = [
            item
            for item in events
            if item[0] == "llm_response" and item[1].get("phase") == "final_decision_final"
        ]
        self.assertTrue(response_events)
        output = response_events[-1][1]["output"]
        self.assertIsInstance(output, dict)
        self.assertEqual(output["decision"], "DEFER")

    def test_empty_local_context_uses_consistent_empty_schema(self) -> None:
        empty_knowledge = LocalKnowledgeBase.from_state_payload(
            {
                "events": [],
                "documents": [],
                "source_paths": {},
            }
        )
        agent = LLMAgent(
            agent_id="disclosure",
            client=FakeChatClient({}),
        )

        response = agent.run(
            query="공시 확인",
            turn_number=1,
            portfolio=_portfolio(),
            knowledge_base=empty_knowledge,
            depth="medium",
        )

        self.assertEqual(response.verdict, AgentVerdict.DIRECT_ANSWER_UNAVAILABLE)
        self.assertEqual(
            response.evidence,
            {
                "found_count": 0,
                "items": [],
                "upcoming_disclosures": [],
            },
        )

    def test_judge_action_normalization_replaces_garbage_call_text(self) -> None:
        orchestrator = JudgeOrchestrator(client=FakeChatClient({}))
        disclosure_response = LLMAgent(
            agent_id="disclosure",
            client=FakeChatClient(
                {
                    "verdict": "PARTIAL_ANSWER",
                    "evidence": {
                        "found_count": 1,
                        "items": [
                            {
                                "ticker": "005930",
                                "type": "분기보고서",
                                "summary": "실적 공시 확인",
                                "date": "20260402",
                            }
                        ],
                    },
                    "direction": 0.1,
                    "strength": 0.5,
                    "urgency": "scheduled",
                    "confidence": 0.8,
                    "reasoning_for_judge_agent": "삼성전자 실적 공시가 확인되었습니다.",
                }
            ),
        ).run(
            query="공시 확인",
            turn_number=1,
            portfolio=_portfolio(),
            knowledge_base=_knowledge_base_with_disclosure(),
            depth="shallow",
        )

        normalized = orchestrator._normalize_judge_action(
            {
                "action": "CALL_AGENT",
                "agent_id": "news",
                "query": "日本語 mixed ???",
                "context": "{'ticker_group':['005930','00066 a']}",
                "depth": "deep",
                "fallback": "bad fallback",
                "note": "삼성전자의 분기보고서 공시가 기술주 비중 조절의 트리ガーとなるため",
            },
            query="포트폴리오 점검",
            portfolio=_portfolio(),
            responses=[disclosure_response],
            called_agents=["disclosure"],
            depth="shallow",
            trigger="pull",
            trigger_event=None,
            candidate_plan=None,
        )

        self.assertIsNotNone(normalized)
        assert normalized is not None
        self.assertEqual(normalized["agent_id"], "news")
        self.assertEqual(
            normalized["query"],
            "최근 공시 이후 시장 반응과 관련 뉴스, 필요시 매크로 배경을 요약해줘.",
        )
        self.assertIn("직전 disclosure 관찰", normalized["context"])
        self.assertNotIn("00066 a", normalized["context"])
        self.assertEqual(
            normalized["fallback"],
            "시장 반응, 교차 확인 여부, 투자 가정 변화 여부를 우선 정리해줘.",
        )
        self.assertEqual(
            normalized["note"],
            "판단 에이전트는 공시 내용이 시장 시각을 바꿨는지, 이미 가격에 반영됐는지 확인합니다.",
        )

    def test_judge_action_normalization_accepts_class_style_agent_ids(self) -> None:
        orchestrator = JudgeOrchestrator(client=FakeChatClient({}))

        normalized = orchestrator._normalize_judge_action(
            {
                "action": "CALL_AGENT",
                "agent_id": "DisclosureAgent",
                "reason": "최신 공시 확인",
                "query": "보유 종목 공시 확인",
                "depth": "medium",
            },
            query="포트폴리오 점검",
            portfolio=_portfolio(),
            responses=[],
            called_agents=[],
            depth="medium",
            trigger="pull",
            trigger_event=None,
            candidate_plan=None,
        )

        self.assertIsNotNone(normalized)
        assert normalized is not None
        self.assertEqual(normalized["agent_id"], "disclosure")


if __name__ == "__main__":
    unittest.main()
