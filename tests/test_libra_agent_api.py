from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from libra_agent.libra.agents import EvaluationAgent
from libra_agent.libra_api import (
    _attach_governance_v1,
    _build_knowledge_base,
    _governance_v1_execution_mode,
    app,
)
from libra_agent.libra_models import PortfolioSnapshot


class LibraAgentApiTests(unittest.TestCase):
    def test_health(self) -> None:
        client = TestClient(app)

        response = client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_judge_run_requires_knowledge_input(self) -> None:
        client = TestClient(app)

        response = client.post(
            "/v1/judge-runs",
            json={
                "query": "포트폴리오 점검",
                "portfolio": {
                    "generated_at": "2026-04-25T00:00:00+09:00",
                    "holdings": [
                        {
                            "ticker": "005930",
                            "company_name": "삼성전자",
                            "weight": 0.5,
                        }
                    ],
                },
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("knowledge_base", response.json()["detail"])

    def test_knowledge_base_accepts_ingest_refresh_options(self) -> None:
        knowledge_base = _build_knowledge_base(
            {
                "knowledge_base": {
                    "events": [],
                    "documents": [],
                    "source_paths": {},
                },
                "allow_ingest_refresh": True,
                "ingest_refresh": {
                    "mode": "live",
                    "root": "D:\\libra-ingest",
                    "out_dir": "D:\\libra-data\\knowledge\\agent_refresh",
                    "rss_limit": 1,
                },
            }
        )

        self.assertEqual(knowledge_base.source_paths["ingest_refresh_enabled"], "true")
        self.assertEqual(knowledge_base.source_paths["ingest_refresh_mode"], "live")
        self.assertEqual(knowledge_base.source_paths["ingest_rss_limit"], "1")

    def test_attach_governance_v1_adds_compliance_branch(self) -> None:
        portfolio = PortfolioSnapshot.from_dict(
            {
                "generated_at": "2026-05-10T09:00:00+09:00",
                "holdings": [
                    {
                        "ticker": "005930",
                        "company_name": "삼성전자",
                        "weight": 0.42,
                        "sector": "반도체",
                    }
                ],
                "cash_weight": 0.02,
                "user_preferences": [
                    "max_single_weight=0.3",
                    "cash_min_weight=0.05",
                    "excluded_sectors=[]",
                ],
            }
        )
        result = {
            "agent_responses": [
                {
                    "agent_id": "risk",
                    "opinion_id": "risk_1",
                    "turn_number": 1,
                    "query_understood": "위험 검토",
                    "verdict": "PARTIAL_ANSWER",
                    "evidence": {},
                    "direction": -0.8,
                    "strength": 0.8,
                    "urgency": "defer",
                    "confidence": 0.9,
                    "reasoning_for_judge_agent": "단일 종목 비중이 높습니다.",
                    "focus_tickers": ["005930"],
                }
            ],
            "decision": {"decision": "HOLD"},
        }

        updated = _attach_governance_v1(result, payload={}, portfolio=portfolio)

        self.assertIn("governance_v1", updated)
        final = updated["governance_v1"]["final_decision"]
        self.assertEqual(final["decision"], "USER_DECISION_REQUIRED")
        self.assertEqual(final["branch"], "COMPLIANCE_VETO")

    def test_governance_v1_primary_mode_is_opt_in(self) -> None:
        self.assertEqual(_governance_v1_execution_mode({}), "attach")
        self.assertEqual(
            _governance_v1_execution_mode({"governance_v1": {"execution_mode": "primary"}}),
            "primary",
        )

    def test_evaluation_endpoint_scores_stored_decision_result(self) -> None:
        client = TestClient(app)

        response = client.post(
            "/v1/evaluations",
            json={
                "horizon": "1w",
                "realized_return_pct": -7.0,
                "cost_pct": 0.1,
                "user_feedback": "rejected: 단기 노이즈",
                "decision_run_result": {
                    "agent_responses": [
                        {
                            "agent_id": "news",
                            "signal_score": -0.5,
                        }
                    ],
                    "decision": {
                        "decision": "REBALANCE",
                        "candidate_rebalance_plan": {
                            "005930": -0.05,
                        },
                    },
                },
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["agent_id"], "evaluation")
        self.assertEqual(payload["verdict"], "USER_WRONG")
        self.assertTrue(payload["direction_accuracy"])
        self.assertEqual(payload["metrics"]["signal_score"], -0.5)

    def test_evaluation_endpoint_accepts_direct_decision_payload(self) -> None:
        client = TestClient(app)

        response = client.post(
            "/v1/evaluations",
            json={
                "horizon": "1w",
                "realized_return_pct": -3.5,
                "cost_pct": 0.1,
                "user_feedback": "rejected: 테스트",
                "signal_score": -0.5,
                "decision": {
                    "decision": "REBALANCE",
                    "candidate_rebalance_plan": {
                        "005930": -0.05,
                    },
                },
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["metrics"]["decision"], "REBALANCE")
        self.assertEqual(payload["verdict"], "USER_WRONG")

    def test_evaluation_agent_is_explicit_subagent(self) -> None:
        result = EvaluationAgent().run(
            decision="HOLD",
            rebalance_plan={},
            signal_score=0.0,
            user_feedback=None,
            realized_return_pct=0.8,
            cost_pct=0.0,
            horizon="1w",
        )

        self.assertEqual(result["agent_id"], "evaluation")
        self.assertEqual(result["horizon"], "1w")
        self.assertEqual(result["verdict"], "HOLD_CORRECT")


if __name__ == "__main__":
    unittest.main()
