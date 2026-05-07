from __future__ import annotations

import unittest

from libra_agent.libra.agents.cost_agent import CostAgent
from libra_agent.libra.agents.news_agent import NewsAgent
from libra_agent.libra.agents.profit_agent import ProfitAgent
from libra_agent.libra_models import PortfolioSnapshot, Urgency


class FakeChatClient:
    model = "fake-model"

    def chat_json(self, **_: object) -> dict[str, object]:
        return {}


class StaticKnowledgeBase:
    def ticker_signal(self, ticker: str, portfolio: PortfolioSnapshot) -> float:
        del portfolio
        return {"005930": 0.5, "000660": -0.2}.get(ticker, 0.0)


def _portfolio() -> PortfolioSnapshot:
    return PortfolioSnapshot.from_dict(
        {
            "generated_at": "2026-04-28T09:00:00+09:00",
            "total_value_krw": 100000000,
            "holdings": [
                {"ticker": "005930", "company_name": "삼성전자", "weight": 0.6},
                {"ticker": "000660", "company_name": "SK하이닉스", "weight": 0.4},
            ],
            "cash_weight": 0.0,
        }
    )


class LibraIndividualAgentTests(unittest.TestCase):
    def test_news_agent_can_shape_request_in_own_file(self) -> None:
        agent = NewsAgent(client=FakeChatClient())

        request = agent.prepare_request(
            query="뉴스 확인",
            context="직전 공시 요약",
            fallback="빈손이면 시장 반응 없음으로 반환",
            note="Judge가 공시 이후 반응 확인을 요청함",
            turn_number=2,
            portfolio=_portfolio(),
            knowledge_base=object(),
            depth="medium",
        )

        self.assertEqual(request.query, "뉴스 확인")
        self.assertEqual(request.context, "직전 공시 요약")
        self.assertEqual(request.fallback, "빈손이면 시장 반응 없음으로 반환")
        self.assertEqual(request.depth, "medium")
        self.assertIn("Judge가 공시 이후 반응 확인을 요청함", request.note or "")
        self.assertIn("Agent owner task:", request.note or "")

    def test_profit_agent_owns_plan_simulation(self) -> None:
        response = ProfitAgent().run(
            query="수익성 검토",
            turn_number=3,
            portfolio=_portfolio(),
            knowledge_base=StaticKnowledgeBase(),
            rebalance_plan={"005930": 0.05, "000660": -0.02},
        )

        self.assertEqual(response.agent_id, "profit")
        self.assertGreater(response.direction, 0.0)
        self.assertEqual(response.evidence["mode"], "plan_simulation")
        self.assertEqual(response.tools_called[0].tool_name, "local_profit.heuristic_plan_simulation")
        self.assertEqual(response.focus_tickers, ["000660", "005930"])

    def test_cost_agent_owns_trade_friction_estimate(self) -> None:
        response = CostAgent().run(
            query="거래비용 검토",
            turn_number=4,
            portfolio=_portfolio(),
            rebalance_plan={"005930": -0.05, "000660": 0.05},
        )

        self.assertEqual(response.agent_id, "cost")
        self.assertEqual(response.direction, 0.0)
        self.assertEqual(response.evidence["mode"], "trade_cost")
        self.assertEqual(response.urgency, Urgency.WATCH)
        self.assertEqual(response.tools_called[0].tool_name, "local_cost.heuristic_trade_cost")


if __name__ == "__main__":
    unittest.main()
