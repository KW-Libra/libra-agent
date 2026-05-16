from __future__ import annotations

import asyncio
import unittest
from datetime import UTC, datetime

from libra_agent.domain_agents._adapter import (
    domain_verdict_to_agent_response,
    portfolio_snapshot_to_domain_context,
)
from libra_agent.domain_agents.base import AgentVerdict as DomainAgentVerdict
from libra_agent.domain_agents.compliance import ComplianceAgent
from libra_agent.domain_agents.esg_agent import ESGAgent
from libra_agent.domain_agents.execution_agent import ExecutionAgent
from libra_agent.domain_agents.liquidity_agent import LiquidityAgent
from libra_agent.domain_agents.macro_agent import MacroAgent
from libra_agent.domain_agents.risk import RiskAgent
from libra_agent.domain_agents.technical_analysis_agent import TechnicalAnalysisAgent
from libra_agent.libra_models import PortfolioHolding, PortfolioSnapshot


class DomainContextAdapterTests(unittest.TestCase):
    def _snapshot(self) -> PortfolioSnapshot:
        return PortfolioSnapshot(
            generated_at=datetime(2026, 5, 25, 0, 0, tzinfo=UTC),
            total_value_krw=60_000_000,
            cash_weight=0.08,
            user_preferences=(
                "ESG 70점 미만 종목은 편입 전 사용자 승인 필요",
                "approval_mode=manual",
                "cash_min_weight=0.08",
                "excluded_sectors=['gambling', 'tobacco', 'coal', 'controversial_platform']",
                "max_single_weight=0.30",
                "esg_min_score=70",
            ),
            holdings=(
                PortfolioHolding(
                    ticker="035720",
                    company_name="카카오",
                    weight=0.22,
                    sector="인터넷",
                    esg_score=58,
                    shares=293,
                    last_price=45_000,
                    average_price=54_000,
                    market_value_krw=13_200_000,
                ),
            ),
        )

    def test_adapter_preserves_sector_esg_and_structured_preferences(self) -> None:
        ctx = portfolio_snapshot_to_domain_context(self._snapshot(), user_id="test")

        self.assertEqual(ctx.holdings[0]["sector"], "인터넷")
        self.assertEqual(ctx.holdings[0]["esg_score"], 58)
        self.assertEqual(ctx.preferences["esg_min_score"], 70)
        self.assertEqual(ctx.preferences["max_single_weight"], 0.30)
        self.assertEqual(ctx.preferences["cash_weight"], 0.08)
        self.assertIn("controversial_platform", ctx.preferences["esg_exclusions"])

    def test_adapter_derives_risk_first_profile_from_preferences(self) -> None:
        snapshot = PortfolioSnapshot(
            generated_at=datetime(2026, 5, 25, 0, 0, tzinfo=UTC),
            total_value_krw=30_000_000,
            cash_weight=1.0,
            user_preferences=("모의투자 기준", "리스크 우선", "무리한 회전율 회피"),
            holdings=(),
        )

        ctx = portfolio_snapshot_to_domain_context(snapshot, user_id="test")

        self.assertEqual(ctx.preferences["risk_profile"], "risk_first")

    def test_domain_adapter_keeps_abstain_neutral(self) -> None:
        response = domain_verdict_to_agent_response(
            DomainAgentVerdict(
                agent_id="risk",
                vote="abstain",
                confidence=0.65,
                rationale="거래 제안이 없어 abstain 합니다.",
            ),
            agent_id="risk",
            turn_number=1,
            query="테스트",
        )

        self.assertEqual(response.verdict.value, "QUIET")
        self.assertEqual(response.opinion, "NEUTRAL")
        self.assertEqual(response.direction, 0.0)
        self.assertEqual(response.signal_score, 0.0)

    def test_empty_portfolio_domain_agents_abstain_without_action_target(self) -> None:
        snapshot = PortfolioSnapshot(
            generated_at=datetime(2026, 5, 25, 0, 0, tzinfo=UTC),
            total_value_krw=30_000_000,
            cash_weight=1.0,
            user_preferences=("리스크 우선",),
            holdings=(),
        )
        ctx = portfolio_snapshot_to_domain_context(snapshot, user_id="test")

        risk = RiskAgent()
        macro = MacroAgent()
        execution = ExecutionAgent()
        risk._ask_llm = lambda *args, **kwargs: ("LLM 원문", "test-risk")  # type: ignore[method-assign]
        macro._ask_llm = lambda *args, **kwargs: ("LLM 원문", "test-macro")  # type: ignore[method-assign]
        execution._ask_llm = lambda *args, **kwargs: (  # type: ignore[method-assign]
            "LLM 원문",
            "test-execution",
        )

        verdicts = [
            asyncio.run(risk.deliberate(ctx)),
            asyncio.run(macro.deliberate(ctx)),
            asyncio.run(ComplianceAgent().deliberate(ctx)),
            asyncio.run(execution.deliberate(ctx)),
        ]

        self.assertEqual([item.vote for item in verdicts], ["abstain"] * 4)

    def test_compliance_rejects_policy_violating_holding(self) -> None:
        ctx = portfolio_snapshot_to_domain_context(self._snapshot(), user_id="test")

        verdict = asyncio.run(ComplianceAgent().deliberate(ctx))

        self.assertEqual(verdict.vote, "reject")
        self.assertIn("035720 ESG score 58.0", verdict.rationale)

    def test_esg_rejects_below_target_holding(self) -> None:
        ctx = portfolio_snapshot_to_domain_context(self._snapshot(), user_id="test")

        verdict = asyncio.run(ESGAgent().deliberate(ctx))

        self.assertEqual(verdict.vote, "reject")
        self.assertIn("ESG", verdict.rationale)

    def test_liquidity_rejects_wide_spread_or_low_free_float(self) -> None:
        ctx = portfolio_snapshot_to_domain_context(self._snapshot(), user_id="test")
        ctx.holdings[0]["bid_ask_spread_bps"] = 75
        ctx.holdings[0]["free_float_ratio_pct"] = 12

        verdict = asyncio.run(LiquidityAgent().deliberate(ctx))

        self.assertEqual(verdict.vote, "reject")
        self.assertIn("스프레드", verdict.rationale)

    def test_technical_analysis_abstains_without_price_history(self) -> None:
        ctx = portfolio_snapshot_to_domain_context(self._snapshot(), user_id="test")

        verdict = asyncio.run(TechnicalAnalysisAgent().deliberate(ctx))

        self.assertEqual(verdict.vote, "abstain")
        self.assertIn("OHLCV", verdict.rationale)


if __name__ == "__main__":
    unittest.main()
