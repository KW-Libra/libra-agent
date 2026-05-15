from __future__ import annotations

import asyncio
import unittest
from datetime import UTC, datetime

from libra_agent.domain_agents._adapter import portfolio_snapshot_to_domain_context
from libra_agent.domain_agents.compliance import ComplianceAgent
from libra_agent.domain_agents.esg_agent import ESGAgent
from libra_agent.domain_agents.liquidity_agent import LiquidityAgent
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
