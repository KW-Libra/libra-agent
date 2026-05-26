"""Week 2 Phase B — FinalJudge 비대칭 라우팅 + Risk vol targeting + DD trigger.

근거: [[16]] §1, §4 — v2 백테스트 평탄화 / MDD -34% 학계 예측 -26% 타깃.
"""

from __future__ import annotations

import asyncio
import unittest
from unittest import mock

from libra_agent.domain_agents._services.portfolio_optimizer import RiskMetrics
from libra_agent.domain_agents.base import PortfolioContext
from libra_agent.domain_agents.risk import RiskAgent
from libra_agent.libra.mediator import classify_branch
from libra_agent.libra.schemas import Direction, Vote


def _votes(*specs: tuple[Direction, float, float]) -> list[Vote]:
    """Make eligible (non-informational) votes for subject 'X'.

    specs: (direction, magnitude_pct, confidence)
    """
    return [Vote("X", d, m, c) for d, m, c in specs]


class AsymmetricClassifyBranchTests(unittest.TestCase):
    """classify_branch regime-aware threshold.

    neutral → 회귀 0 (default 0.6); bear → SELL 임계 0.45, BUY 임계 0.75.
    """

    def test_neutral_regression_strong_consensus_default_threshold(self) -> None:
        # signed = -0.62 > default 0.6 abs → STRONG (neutral 회귀 확인)
        votes = _votes(
            (Direction.DECREASE, -5.0, 0.9),
            (Direction.DECREASE, -5.0, 0.9),
            (Direction.HOLD, 0.0, 0.5),
        )
        with mock.patch.dict("os.environ", {"LIBRA_REGIME": "neutral"}, clear=False):
            branch = classify_branch(votes)
        self.assertEqual(branch.value, "STRONG_CONSENSUS")

    def test_neutral_mid_score_is_weak(self) -> None:
        # signed ≈ -0.533 < 0.6 abs default, > 0.3 weak → WEAK_CONSENSUS
        votes = _votes(
            (Direction.DECREASE, -5.0, 0.4),
            (Direction.DECREASE, -5.0, 0.4),
            (Direction.HOLD, 0.0, 0.7),
        )
        with mock.patch.dict("os.environ", {"LIBRA_REGIME": "neutral"}, clear=False):
            branch = classify_branch(votes)
        self.assertEqual(branch.value, "WEAK_CONSENSUS")

    def test_bear_regime_sell_strong_at_lower_threshold(self) -> None:
        # signed ≈ -0.5 in bear → STRONG (since bear_sell_strong = 0.45)
        # In neutral, same votes would be WEAK (|.5| < .6)
        votes = _votes(
            (Direction.DECREASE, -5.0, 0.5),
            (Direction.DECREASE, -5.0, 0.5),
            (Direction.HOLD, 0.0, 0.5),
        )
        # signed = (-1*0.5 + -1*0.5 + 0*0.5) / 1.5 ≈ -0.667 > 0.45 bear → STRONG
        # Make it specifically test boundary by using lower confidence on bear side
        boundary = _votes(
            (Direction.DECREASE, -5.0, 0.4),
            (Direction.DECREASE, -5.0, 0.4),
            (Direction.HOLD, 0.0, 0.7),
        )
        # signed = -0.8 / 1.5 ≈ -0.533 → abs 0.533; bear 0.45 → STRONG; neutral 0.6 → WEAK
        with mock.patch.dict("os.environ", {"LIBRA_REGIME": "bear"}, clear=False):
            bear_branch = classify_branch(boundary)
        with mock.patch.dict("os.environ", {"LIBRA_REGIME": "neutral"}, clear=False):
            neutral_branch = classify_branch(boundary)
        self.assertEqual(bear_branch.value, "STRONG_CONSENSUS")
        self.assertEqual(neutral_branch.value, "WEAK_CONSENSUS")

    def test_bear_regime_buy_requires_higher_threshold(self) -> None:
        # signed ≈ +0.6 in bear should NOT be STRONG (bear_buy=0.75) but IS STRONG in neutral
        votes = _votes(
            (Direction.INCREASE, 5.0, 0.5),
            (Direction.INCREASE, 5.0, 0.5),
            (Direction.HOLD, 0.0, 0.5),
        )
        # signed = +1.0/1.5 ≈ +0.667; bear 0.75 → NOT strong; neutral 0.6 → STRONG
        with mock.patch.dict("os.environ", {"LIBRA_REGIME": "bear"}, clear=False):
            bear_branch = classify_branch(votes)
        with mock.patch.dict("os.environ", {"LIBRA_REGIME": "neutral"}, clear=False):
            neutral_branch = classify_branch(votes)
        self.assertEqual(bear_branch.value, "WEAK_CONSENSUS")
        self.assertEqual(neutral_branch.value, "STRONG_CONSENSUS")

    def test_bull_regime_mirror_of_bear(self) -> None:
        # signed ≈ +0.53 in bull (buy threshold 0.45) → STRONG; neutral 0.6 → WEAK
        votes = _votes(
            (Direction.INCREASE, 5.0, 0.4),
            (Direction.INCREASE, 5.0, 0.4),
            (Direction.HOLD, 0.0, 0.7),
        )
        with mock.patch.dict("os.environ", {"LIBRA_REGIME": "bull"}, clear=False):
            bull_branch = classify_branch(votes)
        with mock.patch.dict("os.environ", {"LIBRA_REGIME": "neutral"}, clear=False):
            neutral_branch = classify_branch(votes)
        self.assertEqual(bull_branch.value, "STRONG_CONSENSUS")
        self.assertEqual(neutral_branch.value, "WEAK_CONSENSUS")


class RiskVolTargetingTests(unittest.TestCase):
    """RiskAgent vol target 18% + DD -15% trigger ([[16]] §1)."""

    def _ctx(self, *, proposed_trades: list[dict] | None = None) -> PortfolioContext:
        return PortfolioContext(
            user_id="u1",
            holdings=[
                {"symbol": "069500", "weight": 0.30, "quantity": 10},
                {"symbol": "153130", "weight": 0.20, "quantity": 5},
                {"symbol": "379800", "weight": 0.25, "quantity": 8},
                {"symbol": "132030", "weight": 0.15, "quantity": 3},
                {"symbol": "133690", "weight": 0.10, "quantity": 2},
            ],
            preferences={"risk_profile": "balanced"},
            total_value=10_000_000,
            proposed_trades=proposed_trades or [],
        )

    def _patch_llm(self) -> mock._patch:
        return mock.patch.object(
            RiskAgent, "_ask_llm", return_value=("mock rationale", "mock-model")
        )

    def _mock_metrics(self, *, vol: float, mdd: float) -> RiskMetrics:
        return RiskMetrics(
            var_95=50_000.0,  # 0.5% of 10M, under VAR_95_LIMIT_PCT (3%)
            var_99=80_000.0,
            cvar_95=80_000.0,
            mdd=mdd,
            volatility=vol,
            tracking_error=0.0,
            hhi=0.2,
            beta=1.0,
        )

    def _patch_optimizer(self, metrics: RiskMetrics) -> mock._patch:
        opt = mock.MagicMock()
        opt.compute_risk_metrics.return_value = metrics
        return mock.patch(
            "libra_agent.domain_agents.risk.get_optimizer", return_value=opt
        )

    def test_vol_breach_with_buy_triggers_reject(self) -> None:
        ctx = self._ctx(proposed_trades=[{"symbol": "069500", "delta": 0.05}])
        # Provide returns_data so risk_metrics path activates (min 30 days)
        ctx.returns_data = {h["symbol"]: [0.001] * 35 for h in ctx.holdings}
        metrics = self._mock_metrics(vol=0.25, mdd=-0.05)  # vol > 0.18, mdd safe
        with self._patch_llm(), self._patch_optimizer(metrics):
            verdict = asyncio.run(RiskAgent().deliberate(ctx))
        self.assertEqual(verdict.vote, "reject")
        self.assertIn("변동성", verdict.rationale)

    def test_dd_breach_overrides_vol_breach(self) -> None:
        ctx = self._ctx(proposed_trades=[{"symbol": "069500", "delta": 0.05}])
        ctx.returns_data = {h["symbol"]: [0.001] * 35 for h in ctx.holdings}
        metrics = self._mock_metrics(vol=0.25, mdd=-0.20)  # both breached, dd wins
        with self._patch_llm(), self._patch_optimizer(metrics):
            verdict = asyncio.run(RiskAgent().deliberate(ctx))
        self.assertEqual(verdict.vote, "reject")
        self.assertIn("drawdown", verdict.rationale)
        self.assertGreaterEqual(verdict.confidence, 0.80)

    def test_vol_breach_without_buy_does_not_reject(self) -> None:
        # Only SELL proposed → vol breach should not flip to reject
        ctx = self._ctx(proposed_trades=[{"symbol": "069500", "delta": -0.05}])
        ctx.returns_data = {h["symbol"]: [0.001] * 35 for h in ctx.holdings}
        metrics = self._mock_metrics(vol=0.25, mdd=-0.05)
        with self._patch_llm(), self._patch_optimizer(metrics):
            verdict = asyncio.run(RiskAgent().deliberate(ctx))
        self.assertNotEqual(verdict.vote, "reject")

    def test_safe_metrics_no_breach(self) -> None:
        ctx = self._ctx(proposed_trades=[{"symbol": "069500", "delta": 0.05}])
        ctx.returns_data = {h["symbol"]: [0.001] * 35 for h in ctx.holdings}
        metrics = self._mock_metrics(vol=0.10, mdd=-0.05)
        with self._patch_llm(), self._patch_optimizer(metrics):
            verdict = asyncio.run(RiskAgent().deliberate(ctx))
        self.assertEqual(verdict.vote, "approve")


if __name__ == "__main__":
    unittest.main()
