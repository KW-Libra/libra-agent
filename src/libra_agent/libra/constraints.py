from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from ..libra_models import PortfolioSnapshot


@dataclass(frozen=True, slots=True)
class TickerConstraint:
    minimum: float = 0.0
    maximum: float = 1.0


@dataclass(frozen=True, slots=True)
class PortfolioConstraintSet:
    ticker_constraints: Mapping[str, TickerConstraint] = field(default_factory=dict)
    max_trade_per_day: float = 0.20


@dataclass(frozen=True, slots=True)
class ConstraintCheck:
    passed: bool
    reason: str = "PASS"
    adjusted_plan: dict[str, float] = field(default_factory=dict)


def default_constraints_for(portfolio: PortfolioSnapshot) -> PortfolioConstraintSet:
    return PortfolioConstraintSet(
        ticker_constraints={
            holding.ticker: TickerConstraint(minimum=0.0, maximum=1.0)
            for holding in portfolio.holdings
        },
        max_trade_per_day=0.20,
    )


def validate_rebalance_plan(
    *,
    portfolio: PortfolioSnapshot,
    plan: Mapping[str, float],
    constraints: PortfolioConstraintSet | None = None,
) -> ConstraintCheck:
    constraint_set = constraints or default_constraints_for(portfolio)
    current_weights = {holding.ticker: holding.weight for holding in portfolio.holdings}
    adjusted_plan: dict[str, float] = {}

    for ticker, delta in plan.items():
        if ticker not in current_weights:
            return ConstraintCheck(False, f"{ticker} is not in the current portfolio.", dict(plan))
        ticker_constraint = constraint_set.ticker_constraints.get(ticker, TickerConstraint())
        new_weight = current_weights[ticker] + float(delta)
        if new_weight < ticker_constraint.minimum:
            return ConstraintCheck(
                False,
                f"{ticker} would fall below minimum weight {ticker_constraint.minimum:.4f}.",
                dict(plan),
            )
        if new_weight > ticker_constraint.maximum:
            return ConstraintCheck(
                False,
                f"{ticker} would exceed maximum weight {ticker_constraint.maximum:.4f}.",
                dict(plan),
            )
        adjusted_plan[ticker] = round(float(delta), 4)

    total_trade = sum(abs(delta) for delta in adjusted_plan.values())
    if total_trade > constraint_set.max_trade_per_day:
        return ConstraintCheck(
            False,
            f"Total daily trade weight {total_trade:.4f} exceeds max {constraint_set.max_trade_per_day:.4f}.",
            adjusted_plan,
        )
    return ConstraintCheck(True, "PASS", adjusted_plan)
