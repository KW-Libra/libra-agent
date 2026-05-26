from __future__ import annotations

from typing import Any

from libra_agent.libra.governance_config import load_governance_config
from libra_agent.libra_models import (
    AgentResponse,
    AgentVerdict,
    PortfolioSnapshot,
    ToolCall,
    Urgency,
)
from libra_agent.utils import stable_hash


class ProfitAgent:
    agent_id = "profit"
    owner_scope = "Profit Agent"

    def run(
        self,
        *,
        query: str,
        turn_number: int,
        portfolio: PortfolioSnapshot,
        knowledge_base: Any,
        rebalance_plan: dict[str, float],
    ) -> AgentResponse:
        del query
        cfg = load_governance_config()
        signals: dict[str, float] = {}
        gross_change = 0.0
        plan_score = 0.0
        for ticker, delta in rebalance_plan.items():
            signal = float(knowledge_base.ticker_signal(ticker, portfolio))
            signals[ticker] = signal
            gross_change += abs(delta)
            plan_score += delta * signal

        expected_return_1m = plan_score * 40.0
        expected_return_3m = plan_score * 65.0
        sharpe_ratio = plan_score / max(0.05, gross_change * 0.75)
        max_drawdown = -1.0 * ((gross_change * 10.0) + max(0.0, -plan_score * 35.0))
        confidence = _clamp(
            cfg.profit_confidence_base + (len(signals) * cfg.profit_confidence_per_signal),
            0.0,
            cfg.profit_confidence_max,
        )
        recommendation = (
            "Heuristic v1 simulation suggests modest upside relative to trade size."
            if plan_score >= 0
            else "Heuristic v1 simulation suggests the proposed trade is paying for weak expected follow-through."
        )
        opinion_id = f"profit_{stable_hash({'turn': turn_number, 'plan': rebalance_plan})[:12]}"
        return AgentResponse(
            agent_id=self.agent_id,
            opinion_id=opinion_id,
            turn_number=turn_number,
            query_understood="Evaluate the proposed rebalance plan with a heuristic local simulator.",
            verdict=AgentVerdict.DIRECT_ANSWER,
            evidence={
                "mode": "plan_simulation",
                "plan_simulation": {
                    "rebalance_plan": rebalance_plan,
                    "ticker_signals": signals,
                    "expected_return_1m": round(expected_return_1m, 3),
                    "expected_return_3m": round(expected_return_3m, 3),
                    "sharpe_ratio": round(sharpe_ratio, 3),
                    "max_drawdown": round(max_drawdown, 3),
                    "recommendation_text": recommendation,
                },
            },
            direction=_clamp(plan_score * cfg.profit_direction_scale, -1.0, 1.0),
            strength=_clamp(gross_change * 4.0, 0.0, 1.0),
            urgency=Urgency.SCHEDULED,
            confidence=confidence,
            reasoning_for_judge_agent=(
                "This is a static-rule simulator. Use it as a relative check on whether the candidate plan "
                "improves expected follow-through."
            ),
            limits_acknowledged="Profit uses a heuristic local signal model, not a calibrated Monte Carlo engine.",
            tools_called=[
                ToolCall(
                    tool_name="local_profit.heuristic_plan_simulation",
                    purpose="Estimate directional payoff of the candidate rebalance plan",
                    summary=f"Computed heuristic returns for {len(rebalance_plan)} planned position changes.",
                )
            ],
            depth_used="medium",
            focus_tickers=sorted(rebalance_plan),
        )


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))
