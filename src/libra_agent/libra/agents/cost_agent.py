from __future__ import annotations

from libra_agent.libra_models import (
    AgentResponse,
    AgentVerdict,
    PortfolioSnapshot,
    ToolCall,
    Urgency,
)
from libra_agent.utils import stable_hash


class CostAgent:
    agent_id = "cost"
    owner_scope = "Cost Agent"

    def run(
        self,
        *,
        query: str,
        turn_number: int,
        portfolio: PortfolioSnapshot,
        rebalance_plan: dict[str, float],
    ) -> AgentResponse:
        del query
        gross_change = sum(abs(delta) for delta in rebalance_plan.values())
        reference_value = portfolio.total_value_krw or 100000000.0
        traded_notional = reference_value * gross_change
        commission_bp = 1.5
        sell_tax_bp = 18.0
        slippage_bp = 6.0 + (gross_change * 100.0)
        spread_bp = 3.0 + (gross_change * 40.0)
        sells_notional = reference_value * sum(
            abs(delta) for delta in rebalance_plan.values() if delta < 0
        )
        commission_krw = traded_notional * (commission_bp / 10000.0)
        tax_krw = sells_notional * (sell_tax_bp / 10000.0)
        total_friction_bp = (
            commission_bp + slippage_bp + spread_bp + (sell_tax_bp if sells_notional else 0.0)
        )
        opinion_id = f"cost_{stable_hash({'turn': turn_number, 'plan': rebalance_plan})[:12]}"
        return AgentResponse(
            agent_id=self.agent_id,
            opinion_id=opinion_id,
            turn_number=turn_number,
            query_understood="Estimate the execution friction of the candidate rebalance plan.",
            verdict=AgentVerdict.DIRECT_ANSWER,
            evidence={
                "mode": "trade_cost",
                "trade_cost": {
                    "rebalance_plan": rebalance_plan,
                    "commission_krw": round(commission_krw, 0),
                    "tax_krw": round(tax_krw, 0),
                    "estimated_slippage_bp": round(slippage_bp, 3),
                    "spread_state_bp": round(spread_bp, 3),
                    "total_friction_bp": round(total_friction_bp, 3),
                },
            },
            direction=0.0,
            strength=0.0,
            urgency=Urgency.WATCH if total_friction_bp >= 30.0 else Urgency.SCHEDULED,
            confidence=0.58,
            reasoning_for_judge_agent=(
                "Execution friction is manageable for small reallocations but rises quickly with gross turnover."
            ),
            limits_acknowledged="Cost uses configurable heuristic basis-point defaults, not live broker fees or orderbook snapshots.",
            tools_called=[
                ToolCall(
                    tool_name="local_cost.heuristic_trade_cost",
                    purpose="Estimate commission, tax, spread, and slippage",
                    summary=f"Estimated costs for gross turnover {gross_change:.3f} using local heuristic defaults.",
                )
            ],
            depth_used="medium",
            focus_tickers=sorted(rebalance_plan),
        )
