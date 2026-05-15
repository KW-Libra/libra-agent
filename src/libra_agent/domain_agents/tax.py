"""
Tax Agent (Reed) — tax-loss harvesting specialist.

Responsibilities:
- Identifies loss-harvesting pairs (similar factor exposure, different security)
- Estimates wash-sale risk (30-day window)
- Quantifies tax-alpha from harvesting
- Votes on proposals, suggests tax-optimised execution order
"""

from __future__ import annotations

import json

from .base import AgentVerdict, BaseAgent, PortfolioContext


class TaxAgent(BaseAgent):
    agent_id = "tax"
    name = "Reed"
    role = "Tax Strategist"

    # Minimum unrealised loss to bother harvesting (KRW)
    MIN_HARVEST_KRW = 100_000

    async def deliberate(self, ctx: PortfolioContext) -> AgentVerdict:
        # Identify holdings with unrealised losses
        losers = [
            h
            for h in ctx.holdings
            if (h.get("current_price", 0) - h.get("average_price", 0)) * h.get("quantity", 0)
            < -self.MIN_HARVEST_KRW
        ]

        signals = [
            {
                "label": "Harvestable lots",
                "value": str(len(losers)),
                "symbols": [h["symbol"] for h in losers[:5]],
            }
        ]

        summary = json.dumps(
            {
                "losers": losers[:10],
                "proposed_trades": ctx.proposed_trades,
                "user_tax_bracket": ctx.preferences.get("tax_bracket", "unknown"),
            },
            ensure_ascii=False,
            indent=2,
        )

        rationale, model_used = self._ask_llm(
            system=(
                "You are Reed, a tax-loss harvesting specialist. "
                "Review the portfolio losers and proposed trades. "
                "In 1–2 sentences: are there harvesting opportunities in this proposal? "
                "Estimate the tax-alpha in basis points if possible."
            ),
            user=summary,
            ctx=ctx,
        )

        vote = "approve" if len(losers) > 0 else "abstain"

        return AgentVerdict(
            agent_id=self.agent_id,
            vote=vote,
            confidence=0.80,
            rationale=rationale,
            signals=signals,
            llm_used=model_used,
        )
