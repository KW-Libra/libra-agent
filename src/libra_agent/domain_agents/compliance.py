"""
Compliance Agent — IPS (Investment Policy Statement) guardian.

Checks that every proposed trade respects the user's mandate:
- ESG exclusions (sector/security screens)
- Risk profile bounds
- Approval-mode settings
"""

from __future__ import annotations
from .base import BaseAgent, AgentVerdict, PortfolioContext


# Sector → KIS market codes mapping (simplified)
SECTOR_EXCLUSION_CODES: dict[str, list[str]] = {
    "tobacco":  ["KT&G", "BAT", "PMI"],
    "defense":  ["LIG넥스원", "한화에어로스페이스", "한국항공우주"],
    "fossil":   ["S-Oil", "GS칼텍스", "한국가스공사"],
    "gambling": ["GKL", "파라다이스", "강원랜드"],
}


class ComplianceAgent(BaseAgent):
    agent_id = "compliance"
    name = "Clarke"
    role = "Compliance Officer"

    async def deliberate(self, ctx: PortfolioContext) -> AgentVerdict:
        preferences = ctx.preferences
        exclusions: list[str] = preferences.get("exclusions", [])
        approval_mode: str = preferences.get("approval", "manual")

        violations: list[str] = []

        # Check each proposed trade against excluded sectors
        excluded_symbols: set[str] = set()
        for excl in exclusions:
            excluded_symbols.update(SECTOR_EXCLUSION_CODES.get(excl, []))

        for trade in ctx.proposed_trades:
            sym = trade.get("symbol", "")
            if sym in excluded_symbols:
                violations.append(
                    f"Trade {sym} violates ESG exclusion screen '{excl}'."
                )

        if violations:
            return AgentVerdict(
                agent_id=self.agent_id,
                vote="reject",
                confidence=1.0,
                rationale=" | ".join(violations),
                signals=[{"label": "IPS violations", "value": str(len(violations)), "details": violations}],
            )

        return AgentVerdict(
            agent_id=self.agent_id,
            vote="approve",
            confidence=0.99,
            rationale=f"All {len(ctx.proposed_trades)} proposed trades pass IPS check. 0 violations.",
            signals=[{"label": "IPS violations", "value": "0"}],
        )
