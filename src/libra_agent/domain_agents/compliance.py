"""
Compliance Agent — IPS (Investment Policy Statement) guardian.

Checks that every proposed trade respects the user's mandate:
- ESG exclusions (sector/security screens)
- Risk profile bounds
- Approval-mode settings
"""

from __future__ import annotations

from typing import Any

from .base import AgentVerdict, BaseAgent, PortfolioContext

# Sector → KIS market codes mapping (simplified)
SECTOR_EXCLUSION_CODES: dict[str, list[str]] = {
    "tobacco": ["KT&G", "BAT", "PMI"],
    "defense": ["LIG넥스원", "한화에어로스페이스", "한국항공우주"],
    "fossil": ["S-Oil", "GS칼텍스", "한국가스공사"],
    "gambling": ["GKL", "파라다이스", "강원랜드"],
}


def _as_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    try:
        return [str(item) for item in value if str(item).strip()]
    except TypeError:
        return [str(value)]


def _matches_exclusion(item: dict[str, Any], exclusions: list[str]) -> str | None:
    searchable = " ".join(
        str(item.get(key, "") or "") for key in ("symbol", "name", "sector")
    ).lower()
    mapped_symbols: set[str] = set()
    for exclusion in exclusions:
        lowered = exclusion.lower()
        if lowered and lowered in searchable:
            return exclusion
        mapped_symbols.update(SECTOR_EXCLUSION_CODES.get(lowered, []))
    symbol = str(item.get("symbol", "") or "")
    name = str(item.get("name", "") or "")
    if symbol in mapped_symbols or name in mapped_symbols:
        return "mapped_exclusion"
    return None


class ComplianceAgent(BaseAgent):
    agent_id = "compliance"
    name = "Clarke"
    role = "Compliance Officer"

    async def deliberate(self, ctx: PortfolioContext) -> AgentVerdict:
        preferences = ctx.preferences
        exclusions = _as_list(
            preferences.get("exclusions")
            or preferences.get("excluded_sectors")
            or preferences.get("esg_exclusions")
        )
        approval_mode = str(
            preferences.get("approval") or preferences.get("approval_mode") or "manual"
        )
        max_single_weight = _as_float(preferences.get("max_single_weight"), 0.0) or 0.0
        cash_min_weight = _as_float(preferences.get("cash_min_weight"), 0.0) or 0.0
        cash_weight = _as_float(preferences.get("cash_weight"), 0.0) or 0.0
        esg_min_score = _as_float(preferences.get("esg_min_score"), None)

        violations: list[str] = []

        for holding in ctx.holdings:
            symbol = str(holding.get("symbol") or "?")
            weight = _as_float(holding.get("weight"), 0.0) or 0.0
            if max_single_weight and weight > max_single_weight + 1e-9:
                violations.append(
                    f"{symbol} weight {weight:.1%} exceeds max_single_weight {max_single_weight:.1%}."
                )
            exclusion = _matches_exclusion(holding, exclusions)
            if exclusion:
                violations.append(f"{symbol} violates exclusion screen '{exclusion}'.")
            esg_score = _as_float(holding.get("esg_score"), None)
            if esg_min_score is not None and esg_score is not None and esg_score < esg_min_score:
                violations.append(
                    f"{symbol} ESG score {esg_score:.1f} is below esg_min_score {esg_min_score:.1f}."
                )

        if cash_min_weight and cash_weight < cash_min_weight - 1e-9:
            violations.append(
                f"cash weight {cash_weight:.1%} is below cash_min_weight {cash_min_weight:.1%}."
            )

        for trade in ctx.proposed_trades:
            sym = str(trade.get("symbol", "") or "")
            holding = next(
                (item for item in ctx.holdings if item.get("symbol") == sym), {"symbol": sym}
            )
            exclusion = _matches_exclusion(holding, exclusions)
            if exclusion:
                violations.append(f"Trade {sym} violates exclusion screen '{exclusion}'.")

        if violations:
            return AgentVerdict(
                agent_id=self.agent_id,
                vote="reject",
                confidence=1.0,
                rationale=" | ".join(violations),
                signals=[
                    {
                        "label": "IPS violations",
                        "value": str(len(violations)),
                        "details": violations,
                    }
                ],
            )

        if not ctx.holdings and not ctx.proposed_trades:
            return AgentVerdict(
                agent_id=self.agent_id,
                vote="abstain",
                confidence=0.80,
                rationale=(
                    "보유 종목과 제안 거래가 없어 IPS 위반은 없지만 승인할 투자 행동도 없습니다. "
                    f"abstain - approval_mode={approval_mode}, 0 violations."
                ),
                signals=[{"label": "IPS violations", "value": "0"}],
            )

        return AgentVerdict(
            agent_id=self.agent_id,
            vote="approve",
            confidence=0.99,
            rationale=(
                f"All {len(ctx.proposed_trades)} proposed trades and "
                f"{len(ctx.holdings)} holdings pass IPS check. "
                f"approval_mode={approval_mode}. 0 violations."
            ),
            signals=[{"label": "IPS violations", "value": "0"}],
        )
