from __future__ import annotations

from typing import Any

from .base import AgentVerdict, BaseAgent, PortfolioContext


class LiquidityAgent(BaseAgent):
    """Market liquidity and tradability review.

    This adapts the Team B LiquidityAgent into the official domain-agent
    protocol. It stays deterministic when live order-book/ADV data is missing
    and avoids owning execution timing, which remains ExecutionAgent's scope.
    """

    agent_id = "liquidity"
    name = "LiquidityAgent"
    role = "Liquidity Analyst"

    async def deliberate(self, ctx: PortfolioContext) -> AgentVerdict:
        signals: list[dict[str, Any]] = []
        rejecting_reasons: list[str] = []
        observed = 0

        for holding in ctx.holdings:
            symbol = str(holding.get("symbol") or holding.get("ticker") or "").strip()
            if not symbol:
                continue
            adv_krw = _as_float(
                holding.get("avg_daily_turnover_krw")
                or holding.get("average_daily_turnover_krw")
                or holding.get("adv_krw")
            )
            spread_bps = _as_float(holding.get("bid_ask_spread_bps"))
            free_float_pct = _as_float(holding.get("free_float_ratio_pct"))
            market_value = _as_float(holding.get("market_value"))

            if adv_krw > 0:
                observed += 1
                adv_usage = market_value / adv_krw if market_value > 0 else 0.0
                signals.append(
                    {
                        "label": f"{symbol} ADV 사용률",
                        "value": round(adv_usage, 4),
                        "threshold": 0.1,
                        "breached": adv_usage > 0.1,
                    }
                )
                if adv_usage > 0.1:
                    rejecting_reasons.append(
                        f"{symbol}: 보유/거래 규모가 ADV 10%를 초과할 수 있습니다."
                    )

            if spread_bps > 0:
                observed += 1
                signals.append(
                    {
                        "label": f"{symbol} bid-ask spread",
                        "value": spread_bps,
                        "threshold": 50.0,
                        "breached": spread_bps > 50.0,
                    }
                )
                if spread_bps > 50.0:
                    rejecting_reasons.append(f"{symbol}: 호가 스프레드가 50bp를 초과합니다.")

            if free_float_pct > 0:
                observed += 1
                signals.append(
                    {
                        "label": f"{symbol} free float",
                        "value": free_float_pct,
                        "threshold": 20.0,
                        "breached": free_float_pct < 20.0,
                    }
                )
                if free_float_pct < 20.0:
                    rejecting_reasons.append(f"{symbol}: 유통주식 비율이 20% 미만입니다.")

        if rejecting_reasons:
            return AgentVerdict(
                agent_id=self.agent_id,
                vote="reject",
                confidence=0.78,
                rationale=" ".join(rejecting_reasons[:3]),
                signals=signals,
                llm_used="deterministic-liquidity",
            )

        if observed == 0:
            return AgentVerdict(
                agent_id=self.agent_id,
                vote="abstain",
                confidence=0.42,
                rationale="ADV, 호가 스프레드, 유통주식 비율 데이터가 없어 유동성 판단을 보류합니다.",
                signals=signals,
                llm_used="deterministic-liquidity",
            )

        return AgentVerdict(
            agent_id=self.agent_id,
            vote="approve",
            confidence=0.66,
            rationale="확인 가능한 유동성 지표에서 즉시 차단할 만한 제약은 감지되지 않았습니다.",
            signals=signals,
            llm_used="deterministic-liquidity",
        )


def _as_float(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        return float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return 0.0
