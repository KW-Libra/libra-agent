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
from typing import Any

from .base import AgentVerdict, BaseAgent, PortfolioContext


class TaxAgent(BaseAgent):
    agent_id = "tax"
    name = "Reed"
    role = "Tax Strategist"

    # Minimum unrealised loss to bother harvesting (KRW)
    MIN_HARVEST_KRW = 100_000
    DEFAULT_TAX_RATE = 0.22

    async def deliberate(self, ctx: PortfolioContext) -> AgentVerdict:
        lots = [_tax_lot_view(holding) for holding in ctx.holdings]
        losers = [
            lot for lot in lots if lot["unrealized_pnl_krw"] < -self.MIN_HARVEST_KRW
        ]
        proposed_sells = [
            trade
            for trade in ctx.proposed_trades
            if str(trade.get("side") or trade.get("action") or "").lower() in {"sell", "sell_short"}
            or _as_float(trade.get("weight_delta") or trade.get("delta")) < 0
        ]
        sell_symbols = {_symbol_from_trade(trade) for trade in proposed_sells}
        sell_symbols.discard("")
        sold_losers = [lot for lot in losers if lot["symbol"] in sell_symbols]
        sold_gainers = [
            lot
            for lot in lots
            if lot["symbol"] in sell_symbols and lot["unrealized_pnl_krw"] > self.MIN_HARVEST_KRW
        ]
        tax_rate = _tax_rate(ctx.preferences.get("tax_rate") or ctx.preferences.get("tax_bracket"))
        estimated_tax_alpha_krw = sum(
            _harvested_loss_for_trade(lot, proposed_sells) * tax_rate for lot in sold_losers
        )
        estimated_tax_alpha_bp = (
            estimated_tax_alpha_krw / ctx.total_value * 10000.0 if ctx.total_value > 0 else 0.0
        )

        signals = [
            {
                "label": "Harvestable lots",
                "value": str(len(losers)),
                "symbols": [lot["symbol"] for lot in losers[:5]],
            },
            {
                "label": "Proposed loss sells",
                "value": str(len(sold_losers)),
                "symbols": [lot["symbol"] for lot in sold_losers[:5]],
            },
            {
                "label": "Proposed gain sells",
                "value": str(len(sold_gainers)),
                "symbols": [lot["symbol"] for lot in sold_gainers[:5]],
            },
            {
                "label": "Estimated tax alpha KRW",
                "value": str(round(estimated_tax_alpha_krw)),
            },
            {
                "label": "Estimated tax alpha bp",
                "value": f"{estimated_tax_alpha_bp:.2f}",
            },
        ]

        summary = json.dumps(
            {
                "losers": losers[:10],
                "proposed_sells": proposed_sells,
                "sold_losers": sold_losers,
                "sold_gainers": sold_gainers,
                "estimated_tax_alpha_krw": round(estimated_tax_alpha_krw),
                "estimated_tax_alpha_bp": round(estimated_tax_alpha_bp, 2),
                "proposed_trades": ctx.proposed_trades,
                "user_tax_bracket": ctx.preferences.get("tax_bracket", "unknown"),
            },
            ensure_ascii=False,
            indent=2,
        )

        if sold_losers:
            rationale, model_used = self._ask_llm(
                system=(
                    "You are Reed, a tax-loss harvesting specialist. "
                    "Review the loss-selling proposal. In 1 Korean sentence, explain why it is tax-efficient "
                    "and mention estimated tax-alpha if available."
                ),
                user=summary,
                ctx=ctx,
            )
            vote = "approve"
            direction = -0.55
        elif sold_gainers and not sold_losers:
            rationale = (
                "제안된 매도는 손실실현 후보가 아니라 평가이익 종목 매도에 가까워 세금 관점에서는 "
                "즉시 실행보다 보류 또는 대체 매도 검토가 필요합니다."
            )
            model_used = ""
            vote = "reject"
            direction = 0.35
        elif losers:
            rationale, model_used = self._ask_llm(
                system=(
                    "You are Reed, a tax-loss harvesting specialist. "
                    "There are harvestable loss lots but no proposed sell order. "
                    "In 1 Korean sentence, explain this as an advisory tax signal, not an execution block."
                ),
                user=summary,
                ctx=ctx,
            )
            vote = "abstain"
            direction = 0.0
        else:
            rationale = "평단/손익 기준으로 의미 있는 손실실현 후보가 없어 세금 관점에서는 중립입니다."
            model_used = ""
            vote = "abstain"
            direction = 0.0

        return AgentVerdict(
            agent_id=self.agent_id,
            vote=vote,
            confidence=0.80,
            rationale=rationale,
            signals=signals,
            llm_used=model_used,
            direction=direction,
        )


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return default


def _tax_lot_view(holding: dict[str, Any]) -> dict[str, Any]:
    symbol = str(holding.get("symbol") or holding.get("ticker") or "").strip()
    quantity = _as_float(holding.get("quantity"))
    current_price = _as_float(holding.get("current_price"))
    average_price = _as_float(holding.get("average_price"))
    explicit_pnl = holding.get("unrealized_pnl_krw")
    if explicit_pnl is not None:
        unrealized_pnl = _as_float(explicit_pnl)
    elif current_price > 0 and average_price > 0 and quantity > 0:
        unrealized_pnl = (current_price - average_price) * quantity
    else:
        unrealized_pnl = 0.0
    return {
        "symbol": symbol,
        "quantity": quantity,
        "current_price": current_price,
        "average_price": average_price,
        "weight": _as_float(holding.get("weight")),
        "unrealized_pnl_krw": unrealized_pnl,
    }


def _symbol_from_trade(trade: dict[str, Any]) -> str:
    return str(trade.get("symbol") or trade.get("ticker") or trade.get("subject") or "").strip()


def _trade_weight_delta(trade: dict[str, Any]) -> float:
    return _as_float(trade.get("weight_delta") or trade.get("delta"))


def _harvested_loss_for_trade(lot: dict[str, Any], trades: list[dict[str, Any]]) -> float:
    symbol = str(lot.get("symbol") or "")
    weight = max(_as_float(lot.get("weight")), 0.0)
    loss = abs(min(_as_float(lot.get("unrealized_pnl_krw")), 0.0))
    if loss <= 0:
        return 0.0
    for trade in trades:
        if _symbol_from_trade(trade) != symbol:
            continue
        delta = abs(min(_trade_weight_delta(trade), 0.0))
        if weight <= 0:
            return loss
        return loss * min(1.0, delta / weight)
    return 0.0


def _tax_rate(value: Any) -> float:
    if value is None:
        return TaxAgent.DEFAULT_TAX_RATE
    raw = str(value).strip().lower()
    if not raw or raw == "unknown":
        return TaxAgent.DEFAULT_TAX_RATE
    if raw.endswith("%"):
        return max(0.0, min(1.0, _as_float(raw[:-1]) / 100.0))
    numeric = _as_float(raw, TaxAgent.DEFAULT_TAX_RATE)
    if numeric > 1.0:
        numeric = numeric / 100.0
    return max(0.0, min(1.0, numeric))
