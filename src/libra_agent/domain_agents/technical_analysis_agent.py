from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from .base import AgentVerdict, BaseAgent, PortfolioContext


class TechnicalAnalysisAgent(BaseAgent):
    """Momentum and chart-signal review.

    This preserves the Team B technical-analysis role without adding a pandas
    runtime dependency. It consumes either per-holding `ohlcv` rows or
    `returns_data` injected into `PortfolioContext`.
    """

    agent_id = "technical"
    name = "TechnicalAnalysisAgent"
    role = "Technical Analyst"

    async def deliberate(self, ctx: PortfolioContext) -> AgentVerdict:
        signals: list[dict[str, Any]] = []
        scores: list[float] = []

        for holding in ctx.holdings:
            symbol = str(holding.get("symbol") or holding.get("ticker") or "").strip()
            if not symbol:
                continue
            rows = _extract_ohlcv(holding.get("ohlcv"))
            if rows:
                score, row_signals = _score_ohlcv(symbol=symbol, rows=rows)
                scores.append(score)
                signals.extend(row_signals)
                continue

            returns = []
            if isinstance(ctx.returns_data, Mapping):
                raw_returns = ctx.returns_data.get(symbol)
                if isinstance(raw_returns, list):
                    returns = [_as_float(item) for item in raw_returns]
            if returns:
                score = _score_returns(returns)
                scores.append(score)
                signals.append(
                    {
                        "label": f"{symbol} return momentum",
                        "value": round(score, 4),
                        "threshold": 0.0,
                        "breached": score < -0.05,
                    }
                )

        if not scores:
            return AgentVerdict(
                agent_id=self.agent_id,
                vote="abstain",
                confidence=0.4,
                rationale="OHLCV 또는 수익률 히스토리가 없어 기술적 분석 판단을 보류합니다.",
                signals=signals,
                llm_used="deterministic-technical",
            )

        aggregate = sum(scores) / len(scores)
        if aggregate <= -0.08:
            vote = "reject"
            confidence = 0.72
            rationale = "단기/중기 모멘텀 지표가 약세로 기울어 자동 증액 판단을 보수적으로 봅니다."
        elif aggregate >= 0.08:
            vote = "approve"
            confidence = 0.68
            rationale = "확인 가능한 가격 모멘텀이 양호해 기술적 관점에서 차단 신호는 낮습니다."
        else:
            vote = "abstain"
            confidence = 0.55
            rationale = "기술적 지표가 혼조라 독립적인 방향성 판단은 보류합니다."

        return AgentVerdict(
            agent_id=self.agent_id,
            vote=vote,
            confidence=confidence,
            rationale=rationale,
            signals=signals,
            llm_used="deterministic-technical",
        )


def _extract_ohlcv(value: Any) -> list[dict[str, float]]:
    if not isinstance(value, list):
        return []
    rows: list[dict[str, float]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        close = _as_float(item.get("close"))
        volume = _as_float(item.get("volume"))
        if close <= 0:
            continue
        rows.append(
            {
                "close": close,
                "volume": volume,
                "high": _as_float(item.get("high")) or close,
                "low": _as_float(item.get("low")) or close,
            }
        )
    return rows


def _score_ohlcv(
    *, symbol: str, rows: list[dict[str, float]]
) -> tuple[float, list[dict[str, Any]]]:
    closes = [row["close"] for row in rows]
    volumes = [row["volume"] for row in rows if row["volume"] > 0]
    current = closes[-1]
    score = 0.0
    signals: list[dict[str, Any]] = []

    if len(closes) >= 20:
        ma20 = sum(closes[-20:]) / 20
        ma_score = (current - ma20) / ma20 if ma20 > 0 else 0.0
        score += _clamp(ma_score, -0.15, 0.15)
        signals.append(
            {
                "label": f"{symbol} price vs MA20",
                "value": round(ma_score, 4),
                "threshold": 0.0,
                "breached": ma_score < -0.05,
            }
        )

    if len(closes) >= 60:
        ma60 = sum(closes[-60:]) / 60
        ma_score = (current - ma60) / ma60 if ma60 > 0 else 0.0
        score += _clamp(ma_score, -0.15, 0.15)
        signals.append(
            {
                "label": f"{symbol} price vs MA60",
                "value": round(ma_score, 4),
                "threshold": 0.0,
                "breached": ma_score < -0.08,
            }
        )

    if len(closes) >= 15:
        rsi = _rsi(closes[-15:])
        rsi_score = (rsi - 50.0) / 100.0
        score += _clamp(rsi_score, -0.2, 0.2)
        signals.append(
            {
                "label": f"{symbol} RSI14",
                "value": round(rsi, 2),
                "threshold": 30.0,
                "breached": rsi < 30.0,
            }
        )

    if len(volumes) >= 20:
        avg_volume = sum(volumes[-20:]) / 20
        ratio = volumes[-1] / avg_volume if avg_volume > 0 else 1.0
        if ratio > 1.5 and len(closes) >= 2 and closes[-1] > closes[-2]:
            score += 0.04
        signals.append(
            {
                "label": f"{symbol} volume ratio 20d",
                "value": round(ratio, 3),
                "threshold": 1.5,
                "breached": False,
            }
        )

    return _clamp(score, -1.0, 1.0), signals


def _score_returns(values: list[float]) -> float:
    if not values:
        return 0.0
    compounded = 1.0
    for value in values[-60:]:
        compounded *= 1.0 + value
    return _clamp(compounded - 1.0, -1.0, 1.0)


def _rsi(closes: list[float]) -> float:
    gains = 0.0
    losses = 0.0
    for prev, cur in zip(closes, closes[1:], strict=False):
        diff = cur - prev
        if diff >= 0:
            gains += diff
        else:
            losses += abs(diff)
    if math.isclose(losses, 0.0):
        return 100.0
    rs = gains / losses
    return 100.0 - (100.0 / (1.0 + rs))


def _as_float(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        return float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return 0.0


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
