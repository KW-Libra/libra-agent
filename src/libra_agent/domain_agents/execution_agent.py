"""
Execution Agent (Tien) — 최적 체결 전략가

Claude Haiku 활용 이유:
  - 정형적인 수치 계산 + 체결 계획 생성 → Haiku로 충분
  - Almgren-Chriss 수식은 Python으로 직접 계산 (LLM 불필요)
  - LLM은 체결 전략 설명 텍스트 생성에만 사용

담당 역할:
  - 거래 규모 vs 시장 유동성 비교 (ADV 참여율)
  - Almgren-Chriss 시장충격 비용 계산
  - TWAP/VWAP/IS(Implementation Shortfall) 전략 선택
  - 체결 순서 최적화 (매수/매도 순서, 분할 체결 계획)

수식:
  시장충격 = η × (거래량/ADV) × σ × P  (일시적)
             + γ × (거래량/ADV) × σ × P  (영구적)
  참여율 = 거래량 / ADV × 100%
  허용 최대 참여율: 20% (시장충격 최소화)
"""

from __future__ import annotations

import json
import logging

from .base import BaseAgent, AgentVerdict, PortfolioContext
from ._services.portfolio_optimizer import get_optimizer

logger = logging.getLogger(__name__)


class ExecutionAgent(BaseAgent):
    agent_id = "execution"
    name = "Tien"
    role = "Execution Strategist"

    MAX_PARTICIPATION_RATE = 0.20     # 20% 이상 참여 시 시장충격 경고
    MAX_IMPACT_BPS         = 30.0     # 30bps 이상 비용 시 거부

    async def deliberate(self, ctx: PortfolioContext) -> AgentVerdict:
        optimizer = get_optimizer()

        # ── 각 거래의 시장충격 비용 계산 ──────────────────────

        high_impact_trades = []
        total_impact_bps = 0.0
        trade_analyses = []

        for trade in ctx.proposed_trades:
            symbol   = trade.get("symbol", "?")
            delta    = abs(trade.get("delta", 0.0))
            tv       = ctx.total_value
            trade_value_krw = delta * tv

            # ADV, 변동성, 가격은 holding 데이터에서 추출
            holding = next((h for h in ctx.holdings if h["symbol"] == symbol), {})
            current_price = holding.get("current_price", 10_000)
            adv_volume    = holding.get("adv_volume", 100_000)     # 일평균 거래량 (주)
            daily_sigma   = holding.get("daily_volatility", 0.02)  # 일별 변동성

            # 거래 수량 추정 (가격 기준)
            approx_quantity = trade_value_krw / current_price if current_price > 0 else 0

            impact = optimizer.almgren_chriss_cost(
                quantity=approx_quantity,
                adv=adv_volume,
                sigma=daily_sigma,
                price=current_price,
            )

            total_impact_bps += impact["cost_bps"]

            analysis = {
                "symbol": symbol,
                "action": trade.get("action"),
                "trade_value_krw": round(trade_value_krw),
                "participation_rate": round(impact["participation_rate"], 4),
                "impact_bps": round(impact["cost_bps"], 2),
                "strategy": self._recommend_strategy(impact["participation_rate"]),
            }
            trade_analyses.append(analysis)

            if impact["participation_rate"] > self.MAX_PARTICIPATION_RATE:
                high_impact_trades.append(f"{symbol}({impact['participation_rate']:.0%})")

        signals = [
            {
                "label": "총 예상 시장충격 비용",
                "value": f"{total_impact_bps:.1f} bps",
                "threshold": f"{self.MAX_IMPACT_BPS} bps",
                "breached": total_impact_bps > self.MAX_IMPACT_BPS,
            },
            {
                "label": "고충격 거래",
                "value": high_impact_trades,
            },
            {
                "label": "체결 전략 분석",
                "value": trade_analyses[:5],
            },
        ]

        # ── Claude Haiku로 체결 계획 요약 ──────────────────────

        exec_summary = json.dumps({
            "trade_analyses": trade_analyses[:5],
            "total_impact_bps": round(total_impact_bps, 2),
            "high_impact": high_impact_trades,
            "total_value_krw": ctx.total_value,
        }, ensure_ascii=False, indent=2)

        rationale, model_used = self._ask_llm(
            system=(
                "당신은 Tien입니다. 체결 비용 최소화 전문가입니다.\n"
                "Almgren-Chriss 시장충격 분석 결과를 바탕으로:\n"
                "1) 총 체결 비용이 허용 범위인지 (30bps 기준)\n"
                "2) 고충격 거래가 있으면 분할 체결 권장 기간\n"
                "3) approve / reject / abstain\n"
                "2문장으로 간결하게."
            ),
            user=exec_summary,
            ctx=ctx,
        )

        # 투표
        if total_impact_bps > self.MAX_IMPACT_BPS * 2:
            vote, confidence = "reject", 0.85
            rationale += f" 총 시장충격 {total_impact_bps:.1f}bps — 임계값 {self.MAX_IMPACT_BPS}bps 초과 거부."
        elif total_impact_bps > self.MAX_IMPACT_BPS:
            vote, confidence = "abstain", 0.65
        else:
            vote, confidence = "approve", 0.88

        return AgentVerdict(
            agent_id=self.agent_id,
            vote=vote,
            confidence=confidence,
            rationale=rationale,
            signals=signals,
            llm_used=model_used,
        )

    def _recommend_strategy(self, participation_rate: float) -> str:
        if participation_rate < 0.05:
            return "즉시 시장가 체결 (충격 미미)"
        elif participation_rate < 0.10:
            return "TWAP 2시간 분산 체결"
        elif participation_rate < 0.20:
            return "VWAP 1일 분산 체결"
        else:
            return "IS 전략 3-5일 분산 체결 (충격 최소화 필수)"
