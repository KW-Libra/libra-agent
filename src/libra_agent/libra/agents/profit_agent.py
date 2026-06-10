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
            "거래 규모 대비 완만한 상승 여력이 있습니다."
            if plan_score >= 0
            else "제안된 거래는 기대 후속 성과가 약한 데 비해 비용을 지불하는 형태입니다."
        )
        opinion_id = f"profit_{stable_hash({'turn': turn_number, 'plan': rebalance_plan})[:12]}"
        return AgentResponse(
            agent_id=self.agent_id,
            opinion_id=opinion_id,
            turn_number=turn_number,
            query_understood="제안된 리밸런싱안을 휴리스틱 로컬 시뮬레이터로 평가합니다.",
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
            reasoning_for_judge_agent=recommendation,
            limits_acknowledged="수익성 추정은 보정된 몬테카를로 엔진이 아닌 휴리스틱 로컬 신호 모델을 사용합니다.",
            tools_called=[
                ToolCall(
                    tool_name="local_profit.heuristic_plan_simulation",
                    purpose="후보 리밸런싱안의 방향성 손익 추정",
                    summary=f"계획된 포지션 변경 {len(rebalance_plan)}건에 대해 휴리스틱 수익률을 계산했습니다.",
                )
            ],
            depth_used="medium",
            focus_tickers=sorted(rebalance_plan),
        )


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))
