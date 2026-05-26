"""
Risk Agent (Vora) — 포트폴리오 리스크 감시자

변경 사항 (v2):
  - Claude Sonnet 사용 (고위험 판단 — LLM 라우터 자동 배정)
  - 실제 HHI, 히스토리컬 VaR, MDD 계산 (portfolio_optimizer 연동)
  - 실시간 시장 데이터 자동 주입 (market_context_str)
  - _ask_claude() → _ask_llm() 마이그레이션
"""

from __future__ import annotations

import json
import logging

from ._services.portfolio_optimizer import get_optimizer
from .base import AgentVerdict, BaseAgent, PortfolioContext

logger = logging.getLogger(__name__)


class RiskAgent(BaseAgent):
    agent_id = "risk"
    name = "Vora"
    role = "Risk Sentinel"

    CONCENTRATION_THRESHOLD = 0.25  # 상위 5개 비중 합 > 25%
    MAX_SINGLE_DELTA = 0.10  # 단일 거래 delta > 10% 거부
    VAR_95_LIMIT_PCT = 0.03  # 일일 VaR 95% > 총자산 3% 경고
    # Week 2 Phase B ([[16]] §1, 2026-05-27) — MDD 개선 단일 동인.
    # 학계 60/40 표준 변동성 근처. v2 결과 MDD -34% → 학계 예측 -26% 타깃.
    VOL_TARGET = 0.18  # 연환산 변동성 18% 초과 시 신규 BUY 거부
    DD_TRIGGER = -0.15  # MDD -15% 초과 (더 큰 손실) 시 강제 de-risking

    async def deliberate(self, ctx: PortfolioContext) -> AgentVerdict:
        optimizer = get_optimizer()

        # ── 정량 리스크 지표 계산 ──────────────────────────────

        holdings = sorted(ctx.holdings, key=lambda h: h.get("weight", 0), reverse=True)
        top5_weight = sum(h.get("weight", 0) for h in holdings[:5])

        # 수익률 히스토리가 있으면 실제 VaR 계산
        import numpy as np

        risk_metrics = None
        if ctx.returns_data:
            symbols = [h["symbol"] for h in holdings]
            returns_list = [ctx.returns_data.get(s, []) for s in symbols]
            # 최소 30일치가 있는 종목만 사용
            valid = [(s, r) for s, r in zip(symbols, returns_list, strict=False) if len(r) >= 30]
            if valid:
                valid_symbols, valid_returns = zip(*valid, strict=False)
                min_len = min(len(r) for r in valid_returns)
                R = np.array([r[-min_len:] for r in valid_returns]).T
                risk_metrics = optimizer.compute_risk_metrics(
                    holdings=[h for h in holdings if h["symbol"] in valid_symbols],
                    returns_matrix=R,
                    total_value=ctx.total_value,
                )

        # HHI (히스토리 없어도 계산 가능)
        import numpy as np

        weights_arr = np.array([h.get("weight", 0) for h in holdings])
        hhi = float(np.sum(weights_arr**2))

        signals = [
            {
                "label": "상위5 집중도",
                "value": f"{top5_weight:.1%}",
                "threshold": f"{self.CONCENTRATION_THRESHOLD:.0%}",
                "breached": top5_weight > self.CONCENTRATION_THRESHOLD,
            },
            {
                "label": "HHI 집중도 지수",
                "value": round(hhi, 4),
                "note": "0에 가까울수록 분산, 1이면 완전 집중",
            },
        ]

        if risk_metrics:
            signals += [
                {"label": "VaR 95% (일일)", "value": f"{risk_metrics.var_95:,.0f} KRW"},
                {"label": "CVaR 95%", "value": f"{risk_metrics.cvar_95:,.0f} KRW"},
                {
                    "label": "MDD",
                    "value": f"{risk_metrics.mdd:.1%}",
                    "threshold": f"{self.DD_TRIGGER:.0%}",
                    "breached": risk_metrics.mdd < self.DD_TRIGGER,
                },
                {
                    "label": "연환산 변동성",
                    "value": f"{risk_metrics.volatility:.1%}",
                    "threshold": f"{self.VOL_TARGET:.0%}",
                    "breached": risk_metrics.volatility > self.VOL_TARGET,
                },
                {"label": "베타", "value": f"{risk_metrics.beta:.2f}"},
            ]

        # ── Claude Sonnet LLM 호출 ──────────────────────────────
        # 고위험 판단 → Sonnet (라우터가 자동 배정)

        portfolio_summary = json.dumps(
            {
                "total_value_krw": ctx.total_value,
                "top_holdings": holdings[:10],
                "top5_weight": top5_weight,
                "hhi": round(hhi, 4),
                "var_95_krw": risk_metrics.var_95 if risk_metrics else "N/A (히스토리 없음)",
                "mdd": risk_metrics.mdd if risk_metrics else "N/A",
                "volatility": risk_metrics.volatility if risk_metrics else "N/A",
                "proposed_trades": ctx.proposed_trades,
                "user_risk_profile": ctx.preferences.get("risk_profile", "balanced"),
            },
            ensure_ascii=False,
            indent=2,
        )

        rationale, model_used = self._ask_llm(
            system=(
                "당신은 Vora입니다. 다이렉트 인덱싱 포트폴리오의 리스크 감시자입니다.\n"
                "정량 지표(HHI, VaR, MDD, 변동성)와 제안 거래를 검토하여:\n"
                "1) 가장 심각한 리스크 요인 1가지\n"
                "2) approve / reject / abstain 판단\n"
                "2문장으로 간결하게. 수치를 직접 인용하세요."
            ),
            user=f"포트폴리오 스냅샷:\n{portfolio_summary}",
            ctx=ctx,
        )

        # ── 투표 결정 ───────────────────────────────────────────

        vote = "abstain"
        confidence = 0.60

        if not holdings and not ctx.proposed_trades:
            rationale = (
                "보유 종목과 제안 거래가 없어 집중도, VaR, MDD를 평가할 투자 리스크 대상이 없습니다. "
                "abstain - 초기 포트폴리오 후보가 생성된 뒤 리스크 한도 위반 여부를 평가해야 합니다."
            )

        elif top5_weight > self.CONCENTRATION_THRESHOLD:
            vote = "approve"  # 집중도 위반이 리밸런싱 이유
            confidence = 0.90
        elif holdings or ctx.proposed_trades:
            vote = "approve"
            confidence = 0.85

        if risk_metrics and ctx.total_value > 0:
            var_pct = risk_metrics.var_95 / ctx.total_value
            if var_pct > self.VAR_95_LIMIT_PCT:
                vote = "abstain"
                confidence = 0.70
                rationale += f" VaR 95% {var_pct:.1%} — 리스크 허용 한도 검토 필요."

        # Week 2 Phase B — vol targeting + drawdown trigger ([[16]] §1).
        # DD trigger 는 단일 거래 체크 위에 두어 새 BUY 차단; vol breach 는 abstain.
        vol_breach = bool(risk_metrics and risk_metrics.volatility > self.VOL_TARGET)
        dd_breach = bool(risk_metrics and risk_metrics.mdd < self.DD_TRIGGER)
        has_buy = any(t.get("delta", 0) > 0 for t in ctx.proposed_trades)

        if dd_breach and has_buy:
            vote = "reject"
            confidence = 0.85
            rationale += (
                f" MDD {risk_metrics.mdd:.1%} ≤ {self.DD_TRIGGER:.0%} — drawdown 트리거 활성,"
                " 신규 매수 차단 de-risk."
            )
        elif vol_breach and has_buy:
            vote = "reject"
            confidence = 0.75
            rationale += (
                f" 연환산 변동성 {risk_metrics.volatility:.1%} > 목표 {self.VOL_TARGET:.0%}"
                " — 신규 매수 거부 (vol targeting)."
            )

        for trade in ctx.proposed_trades:
            if abs(trade.get("delta", 0)) > self.MAX_SINGLE_DELTA:
                vote = "reject"
                confidence = 0.75
                rationale += f" 단일 거래 delta {trade.get('delta', 0):.1%} — 10% 초과 거부."
                break

        return AgentVerdict(
            agent_id=self.agent_id,
            vote=vote,
            confidence=confidence,
            rationale=rationale,
            signals=signals,
            llm_used=model_used,
        )
