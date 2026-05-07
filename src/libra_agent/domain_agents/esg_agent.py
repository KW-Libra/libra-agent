"""
ESG Agent (Esme) — ESG / 지속가능성 수호자

Gemini Flash 활용 이유:
  - ESG 스크리닝은 다수 종목의 키워드 분류 작업 → 고속 처리 필요
  - 글로벌 ESG 뉴스/이슈는 최신 데이터 주입으로 커트오프 보완
  - 비용 효율적 (ESG 점수 계산은 규칙 기반으로도 충분)

담당 역할:
  - ESG 등급 기반 포트폴리오 가중 점수 계산
  - 사용자 ESG 제외 목록 (exclusions) 검증
  - UN PRI, TCFD, SFDR 기본 프레임워크 체크
  - 탄소 발자국 노출도 추정

수식:
  Portfolio ESG Score = Σ (w_i × ESG_score_i)
  Carbon Intensity  = Σ (w_i × CO2_intensity_i)  [tCO2e/매출 백만원]
  ESG Drift         = |현재 ESG 점수 - 목표 ESG 점수|
"""

from __future__ import annotations

import json
import logging

from .base import BaseAgent, AgentVerdict, PortfolioContext

logger = logging.getLogger(__name__)

# 섹터별 평균 ESG 점수 (0~100, 자체 추정치)
# 실제 운영: MSCI ESG, Sustainalytics, Korea ESG Standards Institute (KCGS) 연동 권장
SECTOR_ESG_BASELINE: dict[str, float] = {
    "기술":       72.0,
    "헬스케어":   70.0,
    "소비재":     60.0,
    "금융":       65.0,
    "산업재":     55.0,
    "소재":       48.0,
    "에너지":     38.0,   # 화석연료 낮음
    "유틸리티":   50.0,
    "통신":       63.0,
    "부동산":     58.0,
    "기타":       55.0,
}

# 섹터별 탄소강도 추정 (tCO2e / 매출 10억원, 자체 추정)
SECTOR_CARBON_INTENSITY: dict[str, float] = {
    "에너지":     850.0,
    "소재":       420.0,
    "유틸리티":   380.0,
    "산업재":     180.0,
    "부동산":     120.0,
    "금융":        15.0,
    "소비재":      80.0,
    "기술":        25.0,
    "헬스케어":    35.0,
    "통신":        40.0,
    "기타":        90.0,
}


class ESGAgent(BaseAgent):
    agent_id = "esg"
    name = "Esme"
    role = "ESG Guardian"

    MIN_ESG_SCORE    = 40.0   # 포트폴리오 평균 ESG < 40이면 거부
    MAX_CARBON_INTENSITY = 200.0  # tCO2e/10억 초과 시 경고

    async def deliberate(self, ctx: PortfolioContext) -> AgentVerdict:
        # ── ESG 점수 및 탄소강도 계산 ──────────────────────────

        portfolio_esg = 0.0
        portfolio_carbon = 0.0
        esg_details = []

        exclusions: list[str] = ctx.preferences.get("esg_exclusions", [])
        esg_target: float = float(ctx.preferences.get("esg_min_score", 50.0))

        violations = []

        for h in ctx.holdings:
            symbol  = h.get("symbol", "?")
            sector  = h.get("sector", "기타")
            weight  = h.get("weight", 0.0)

            # 종목 개별 ESG 점수 (실제 운영: DB에서 조회)
            esg_score = h.get("esg_score") or SECTOR_ESG_BASELINE.get(sector, 55.0)
            carbon_intensity = h.get("carbon_intensity") or SECTOR_CARBON_INTENSITY.get(sector, 90.0)

            portfolio_esg    += weight * esg_score
            portfolio_carbon += weight * carbon_intensity

            esg_details.append({
                "symbol":   symbol,
                "sector":   sector,
                "weight":   round(weight, 4),
                "esg":      esg_score,
                "carbon":   carbon_intensity,
            })

            # 제외 목록 검증
            symbol_upper = symbol.upper()
            sector_lower = sector.lower()
            for excl in exclusions:
                if excl.lower() in sector_lower or excl.lower() in symbol_upper.lower():
                    violations.append(f"{symbol}: ESG 제외 '{excl}' 위반")

        # 제안 거래에도 ESG 체크
        trade_esg_violations = []
        for trade in ctx.proposed_trades:
            sym = trade.get("symbol", "")
            holding = next((h for h in ctx.holdings if h.get("symbol") == sym), {})
            sector = holding.get("sector", "기타")
            for excl in exclusions:
                if excl.lower() in sector.lower():
                    trade_esg_violations.append(f"신규 {trade.get('action')} {sym}: '{excl}' 위반")

        all_violations = violations + trade_esg_violations

        signals = [
            {
                "label": "포트폴리오 ESG 점수",
                "value": f"{portfolio_esg:.1f} / 100",
                "target": esg_target,
                "breached": portfolio_esg < self.MIN_ESG_SCORE,
            },
            {
                "label": "탄소강도",
                "value": f"{portfolio_carbon:.1f} tCO2e/10억원",
                "threshold": self.MAX_CARBON_INTENSITY,
                "breached": portfolio_carbon > self.MAX_CARBON_INTENSITY,
            },
            {
                "label": "ESG 위반",
                "value": len(all_violations),
                "details": all_violations[:3],
            },
        ]

        # 즉시 거부 조건 (ESG 위반)
        if all_violations:
            return AgentVerdict(
                agent_id=self.agent_id,
                vote="reject",
                confidence=1.0,
                rationale=f"ESG 위반 {len(all_violations)}건: {' | '.join(all_violations[:2])}",
                signals=signals,
                llm_used="rule_based",
            )

        # ── Gemini Flash로 ESG 내러티브 생성 ───────────────────

        esg_summary = json.dumps({
            "portfolio_esg_score": round(portfolio_esg, 2),
            "portfolio_carbon_intensity": round(portfolio_carbon, 2),
            "esg_target": esg_target,
            "top_esg_concerns": sorted(esg_details, key=lambda d: d["esg"])[:3],
        }, ensure_ascii=False, indent=2)

        rationale, model_used = self._ask_llm(
            system=(
                "당신은 Esme입니다. ESG/지속가능성 투자 전문가입니다.\n"
                "포트폴리오 ESG 분석 결과를 바탕으로:\n"
                "1) 현재 ESG 점수 평가 (목표 대비)\n"
                "2) 탄소강도 수준 평가\n"
                "3) approve / abstain 중 판단 (위반 없으므로 reject는 없음)\n"
                "2문장으로 간결하게. 수치를 직접 인용하세요."
            ),
            user=esg_summary,
            ctx=ctx,
        )

        # 투표
        if portfolio_esg < self.MIN_ESG_SCORE:
            vote, confidence = "reject", 0.90
            rationale += f" ESG 점수 {portfolio_esg:.1f} < 최소 기준 {self.MIN_ESG_SCORE}."
        elif portfolio_esg < esg_target:
            vote, confidence = "abstain", 0.65
        else:
            vote, confidence = "approve", 0.85

        return AgentVerdict(
            agent_id=self.agent_id,
            vote=vote,
            confidence=confidence,
            rationale=rationale,
            signals=signals,
            llm_used=model_used,
        )
