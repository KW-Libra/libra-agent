"""
Macro Agent (Halden) — 거시경제 환경 분석가

Gemini Flash 활용 이유:
  - 뉴스 헤드라인 대량 처리에 Gemini Flash가 Claude Haiku보다 2-3배 저렴
  - 최신 뉴스/매크로 데이터는 RAG(실시간 주입)로 커트오프 문제 해결
  - cross_validate=True로 Gemini+Claude 교차 검증 → 할루시네이션 방지

담당 역할:
  - 금리/환율/GDP/인플레이션 사이클 평가
  - 거시 리스크가 포트폴리오 섹터 노출도에 미치는 영향 분석
  - 경기 사이클 국면 (확장/정점/수축/저점) 판단

수식:
  Macro Impact Score = Σ (섹터비중_i × 섹터_금리민감도_i × 금리변화)
  Sector Exposure = Σ w_i × β_i_sector (섹터별 가중 베타)
"""

from __future__ import annotations

import json
import logging

from .base import AgentVerdict, BaseAgent, PortfolioContext

logger = logging.getLogger(__name__)


# 섹터별 금리 민감도 (부호: + = 금리상승 시 악영향)
RATE_SENSITIVITY: dict[str, float] = {
    "금융": -0.3,  # 은행은 금리 상승 수혜
    "부동산": 0.8,  # 금리 상승 취약
    "유틸리티": 0.7,  # 채권 대체재, 금리 상승 취약
    "기술": 0.4,  # 성장주 할인율 상승
    "에너지": 0.1,  # 인플레와 상관
    "소비재": 0.2,
    "헬스케어": 0.0,  # 비교적 중립
    "산업재": 0.3,
    "통신": 0.5,
    "소재": 0.2,
}


class MacroAgent(BaseAgent):
    agent_id = "macro"
    name = "Halden"
    role = "Macro Economist"

    async def deliberate(self, ctx: PortfolioContext) -> AgentVerdict:
        # ── 섹터 노출도 계산 ────────────────────────────────────

        sector_exposure: dict[str, float] = {}
        for h in ctx.holdings:
            sector = h.get("sector", "기타")
            weight = h.get("weight", 0.0)
            sector_exposure[sector] = sector_exposure.get(sector, 0.0) + weight

        # 거시 충격 점수 (금리 민감도 가중합)
        # 실제 운영에서는 MarketDataInjector에서 주입된 금리 변화를 사용
        macro_data = {}
        if ctx.market_context_str and "기준금리" in ctx.market_context_str:
            # 파싱 시도
            for line in ctx.market_context_str.split("\n"):
                if "기준금리" in line:
                    try:
                        rate_str = line.split(":")[-1].strip().rstrip("%")
                        macro_data["base_rate"] = float(rate_str)
                    except ValueError:
                        pass

        # 실질 금리변화 없으면 중립 가정
        rate_delta = macro_data.get("rate_change", 0.0)
        macro_impact_score = sum(
            sector_exposure.get(sector, 0) * sensitivity * rate_delta
            for sector, sensitivity in RATE_SENSITIVITY.items()
        )

        signals = [
            {
                "label": "섹터 노출도",
                "value": {k: f"{v:.1%}" for k, v in sector_exposure.items()},
            },
            {
                "label": "매크로 충격 점수",
                "value": f"{macro_impact_score:.4f}",
                "note": "양수 = 금리상승 불리, 음수 = 유리",
            },
        ]

        # ── Gemini Flash LLM 호출 (뉴스 + 매크로 분석) ─────────
        # cross_validate=True → Gemini + Claude 교차 검증 (할루시네이션 방지)

        portfolio_summary = json.dumps(
            {
                "sector_exposure": sector_exposure,
                "macro_impact_score": round(macro_impact_score, 4),
                "base_rate": macro_data.get("base_rate", "N/A"),
                "proposed_trades": ctx.proposed_trades[:5],
                "total_value_krw": ctx.total_value,
            },
            ensure_ascii=False,
            indent=2,
        )

        rationale, model_used = self._ask_llm(
            system=(
                "당신은 Halden입니다. 거시경제 분석 전문가로서 포트폴리오의 매크로 리스크를 평가합니다.\n"
                "아래 주입된 실시간 뉴스/매크로 데이터와 포트폴리오 스냅샷을 바탕으로:\n"
                "1) 현재 경기 사이클 국면 판단 (확장/정점/수축/저점)\n"
                "2) 이 포트폴리오의 주요 거시 리스크 1가지\n"
                "3) approve / reject / abstain 중 하나와 그 이유\n"
                "2-3문장으로 간결하게. 실제 데이터 수치를 인용하세요."
            ),
            user=f"포트폴리오 분석:\n{portfolio_summary}",
            ctx=ctx,
            cross_validate=True,  # Gemini + Claude 교차 검증
        )

        # 투표 결정 (매크로 충격 임계값 기반)
        if abs(macro_impact_score) < 0.05:
            vote, confidence = "approve", 0.75
        elif macro_impact_score > 0.10:
            vote, confidence = "abstain", 0.60  # 불리하지만 차단할 수준은 아님
        else:
            vote, confidence = "approve", 0.70

        return AgentVerdict(
            agent_id=self.agent_id,
            vote=vote,
            confidence=confidence,
            rationale=rationale,
            signals=signals,
            llm_used=model_used,
        )
