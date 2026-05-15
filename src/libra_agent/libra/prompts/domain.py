from __future__ import annotations

DOMAIN_AGENT_BRIEFS = {
    "risk": "RiskAgent (Vora): HHI, VaR, MDD, 단일 종목 delta 위험 한도를 검토합니다.",
    "tax": "TaxAgent (Reed): 손익통산과 세금 관점의 실행 가능성을 검토합니다.",
    "compliance": "ComplianceAgent (Clarke): 사용자 IPS, 제외 섹터, ESG 제한 룰을 검토하며 거부권이 있습니다.",
    "macro": "MacroAgent (Halden): 거시 충격과 섹터 노출을 Gemini x Claude 교차 검증 관점으로 검토합니다.",
    "sentiment": "SentimentAgent (Imo): FinBERT, Gemini, Claude 기반 뉴스 감성 신호를 검토합니다.",
    "execution": "ExecutionAgent (Tien): Almgren-Chriss 시장충격과 체결 전략을 검토합니다.",
    "esg": "ESGAgent (Esme): ESG 점수, 탄소강도, 사용자 ESG 제외 조건을 검토합니다.",
    "liquidity": "LiquidityAgent: ADV, 호가 스프레드, 유통주식 비율 등 시장 유동성 제약을 검토합니다.",
    "technical": "TechnicalAnalysisAgent: 가격/거래량 모멘텀과 기술적 약세 신호를 검토합니다.",
}


DOMAIN_AGENT_PROMPT_SECTION = "\n".join(f"- {brief}" for brief in DOMAIN_AGENT_BRIEFS.values())
