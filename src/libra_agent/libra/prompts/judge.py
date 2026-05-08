from __future__ import annotations


JUDGE_ACTION_RULES = [
    "Choose exactly one next action.",
    "Judge must decide the first and every later call from the current trigger, user request, and observations.",
    "Never follow a precomputed collection order.",
    "Do not call an agent that already answered unless there is a very strong reason.",
    "Push events already include News pre-screening in trigger_event.",
    "Use Profit and Cost only when you have a candidate rebalance plan or execution question.",
    "When information is sufficient or more calls are not justified, choose FINALIZE.",
    "On calm pull checks, Disclosure plus shallow News can already be enough.",
    "Do not call Report on pull unless there is conflict, a meaningful directional signal, or an explicit report request.",
    "Do not call Report just because the local cache is empty or an agent returned DIRECT_ANSWER_UNAVAILABLE.",
    "When push trigger_event already has cross-check and market reaction, avoid re-running Disclosure or News by default.",
    "Domain agents are a separate decision-review council layer; do not treat them as first-layer information gathering agents.",
    "After the core loop has enough evidence or a candidate plan, domain council routing may call Risk, Tax, Compliance, Macro, Sentiment, Execution, and ESG one at a time.",
]

JUDGE_ACTION_SYSTEM_PROMPT = (
    "You are the LIBRA Judge orchestrator. "
    "Choose the next best action in an agentic loop. "
    "Return one JSON object with keys: action, reason, agent_id, query, context, depth, fallback, note, candidate_rebalance_plan. "
    "If action is FINALIZE, omit agent_id/query/context. "
    "Respect dynamic orchestration: observe state, choose one agent or FINALIZE, then wait for the next observation. "
    "Write every natural-language value only in Korean. Do not use Japanese kana. "
    "English is allowed only for enum values, JSON keys, tickers, URLs, and source names."
    "\nAvailable first-layer agents: DisclosureAgent, NewsAgent, ReportAgent, ProfitAgent, CostAgent. "
    "Domain agents are handled only in the separate domain council layer. "
    "EvaluationAgent is used only after outcomes are known, not during CALL_AGENT routing."
)

JUDGE_DOMAIN_ACTION_SYSTEM_PROMPT = (
    "You are the LIBRA Judge orchestrating the domain council layer. "
    "The first-layer information and trade-review loop has already run. "
    "Choose exactly one domain agent to consult next or finalize the domain review. "
    "Return one JSON object with keys: action, reason, agent_id, query, context, depth, fallback, note. "
    "Use action CALL_AGENT or FINALIZE_DOMAIN_REVIEW. "
    "Available domain agents: RiskAgent, TaxAgent, ComplianceAgent, MacroAgent, SentimentAgent, ExecutionAgent, ESGAgent. "
    "Write every natural-language value only in Korean. Do not use Japanese kana. "
    "English is allowed only for enum values, JSON keys, tickers, URLs, and source names."
)

_DOMAIN_AGENT_QUERIES = {
    "risk": "후보 리밸런싱 또는 현재 포트폴리오의 집중도, 손실 위험, 하방 노출을 평가해줘.",
    "tax": "후보 리밸런싱의 세금 영향과 손실실현 가능성을 평가해줘.",
    "compliance": "후보 리밸런싱이 사용자 투자정책, 제외 조건, 승인 조건을 위반하는지 검토해줘.",
    "macro": "현재 포트폴리오와 후보 리밸런싱이 거시 환경과 경기 국면에 비추어 적절한지 평가해줘.",
    "sentiment": "보유 종목과 후보 리밸런싱에 대한 최근 시장 심리와 뉴스 감성 리스크를 평가해줘.",
    "execution": "후보 리밸런싱의 체결 가능성, 유동성, 시장충격, 주문 전략을 평가해줘.",
    "esg": "보유 종목과 후보 리밸런싱이 ESG 기준과 사용자 제외 조건을 충족하는지 평가해줘.",
}

_DOMAIN_AGENT_FALLBACKS = {
    "risk": "정량 데이터가 부족하면 보유 비중, 단일 종목 집중도, 거래 비중 변화만으로 보수적으로 판단해줘.",
    "tax": "세율 정보가 부족하면 과세 효과의 방향성과 확인해야 할 항목을 중심으로 판단해줘.",
    "compliance": "사용자 정책이 불명확하면 자동 승인하지 말고 사용자 확인이 필요한 조건을 짚어줘.",
    "macro": "거시 데이터가 부족하면 판단 불가 근거와 재확인해야 할 지표를 명시해줘.",
    "sentiment": "뉴스가 부족하면 감성 판단을 보류하고 필요한 데이터 조건을 명시해줘.",
    "execution": "실시간 호가가 부족하면 거래 규모와 일반 유동성 가정에 따른 실행 리스크를 보수적으로 판단해줘.",
    "esg": "ESG 데이터가 부족하면 사용자 기준 위반 가능성과 확인해야 할 항목을 명시해줘.",
}

_DOMAIN_AGENT_NOTES = {
    "risk": "Judge는 다음 판단 전에 포트폴리오 위험 관점의 독립 의견을 요청합니다.",
    "tax": "Judge는 실행 전 세금 효과와 손실실현 가능성을 별도 관점으로 확인합니다.",
    "compliance": "Judge는 자동 판단 전에 사용자 정책과 제약 위반 가능성을 확인합니다.",
    "macro": "Judge는 종목 신호가 거시 환경과 충돌하는지 확인합니다.",
    "sentiment": "Judge는 정량 신호만으로 부족한 시장 심리 변화를 확인합니다.",
    "execution": "Judge는 후보 초안이 실제 시장에서 무리 없이 체결 가능한지 확인합니다.",
    "esg": "Judge는 사용자 ESG 기준과 비재무 제약을 독립적으로 확인합니다.",
}

JUDGE_PHASE_SYSTEM_PROMPT = (
    "You are the LIBRA Judge orchestrator. Respond only with one JSON object. "
    "Decide among HOLD, DEFER, USER_DECISION_REQUIRED, REBALANCE. "
    "Use candidate_rebalance_plan only when you can justify a specific weight change. "
    "For HOLD prefer notification level silent. For DEFER prefer info. For USER_DECISION_REQUIRED prefer push. "
    "On a calm pull check with no meaningful supplied signal and no trade draft, prefer HOLD over DEFER. "
    "Write every natural-language value only in Korean. Do not use Japanese kana. "
    "English is allowed only for enum values, JSON keys, tickers, URLs, and source names."
    " If domain agent responses are present, treat ComplianceAgent reject as a hard veto that requires USER_DECISION_REQUIRED."
)

JUDGE_PHASE_REQUIRED_KEYS = [
    "decision",
    "summary",
    "confidence",
    "urgency",
    "reasoning",
    "candidate_rebalance_plan",
    "needs_trade_evaluation",
    "follow_up_at",
    "feedback_checkpoint",
    "user_notification",
]

JUDGE_NOTIFICATION_LEVELS = ["silent", "info", "watch", "push"]


def default_agent_query(
    *,
    agent_id: str,
    trigger: str,
    has_disclosure_context: bool = False,
) -> str:
    if agent_id in _DOMAIN_AGENT_QUERIES:
        return _DOMAIN_AGENT_QUERIES[agent_id]
    if agent_id == "disclosure":
        return "포트폴리오 관련 신규 공시와 실적 신호를 요약해줘."
    if agent_id == "news":
        if has_disclosure_context:
            return "최근 공시 이후 시장 반응과 관련 뉴스, 필요시 매크로 배경을 요약해줘."
        return "포트폴리오 관련 뉴스, 시장 반응, 필요시 매크로 배경을 요약해줘."
    if agent_id == "report":
        return "포트폴리오 관련 증권사 리포트와 컨센서스 변화, 사업부 단서를 요약해줘."
    if agent_id == "profit":
        return "후보 리밸런싱 초안의 기대수익과 위험을 평가해줘."
    if trigger == "push":
        return "후보 리밸런싱 초안의 거래비용, 슬리피지, 유동성을 평가해줘."
    return "후보 리밸런싱 초안의 거래비용과 실행 마찰을 평가해줘."


def default_agent_fallback(*, agent_id: str, trigger: str) -> str | None:
    if agent_id in _DOMAIN_AGENT_FALLBACKS:
        return _DOMAIN_AGENT_FALLBACKS[agent_id]
    if agent_id == "disclosure":
        return "단기 투자 가정이나 판단 시점을 바꿀 수 있는 공시만 우선 정리해줘."
    if agent_id == "news":
        return "시장 반응, 교차 확인 여부, 투자 가정 변화 여부를 우선 정리해줘."
    if agent_id == "report":
        return "코멘트 리포트가 없으면 프리뷰 리포트나 간접 단서라도 정리해줘."
    if agent_id == "profit":
        return "현 상태 유지와 비교해 초안이 기대수익과 위험을 개선하는지 검토해줘."
    if trigger == "push":
        return "거래 마찰, 유동성, 중단 조건 중심으로 검토해줘."
    return "수수료, 세금, 슬리피지, 스프레드와 실제 실행 마찰을 추정해줘."


def default_agent_note(
    *,
    agent_id: str,
    latest_agent_id: str | None = None,
    trigger: str,
    has_candidate_plan: bool = False,
) -> str | None:
    if agent_id in _DOMAIN_AGENT_NOTES:
        return _DOMAIN_AGENT_NOTES[agent_id]
    if agent_id == "disclosure":
        return "판단 에이전트는 원천 정보인 공시에서 투자 가정 변화가 있는지 먼저 확인합니다."
    if agent_id == "news":
        if latest_agent_id == "disclosure":
            return "판단 에이전트는 공시 내용이 시장 시각을 바꿨는지, 이미 가격에 반영됐는지 확인합니다."
        if trigger == "push":
            return "판단 에이전트는 속보가 여러 출처에서 확인됐고 시장에 중요한지 점검합니다."
        return "판단 에이전트는 추가 수집이 필요한지 결정하기 전에 시장 반응과 교차 확인을 봅니다."
    if agent_id == "report":
        return "직접 근거가 충돌하거나 부족하거나 아직 모호해 증권사 해석을 확인합니다."
    if agent_id == "profit":
        if has_candidate_plan:
            return "판단 에이전트는 현재 초안이 추가 위험 대비 기대수익을 개선하는지 확인합니다."
        return "수익 에이전트는 구체적인 리밸런싱 초안이 있을 때만 호출합니다."
    if has_candidate_plan:
        return "초안이 실제로 실행 가능한지 판단하기 위해 거래 마찰을 확인합니다."
    return "비용 에이전트는 구체적인 리밸런싱 초안이 있을 때만 호출합니다."
