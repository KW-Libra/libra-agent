from __future__ import annotations


JUDGE_ACTION_RULES = [
    "Choose exactly one next action.",
    "Do not call an agent that already answered unless there is a very strong reason.",
    "Push events already include News pre-screening in trigger_event.",
    "Use Profit and Cost only when you have a candidate rebalance plan or execution question.",
    "When information is sufficient or more calls are not justified, choose FINALIZE.",
    "On calm pull checks, Disclosure plus shallow News can already be enough.",
    "Do not call Report on pull unless there is conflict, missing evidence, or an explicit report request.",
    "When push trigger_event already has cross-check and market reaction, avoid re-running Disclosure or News by default.",
]

JUDGE_ACTION_SYSTEM_PROMPT = (
    "You are the LIBRA Judge orchestrator. "
    "Choose the next best action in an agentic loop. "
    "Return one JSON object with keys: action, reason, agent_id, query, context, depth, fallback, note, candidate_rebalance_plan. "
    "If action is FINALIZE, omit agent_id/query/context. "
    "Respect dynamic orchestration: decide based on current observations, not a fixed pipeline."
)

JUDGE_PHASE_SYSTEM_PROMPT = (
    "You are the LIBRA Judge orchestrator. Respond only with one JSON object. "
    "Decide among HOLD, DEFER, USER_DECISION_REQUIRED, REBALANCE. "
    "Use candidate_rebalance_plan only when you can justify a specific weight change. "
    "For HOLD prefer notification level silent. For DEFER prefer info. For USER_DECISION_REQUIRED prefer push."
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
    if agent_id == "disclosure":
        return "Focus on filings that materially change near-term thesis or timing."
    if agent_id == "news":
        return "Focus on market reaction, cross-checks, and whether the thesis changed."
    if agent_id == "report":
        return "코멘트 리포트가 없으면 프리뷰 리포트나 간접 단서라도 정리해줘."
    if agent_id == "profit":
        return "Stress-test whether the plan improves follow-through versus staying put."
    if trigger == "push":
        return "Focus on friction, liquidity, and tripwire-style safeguards."
    return "Estimate commission, tax, slippage, spread, and practical execution friction."


def default_agent_note(
    *,
    agent_id: str,
    latest_agent_id: str | None = None,
    trigger: str,
    has_candidate_plan: bool = False,
) -> str | None:
    if agent_id == "disclosure":
        return "Judge starts with disclosures to check whether the thesis changed at the source."
    if agent_id == "news":
        if latest_agent_id == "disclosure":
            return "Judge wants to know whether the latest disclosure changed the market view or is already priced."
        if trigger == "push":
            return "Judge is checking whether the push event is broadly confirmed and market-relevant."
        return "Judge wants market reaction and cross-checks before deciding whether more collection is justified."
    if agent_id == "report":
        return "Judge needs sell-side interpretation because direct evidence is conflicting, thin, or still ambiguous."
    if agent_id == "profit":
        if has_candidate_plan:
            return "Judge is checking whether the current draft plan improves expected return relative to added risk."
        return "Judge only calls Profit when a concrete draft plan exists."
    if has_candidate_plan:
        return "Judge needs execution friction before deciding whether the draft plan is practical."
    return "Judge only calls Cost when a concrete draft plan exists."
