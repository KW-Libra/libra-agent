"""JY 7-에이전트 → LIBRA Judge 시스템 어댑터.

JY 패턴 ``BaseAgent.deliberate(ctx) → DomainAgentVerdict`` 을
LIBRA ``InformationAgentProtocol.run(...) → AgentResponse`` 로 변환.

매핑 규칙:
    vote=approve  → verdict=DIRECT_ANSWER, opinion=POSITIVE, direction=+0.6
    vote=reject   → verdict=DIRECT_ANSWER, opinion=NEGATIVE, direction=-0.7
    vote=abstain  → verdict=QUIET,         opinion=NEUTRAL,  direction= 0.0
    confidence    → confidence + strength
    rationale     → reasoning_for_judge_agent
    signals[]     → evidence dict (label → value)
"""

from __future__ import annotations

import ast
import asyncio
import threading
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from libra_agent.libra_models import (
    AgentResponse,
    PortfolioSnapshot,
    Urgency,
)
from libra_agent.libra_models import (
    AgentVerdict as LibraVerdict,
)

from .base import BaseAgent as DomainBaseAgent
from .base import PortfolioContext as DomainPortfolioContext

if TYPE_CHECKING:
    from libra_agent.libra_runtime import LocalKnowledgeBase


_VOTE_TO_DIRECTION: dict[str, float] = {
    "approve": +0.6,
    "reject": -0.7,
    "abstain": 0.0,
}

_VOTE_TO_VERDICT: dict[str, LibraVerdict] = {
    "approve": LibraVerdict.DIRECT_ANSWER,
    "reject": LibraVerdict.DIRECT_ANSWER,
    "abstain": LibraVerdict.QUIET,
}

_VOTE_TO_OPINION: dict[str, str] = {
    "approve": "POSITIVE",
    "reject": "NEGATIVE",
    "abstain": "NEUTRAL",
}


def _parse_preference_value(raw: str) -> Any:
    text = raw.strip()
    if not text:
        return ""
    lowered = text.lower()
    if lowered in {"none", "null"}:
        return None
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        pass
    try:
        return float(text) if "." in text else int(text)
    except ValueError:
        return text


def _as_preference_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, Iterable):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)]


def _parse_user_preferences(items: Iterable[str]) -> dict[str, Any]:
    preferences: dict[str, Any] = {}
    notes: list[str] = []
    for raw_item in items:
        item = str(raw_item).strip()
        if not item:
            continue
        if "=" in item:
            key, raw_value = item.split("=", 1)
            key = key.strip()
            if key:
                preferences[key] = _parse_preference_value(raw_value)
                continue
        preferences[item] = True
        notes.append(item)

    exclusions = _as_preference_list(
        preferences.get("excluded_sectors")
        or preferences.get("exclusions")
        or preferences.get("esg_exclusions")
    )
    preferences["excluded_sectors"] = exclusions
    preferences["exclusions"] = exclusions
    preferences["esg_exclusions"] = exclusions
    preferences["preference_notes"] = notes
    if "approval_mode" in preferences and "approval" not in preferences:
        preferences["approval"] = preferences["approval_mode"]
    return preferences


def portfolio_snapshot_to_domain_context(
    portfolio: PortfolioSnapshot,
    *,
    user_id: str = "libra",
    proposed_trades: list[dict[str, Any]] | None = None,
    market_context_str: str = "",
) -> DomainPortfolioContext:
    """LIBRA ``PortfolioSnapshot`` → JY ``DomainPortfolioContext``.

    LIBRA holdings 는 frozen tuple of ``PortfolioHolding``,
    JY holdings 는 ``list[dict]`` 이므로 변환이 필요하다.
    """
    jy_holdings: list[dict[str, Any]] = []
    for h in portfolio.holdings:
        jy_holdings.append(
            {
                "symbol": h.ticker,
                "name": h.company_name,
                "weight": float(h.weight),
                "quantity": float(h.shares or 0),
                "current_price": float(h.last_price or 0),
                "average_price": float(h.average_price or 0),
                "market_value": float(h.market_value_krw or 0),
                "sector": h.sector or "기타",
                "esg_score": h.esg_score,
                "carbon_intensity": h.carbon_intensity,
            }
        )

    preferences = _parse_user_preferences(portfolio.user_preferences)
    preferences["cash_weight"] = float(portfolio.cash_weight or 0.0)

    return DomainPortfolioContext(
        user_id=user_id,
        holdings=jy_holdings,
        preferences=preferences,
        total_value=float(portfolio.total_value_krw or 0.0),
        proposed_trades=proposed_trades or [],
        market_context_str=market_context_str,
    )


def domain_verdict_to_agent_response(
    verdict: Any,
    *,
    agent_id: str,
    turn_number: int,
    query: str,
) -> AgentResponse:
    """JY ``DomainAgentVerdict`` → LIBRA ``AgentResponse``.

    monorepo / split-repo 사이의 ``AgentResponse`` 필드 차이(``signal_score``,
    ``risk_level``, ``opinion`` 같은 추가 필드 유무)를 흡수하기 위해
    dataclass ``fields()`` 로 실제 받는 필드만 동적으로 전달한다.
    """
    from dataclasses import fields as dc_fields

    vote = str(getattr(verdict, "vote", "abstain") or "abstain").lower()
    direction = _VOTE_TO_DIRECTION.get(vote, 0.0)
    confidence = float(getattr(verdict, "confidence", 0.5) or 0.5)

    evidence: dict[str, Any] = {}
    domain_signals: list[dict[str, Any]] = []
    for sig in getattr(verdict, "signals", []) or []:
        if isinstance(sig, dict):
            domain_signals.append(dict(sig))
            label = str(sig.get("label", "")).strip()
            if label:
                evidence[label] = sig.get("value")
    llm_used = str(getattr(verdict, "llm_used", "") or "").strip()
    evidence.update(
        {
            "vote": vote,
            "domain_signals": domain_signals,
            "llm_used": llm_used,
        }
    )

    candidate = {
        "agent_id": agent_id,
        "opinion_id": f"{agent_id}-domain-t{turn_number}",
        "turn_number": turn_number,
        "query_understood": query,
        "verdict": _VOTE_TO_VERDICT.get(vote, LibraVerdict.PARTIAL_ANSWER),
        "evidence": evidence,
        "direction": direction,
        "strength": confidence,
        "urgency": Urgency.SCHEDULED,
        "confidence": confidence,
        "reasoning_for_judge_agent": str(getattr(verdict, "rationale", "") or ""),
        "signal_score": direction,
        "source_trust": 0.7,
        "risk_level": "medium" if vote == "reject" else "low",
        "opinion": _VOTE_TO_OPINION.get(vote, "NEUTRAL"),
    }
    valid = {f.name for f in dc_fields(AgentResponse)}
    return AgentResponse(**{k: v for k, v in candidate.items() if k in valid})


def _run_async(coro: Any) -> Any:
    """Sync 컨텍스트에서 async 코루틴을 실행. 이미 loop 안이면 별 thread 사용."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    else:
        result: list[Any] = []
        error: list[BaseException] = []

        def runner() -> None:
            try:
                loop = asyncio.new_event_loop()
                try:
                    result.append(loop.run_until_complete(coro))
                finally:
                    loop.close()
            except BaseException as exc:
                error.append(exc)

        t = threading.Thread(target=runner)
        t.start()
        t.join()
        if error:
            raise error[0]
        return result[0]


class JyDomainAgentAdapter:
    """JY ``BaseAgent`` 를 LIBRA ``InformationAgentProtocol`` 로 노출.

    Judge 가 ``run(...)`` 을 동기 호출하면 내부에서 async ``deliberate(ctx)``
    실행 후 ``AgentResponse`` 로 변환한다. ``proposed_trades`` 는 Judge 의
    candidate plan 에서 주입할 수 있도록 ``portfolio.cash_weight`` 같은
    PortfolioSnapshot 필드와 별도 채널 (``note``) 로 받는다.
    """

    def __init__(
        self,
        jy_agent: DomainBaseAgent,
        *,
        agent_id: str,
        owner_scope: str,
    ) -> None:
        self._jy = jy_agent
        self.agent_id = agent_id
        self.owner_scope = owner_scope

    def run(
        self,
        *,
        query: str,
        context: str | None = None,
        fallback: str | None = None,
        note: str | None = None,
        turn_number: int,
        portfolio: PortfolioSnapshot,
        knowledge_base: LocalKnowledgeBase,
        depth: str = "medium",
    ) -> AgentResponse:
        del fallback, knowledge_base, depth

        ctx = portfolio_snapshot_to_domain_context(
            portfolio,
            market_context_str=str(context or ""),
        )
        verdict = _run_async(self._jy.deliberate(ctx))
        return domain_verdict_to_agent_response(
            verdict,
            agent_id=self.agent_id,
            turn_number=turn_number,
            query=query,
        )

    async def deliberate(self, ctx: DomainPortfolioContext) -> Any:
        """원본 JY 패턴 호출이 필요할 때를 위한 직통 패스."""
        return await self._jy.deliberate(ctx)


def build_domain_agent_adapters() -> dict[str, JyDomainAgentAdapter]:
    """도메인 에이전트의 LIBRA 어댑터 묶음을 생성."""
    from .compliance import ComplianceAgent
    from .esg_agent import ESGAgent
    from .execution_agent import ExecutionAgent
    from .liquidity_agent import LiquidityAgent
    from .macro_agent import MacroAgent
    from .risk import RiskAgent
    from .sentiment_agent import SentimentAgent
    from .tax import TaxAgent
    from .technical_analysis_agent import TechnicalAnalysisAgent

    return {
        "risk": JyDomainAgentAdapter(RiskAgent(), agent_id="risk", owner_scope="Risk Agent (Vora)"),
        "tax": JyDomainAgentAdapter(TaxAgent(), agent_id="tax", owner_scope="Tax Agent (Reed)"),
        "compliance": JyDomainAgentAdapter(
            ComplianceAgent(), agent_id="compliance", owner_scope="Compliance Agent (Clarke)"
        ),
        "macro": JyDomainAgentAdapter(
            MacroAgent(), agent_id="macro", owner_scope="Macro Agent (Halden)"
        ),
        "sentiment": JyDomainAgentAdapter(
            SentimentAgent(), agent_id="sentiment", owner_scope="Sentiment Agent (Imo)"
        ),
        "execution": JyDomainAgentAdapter(
            ExecutionAgent(), agent_id="execution", owner_scope="Execution Agent (Tien)"
        ),
        "esg": JyDomainAgentAdapter(ESGAgent(), agent_id="esg", owner_scope="ESG Agent (Esme)"),
        "liquidity": JyDomainAgentAdapter(
            LiquidityAgent(), agent_id="liquidity", owner_scope="Liquidity Agent"
        ),
        "technical": JyDomainAgentAdapter(
            TechnicalAnalysisAgent(),
            agent_id="technical",
            owner_scope="Technical Analysis Agent",
        ),
    }
