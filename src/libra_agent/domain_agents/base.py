"""
Base class for all LIBRA agents.
Each agent receives a portfolio snapshot and returns a verdict.

변경 사항 (v2):
  - LLMRouter 통합 — Claude + Gemini 자동 라우팅
  - MarketDataInjector — 실시간 데이터를 시스템 프롬프트에 자동 주입
  - PortfolioContext 필드 정규화 (holdings / total_value 명시적 요구)
  - _ask_llm() → agent_id 기반 자동 모델 선택
  - data_freshness_note — stale 데이터 경고를 LLM에 자동 전달

버그 수정:
  - event_processor.py가 PortfolioContext(portfolio=...) 로 생성하던 타입 불일치
    → from_supabase_raw() 팩토리 메서드 추가
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ._services.llm_router import LLMRouter

logger = logging.getLogger(__name__)


# ── 포트폴리오 컨텍스트 ────────────────────────────────────────────

@dataclass
class PortfolioContext:
    """
    에이전트에 전달되는 포트폴리오 스냅샷.

    주의: event_processor.py에서 `portfolio=` 키워드로 생성하던 버그 수정됨.
    이 클래스는 `holdings` + `total_value`를 명시적으로 요구함.
    """
    user_id:         str
    holdings:        list[dict[str, Any]]    # [{symbol, quantity, weight, current_price, average_price, ...}]
    preferences:     dict[str, Any]          # investment_preferences 테이블 데이터
    total_value:     float
    proposed_trades: list[dict[str, Any]] = field(default_factory=list)

    # 선택적: 실시간 데이터 (MarketDataInjector가 주입)
    market_context_str: str = ""             # MarketContext.to_prompt_string() 결과

    # 선택적: 수익률 히스토리 (금융공학 엔진용)
    returns_data: dict[str, list[float]] | None = None   # symbol → daily_returns

    # 선택적: 사용자별 LLM 라우터 (오케스트레이터가 deliberate() 시작 시 주입)
    # TYPE_CHECKING import로 순환 의존 없이 타입 힌트 지원
    router: "LLMRouter | None" = field(default=None, repr=False)

    @classmethod
    def from_supabase_raw(
        cls,
        user_id: str,
        portfolio_data: dict[str, Any],
        prefs_data: dict[str, Any],
        proposed_trades: list[dict[str, Any]] | None = None,
    ) -> "PortfolioContext":
        """
        Supabase 쿼리 결과에서 PortfolioContext 생성.
        event_processor.py의 PortfolioContext(portfolio=...) 버그를 수정한 팩토리.
        """
        holdings = portfolio_data.get("holdings", [])
        total_value = float(portfolio_data.get("total_value", 0.0))

        # holdings에 weight 없으면 시장가치 기반 자동 계산
        tv = total_value or sum(
            h.get("market_value", h.get("current_price", 0) * h.get("quantity", 0))
            for h in holdings
        )
        if tv > 0:
            for h in holdings:
                if not h.get("weight"):
                    mv = h.get("market_value", h.get("current_price", 0) * h.get("quantity", 0))
                    h["weight"] = mv / tv

        return cls(
            user_id=user_id,
            holdings=holdings,
            preferences=prefs_data or {},
            total_value=tv,
            proposed_trades=proposed_trades or [],
        )


# ── 에이전트 판결 ────────────────────────────────────────────────

@dataclass
class AgentVerdict:
    agent_id:    str
    vote:        str        # "approve" | "reject" | "abstain"
    confidence:  float      # 0.0 – 1.0
    rationale:   str
    signals:     list[dict[str, Any]] = field(default_factory=list)
    llm_used:    str = ""   # 실제 사용된 LLM 모델명 (감사 로그용)


# ── 기본 에이전트 클래스 ─────────────────────────────────────────

class BaseAgent(ABC):
    """
    모든 전문 에이전트의 기반 클래스.
    서브클래스는 `deliberate()` 구현 필요.

    LLM 호출:
        text, model = self._ask_llm(system, user, ctx=ctx)
        # agent_id 기반 자동 라우팅: Claude Sonnet/Haiku/Gemini Flash 등
    """

    agent_id: str
    name: str
    role: str

    def __init__(self) -> None:
        self._router = None   # 지연 초기화 (순환 import 방지)

    def _get_router(self):
        if self._router is None:
            from ._services.llm_router import get_router
            self._router = get_router()
        return self._router

    @abstractmethod
    async def deliberate(self, ctx: PortfolioContext) -> AgentVerdict:
        ...

    def _ask_llm(
        self,
        system: str,
        user: str,
        max_tokens: int = 1024,
        cross_validate: bool = False,
        ctx: PortfolioContext | None = None,
    ) -> tuple[str, str]:
        """
        멀티 LLM 호출. agent_id에 따라 최적 모델 자동 선택.

        우선순위:
          1. ctx.router — 오케스트레이터가 사용자별 DB 키로 생성한 라우터
          2. self._router — 캐시된 글로벌 라우터 (env 키 기반)

        Returns:
            (response_text, model_name_used)
        """
        # 사용자별 라우터가 ctx에 주입되어 있으면 우선 사용
        if ctx is not None and ctx.router is not None:
            router = ctx.router
        else:
            router = self._get_router()

        # 실시간 시장 데이터 + 신선도 경고 주입
        enriched_user = user
        enriched_system = system

        if ctx and ctx.market_context_str:
            enriched_user = f"### 실시간 시장 데이터\n{ctx.market_context_str}\n\n---\n{user}"
            # 신선도 경고가 있으면 시스템 프롬프트에도 삽입
            warnings = [
                line for line in ctx.market_context_str.split("\n") if "⚠️" in line
            ]
            if warnings:
                enriched_system = "\n".join(warnings) + "\n\n" + system

        response = router.ask(
            agent_id=self.agent_id,
            system=enriched_system,
            user=enriched_user,
            max_tokens=max_tokens,
            cross_validate=cross_validate,
        )

        if hasattr(router, "model_name_for"):
            model_used = router.model_name_for(self.agent_id)
        else:
            model_used = "unknown"

        return response, model_used

    # ── 하위 호환성 ─────────────────────────────────────────────

    def _ask_claude(
        self,
        system: str,
        user: str,
        model: str = "claude-haiku-4-5-20251001",
        ctx: PortfolioContext | None = None,
    ) -> str:
        """
        [Deprecated] 기존 코드 호환용. _ask_llm()으로 위임하여 정책·폴백 완전 지원.
        신규 코드는 _ask_llm() 사용 권장.

        변경: 이전에는 router._call_claude()를 직접 호출하여 사용자가 Gemini-only
        정책을 설정했을 때 RuntimeError가 발생했음. 이제 _ask_llm()을 경유하므로
        라우터의 폴백 로직과 사용자별 ctx.router를 모두 올바르게 활용함.
        """
        logger.debug(f"[{self.agent_id}] _ask_claude() → _ask_llm() 위임 (policy-aware)")
        rationale, _ = self._ask_llm(system=system, user=user, max_tokens=1024, ctx=ctx)
        return rationale
