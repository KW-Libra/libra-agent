"""NewsAgent — LangGraph Judge 안에서 호출되는 텍스트 분석 에이전트.

확장 (2026-05-07):
  - 보조 정량 신호: ``analyze_with_collab()`` 헬퍼로
    FinBERT → Gemini → Claude 적대 검토 결과를 dict 로 반환.
    Judge 또는 NewsAgent.run() 흐름에서 이 점수를 텍스트 분석과
    결합하여 의사결정에 반영 가능.
"""

from __future__ import annotations

from typing import Any

from .base import DelegatingInformationAgent
from ..prompts import NEWS_PROMPT_PROFILE


class NewsAgent(DelegatingInformationAgent):
    agent_id = "news"
    owner_scope = "News Agent"
    prompt_profile = NEWS_PROMPT_PROFILE
    owner_task_brief = (
        "보유 종목 관련 뉴스, 교차 확인 여부, 가격/시장 반응을 우선 본다. "
        "매크로는 관련성이 있을 때만 보조 근거로 쓴다. "
        "정량 sentiment 가 필요하면 libra_agent.libra.agents.news_agent.analyze_with_collab "
        "을 통해 FinBERT → Gemini → Claude 적대 검토 결과를 가져와 텍스트 분석을 보강한다."
    )


async def analyze_with_collab(
    headlines: list[str],
    portfolio_summary: str = "",
) -> dict[str, Any] | None:
    """NewsAgent 보조 헬퍼.

    sentiment 파이프라인을 호출하여 정량 신호를 dict 로 반환한다.

    Returns:
        성공 시 ``{"score", "vote", "rationale", "model_used",
        "positive_count", "negative_count"}``. 실패 시 ``None``.
    """
    from libra_agent.sentiment.news_analyzer import analyze_news

    result = await analyze_news(
        headlines=headlines,
        portfolio_summary=portfolio_summary,
    )
    if result is None:
        return None

    return {
        "score": result.portfolio_sentiment_score,
        "vote": result.vote,
        "rationale": result.rationale,
        "model_used": result.model_used,
        "positive_count": result.positive_count,
        "negative_count": result.negative_count,
    }
