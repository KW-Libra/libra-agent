"""
Sentiment Agent (Imo) -- Phase 2: FinBERT+Ollama, Phase 1: Gemini Flash fallback
"""

from __future__ import annotations

import json
import logging
import re

from .base import AgentVerdict, BaseAgent, PortfolioContext

logger = logging.getLogger(__name__)


class SentimentAgent(BaseAgent):
    agent_id = "sentiment"
    name = "Imo"
    role = "Sentiment Analyst"
    FEAR_THRESHOLD = -0.30
    GREED_THRESHOLD = 0.50

    def _extract_headlines(self, ctx: PortfolioContext) -> list[str]:
        """ctx.market_context_str 에서 " - " 접두사 뉴스 헤드라인 추출."""
        lines = []
        for line in (ctx.market_context_str or "").splitlines():
            s = line.strip()
            if s.startswith("- ") and len(s) > 10:
                lines.append(s[2:].strip())
        return lines[:30]

    # -- Phase 2: FinBERT + Ollama --------------------------------

    async def _try_phase2(self, ctx: PortfolioContext) -> AgentVerdict | None:
        try:
            from libra_agent.sentiment.news_analyzer import analyze_news

            headlines = self._extract_headlines(ctx)
            if not headlines:
                return None
            top_holdings = sorted(ctx.holdings, key=lambda h: h.get("weight", 0), reverse=True)[:5]
            portfolio_summary = json.dumps(
                {
                    "top_holdings": [
                        {
                            "symbol": h["symbol"],
                            "weight": h.get("weight", 0),
                            "sector": h.get("sector", ""),
                        }
                        for h in top_holdings
                    ],
                },
                ensure_ascii=False,
            )
            result = await analyze_news(headlines=headlines, portfolio_summary=portfolio_summary)
            if result is None:
                return None
            score = result.portfolio_sentiment_score
            vote = result.vote
            if score < self.FEAR_THRESHOLD:
                vote = "abstain"
            elif score > self.GREED_THRESHOLD:
                vote = "approve"
            confidence = abs(score) * 0.4 + 0.5
            signals = [
                {
                    "label": "감성 점수",
                    "value": str(round(score, 2)),
                    "range": "[-1.0 공포~+1.0 탐욕]",
                },
                {"label": "부정 기사 수", "value": result.negative_count},
                {"label": "긍정 기사 수", "value": result.positive_count},
                {"label": "분석 모델", "value": result.model_used},
            ]
            logger.info("[SentimentAgent] Phase2: vote=%s score=%.2f", vote, score)
            return AgentVerdict(
                agent_id=self.agent_id,
                vote=vote,
                confidence=round(confidence, 2),
                rationale=result.rationale,
                signals=signals,
                llm_used=result.model_used,
            )
        except Exception as e:
            logger.warning("[SentimentAgent] Phase2 오류: %s", e)
            return None

    # -- Phase 1: Gemini Flash (fallback) -------------------------

    async def _run_phase1(self, ctx: PortfolioContext) -> AgentVerdict:
        symbols = [h["symbol"] for h in ctx.holdings[:10]]
        top = sorted(ctx.holdings, key=lambda h: h.get("weight", 0), reverse=True)[:5]
        portfolio_summary = json.dumps(
            {
                "top_holdings": [
                    {"symbol": h["symbol"], "weight": h.get("weight", 0)} for h in top
                ],
                "proposed_trades": ctx.proposed_trades[:5],
            },
            ensure_ascii=False,
            indent=2,
        )

        system_msg = (
            "당신은 Imo입니다. 포트폴리오 감성 분석 전문가입니다. "
            "주입된 최신 뉴스와 포트폴리오를 분석하여 "
            "approve / reject / abstain 중 하나를 포함한 JSON으로 답하세요: "
            '{"sentiment_score": 0.0, "vote": "approve", "rationale": "..."}'
        )
        user_msg = "포트폴리오:" + chr(10) + portfolio_summary
        rationale, model_used = self._ask_llm(system=system_msg, user=user_msg, ctx=ctx)

        sentiment_score = 0.0
        vote = "abstain"
        try:
            m = re.search(r"\{.*\}", rationale, re.DOTALL)
            if m:
                p = json.loads(m.group())
                sentiment_score = float(p.get("sentiment_score", 0.0))
                vote = p.get("vote", "abstain")
                rationale = p.get("rationale", rationale)
        except (json.JSONDecodeError, ValueError, AttributeError):
            lower = rationale.lower()
            if "reject" in lower:
                vote = "reject"
            elif "approve" in lower:
                vote = "approve"

        if sentiment_score < self.FEAR_THRESHOLD:
            vote = "abstain"
        elif sentiment_score > self.GREED_THRESHOLD:
            vote = "approve"
        confidence = abs(sentiment_score) * 0.4 + 0.5
        return AgentVerdict(
            agent_id=self.agent_id,
            vote=vote,
            confidence=round(confidence, 2),
            rationale=rationale,
            signals=[
                {"label": "감성 점수", "value": str(round(sentiment_score, 2))},
                {"label": "분석 종목", "value": len(symbols)},
            ],
            llm_used=model_used,
        )

    # -- 메인 진입점 -----------------------------------------------

    async def deliberate(self, ctx: PortfolioContext) -> AgentVerdict:
        result = await self._try_phase2(ctx)
        if result is not None:
            return result
        logger.info("[SentimentAgent] Phase1 (Gemini) 실행")
        return await self._run_phase1(ctx)
