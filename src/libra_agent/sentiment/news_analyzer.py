"""
뉴스 분석기 — Phase 2 파이프라인 오케스트레이터 (v2)

설계안 §4 + 사용자 confirm #2 에 따라 SENTIMENT_MODE 환경변수로
세 가지 운영 모드를 지원한다:

  - "gemini-claude"  (default)  : FinBERT → Gemini Flash → Claude 적대 검토 → 합의
  - "finbert-ollama"             : FinBERT → Ollama (legacy llama3.2:3b)
  - "fingpt-local"               : FinBERT → Ollama(fingpt-8b GGUF, 차후 스왑 대상)

각 모드는 동일한 NewsAnalysisResult 스키마를 반환하므로 상위 SentimentAgent 가
모드 변경에 영향받지 않는다.

Returns:
    NewsAnalysisResult — vote, rationale, portfolio_sentiment_score, model_used
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass

from .finbert_service import FinBERTService, HeadlineSentiment, get_finbert
from .ollama_client import OllamaClient, OllamaUnavailableError, get_ollama

logger = logging.getLogger(__name__)

# ── 운영 모드 (CLAUDE.md: 모든 결정 문서화) ───────────────────
SENTIMENT_MODE = os.environ.get("SENTIMENT_MODE", "gemini-claude").lower()
ALLOWED_MODES = {"gemini-claude", "finbert-ollama", "fingpt-local"}
if SENTIMENT_MODE not in ALLOWED_MODES:
    logger.warning(
        "[NewsAnalyzer] 알 수 없는 SENTIMENT_MODE=%s — 'gemini-claude' 로 폴백",
        SENTIMENT_MODE,
    )
    SENTIMENT_MODE = "gemini-claude"

# 파이프라인 필터 임계값
NEGATIVE_KEEP_RATIO = 0.3  # 전체 헤드라인 중 상위 부정 30% 를 Ollama에 전달
MIN_HEADLINES_TO_PASS = 3  # 최소 이 이상이 필터 통과해야 Ollama 호출

_NL = "\n"  # 개행 상수 — 중첩 문자열 내 가독성 확보용


@dataclass
class NewsAnalysisResult:
    portfolio_sentiment_score: float  # -1.0 (극단 공포) ~ +1.0 (극단 탐욕)
    vote: str  # approve | reject | abstain
    rationale: str  # 분석 근거 텍스트
    finbert_summary: list[HeadlineSentiment]  # FinBERT 필터링 결과
    model_used: str  # "finbert+ollama:llama3.2:3b" 등
    negative_count: int = 0
    positive_count: int = 0


_SYSTEM_PROMPT = (
    "당신은 포트폴리오 감성 분석 AI입니다.\n"
    "주어진 뉴스 헤드라인과 포트폴리오 컨텍스트를 분석하여 정확히 아래 JSON 형식으로만 응답하세요.\n"
    "다른 텍스트는 절대 출력하지 마세요.\n\n"
    '{"sentiment_score": <-1.0~+1.0>, "vote": "<approve|reject|abstain>", "rationale": "<한국어 분석 근거>"}\n\n'
    "판단 기준:\n"
    "- approve: 리밸런싱에 유리한 감성 (중립~긍정)\n"
    "- reject:  강한 부정 뉴스 (전쟁, 금리 급등, 규제 충격 등)\n"
    "- abstain: 정보 불충분 또는 혼합 신호"
)


async def analyze_news(
    headlines: list[str],
    portfolio_summary: str = "",
    finbert: FinBERTService | None = None,
    ollama: OllamaClient | None = None,
) -> NewsAnalysisResult | None:
    """
    Phase 2 뉴스 감성 파이프라인 실행 (모드별 분기).

    Returns:
        NewsAnalysisResult — 성공시.
        None — 완전 실패시 (상위 호출자가 Gemini 폴백 실행).
    """
    if not headlines:
        return None

    fb = finbert or get_finbert()

    # ── Step 1: FinBERT 스크리닝 (모든 모드 공통) ──────────────
    all_sentiments = await fb.score_headlines(headlines)
    agg_score = FinBERTService.aggregate_score(all_sentiments)

    negatives = [s for s in all_sentiments if s.label == "negative"]
    positives = [s for s in all_sentiments if s.label == "positive"]

    keep_n = max(MIN_HEADLINES_TO_PASS, int(len(headlines) * NEGATIVE_KEEP_RATIO))
    top_negatives = sorted(negatives, key=lambda s: s.score, reverse=True)[:keep_n]
    top_positives = sorted(positives, key=lambda s: s.score, reverse=True)[:3]
    filtered = top_negatives + top_positives

    logger.info(
        "[NewsAnalyzer] FinBERT: 전체=%d 부정=%d 긍정=%d 점수=%.2f mode=%s",
        len(headlines),
        len(negatives),
        len(positives),
        agg_score,
        SENTIMENT_MODE,
    )

    # ── 헤드라인 부족 처리 ────────────────────────────────────
    if not filtered or len(filtered) < 2:
        return NewsAnalysisResult(
            portfolio_sentiment_score=agg_score,
            vote="abstain",
            rationale="뉴스 헤드라인이 부족하거나 모두 중립으로 분류되었습니다.",
            finbert_summary=filtered,
            model_used="finbert-only",
            negative_count=len(negatives),
            positive_count=len(positives),
        )

    # ── Step 2 분기: SENTIMENT_MODE 별 ─────────────────────────
    if SENTIMENT_MODE == "gemini-claude":
        try:
            from .gemini_claude_collab import run_collab

            result = await run_collab(
                filtered=filtered,
                portfolio_summary=portfolio_summary,
                fallback_score=agg_score,
            )
            if result is not None:
                result.negative_count = len(negatives)
                result.positive_count = len(positives)
                return result
            logger.warning("[NewsAnalyzer] gemini-claude 협업 결과 없음 → Ollama 폴백")
        except Exception as e:
            logger.warning("[NewsAnalyzer] gemini-claude 협업 실패: %s → Ollama 폴백", e)
        # 폴백 — Ollama (legacy)

    # SENTIMENT_MODE in {"finbert-ollama", "fingpt-local"} 또는 위 폴백
    oc = ollama or get_ollama()

    # fingpt-local 모드는 OLLAMA_MODEL 환경변수가 별도로 fingpt-8b 로 지정돼 있어야 함.
    if (
        SENTIMENT_MODE == "fingpt-local"
        and "fingpt" not in (getattr(oc, "model", "") or "").lower()
    ):
        logger.warning(
            "[NewsAnalyzer] fingpt-local 모드인데 OLLAMA_MODEL=%s — Modelfile 등록 후 재기동 필요",
            getattr(oc, "model", "?"),
        )

    # ── Step 2: Ollama 딥 분석 ───────────────────────────────
    parts = []
    for s in filtered:
        parts.append("- [" + s.label.upper() + " " + str(round(s.score * 100)) + "%] " + s.headline)
    headlines_text = _NL.join(parts)

    user_prompt = (
        "뉴스 헤드라인:"
        + _NL
        + headlines_text
        + _NL
        + _NL
        + "포트폴리오 현황:"
        + _NL
        + (portfolio_summary or "(정보 없음)")
        + _NL
        + _NL
        + "위 정보를 바탕으로 JSON 형식으로 감성 분석하세요."
    )

    try:
        raw = await oc.generate(
            prompt=user_prompt,
            system=_SYSTEM_PROMPT,
            max_tokens=400,
        )
        json_match = re.search(r"\{[\s\S]*?\}", raw)
        if not json_match:
            raise ValueError("JSON 파싱 실패: " + raw[:100])

        parsed = json.loads(json_match.group())
        score = float(parsed.get("sentiment_score", agg_score))
        vote = str(parsed.get("vote", "abstain")).lower()
        if vote not in ("approve", "reject", "abstain"):
            vote = "abstain"
        rationale = str(parsed.get("rationale", raw))
        model_label = "finbert+" + oc.model

        logger.info("[NewsAnalyzer] Ollama 완료: vote=%s score=%.2f", vote, score)

        return NewsAnalysisResult(
            portfolio_sentiment_score=score,
            vote=vote,
            rationale=rationale,
            finbert_summary=filtered,
            model_used=model_label,
            negative_count=len(negatives),
            positive_count=len(positives),
        )

    except OllamaUnavailableError as e:
        logger.warning("[NewsAnalyzer] Ollama 미사용: %s", e)
        return None

    except Exception as e:
        logger.warning("[NewsAnalyzer] 오류: %s → Gemini 폴백 예정", e)
        return None
