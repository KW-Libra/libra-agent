"""
gemini_claude_collab.py — Gemini × Claude 적대 검토 협업 모드

설계안 §4 + 사용자 confirm #2 의 "Gemini × Claude prompt-engineering collaboration"
을 구현한다. 절차:

  1. Gemini Flash 가 FinBERT 필터 결과 + 포트폴리오 컨텍스트를 받아
     1차 요약·sentiment_score·suggested_action 을 JSON 으로 산출.
  2. Claude (Anthropic API) 가 동일 입력 + Gemini 산출물을 받아
     "adversarial review" — 약점/근거 부족/대안 vote — 를 JSON 으로 산출.
  3. 두 결과를 머지: vote 가 일치하면 confidence 가산, 불일치 시 더 보수적인
     쪽(reject > abstain > approve) 을 채택하고 rationale 에 양측 의견 명시.

API 키 부재 시:
  - GEMINI_API_KEY 미설정 → 함수 진입 직후 None 반환 → 상위 analyze_news 가
    Ollama 경로로 폴백.
  - ANTHROPIC_API_KEY 만 부재 → Gemini 결과만 채택 (review_skipped=True).

키는 env 에서만 읽는다 (CLAUDE.md: never hard-code secrets).
"""

from __future__ import annotations

import json
import logging
import os
import re

import httpx

from .finbert_service import HeadlineSentiment

logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
)
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")

_HTTP_TIMEOUT = 25.0


# ──────────────────────────────────────────────────────────────
# 프롬프트
# ──────────────────────────────────────────────────────────────

GEMINI_SYSTEM_PROMPT = (
    "You are FinGPT-Lite, a financial sentiment analyst.\n"
    "Given filtered news headlines and a portfolio summary, return ONLY this JSON:\n"
    '{"sentiment_score": <-1.0~+1.0>, '
    '"vote": "<approve|reject|abstain>", '
    '"summary": "<1~3 Korean sentences>", '
    '"affected_sectors": [{"sector": "<name>", "impact": "negative|positive|neutral", "confidence": <0~1>}], '
    '"suggested_action": "<hold|rebalance_to_target|reduce_growth_tilt|defensive_rotation|tax_harvest>", '
    '"risk_level": "<low|medium|high>"}\n'
    "Do NOT output anything outside the JSON object."
)

CLAUDE_SYSTEM_PROMPT = (
    "당신은 다른 AI(Gemini)의 포트폴리오 감성 분석을 적대적으로 검토하는 시니어 애널리스트입니다.\n"
    "Gemini의 출력에 (a) 근거가 약하거나 (b) 데이터와 모순되거나 (c) 더 보수적인 vote 가\n"
    "필요한지 점검하고, 다음 JSON 스키마로만 답하세요:\n"
    '{"agree": <true|false>, '
    '"final_vote": "<approve|reject|abstain>", '
    '"final_score": <-1.0~+1.0>, '
    '"rationale_ko": "<한국어 근거 2~4문장>", '
    '"weaknesses": ["<지적 사항>"]}\n'
    "다른 텍스트는 출력하지 마세요."
)


# ──────────────────────────────────────────────────────────────
# Gemini 호출
# ──────────────────────────────────────────────────────────────


async def _call_gemini(headlines_block: str, portfolio_summary: str) -> dict | None:
    if not GEMINI_API_KEY:
        return None
    user = (
        "Filtered headlines:\n"
        f"{headlines_block}\n\n"
        f"Portfolio:\n{portfolio_summary or '(no info)'}\n\n"
        "Return JSON now."
    )
    payload = {
        "system_instruction": {"parts": [{"text": GEMINI_SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 1500,
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }
    try:
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
            r = await client.post(
                f"{GEMINI_URL}?key={GEMINI_API_KEY}",
                json=payload,
            )
            r.raise_for_status()
            data = r.json()
            text = (
                data.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
            )
        parsed = _extract_json(text)
        if parsed is None:
            logger.warning(
                "[gemini-claude] Gemini 응답 JSON 파싱 실패. raw text (앞 500자):\n%s",
                text[:500] if text else "(빈 응답)",
            )
            logger.warning(
                "[gemini-claude] Gemini API 응답 구조 (앞 800자):\n%s",
                json.dumps(data, ensure_ascii=False)[:800] if data else "(데이터 없음)",
            )
        return parsed
    except Exception as e:
        logger.warning("[gemini-claude] Gemini 호출 실패: %s", e)
        return None


# ──────────────────────────────────────────────────────────────
# Claude 호출 (adversarial review)
# ──────────────────────────────────────────────────────────────


async def _call_claude_review(
    headlines_block: str,
    portfolio_summary: str,
    gemini_out: dict,
) -> dict | None:
    if not ANTHROPIC_API_KEY:
        return None
    user = (
        "원본 헤드라인:\n"
        f"{headlines_block}\n\n"
        "포트폴리오:\n"
        f"{portfolio_summary or '(정보 없음)'}\n\n"
        "Gemini 출력:\n"
        f"{json.dumps(gemini_out, ensure_ascii=False)}\n\n"
        "위를 검토하여 JSON 으로만 답하세요."
    )
    payload = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": 600,
        "system": CLAUDE_SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": user}],
    }
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    try:
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
            r = await client.post(ANTHROPIC_URL, json=payload, headers=headers)
            r.raise_for_status()
            data = r.json()
            blocks = data.get("content", [])
            text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        return _extract_json(text)
    except Exception as e:
        logger.warning("[gemini-claude] Claude 검토 실패: %s", e)
        return None


def _extract_json(text: str) -> dict | None:
    if not text:
        return None
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        return json.loads(m.group())
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────
# 머지 로직
# ──────────────────────────────────────────────────────────────

_VOTE_RANK = {"reject": 0, "abstain": 1, "approve": 2}


def _merge(gemini_out: dict, claude_out: dict | None) -> dict:
    """두 모델 결과를 보수적으로 합친다."""
    g_vote = str(gemini_out.get("vote", "abstain")).lower()
    g_score = float(gemini_out.get("sentiment_score", 0.0))
    g_summary = str(gemini_out.get("summary", ""))

    if claude_out is None:
        return {
            "vote": g_vote if g_vote in _VOTE_RANK else "abstain",
            "score": g_score,
            "rationale": g_summary,
            "review_skipped": True,
            "gemini": gemini_out,
        }

    c_vote = str(claude_out.get("final_vote", g_vote)).lower()
    c_score = float(claude_out.get("final_score", g_score))
    c_rationale = str(claude_out.get("rationale_ko", ""))
    agree = bool(claude_out.get("agree", False))

    # 합의 시 그대로, 불합치 시 더 보수적인 쪽
    if g_vote == c_vote:
        final_vote = g_vote
    else:
        final_vote = g_vote if _VOTE_RANK.get(g_vote, 1) <= _VOTE_RANK.get(c_vote, 1) else c_vote

    final_score = (g_score + c_score) / 2.0
    merged_rationale = (
        f"[Gemini] {g_summary}\n[Claude 검토] {c_rationale}"
        if not agree
        else f"{c_rationale} (Gemini·Claude 합의)"
    )

    return {
        "vote": final_vote if final_vote in _VOTE_RANK else "abstain",
        "score": max(-1.0, min(1.0, final_score)),
        "rationale": merged_rationale,
        "review_skipped": False,
        "gemini": gemini_out,
        "claude": claude_out,
    }


# ──────────────────────────────────────────────────────────────
# 진입점
# ──────────────────────────────────────────────────────────────


async def run_collab(
    filtered: list[HeadlineSentiment],
    portfolio_summary: str,
    fallback_score: float,
):
    """
    상위 news_analyzer.analyze_news 가 호출.
    실패 시 None 반환 → 상위가 Ollama 경로로 폴백.
    """
    # 지연 import — 순환참조 회피
    from .news_analyzer import NewsAnalysisResult

    if not GEMINI_API_KEY:
        logger.info("[gemini-claude] GEMINI_API_KEY 없음 — 모드 비활성, 상위에서 폴백 처리")
        return None

    headlines_block = "\n".join(
        f"- [{s.label.upper()} {round(s.score * 100)}%] {s.headline}" for s in filtered
    )

    gemini_out = await _call_gemini(headlines_block, portfolio_summary)
    if not gemini_out:
        return None

    claude_out = await _call_claude_review(headlines_block, portfolio_summary, gemini_out)

    merged = _merge(gemini_out, claude_out)

    model_used = (
        "finbert+gemini-flash+claude" if not merged["review_skipped"] else "finbert+gemini-flash"
    )

    return NewsAnalysisResult(
        portfolio_sentiment_score=merged["score"]
        if merged["score"] is not None
        else fallback_score,
        vote=merged["vote"],
        rationale=merged["rationale"],
        finbert_summary=filtered,
        model_used=model_used,
    )
