"""
LIBRA — Gemini × Claude 적대 검토 파이프라인 단위 검증

JYlibra-sample_v1#fix/sentiment-gemini-collab (PRAHE) 통합본.
sentiment 모듈만 단위 테스트 — Kafka/Supabase/Orchestrator 우회.

실행:
    cd D:\\libra-agent
    python scripts/test_news_pipeline.py            # 전체
    python scripts/test_news_pipeline.py --step 3   # 적대 검토만 검증

요구 사항:
    - .env (또는 환경변수) ANTHROPIC_API_KEY  (Step 3, 4)
    - .env (또는 환경변수) GEMINI_API_KEY     (Step 3, 4 — 없으면 Ollama 폴백)
    - transformers + torch 설치               (Step 2, 3 — 없으면 모두 neutral 폴백)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    # .env 후보: 스크립트 인접 → cwd → 형제 libra-agent (모노레포 ↔ 분리레포 양쪽 지원)
    _here = Path(__file__).resolve()
    for _candidate in (
        _here.parent.parent / ".env",
        Path.cwd() / ".env",
        _here.parent.parent.parent / "libra-agent" / ".env",
    ):
        if _candidate.exists():
            # override=True: 시스템 env 에 빈 값으로 미리 set 돼 있어도 .env 값으로 덮어씀
            load_dotenv(_candidate, override=True)
            break
except ImportError:
    pass

# Windows 콘솔 cp949 환경에서도 UTF-8 출력 강제
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
)
log = logging.getLogger("test_news")


SAMPLE_HEADLINES_KOR = [
    "삼성전자, 3분기 영업이익 9조원 돌파 — 메모리 반도체 수요 회복",
    "SK하이닉스 실적 호조에 외국인 매수세 유입",
    "한국은행 기준금리 동결 — 인플레이션 둔화 신호",
    "현대차 미국 EV 보조금 전액 수령 확정 — 가격 경쟁력 회복",
    "카카오 경영진 횡령 혐의로 추가 조사 — 주가 하방 압력",
]

SAMPLE_HEADLINES_ENG = [
    "Samsung Electronics Q3 operating profit surges past 9 trillion won on memory rebound",
    "SK Hynix beats earnings estimates, foreign investors pile in",
    "Bank of Korea holds interest rates as inflation cools",
    "Hyundai secures full US EV tax credit — price competitiveness restored",
    "Kakao executives face additional probe over embezzlement charges, stock under pressure",
]

SAMPLE_PORTFOLIO_SUMMARY = """
{
  "top_holdings": [
    {"symbol": "005930", "weight": 0.40, "sector": "Technology"},
    {"symbol": "000660", "weight": 0.25, "sector": "Technology"},
    {"symbol": "035420", "weight": 0.15, "sector": "Communication"},
    {"symbol": "035720", "weight": 0.12, "sector": "Communication"},
    {"symbol": "005380", "weight": 0.08, "sector": "Consumer Cyclical"}
  ]
}
"""


# ────────────────────────────────────────────────────────────────────
# Step 1 — 데이터 수집 레이어 안내
# ────────────────────────────────────────────────────────────────────

async def step1_rss():
    print("\n" + "=" * 70)
    print("  Step 1: 뉴스 수집 레이어 위치 안내")
    print("=" * 70)
    print()
    print("  본 sentiment 모듈은 헤드라인 입력만 받음.")
    print("  RSS/공시/리포트 수집은 별도 레포 libra-ingest 가 담당:")
    print("    - libra-ingest/src/libra_ingest/pipeline/")
    print("    - libra-ingest/src/libra_ingest/monitor/")
    print()
    print("  알려진 외부 이슈 (PR 코멘트 후속):")
    print("    - 네이버 금융 RSS 엔드포인트 deprecated (200 OK + HTML 에러 페이지)")
    print("    - 대체 소스 후보: 한경/매경 RSS, 연합뉴스, Yahoo Finance")
    print()
    print("  본 스크립트는 SAMPLE_HEADLINES_ENG/KOR 로 단위 테스트만 수행.")
    print("\n  [OK] 안내 완료")
    return True


# ────────────────────────────────────────────────────────────────────
# Step 2 — FinBERT 분류
# ────────────────────────────────────────────────────────────────────

async def _classify_and_print(label_set: str, headlines: list[str]):
    from libra_agent.sentiment.finbert_service import get_finbert
    fb = get_finbert()
    results = await fb.score_headlines(headlines)

    print(f"\n  ── {label_set} ────────────────────────────────────────────")
    for h, r in zip(headlines, results):
        lab = r.label.upper().ljust(8)
        sc = f"{r.score:.3f}"
        print(f"  [{lab}] {sc}  {h[:55]}")

    pos = sum(1 for r in results if r.label == "positive")
    neg = sum(1 for r in results if r.label == "negative")
    neu = len(results) - pos - neg
    print(f"  → positive={pos}  negative={neg}  neutral={neu}")
    return pos, neg, neu


async def step2_finbert():
    print("\n" + "=" * 70)
    print("  Step 2: FinBERT 헤드라인 분류 (한국어 vs 영어)")
    print("=" * 70)
    print("\n  로컬 모델 로딩 (첫 실행은 다운로드)...")

    pos_k, neg_k, neu_k = await _classify_and_print(
        "한국어 헤드라인", SAMPLE_HEADLINES_KOR
    )
    pos_e, neg_e, neu_e = await _classify_and_print(
        "영어 헤드라인 (동일 의미)", SAMPLE_HEADLINES_ENG
    )

    print("\n  ## 한 vs 영 비교:")
    print(f"     한국어: pos={pos_k} neg={neg_k} neu={neu_k}")
    print(f"     영어:   pos={pos_e} neg={neg_e} neu={neu_e}")

    if (pos_k + neg_k) == 0 and (pos_e + neg_e) > 0:
        print("\n  [WARN]  결론: FinBERT 는 영어만 정상 분류, 한국어는 모두 neutral")
        print("     → 한국 시장 시스템에는 KoFinBERT / 한국어 금융 sentiment 모델 필요")
    elif (pos_k + neg_k) == 0 and (pos_e + neg_e) == 0:
        print("\n  [FAIL] 영어도 모두 neutral — transformers/torch 미설치 가능성")
        print("     → pip install transformers torch")
        return False

    print(f"\n  [OK] FinBERT 동작 확인 완료")
    return True


# ────────────────────────────────────────────────────────────────────
# Step 3 — NewsAnalyzer 종합 (FinBERT → Gemini × Claude)
# ────────────────────────────────────────────────────────────────────

async def step3_analyzer():
    print("\n" + "=" * 70)
    print("  Step 3: NewsAnalyzer — Gemini × Claude 적대 검토")
    print("=" * 70)

    if not os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("GEMINI_API_KEY"):
        print("\n  [FAIL] ANTHROPIC_API_KEY / GEMINI_API_KEY 둘 다 없음 — 스킵")
        return False

    from libra_agent.sentiment.news_analyzer import analyze_news

    print(f"\n  SENTIMENT_MODE = {os.environ.get('SENTIMENT_MODE', 'gemini-claude')}")
    print(f"  Anthropic key  = {'있음' if os.environ.get('ANTHROPIC_API_KEY') else '없음'}")
    print(f"  Gemini key     = {'있음' if os.environ.get('GEMINI_API_KEY') else '없음'}")
    print(f"\n  영어 헤드라인 {len(SAMPLE_HEADLINES_ENG)} 건 분석 중...")
    print(f"  (한국어로는 FinBERT 가 부정 0건 → Gemini × Claude 단계 발동 안 함)\n")

    result = await analyze_news(
        headlines=SAMPLE_HEADLINES_ENG,
        portfolio_summary=SAMPLE_PORTFOLIO_SUMMARY,
    )

    if result is None:
        print("  [FAIL] analyze_news() 가 None 반환")
        print("     → news_analyzer.py 의 모드 분기에서 모든 경로가 실패")
        return False

    print(f"  최종 vote          : {result.vote}")
    print(f"  sentiment score    : {result.portfolio_sentiment_score:+.3f}")
    print(f"  positive count     : {result.positive_count}")
    print(f"  negative count     : {result.negative_count}")
    print(f"  사용된 모델         : {result.model_used}")
    print(f"\n  rationale:")
    print(f"    {result.rationale[:300]}")
    if len(result.rationale) > 300:
        print(f"    ... (총 {len(result.rationale)}자)")

    if "claude" in result.model_used:
        print("\n  [OK] Gemini × Claude 적대 검토 동작 확인 (PR 의도대로)")
    elif "gemini" in result.model_used:
        print("\n  [WARN]  Gemini 만 동작, Claude 검토 단계 미실행 (ANTHROPIC_API_KEY 확인)")
    else:
        print("\n  [WARN]  finbert-only 폴백 — 적대 검토 미발동")

    print(f"\n  [OK] NewsAnalyzer OK")
    return True


# ────────────────────────────────────────────────────────────────────
# Step 4 — NewsAgent 활용 시뮬레이션
# ────────────────────────────────────────────────────────────────────

async def step4_news_agent_hook():
    print("\n" + "=" * 70)
    print("  Step 4: LIBRA NewsAgent 통합 시뮬레이션")
    print("=" * 70)
    print()
    print("  NewsAgent (libra/agents/news_agent.py) 는 LangGraph Judge 흐름")
    print("  안에서만 동작하므로 단위 테스트에서는 직접 호출이 어려움.")
    print()
    print("  대신 sentiment 모듈을 NewsAgent 보조 도구로 활용하는 흐름을")
    print("  실제 Judge run 에서 어떻게 호출하는지 시뮬레이션:")
    print()
    print("    1. Judge 가 NewsAgent.run() 호출")
    print("    2. NewsAgent 가 LLM 으로 1차 텍스트 분석 (Claude)")
    print("    3. (선택) news_agent.analyze_with_collab() 호출하여")
    print("       FinBERT → Gemini → Claude 적대 검토 점수 획득")
    print("    4. 두 결과를 NewsAgent 의 AgentResponse 에 결합")
    print()

    if not os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("GEMINI_API_KEY"):
        print("  (API 키 없음 — 시뮬레이션 안내만 출력)")
        return True

    from libra_agent.libra.agents.news_agent import analyze_with_collab

    result = await analyze_with_collab(
        headlines=SAMPLE_HEADLINES_ENG[:3],
        portfolio_summary=SAMPLE_PORTFOLIO_SUMMARY,
    )

    if result is None:
        print("  [FAIL] analyze_with_collab() None — Step 3 결과 확인")
        return False

    print(f"  sentiment score        : {result['score']:+.3f}")
    print(f"  vote                   : {result['vote']}")
    print(f"  사용된 모델             : {result['model_used']}")
    print(f"  positive / negative    : {result['positive_count']} / {result['negative_count']}")
    print()
    print(f"  NewsAgent 가 결합할 신호:")
    print(f"    - 정량 sentiment score (Gemini × Claude 합의)")
    print(f"    - vote (approve/reject/abstain)")
    print(f"    - rationale (양 LLM 의견 + Claude 적대 검토)")
    print()
    print(f"  [OK] NewsAgent ↔ sentiment 모듈 연동 경로 확인")
    return True


# ────────────────────────────────────────────────────────────────────
# 진입점
# ────────────────────────────────────────────────────────────────────

STEPS = {
    1: ("RSS layer note", step1_rss),
    2: ("FinBERT classify", step2_finbert),
    3: ("Gemini x Claude", step3_analyzer),
    4: ("NewsAgent hook", step4_news_agent_hook),
}


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--step", type=int, choices=[1, 2, 3, 4],
                   help="특정 단계만 실행 (생략 시 전체)")
    args = p.parse_args()

    steps_to_run = [args.step] if args.step else [1, 2, 3, 4]

    results: dict[int, bool] = {}
    for n in steps_to_run:
        name, fn = STEPS[n]
        try:
            ok = await fn()
            results[n] = ok
        except Exception as e:
            log.exception(f"Step {n} ({name}) 예외: {e}")
            results[n] = False

    print("\n" + "=" * 70)
    print("  결과 요약")
    print("=" * 70)
    for n, ok in results.items():
        name = STEPS[n][0]
        flag = "[PASS]" if ok else "[FAIL]"
        print(f"  Step {n}  {name:25}  {flag}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
