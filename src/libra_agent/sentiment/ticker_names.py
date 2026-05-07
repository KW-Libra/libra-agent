"""
종목코드 → 사용자 친화적 이름 매핑.

HJ-agent-main/scripts/orchestrator.py 의 TICKER_NAMES 를 단일 소스로 분리.
한국 종목(KS)과 미국 종목을 모두 커버.
"""

TICKER_NAMES: dict[str, str] = {
    # ── 한국 (KOSPI/KOSDAQ) ────────────────────────
    "005930": "삼성전자",
    "000660": "SK하이닉스",
    "035420": "네이버",
    "035720": "카카오",
    "005380": "현대차",
    "051910": "LG화학",
    "006400": "삼성SDI",
    "207940": "삼성바이오로직스",
    "068270": "셀트리온",
    "005490": "POSCO홀딩스",
    "373220": "LG에너지솔루션",
    # ── 미국 ───────────────────────────────────────
    "AAPL":  "Apple",
    "MSFT":  "Microsoft",
    "GOOGL": "Alphabet",
    "NVDA":  "NVIDIA",
    "AMZN":  "Amazon",
    "META":  "Meta Platforms",
    "TSLA":  "Tesla",
    "AVGO":  "Broadcom",
}


def ticker_name(ticker: str) -> str:
    """종목코드로 표시명 조회. 매핑 없으면 코드 그대로 반환."""
    return TICKER_NAMES.get(ticker, ticker)
