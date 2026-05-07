"""
Market Data Injector — 실시간 데이터 주입 레이어

LLM 호출 전에 최신 시장 데이터를 수집하여 컨텍스트에 주입합니다.

해결 문제:
  - LLM의 학습 데이터 커트오프 이후의 최신 정보 부재
  - 포트폴리오 스냅샷만으로는 매크로/뉴스 영향 판단 불가
  - 데이터 신선도 미검증 시 stale 데이터로 잘못된 추론 유발

데이터 소스:
  1. KIS API          — 실시간 시세 (Kafka → ksqlDB)
  2. 한국은행 OpenAPI  — 금리/환율/경제지표 (무료)
  3. 네이버 금융 RSS   — 국내 금융 뉴스 (무료, 실시간)
  4. Finnhub / EODHD  — 글로벌 매크로 (옵션, 유료)

데이터 신선도 검증 (DataFreshnessGuard):
  - 시세: 5분 이상 경과 시 재조회
  - 뉴스: 30분 이상 경과 시 재조회
  - 경제지표: 24시간 이상 경과 시 재조회
  - 신선도 실패 시 에이전트 시스템 프롬프트에 경고 주입
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)

# ── 신선도 임계값 (초) ────────────────────────────────────────────
FRESHNESS_PRICE_SEC  = int(os.environ.get("DATA_FRESHNESS_THRESHOLD_SEC", 300))     # 5분
FRESHNESS_NEWS_SEC   = int(os.environ.get("NEWS_FRESHNESS_THRESHOLD_SEC", 1800))    # 30분
FRESHNESS_MACRO_SEC  = 86400                                                         # 24시간


# ── 데이터 신선도 검증 ────────────────────────────────────────────

@dataclass
class DataPoint:
    """단일 데이터 포인트 + 타임스탬프"""
    value: Any
    fetched_at: float = field(default_factory=time.time)
    source: str = "unknown"

    @property
    def age_seconds(self) -> float:
        return time.time() - self.fetched_at

    def is_stale(self, threshold_sec: float) -> bool:
        return self.age_seconds > threshold_sec


class DataFreshnessGuard:
    """
    LLM 호출 전에 데이터 신선도를 검증합니다.
    stale 데이터가 있으면 시스템 프롬프트에 경고를 추가하여
    LLM이 불확실성을 인지하고 과신하지 않도록 합니다.
    """

    def check_prices(self, price_data: list[dict]) -> tuple[bool, str]:
        """
        시세 데이터 신선도 검증.
        Returns: (is_fresh, warning_message)
        """
        if not price_data:
            return False, "⚠️ [데이터 경고] 시세 데이터 없음. 최신 가격을 확인할 수 없습니다."

        stale_symbols = []
        for p in price_data:
            ts = p.get("last_updated_at") or p.get("ts", 0)
            if isinstance(ts, (int, float)):
                age = time.time() - ts
                if age > FRESHNESS_PRICE_SEC:
                    stale_symbols.append(f"{p.get('symbol', '?')}({age/60:.0f}분전)")

        if stale_symbols:
            warning = (
                f"⚠️ [데이터 경고] 다음 종목의 시세가 {FRESHNESS_PRICE_SEC//60}분 이상 지연되었습니다: "
                f"{', '.join(stale_symbols[:5])}. "
                f"실시간 판단 시 주의하십시오."
            )
            return False, warning

        return True, ""

    def check_news(self, news_items: list[dict]) -> tuple[bool, str]:
        """뉴스 신선도 검증"""
        if not news_items:
            return False, "⚠️ [데이터 경고] 최신 뉴스 데이터 없음. 이벤트 리스크를 판단할 수 없습니다."

        oldest = min(n.get("published_at", time.time()) for n in news_items)
        age = time.time() - oldest

        if age > FRESHNESS_NEWS_SEC:
            return False, (
                f"⚠️ [데이터 경고] 뉴스 데이터가 {age/60:.0f}분 전 데이터입니다. "
                f"최신 시장 이벤트가 반영되지 않을 수 있습니다."
            )

        return True, ""

    def check_macro(self, macro_data: dict) -> tuple[bool, str]:
        """매크로 데이터 신선도 검증"""
        if not macro_data:
            return False, "⚠️ [데이터 경고] 매크로 경제지표 없음. 거시 환경을 고려하지 못합니다."

        ts = macro_data.get("fetched_at", 0)
        age = time.time() - ts

        if age > FRESHNESS_MACRO_SEC:
            return False, (
                f"⚠️ [데이터 경고] 경제지표가 {age/3600:.0f}시간 전 데이터입니다."
            )

        return True, ""

    def build_data_quality_note(
        self,
        price_data: list[dict],
        news_items: list[dict],
        macro_data: dict,
    ) -> str:
        """
        LLM 시스템 프롬프트 앞에 붙일 데이터 품질 노트 생성.
        모든 데이터가 신선하면 빈 문자열 반환.
        """
        notes = []

        fresh_p, warn_p = self.check_prices(price_data)
        if not fresh_p:
            notes.append(warn_p)

        fresh_n, warn_n = self.check_news(news_items)
        if not fresh_n:
            notes.append(warn_n)

        fresh_m, warn_m = self.check_macro(macro_data)
        if not fresh_m:
            notes.append(warn_m)

        if notes:
            return "\n".join(notes) + "\n\n"
        return ""


# ── 시장 데이터 수집기 ────────────────────────────────────────────

class MarketDataInjector:
    """
    에이전트 deliberate() 호출 전에 최신 시장 데이터를 수집합니다.

    수집 항목:
    - 보유 종목 실시간 시세 (KIS → ksqlDB PULL)
    - 국내 금융 뉴스 헤드라인 (네이버 금융 RSS)
    - 한국은행 기준금리 / 환율 (BOK OpenAPI)
    - KOSPI 200 지수 변화율

    사용법:
        injector = MarketDataInjector()
        market_ctx = await injector.fetch_context(symbols=["005930", "000660"])
        # market_ctx를 에이전트 user prompt에 주입
    """

    def __init__(self) -> None:
        self._ksqldb_url = os.environ.get("KSQLDB_URL", "http://localhost:8088")
        self._bok_api_key = os.environ.get("BOK_API_KEY", "")
        self._backend_url = os.environ.get("BACKEND_URL", "http://localhost:4000")
        self._freshness  = DataFreshnessGuard()
        self._cache: dict[str, DataPoint] = {}

    # ── 통합 컨텍스트 수집 ────────────────────────────────────────

    async def fetch_context(self, symbols: list[str]) -> "MarketContext":
        """
        병렬로 모든 데이터 소스를 조회하고 MarketContext를 반환합니다.

        뉴스 우선순위:
          1. Kafka libra.market.news 버퍼 (NewsConsumer — 실시간, 2시간 이내)
          2. 네이버 금융 RSS 직접 조회 (폴백)
        """
        prices_task = self._fetch_prices(symbols)
        news_task   = self._fetch_news_headlines()
        macro_task  = self._fetch_macro_indicators()

        prices, news, macro = await asyncio.gather(
            prices_task, news_task, macro_task,
            return_exceptions=True,
        )

        # 예외 처리 — 데이터 수집 실패는 치명적이지 않음
        if isinstance(prices, Exception):
            logger.warning(f"[DataInjector] 시세 조회 실패: {prices}")
            prices = []
        if isinstance(news, Exception):
            logger.warning(f"[DataInjector] 뉴스 조회 실패: {news}")
            news = []
        if isinstance(macro, Exception):
            logger.warning(f"[DataInjector] 매크로 조회 실패: {macro}")
            macro = {}

        # ── Kafka 뉴스 버퍼 병합 ─────────────────────────────
        # NewsConsumer가 사전에 start()돼 있으면 실시간 헤드라인을 주입
        kafka_headlines: list[str] = []
        try:
            from .news_consumer import get_news_consumer
            nc = get_news_consumer()
            kafka_headlines = nc.get_recent_headlines()
            if kafka_headlines:
                logger.info(
                    "[DataInjector] Kafka 뉴스 버퍼: %d개 헤드라인 주입",
                    len(kafka_headlines),
                )
        except Exception as e:
            logger.debug("[DataInjector] Kafka 뉴스 버퍼 접근 실패 (무시): %s", e)

        quality_note = self._freshness.build_data_quality_note(prices, news, macro)

        return MarketContext(
            prices=prices,
            news=news,
            macro=macro,
            data_quality_note=quality_note,
            kafka_headlines=kafka_headlines,
        )

    # ── 시세 조회 (ksqlDB PULL) ──────────────────────────────────

    async def _fetch_prices(self, symbols: list[str]) -> list[dict]:
        """ksqlDB PRICE_SNAPSHOT_TABLE에서 최신 시세 조회"""
        results = []
        try:
            async with aiohttp.ClientSession() as session:
                for sym in symbols[:20]:  # 최대 20개 종목
                    cache_key = f"price:{sym}"
                    if cache_key in self._cache:
                        dp = self._cache[cache_key]
                        if not dp.is_stale(FRESHNESS_PRICE_SEC):
                            results.append(dp.value)
                            continue

                    try:
                        url = f"{self._backend_url}/api/stream/price/{sym}"
                        async with session.get(url, timeout=aiohttp.ClientTimeout(total=3)) as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                data["last_updated_at"] = time.time()
                                self._cache[cache_key] = DataPoint(data, source="ksqldb")
                                results.append(data)
                    except Exception as e:
                        logger.debug(f"[DataInjector] 시세 조회 실패 {sym}: {e}")

        except Exception as e:
            logger.warning(f"[DataInjector] 시세 수집 중 오류: {e}")

        return results

    # ── 뉴스 조회 (네이버 금융 RSS) ──────────────────────────────

    async def _fetch_news_headlines(self) -> list[dict]:
        """
        네이버 금융 RSS에서 최신 금융 뉴스 수집.
        실제 운영에서는 Bloomberg/Reuters 유료 피드 권장.
        """
        cache_key = "news:naver_finance"
        if cache_key in self._cache:
            dp = self._cache[cache_key]
            if not dp.is_stale(FRESHNESS_NEWS_SEC):
                return dp.value  # type: ignore[return-value]

        rss_urls = [
            "https://finance.naver.com/news/news_list.naver?mode=RSS&section=mainnews",
        ]

        articles: list[dict] = []
        try:
            async with aiohttp.ClientSession() as session:
                for url in rss_urls:
                    try:
                        async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                            if resp.status == 200:
                                text = await resp.text()
                                articles.extend(self._parse_rss(text))
                    except Exception as e:
                        logger.debug(f"[DataInjector] RSS 조회 실패 {url}: {e}")
        except Exception as e:
            logger.warning(f"[DataInjector] 뉴스 수집 중 오류: {e}")

        # feedparser 사용 가능하면 더 정교하게 파싱
        if not articles:
            articles = await self._fetch_news_feedparser()

        if articles:
            self._cache[cache_key] = DataPoint(articles[:20], source="rss")

        return articles[:20]

    def _parse_rss(self, xml_text: str) -> list[dict]:
        """간단한 RSS XML 파싱"""
        import re
        items = []
        titles = re.findall(r"<title><!\[CDATA\[(.*?)\]\]></title>", xml_text)
        links  = re.findall(r"<link>(.*?)</link>", xml_text)
        for i, title in enumerate(titles[:10]):
            items.append({
                "title": title.strip(),
                "url": links[i] if i < len(links) else "",
                "published_at": time.time(),
                "source": "naver_finance",
            })
        return items

    async def _fetch_news_feedparser(self) -> list[dict]:
        """feedparser 라이브러리 활용 (더 강건한 파싱)"""
        try:
            import feedparser
            rss_url = "https://finance.naver.com/news/news_list.naver?mode=RSS&section=mainnews"
            loop = asyncio.get_event_loop()
            feed = await loop.run_in_executor(None, feedparser.parse, rss_url)
            return [
                {
                    "title": entry.get("title", ""),
                    "url":   entry.get("link", ""),
                    "published_at": time.time(),
                    "source": "naver_finance_fp",
                }
                for entry in feed.entries[:20]
            ]
        except Exception as e:
            logger.debug(f"[DataInjector] feedparser 실패: {e}")
            return []

    # ── 매크로 지표 (한국은행 OpenAPI) ──────────────────────────

    async def _fetch_macro_indicators(self) -> dict:
        """
        한국은행 OpenAPI에서 기준금리, 환율, CPI 조회.
        API 키 없으면 캐시된 더미 데이터 반환.

        한국은행 OpenAPI: https://ecos.bok.or.kr/api/
        무료 회원가입 후 API 키 발급 가능.
        """
        cache_key = "macro:bok"
        if cache_key in self._cache:
            dp = self._cache[cache_key]
            if not dp.is_stale(FRESHNESS_MACRO_SEC):
                return dp.value  # type: ignore[return-value]

        if not self._bok_api_key:
            # API 키 없으면 LLM에 알림
            fallback = {
                "source": "fallback",
                "note": "한국은행 API 키 미설정 — 매크로 지표 없음. BOK_API_KEY 환경변수 설정 권장.",
                "fetched_at": time.time(),
            }
            return fallback

        macro: dict = {"fetched_at": time.time(), "source": "bok"}

        try:
            async with aiohttp.ClientSession() as session:
                # 기준금리 (통계코드: 722Y001)
                url = (
                    f"https://ecos.bok.or.kr/api/StatisticSearch/{self._bok_api_key}"
                    f"/json/kr/1/1/722Y001/MM/2024-01/2024-12/0101000"
                )
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        rows = data.get("StatisticSearch", {}).get("row", [])
                        if rows:
                            macro["base_rate"] = float(rows[-1].get("DATA_VALUE", 0))
                            macro["base_rate_date"] = rows[-1].get("TIME", "")

                # 원/달러 환율 (통계코드: 731Y003)
                url2 = (
                    f"https://ecos.bok.or.kr/api/StatisticSearch/{self._bok_api_key}"
                    f"/json/kr/1/1/731Y003/MM/2024-01/2024-12/0000001"
                )
                async with session.get(url2, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        rows = data.get("StatisticSearch", {}).get("row", [])
                        if rows:
                            macro["usd_krw"] = float(rows[-1].get("DATA_VALUE", 0))

        except Exception as e:
            logger.warning(f"[DataInjector] BOK API 오류: {e}")
            macro["error"] = str(e)

        if macro:
            self._cache[cache_key] = DataPoint(macro, source="bok")

        return macro


# ── MarketContext 데이터 클래스 ──────────────────────────────────

@dataclass
class MarketContext:
    """에이전트에게 주입할 시장 컨텍스트"""
    prices:           list[dict]
    news:             list[dict]
    macro:            dict
    data_quality_note: str = ""         # 신선도 경고 문자열
    kafka_headlines:  list[str] = field(default_factory=list)  # Kafka 실시간 뉴스

    def to_prompt_string(self, max_news: int = 5) -> str:
        """LLM user prompt에 삽입할 마크다운 문자열 생성"""
        parts = []

        if self.data_quality_note:
            parts.append(self.data_quality_note)

        # 현재 시세
        if self.prices:
            parts.append("### 현재 시세")
            for p in self.prices[:10]:
                sym = p.get("symbol", "?")
                price = p.get("close_price") or p.get("current_price", "N/A")
                chg = p.get("change_rate", "N/A")
                parts.append(f"- {sym}: {price:,} KRW ({chg:+.2f}%)" if isinstance(chg, (int, float)) else f"- {sym}: {price}")

        # Kafka 실시간 뉴스 우선 (있으면 RSS 뉴스 대체)
        if self.kafka_headlines:
            parts.append("\n### 최신 뉴스 헤드라인 (Kafka 실시간)")
            for h in self.kafka_headlines[:max_news]:
                parts.append(f"- {h}")
        elif self.news:
            parts.append("\n### 주요 금융 뉴스 (RSS)")
            for n in self.news[:max_news]:
                parts.append(f"- {n.get('title', '')}")

        # 매크로 지표
        if self.macro and "error" not in self.macro:
            parts.append("\n### 매크로 지표")
            if "base_rate" in self.macro:
                parts.append(f"- 한국은행 기준금리: {self.macro['base_rate']}%")
            if "usd_krw" in self.macro:
                parts.append(f"- 원/달러 환율: {self.macro['usd_krw']:,.0f}원")
            if "note" in self.macro:
                parts.append(f"- {self.macro['note']}")

        return "\n".join(parts)


# ── 글로벌 싱글턴 ────────────────────────────────────────────────
_injector: MarketDataInjector | None = None


def get_injector() -> MarketDataInjector:
    global _injector
    if _injector is None:
        _injector = MarketDataInjector()
    return _injector
