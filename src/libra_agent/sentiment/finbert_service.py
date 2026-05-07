"""
FinBERT 감성 스크리닝 서비스 — Phase 2 뉴스 파이프라인 1단계

ProsusAI/finbert 모델로 뉴스 헤드라인을 빠르게 분류:
  positive / negative / neutral + confidence score

특징:
  - 완전 로컬 실행 (API 비용 없음, CPU-only)
  - 배치 처리 (최대 BATCH_SIZE 헤드라인 동시 처리)
  - transformers 미설치 시 graceful degradation → 전부 neutral 반환
  - 첫 호출 시 모델 자동 다운로드 (~420 MB, 캐시됨)

사용:
    from services.finbert_service import FinBERTService
    fb = FinBERTService()
    results = await fb.score_headlines(["Fed raises rates", "AAPL beats EPS"])
    # [{"label": "negative", "score": 0.91, "headline": "Fed raises rates"}, ...]
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

FINBERT_MODEL = "ProsusAI/finbert"
BATCH_SIZE    = 32   # 한 번에 처리할 최대 헤드라인 수
MAX_LEN       = 128  # 토크나이저 최대 길이 (뉴스 헤드라인 기준 충분)


@dataclass
class HeadlineSentiment:
    headline: str
    label:    str    # "positive" | "negative" | "neutral"
    score:    float  # 해당 label 의 confidence (0.0~1.0)


class FinBERTService:
    """
    FinBERT 기반 뉴스 헤드라인 감성 분류기.
    싱글턴처럼 사용하되 첫 호출 시 모델을 레이지 로드합니다.
    """

    def __init__(self) -> None:
        self._pipeline: Any | None = None
        self._available: bool | None = None  # None = 아직 확인 안 됨

    # ── 레이지 초기화 ─────────────────────────────────────────

    def _load(self) -> bool:
        """transformers 임포트 + 파이프라인 생성. 실패하면 False 반환."""
        if self._available is not None:
            return self._available
        try:
            from transformers import pipeline  # type: ignore[import]
            logger.info("[FinBERT] 모델 로딩 중: %s (첫 실행시 다운로드)", FINBERT_MODEL)
            self._pipeline = pipeline(
                "text-classification",
                model=FINBERT_MODEL,
                tokenizer=FINBERT_MODEL,
                max_length=MAX_LEN,
                truncation=True,
                device=-1,        # CPU 강제 (-1 = CPU, 0+ = GPU 인덱스)
                top_k=None,       # 모든 label score 반환
            )
            logger.info("[FinBERT] 모델 로드 완료")
            self._available = True
        except ImportError:
            logger.warning("[FinBERT] transformers 미설치 — pip install transformers torch")
            self._available = False
        except Exception as e:
            logger.warning("[FinBERT] 모델 로드 실패: %s", e)
            self._available = False
        return self._available

    # ── 배치 처리 ─────────────────────────────────────────────

    async def score_headlines(
        self,
        headlines: list[str],
    ) -> list[HeadlineSentiment]:
        """
        헤드라인 리스트를 FinBERT로 분류합니다.
        transformers 미설치시 모든 항목을 neutral(0.5)로 반환합니다.

        Args:
            headlines: 분류할 뉴스 헤드라인 리스트

        Returns:
            HeadlineSentiment 리스트 (입력 순서 유지)
        """
        if not headlines:
            return []

        # CPU 모델 호출은 동기 — 이벤트 루프 블로킹 방지를 위해 executor 사용
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._score_sync, headlines)

    def _score_sync(self, headlines: list[str]) -> list[HeadlineSentiment]:
        """동기 배치 처리 (run_in_executor에서 호출)."""
        if not self._load() or self._pipeline is None:
            # 폴백: 모두 neutral
            return [
                HeadlineSentiment(headline=h, label="neutral", score=0.5)
                for h in headlines
            ]

        results: list[HeadlineSentiment] = []
        # 배치 분할
        for i in range(0, len(headlines), BATCH_SIZE):
            batch = headlines[i : i + BATCH_SIZE]
            try:
                raw_outputs = self._pipeline(batch)   # list[list[dict]]
                for headline, label_scores in zip(batch, raw_outputs):
                    best = max(label_scores, key=lambda x: x["score"])
                    results.append(HeadlineSentiment(
                        headline=headline,
                        label=best["label"].lower(),   # FinBERT는 POSITIVE 대문자 반환
                        score=round(float(best["score"]), 4),
                    ))
            except Exception as e:
                logger.warning("[FinBERT] 배치 처리 오류: %s", e)
                results.extend([
                    HeadlineSentiment(headline=h, label="neutral", score=0.5)
                    for h in batch
                ])

        return results

    # ── 포트폴리오 감성 점수 계산 ──────────────────────────────

    @staticmethod
    def aggregate_score(sentiments: list[HeadlineSentiment]) -> float:
        """
        HeadlineSentiment 리스트를 단일 포트폴리오 감성 점수 (-1.0 ~ +1.0)로 집계.
        positive=+score, negative=-score, neutral=0 으로 가중 평균.
        """
        if not sentiments:
            return 0.0
        total = 0.0
        for s in sentiments:
            if s.label == "positive":
                total += s.score
            elif s.label == "negative":
                total -= s.score
            # neutral: 0 기여
        return round(total / len(sentiments), 4)


# ── 모듈 수준 싱글턴 (지연 초기화) ─────────────────────────────

_service: FinBERTService | None = None


def get_finbert() -> FinBERTService:
    global _service
    if _service is None:
        _service = FinBERTService()
    return _service
