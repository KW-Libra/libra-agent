"""리밸런싱 카덴스 (cadence) 단일 진입점.

v2 백테스트 ([[15_v2_결과_2020-2023]]) 결과 — 일일 리밸런싱은
LLM 호출 비용 대비 신호 가치가 낮음. 주·월 단위로 전환하면서
freshness 임계와 데이터 소스 선택을 카덴스에 맞춰 조정한다.

설계 원칙:
- 카덴스가 길어질수록 freshness 임계를 비례 완화 (잘못된 stale 경고 방지).
- 카덴스가 길어지면 실시간 Kafka/ksqlDB 경로를 비활성화 (불필요한 인프라 의존).
- 기본값(default=daily)은 회귀 0. 운영 권장값은 monthly — env 로 명시 전환.
  (governance_config 와 동일한 패턴: 기존 상수와 일치하는 기본값.)

환경 변수:
    LIBRA_REBALANCE_CADENCE = daily | weekly | biweekly | monthly   # 기본 daily
    LIBRA_CADENCE_ENABLE_REALTIME = true | false                    # 기본 cadence별 추천값 사용
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CadenceConfig:
    name: str  # "daily" | "weekly" | "biweekly" | "monthly"
    # Freshness 임계 (초). market_data_injector 의 FRESHNESS_* 상수를 override.
    freshness_price_sec: int
    freshness_news_sec: int
    freshness_macro_sec: int
    # 실시간 스트림 (Kafka 헤드라인, ksqlDB pull) 사용 여부.
    enable_realtime_stream: bool
    # 추천 데이터 소스 라벨 — 로깅/감사용. 실제 fetcher 선택에 활용 가능.
    data_source: str  # "stream" | "batch_eod"


# 카덴스별 기본 프로필.
# 핵심 휴리스틱:
#   - price freshness: 카덴스 한 주기의 1/4 까지 허용 (예: 주간이면 시세는 약 1.5일).
#   - news freshness:  카덴스 한 주기의 1/2 까지 허용.
#   - macro freshness: 카덴스보다 길게 (분기 발표주기 대비 보수적으로 1주 이상).
_PROFILES: dict[str, CadenceConfig] = {
    "daily": CadenceConfig(
        name="daily",
        freshness_price_sec=300,  # 5분 — 기존 값 유지 (회귀 0)
        freshness_news_sec=1800,  # 30분
        freshness_macro_sec=86_400,  # 24시간
        enable_realtime_stream=True,
        data_source="stream",
    ),
    "weekly": CadenceConfig(
        name="weekly",
        freshness_price_sec=6 * 3600,  # 6시간
        freshness_news_sec=24 * 3600,  # 24시간
        freshness_macro_sec=7 * 86_400,  # 7일
        enable_realtime_stream=False,
        data_source="batch_eod",
    ),
    "biweekly": CadenceConfig(
        name="biweekly",
        freshness_price_sec=24 * 3600,  # 1일
        freshness_news_sec=3 * 24 * 3600,  # 3일
        freshness_macro_sec=14 * 86_400,
        enable_realtime_stream=False,
        data_source="batch_eod",
    ),
    "monthly": CadenceConfig(
        name="monthly",
        freshness_price_sec=3 * 24 * 3600,  # 3일
        freshness_news_sec=7 * 24 * 3600,  # 7일
        freshness_macro_sec=30 * 86_400,
        enable_realtime_stream=False,
        data_source="batch_eod",
    ),
}

_DEFAULT_NAME = "daily"


def load_cadence_config() -> CadenceConfig:
    """매 호출마다 환경을 다시 읽는다 (테스트 monkeypatch 지원)."""
    raw = os.environ.get("LIBRA_REBALANCE_CADENCE", _DEFAULT_NAME).strip().lower()
    profile = _PROFILES.get(raw, _PROFILES[_DEFAULT_NAME])

    realtime_override = os.environ.get("LIBRA_CADENCE_ENABLE_REALTIME")
    if realtime_override is not None:
        flag = realtime_override.strip().lower() in {"1", "true", "yes", "on"}
        if flag != profile.enable_realtime_stream:
            from dataclasses import replace

            profile = replace(profile, enable_realtime_stream=flag)
    return profile


def known_cadences() -> tuple[str, ...]:
    return tuple(_PROFILES.keys())
