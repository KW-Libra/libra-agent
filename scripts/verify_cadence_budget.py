"""LIBRA — 카덴스 + LLM budget guard 동작 검증 스크립트.

새로 추가된 두 설정의 효과를 한눈에 보여준다:
  - LIBRA_REBALANCE_CADENCE  : daily | weekly | biweekly | monthly
  - LIBRA_LLM_BUDGET         : paid | free
  - LIBRA_LOCAL_LLM_BACKEND  : ollama | llama_cpp (free 모드에서 유료 모델 대체)

외부 API 호출 없음. 라우팅·threshold 계산만 표시.

사용법:
    # 기본값 (회귀 0 확인)
    python scripts/verify_cadence_budget.py

    # 운영 권장 시나리오
    LIBRA_REBALANCE_CADENCE=monthly \
    LIBRA_LLM_BUDGET=free \
    LIBRA_LOCAL_LLM_BACKEND=ollama \
        python scripts/verify_cadence_budget.py

    # 모든 시나리오 일괄 출력
    python scripts/verify_cadence_budget.py --matrix
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


AGENTS = ("risk", "compliance", "tax", "execution", "macro", "sentiment", "esg")


def _show_current() -> None:
    from libra_agent.domain_agents._services import market_data_injector as m
    from libra_agent.domain_agents._services.llm_router import LLMRouter
    from libra_agent.libra.cadence_config import load_cadence_config

    print("─" * 60)
    print(" 환경 변수")
    print("─" * 60)
    for k in (
        "LIBRA_REBALANCE_CADENCE",
        "LIBRA_CADENCE_ENABLE_REALTIME",
        "LIBRA_LLM_BUDGET",
        "LIBRA_LOCAL_LLM_BACKEND",
        "LIBRA_OLLAMA_MODEL",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
    ):
        val = os.environ.get(k)
        if k.endswith("_API_KEY") and val:
            val = val[:6] + "…(set)"
        print(f"  {k:<32} = {val or '<unset>'}")

    cad = load_cadence_config()
    print()
    print("─" * 60)
    print(" 카덴스 설정")
    print("─" * 60)
    print(f"  name                    = {cad.name}")
    print(f"  freshness_price_sec     = {cad.freshness_price_sec:>10,}   ({cad.freshness_price_sec / 3600:.1f}h)")
    print(f"  freshness_news_sec      = {cad.freshness_news_sec:>10,}   ({cad.freshness_news_sec / 3600:.1f}h)")
    print(f"  freshness_macro_sec     = {cad.freshness_macro_sec:>10,}   ({cad.freshness_macro_sec / 86400:.1f}d)")
    print(f"  enable_realtime_stream  = {cad.enable_realtime_stream}")
    print(f"  data_source             = {cad.data_source}")

    print()
    print(" Market Data Injector freshness (런타임 lookup)")
    print(f"  price                   = {m._freshness_price_sec():>10,} sec")
    print(f"  news                    = {m._freshness_news_sec():>10,} sec")
    print(f"  macro                   = {m._freshness_macro_sec():>10,} sec")

    print()
    print("─" * 60)
    print(" LLM 라우팅 (도메인 에이전트별)")
    print("─" * 60)
    router = LLMRouter()
    for agent in AGENTS:
        try:
            model = router._select_model(agent)
            tag = " (LOCAL)" if model.value.startswith("local-") else ""
            print(f"  {agent:<12} → {model.value}{tag}")
        except RuntimeError as exc:
            print(f"  {agent:<12} → ERROR: {exc}")
    print()


_SCENARIOS = [
    ("기본(daily, paid)", {}),
    ("monthly + free + ollama", {
        "LIBRA_REBALANCE_CADENCE": "monthly",
        "LIBRA_LLM_BUDGET": "free",
        "LIBRA_LOCAL_LLM_BACKEND": "ollama",
    }),
    ("weekly + free + no backend", {
        "LIBRA_REBALANCE_CADENCE": "weekly",
        "LIBRA_LLM_BUDGET": "free",
    }),
    ("monthly + paid (실시간 강제)", {
        "LIBRA_REBALANCE_CADENCE": "monthly",
        "LIBRA_CADENCE_ENABLE_REALTIME": "true",
    }),
]


def _matrix() -> None:
    saved = {k: os.environ.get(k) for k in (
        "LIBRA_REBALANCE_CADENCE",
        "LIBRA_CADENCE_ENABLE_REALTIME",
        "LIBRA_LLM_BUDGET",
        "LIBRA_LOCAL_LLM_BACKEND",
    )}
    try:
        for label, env in _SCENARIOS:
            for k in saved:
                os.environ.pop(k, None)
            os.environ.update(env)
            print()
            print("=" * 60)
            print(f" SCENARIO: {label}")
            print("=" * 60)
            _show_current()
    finally:
        for k, v in saved.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matrix",
        action="store_true",
        help="모든 시나리오를 순차 출력",
    )
    args = parser.parse_args()
    if args.matrix:
        _matrix()
    else:
        _show_current()


if __name__ == "__main__":
    main()
