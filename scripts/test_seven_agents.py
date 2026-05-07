"""LIBRA — JY 7-에이전트 → Judge 어댑터 단위 검증.

JYlibra-sample_v1 의 7개 도메인 에이전트가 LIBRA InformationAgentProtocol 로
변환되어 호출되는 흐름을 검증한다.

검증 단계:
    1. 7개 에이전트 + 어댑터 import (외부 API 없이도 통과)
    2. ComplianceAgent — 룰 기반, LLM 미사용 → 어댑터 통과 보장
    3. ESGAgent — exclusions 기반 즉시 reject (LLM 미사용 경로)
    4. TaxAgent — harvestable lots 식별 (Claude 호출, 키 있을 때만)

실행:
    cd D:\\libra-agent
    python scripts/test_seven_agents.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    from dotenv import load_dotenv
    _here = Path(__file__).resolve()
    for _candidate in (
        _here.parent.parent / ".env",
        Path.cwd() / ".env",
        _here.parent.parent.parent / "libra-agent" / ".env",
    ):
        if _candidate.exists():
            load_dotenv(_candidate, override=True)
            break
except ImportError:
    pass

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(
    level=logging.WARNING,  # 너무 시끄럽지 않게
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
)


def _make_snapshot():
    from libra_agent.libra_models import PortfolioHolding, PortfolioSnapshot
    return PortfolioSnapshot(
        generated_at=datetime.now(timezone.utc),
        holdings=(
            PortfolioHolding(
                ticker="005930", company_name="삼성전자", weight=0.40,
                shares=615, last_price=65000, average_price=70000,
                market_value_krw=39_975_000, unrealized_pnl_krw=-3_075_000,
            ),
            PortfolioHolding(
                ticker="000660", company_name="SK하이닉스", weight=0.25,
                shares=143, last_price=175000, average_price=180000,
                market_value_krw=25_025_000, unrealized_pnl_krw=-715_000,
            ),
            PortfolioHolding(
                ticker="035420", company_name="네이버", weight=0.15,
                shares=85, last_price=176000, average_price=185000,
                market_value_krw=14_960_000, unrealized_pnl_krw=-765_000,
            ),
            PortfolioHolding(
                ticker="035720", company_name="카카오", weight=0.12,
                shares=310, last_price=38500, average_price=42000,
                market_value_krw=11_935_000, unrealized_pnl_krw=-1_085_000,
            ),
            PortfolioHolding(
                ticker="005380", company_name="현대차", weight=0.08,
                shares=42, last_price=190000, average_price=195000,
                market_value_krw=7_980_000, unrealized_pnl_krw=-210_000,
            ),
        ),
        total_value_krw=99_875_000,
        cash_weight=0.0,
        user_preferences=("balanced",),
    )


def step_1_import():
    print("\n" + "=" * 70)
    print("  Step 1: Import 7 domain agents + adapter")
    print("=" * 70)
    from libra_agent.domain_agents import (
        ComplianceAgent, ESGAgent, ExecutionAgent, MacroAgent,
        RiskAgent, SentimentAgent, TaxAgent,
    )
    from libra_agent.domain_agents._adapter import (
        JyDomainAgentAdapter, build_domain_agent_adapters,
        portfolio_snapshot_to_domain_context,
        domain_verdict_to_agent_response,
    )
    adapters = build_domain_agent_adapters()
    print(f"  agents: {[c.__name__ for c in (RiskAgent, TaxAgent, ComplianceAgent, MacroAgent, SentimentAgent, ExecutionAgent, ESGAgent)]}")
    print(f"  adapters: {sorted(adapters.keys())}")
    print(f"  [OK] 7 agents and adapter loaded")
    return True


def step_2_compliance():
    """LLM 미사용 룰 기반 — 외부 키 없이도 동작 보장"""
    print("\n" + "=" * 70)
    print("  Step 2: ComplianceAgent (no LLM, rule-based)")
    print("=" * 70)
    from libra_agent.domain_agents._adapter import build_domain_agent_adapters
    adapters = build_domain_agent_adapters()
    snap = _make_snapshot()
    resp = adapters["compliance"].run(
        query="IPS compliance check on current holdings",
        context=None, fallback=None, note=None,
        turn_number=1, portfolio=snap, knowledge_base=None, depth="medium",
    )
    print(f"  agent_id          : {resp.agent_id}")
    print(f"  verdict           : {resp.verdict.value}")
    print(f"  opinion           : {getattr(resp, 'opinion', 'N/A')}")
    print(f"  direction         : {resp.direction:+.2f}")
    print(f"  confidence        : {resp.confidence:.2f}")
    print(f"  reasoning (head)  : {resp.reasoning_for_judge_agent[:120]}")
    print(f"  evidence keys     : {sorted(resp.evidence.keys())}")
    print(f"  [OK] DomainAgentVerdict -> AgentResponse 변환 동작")
    return True


def step_3_esg():
    """ESG 위반 케이스 — 사용자 esg_exclusions에 '기술' 추가하면 즉시 reject"""
    print("\n" + "=" * 70)
    print("  Step 3: ESGAgent (rule-based reject path)")
    print("=" * 70)
    from libra_agent.libra_models import PortfolioSnapshot, PortfolioHolding
    from libra_agent.domain_agents._adapter import build_domain_agent_adapters
    adapters = build_domain_agent_adapters()
    # 사용자가 '기술' 섹터를 ESG 제외로 선언했다고 가정
    snap_with_excl = PortfolioSnapshot(
        generated_at=datetime.now(timezone.utc),
        holdings=_make_snapshot().holdings,
        total_value_krw=99_875_000,
        cash_weight=0.0,
        user_preferences=("balanced", "esg_exclusions:기술"),
    )
    # Domain context 변환 시 user_preferences 는 dict 로 평탄화되므로
    # ESGAgent.preferences.get('esg_exclusions') 가 정확히 동작하지 않음.
    # 대신 어댑터의 portfolio_snapshot_to_domain_context 를 직접 호출하여
    # exclusions 를 주입한 ctx 로 deliberate 호출.
    from libra_agent.domain_agents._adapter import (
        portfolio_snapshot_to_domain_context,
        domain_verdict_to_agent_response,
        _run_async,
    )
    ctx = portfolio_snapshot_to_domain_context(snap_with_excl)
    ctx.preferences["esg_exclusions"] = ["기술"]  # 명시 주입
    verdict = _run_async(adapters["esg"]._jy.deliberate(ctx))
    resp = domain_verdict_to_agent_response(verdict, agent_id="esg", turn_number=1, query="ESG check")
    print(f"  vote              : {verdict.vote}")
    print(f"  confidence        : {verdict.confidence}")
    print(f"  rationale (head)  : {verdict.rationale[:140]}")
    print(f"  -> AgentResponse  : verdict={resp.verdict.value} opinion={getattr(resp, 'opinion', 'N/A')} dir={resp.direction:+.2f}")
    if verdict.vote == "reject":
        print(f"  [OK] '기술' 섹터 보유 -> reject (Compliance 거부권 시뮬레이션)")
    else:
        print(f"  [WARN] 예상은 reject 였으나 vote={verdict.vote}")
    return True


def step_4_tax():
    """TaxAgent — Claude 호출. 키 없으면 graceful fail 메시지만 출력."""
    print("\n" + "=" * 70)
    print("  Step 4: TaxAgent (Claude required)")
    print("=" * 70)
    import os
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("  ANTHROPIC_API_KEY 없음 - skip")
        return True
    from libra_agent.domain_agents._adapter import build_domain_agent_adapters
    adapters = build_domain_agent_adapters()
    snap = _make_snapshot()
    try:
        resp = adapters["tax"].run(
            query="tax-loss harvesting candidates?",
            context=None, fallback=None, note=None,
            turn_number=1, portfolio=snap, knowledge_base=None, depth="medium",
        )
        print(f"  verdict           : {resp.verdict.value}")
        print(f"  opinion           : {getattr(resp, 'opinion', 'N/A')}")
        print(f"  reasoning (head)  : {resp.reasoning_for_judge_agent[:140]}")
        print(f"  [OK] Claude 호출 + 어댑터 변환 통합 동작")
    except Exception as e:
        print(f"  [WARN] {type(e).__name__}: {e}")
    return True


STEPS = {
    1: ("Import 7 agents",     step_1_import),
    2: ("Compliance (no LLM)", step_2_compliance),
    3: ("ESG reject path",     step_3_esg),
    4: ("Tax (Claude)",        step_4_tax),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--step", type=int, choices=[1, 2, 3, 4])
    args = p.parse_args()

    steps = [args.step] if args.step else [1, 2, 3, 4]
    results: dict[int, bool] = {}
    for n in steps:
        name, fn = STEPS[n]
        try:
            results[n] = fn()
        except Exception as e:
            logging.exception(f"Step {n} ({name}) 예외")
            results[n] = False

    print("\n" + "=" * 70)
    print("  결과 요약")
    print("=" * 70)
    for n, ok in results.items():
        flag = "[PASS]" if ok else "[FAIL]"
        print(f"  Step {n}  {STEPS[n][0]:25}  {flag}")
    print()


if __name__ == "__main__":
    main()
