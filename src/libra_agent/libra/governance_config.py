"""거버넌스 임계·계수의 단일 진입점 (Team A — Core 5 + Mediator 영역).

코드 변경 없이 환경변수로 1차 Judge(Mediator)까지의 동작을 토글하기 위한 config.
기본값(default)이 기존 코드의 상수와 정확히 일치하므로 회귀 없음.

영역 경계:
- 본 config 는 **Team A 책임 영역만** 다룬다 — Core 5 발화 변환(committee.py),
  합의 분류(mediator/consensus.py), Profit 휴리스틱(agents/profit_agent.py),
  Core prompt 변형(prompts/base.py).
- ``judge/final.py`` (Final Judge — 2차) 의 ``WEAK_CONSERVATIVE_COEF``,
  ``trade_min_delta_pct`` 같은 항목은 **Team B 영역** 이라 본 config 에 포함하지 않는다.
  Team B 가 같은 패턴으로 별도 config(또는 본 config 확장) 를 만들 수 있다.

선택:
    LIBRA_GOVERNANCE_PRESET = default | aggressive | noise_resist | balanced
                            | info_expand | maximum_aggressive

Regime asymmetry (Week 2 Phase B — [[16]] §4, 2026-05-27):
    LIBRA_REGIME = neutral | bear | bull               # 기본 neutral, MacroAgent 미도입 시 수동 override
    LIBRA_GOV_BEAR_SELL_STRONG_THRESHOLD = 0.45        # bear regime에서 SELL signal의 STRONG 임계
    LIBRA_GOV_BEAR_BUY_STRONG_THRESHOLD = 0.75         # bear regime에서 BUY signal의 STRONG 임계
    LIBRA_GOV_BULL_BUY_STRONG_THRESHOLD = 0.45         # bull regime mirror
    LIBRA_GOV_BULL_SELL_STRONG_THRESHOLD = 0.75

개별 항목 override (프리셋과 곱해서 적용):
    LIBRA_GOV_STRONG_CONSENSUS_THRESHOLD = 0.55
    LIBRA_GOV_WEAK_CONSENSUS_THRESHOLD = 0.3
    LIBRA_GOV_STRONG_HOLD_RATIO_THRESHOLD = 0.6
    LIBRA_GOV_INSUFFICIENT_VOTES_CONFIDENCE_SUM = 1.0
    LIBRA_GOV_R2_MIN_CONFIDENCE = 0.4
    LIBRA_GOV_R2_MAX_TARGETS = 4
    LIBRA_GOV_DIRECTION_THRESHOLD = 0.1
    LIBRA_GOV_MAGNITUDE_SCALE_FACTOR = 10.0
    LIBRA_GOV_MAGNITUDE_CAP_PCT = 10.0
    LIBRA_GOV_MAGNITUDE_FLOOR = 0.1
    LIBRA_GOV_INFO_AGENTS = "cost,execution"               # 콤마 구분
    LIBRA_GOV_HOLD_SILENCE_AGENTS = "news,report,sentiment"
    LIBRA_GOV_HOLD_SILENCE_CONFIDENCE_THRESHOLD = 0.05
    LIBRA_GOV_PROFIT_CONFIDENCE_BASE = 0.42
    LIBRA_GOV_PROFIT_CONFIDENCE_PER_SIGNAL = 0.08
    LIBRA_GOV_PROFIT_CONFIDENCE_MAX = 0.75
    LIBRA_GOV_PROFIT_DIRECTION_SCALE = 8.0
    LIBRA_PROMPT_VARIANT = default | calibrated

호출 시점에 환경을 읽으므로 테스트에서 ``monkeypatch.setenv`` 가능.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields, replace


@dataclass(frozen=True, slots=True)
class GovernanceConfig:
    # --- Mediator (1차 Judge) — consensus 분류 임계 ---
    strong_consensus_threshold: float = 0.6
    weak_consensus_threshold: float = 0.3
    strong_hold_ratio_threshold: float = 0.6
    insufficient_votes_confidence_sum: float = 1.0
    r2_min_confidence: float = 0.4
    r2_max_targets: int = 4
    # --- Core 5 AgentResponse → Vote 변환 (committee.py 어댑터) ---
    direction_threshold: float = 0.1
    magnitude_scale_factor: float = 10.0
    magnitude_cap_pct: float = 10.0
    magnitude_floor: float = 0.1
    info_agents: frozenset[str] = frozenset({"disclosure", "cost", "execution"})
    hold_silence_agents: frozenset[str] = frozenset({"news", "report", "sentiment"})
    hold_silence_confidence_threshold: float = 0.05
    # --- Profit 휴리스틱 ---
    profit_confidence_base: float = 0.42
    profit_confidence_per_signal: float = 0.08
    profit_confidence_max: float = 0.75
    profit_direction_scale: float = 8.0
    # --- Prompt 변형 ---
    prompt_variant: str = "default"
    # --- Regime asymmetry (Week 2 Phase B, [[16]] §4) ---
    # neutral 일 때는 strong_consensus_threshold 사용 (회귀 0).
    # bear/bull 일 때는 sign-aware 임계 적용.
    regime: str = "neutral"
    bear_sell_strong_threshold: float = 0.45
    bear_buy_strong_threshold: float = 0.75
    bull_buy_strong_threshold: float = 0.45
    bull_sell_strong_threshold: float = 0.75


_DEFAULT = GovernanceConfig()

PRESETS: dict[str, GovernanceConfig] = {
    "default": _DEFAULT,
    "aggressive": replace(
        _DEFAULT,
        strong_consensus_threshold=0.5,
        magnitude_scale_factor=15.0,
        info_agents=frozenset({"cost", "execution"}),
        hold_silence_agents=frozenset(),  # 침묵 처리 끔 — 모든 발화 점수에 반영
        profit_confidence_base=0.55,
        profit_confidence_max=0.9,
        profit_direction_scale=12.0,
    ),
    "noise_resist": replace(
        _DEFAULT,
        direction_threshold=0.2,
        strong_hold_ratio_threshold=0.5,
        magnitude_floor=0.0,  # 작은 신호 그대로 작게 — 강제 1% floor 제거
        hold_silence_confidence_threshold=0.15,  # 더 적극적 침묵 처리
    ),
    "balanced": replace(
        _DEFAULT,
        strong_consensus_threshold=0.55,
        direction_threshold=0.15,
        profit_confidence_base=0.45,
        profit_confidence_max=0.8,
        profit_direction_scale=10.0,
    ),
    "info_expand": replace(
        _DEFAULT,
        info_agents=frozenset({"cost", "execution"}),
    ),
    "maximum_aggressive": replace(
        _DEFAULT,
        strong_consensus_threshold=0.4,
        weak_consensus_threshold=0.2,
        direction_threshold=0.05,
        magnitude_scale_factor=20.0,
        info_agents=frozenset({"execution"}),  # disclosure·cost 모두 점수 반영
        hold_silence_agents=frozenset(),
        profit_confidence_base=0.6,
        profit_confidence_max=0.95,
        profit_direction_scale=15.0,
    ),
}


def load_governance_config() -> GovernanceConfig:
    preset_name = os.environ.get("LIBRA_GOVERNANCE_PRESET", "default").strip().lower()
    cfg = PRESETS.get(preset_name, _DEFAULT)

    overrides: dict[str, object] = {}
    for field_obj in fields(GovernanceConfig):
        env_key = f"LIBRA_GOV_{field_obj.name.upper()}"
        raw = os.environ.get(env_key)
        if raw is None:
            continue
        overrides[field_obj.name] = _coerce(field_obj.type, field_obj.name, raw, getattr(cfg, field_obj.name))

    prompt_variant_env = os.environ.get("LIBRA_PROMPT_VARIANT")
    if prompt_variant_env is not None:
        overrides["prompt_variant"] = prompt_variant_env.strip().lower() or _DEFAULT.prompt_variant

    regime_env = os.environ.get("LIBRA_REGIME")
    if regime_env is not None:
        regime_clean = regime_env.strip().lower()
        if regime_clean in {"neutral", "bear", "bull"}:
            overrides["regime"] = regime_clean

    if overrides:
        return replace(cfg, **overrides)
    return cfg


def _coerce(type_str: str, name: str, raw: str, default_value: object) -> object:
    text = raw.strip()
    if name in {"info_agents", "hold_silence_agents"}:
        return frozenset(part.strip() for part in text.split(",") if part.strip())
    if isinstance(default_value, bool):
        return text.lower() in {"1", "true", "yes", "on"}
    if isinstance(default_value, int) and not isinstance(default_value, bool):
        return int(text)
    if isinstance(default_value, float):
        return float(text)
    return text
