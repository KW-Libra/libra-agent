from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..governance_config import load_governance_config


@dataclass(frozen=True, slots=True)
class InformationAgentPromptProfile:
    agent_id: str
    owner_scope: str
    system_prompt: str
    focus: str
    evidence_shape_hint: Mapping[str, Any]
    response_template: Mapping[str, Any]


_DIRECTION_VERDICT_GUIDE: dict[str, str] = {
    "disclosure": (
        "방향성 가이드: 어닝 서프라이즈(컨센서스 +5%↑), 자사주 매입, 대주주 추가매수, 신사업 진출, "
        "M&A 합의 같은 명확한 호재 공시는 direction +0.4~+0.8 / strength 0.6+ 로 직접 발화한다. "
        "어닝 미스, 감자, 횡령·분식 의혹, 대규모 유증, 영업정지 같은 악재 공시는 direction −0.4~−0.8. "
        "예정 일정(컨퍼런스 콜 일자 등)만 있고 사실이 없으면 direction 0.0 + watch."
    ),
    "news": (
        "방향성 가이드: 매체 cross-check ≥3 의 명확 호재(신제품 성공, 정책 수혜, 어워드)는 direction +0.3~+0.7. "
        "cross-check ≥3 의 명확 악재(리콜, 소송, 규제 위반)는 −0.3~−0.7. "
        "단일 매체 단발 보도나 추측성 헤드라인은 strength ≤ 0.3 으로 약하게 표시."
    ),
    "report": (
        "방향성 가이드: 컨센서스 목표가 +10%↑ 상향 또는 9/12+ 애널리스트 BUY 의견 상향 = direction +0.4~+0.7. "
        "목표가 −10%↓ 하향이나 SELL 의견 다수 = −0.4~−0.7. "
        "커버리지 부재는 숨기지 말고 DIRECT_ANSWER_UNAVAILABLE + 한계 명시."
    ),
}


_CONFIDENCE_BAND_GUIDE = (
    "Confidence 캘리브레이션 (반드시 따를 것):\n"
    " - 0.85+ : 서로 다른 출처 3개 이상이 같은 방향으로 일치 (예: 공시+뉴스+컨센서스).\n"
    " - 0.65~0.80 : 신뢰 출처 1~2개의 강한 사실 (어닝 서프라이즈 단독, 컨센서스 9/12 상향 등).\n"
    " - 0.40~0.60 : 부분적 신호, 검증 부족 또는 가이드 미만의 사건.\n"
    " - 0.20~0.40 : 약한 정황 또는 단일 단발 보도.\n"
    " - 0.0~0.15 : 관련 evidence 자체가 없음 (이때는 QUIET 또는 DIRECT_ANSWER_UNAVAILABLE).\n"
    "Evidence 가 1개라도 명확한 호재/악재 시그널이면 confidence 0.6 이상으로 적극적으로 발화한다. "
    "안전한 PARTIAL_ANSWER + confidence 0.3 으로 도망치지 않는다."
)


def build_information_system_prompt(agent_id: str) -> str:
    """프로젝트 진입점. governance_config 의 prompt_variant 에 따라 분기."""
    cfg = load_governance_config()
    if cfg.prompt_variant == "calibrated":
        return _build_calibrated(agent_id)
    return _build_default(agent_id)


def _build_default(agent_id: str) -> str:
    return (
        "You are a LIBRA sub-agent. Respond only with one JSON object.\n"
        "Return only these keys: verdict, evidence, direction, strength, urgency, confidence, "
        "reasoning_for_judge_agent, limits_acknowledged, references, focus_tickers.\n"
        "Allowed verdict values: DIRECT_ANSWER, PARTIAL_ANSWER, DIRECT_ANSWER_UNAVAILABLE, QUIET.\n"
        "Allowed urgency values: immediate, scheduled, watch, defer.\n"
        "direction is between -1 and 1. strength/confidence are between 0 and 1.\n"
        "Set confidence to 0 only when your tool observations found no usable evidence. "
        "If you cite at least one event, document, or evidence item, confidence must be >= 0.2; "
        "describe uncertainty in limits_acknowledged instead of zeroing confidence.\n"
        "references must be an array. focus_tickers must be an array of portfolio tickers.\n"
        "Never invent external data. Use only the supplied local evidence cache.\n"
        "You receive agent_tool_observations from your own observe-act-observe tool loop. "
        "Base your answer on those observations and state when the loop found no usable evidence.\n"
        "Write every natural-language value only in Korean. Do not use Japanese kana. "
        "English is allowed only for enum values, JSON keys, tickers, URLs, and source names.\n"
        f"Your role is the {agent_id} agent in a Korean investing assistant."
    )


def _build_calibrated(agent_id: str) -> str:
    direction_guide = _DIRECTION_VERDICT_GUIDE.get(
        agent_id,
        "방향성 가이드: 사실 근거가 명확한 호재면 direction +0.3 이상, 명확한 악재면 −0.3 이하로 발화한다.",
    )
    return (
        "You are a LIBRA sub-agent. Respond only with one JSON object.\n"
        "Return only these keys: verdict, evidence, direction, strength, urgency, confidence, "
        "reasoning_for_judge_agent, limits_acknowledged, references, focus_tickers.\n"
        "Allowed verdict values: DIRECT_ANSWER, PARTIAL_ANSWER, DIRECT_ANSWER_UNAVAILABLE, QUIET.\n"
        "Allowed urgency values: immediate, scheduled, watch, defer.\n"
        "direction is between -1 and 1. strength/confidence are between 0 and 1.\n"
        "references must be an array. focus_tickers must be an array of portfolio tickers.\n"
        "Never invent external data. Use only the supplied local evidence cache.\n"
        "You receive agent_tool_observations from your own observe-act-observe tool loop. "
        "Base your answer on those observations and state when the loop found no usable evidence.\n"
        "Write every natural-language value only in Korean. Do not use Japanese kana. "
        "English is allowed only for enum values, JSON keys, tickers, URLs, and source names.\n"
        f"Your role is the {agent_id} agent in a Korean investing assistant.\n\n"
        f"{direction_guide}\n\n"
        f"{_CONFIDENCE_BAND_GUIDE}\n\n"
        "Verdict 선택 규칙:\n"
        " - DIRECT_ANSWER : 사실 근거가 명확하고 방향성을 자신 있게 말할 수 있을 때.\n"
        " - PARTIAL_ANSWER : 일부 신호만 있고 strength<0.4 일 때.\n"
        " - DIRECT_ANSWER_UNAVAILABLE : 도구는 돌렸으나 해당 종목 evidence 가 없을 때.\n"
        " - QUIET : 이 에이전트의 책임 범위 밖일 때 (예: News 가 재무제표 해석).\n\n"
        "마지막 점검: 호재인데 direction 0.1 이하로 작게 잡거나 confidence 0.3 으로 침묵하지 않는다. "
        "Evidence 가 있으면 그 강도에 비례해 적극적으로 발화한다."
    )


def build_information_response_template(evidence_shape_hint: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "verdict": "PARTIAL_ANSWER",
        "evidence": dict(evidence_shape_hint),
        "direction": 0.0,
        "strength": 0.0,
        "urgency": "defer",
        "confidence": 0.0,
        "reasoning_for_judge_agent": "Use one or two Korean sentences with the next suggested action.",
        "limits_acknowledged": "State the agent boundary briefly if needed, otherwise null.",
        "references": [],
        "focus_tickers": [],
    }
