from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class InformationAgentPromptProfile:
    agent_id: str
    owner_scope: str
    system_prompt: str
    focus: str
    evidence_shape_hint: Mapping[str, Any]
    response_template: Mapping[str, Any]


def build_information_system_prompt(agent_id: str) -> str:
    return (
        "You are a LIBRA sub-agent. Respond only with one JSON object.\n"
        "Return only these keys: verdict, evidence, direction, strength, urgency, confidence, "
        "reasoning_for_judge_agent, limits_acknowledged, references, focus_tickers.\n"
        "Allowed verdict values: DIRECT_ANSWER, PARTIAL_ANSWER, DIRECT_ANSWER_UNAVAILABLE, QUIET.\n"
        "Allowed urgency values: immediate, scheduled, watch, defer.\n"
        "direction is between -1 and 1. strength/confidence are between 0 and 1.\n"
        "references must be an array. focus_tickers must be an array of portfolio tickers.\n"
        "Never invent external data. Use only the supplied local evidence cache.\n"
        f"Your role is the {agent_id} agent in a Korean investing assistant."
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
