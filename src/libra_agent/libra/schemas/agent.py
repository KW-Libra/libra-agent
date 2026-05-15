from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal


class Direction(StrEnum):
    INCREASE = "INCREASE"
    HOLD = "HOLD"
    DECREASE = "DECREASE"


@dataclass(slots=True)
class Vote:
    subject: str
    direction: Direction
    magnitude_pct: float
    confidence: float
    concerns: list[str] = field(default_factory=list)
    informational: bool = False

    def __post_init__(self) -> None:
        if not self.subject:
            raise ValueError("vote subject is required")
        self.magnitude_pct = max(-100.0, min(100.0, float(self.magnitude_pct)))
        self.confidence = max(0.0, min(1.0, float(self.confidence)))
        self.concerns = [str(item) for item in self.concerns[:5] if str(item).strip()]

    def to_dict(self) -> dict[str, object]:
        return {
            "subject": self.subject,
            "direction": self.direction.value,
            "magnitude_pct": self.magnitude_pct,
            "confidence": self.confidence,
            "concerns": list(self.concerns),
            "informational": self.informational,
        }


@dataclass(slots=True)
class AgentOpinion:
    agent: str
    round: Literal[1, 2] = 1
    votes: list[Vote] = field(default_factory=list)
    silence_reason: str | None = None
    reasoning: str = ""
    evidence_refs: list[str] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)
    previous_round_summary: str | None = None
    exposed_signals: list[str] = field(default_factory=list)
    delta_from_round1: Literal["UNCHANGED", "STRENGTHENED", "WEAKENED", "REVERSED"] | None = None
    delta_rationale: str | None = None

    def __post_init__(self) -> None:
        if not self.agent:
            raise ValueError("agent is required")
        if self.round not in (1, 2):
            raise ValueError("round must be 1 or 2")
        if not self.votes and not self.silence_reason:
            self.silence_reason = "발화 가능한 신호가 없습니다."
        if self.round == 2 and not self.delta_from_round1:
            self.delta_from_round1 = "UNCHANGED"

    def to_dict(self) -> dict[str, object]:
        return {
            "agent": self.agent,
            "round": self.round,
            "votes": [vote.to_dict() for vote in self.votes],
            "silence_reason": self.silence_reason,
            "reasoning": self.reasoning,
            "evidence_refs": list(self.evidence_refs),
            "metadata": dict(self.metadata),
            "previous_round_summary": self.previous_round_summary,
            "exposed_signals": list(self.exposed_signals),
            "delta_from_round1": self.delta_from_round1,
            "delta_rationale": self.delta_rationale,
        }
