from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .agent import Direction
from .compliance import ComplianceCheck


class ConsensusBranch(str, Enum):
    STRONG_CONSENSUS = "STRONG_CONSENSUS"
    STRONG_HOLD = "STRONG_HOLD"
    WEAK_CONSENSUS = "WEAK_CONSENSUS"
    CONFLICT = "CONFLICT"
    INSUFFICIENT_VOTES = "INSUFFICIENT_VOTES"


class DecisionType(str, Enum):
    HOLD = "HOLD"
    DEFER = "DEFER"
    REBALANCE = "REBALANCE"
    USER_DECISION_REQUIRED = "USER_DECISION_REQUIRED"


class DecisionBranch(str, Enum):
    CONSENSUS = "CONSENSUS"
    WEAK_CONSENSUS_CONSERVATIVE = "WEAK_CONSERVATIVE"
    STRONG_CONFLICT = "STRONG_CONFLICT"
    COMPLIANCE_VETO = "COMPLIANCE_VETO"
    HOLD = "HOLD"


@dataclass(slots=True)
class ConsensusScore:
    subject: str
    weighted_score: float
    confidence_sum: float
    vote_distribution: dict[Direction, int]
    branch: ConsensusBranch

    def to_dict(self) -> dict[str, object]:
        return {
            "subject": self.subject,
            "weighted_score": self.weighted_score,
            "confidence_sum": self.confidence_sum,
            "vote_distribution": {key.value: value for key, value in self.vote_distribution.items()},
            "branch": self.branch.value,
        }


@dataclass(slots=True)
class MediatorDecision:
    consensus_per_subject: dict[str, ConsensusScore]
    targets_to_recall: list[str]
    skip_round_2: bool
    rationale: str

    def to_dict(self) -> dict[str, object]:
        return {
            "consensus_per_subject": {
                subject: score.to_dict() for subject, score in self.consensus_per_subject.items()
            },
            "targets_to_recall": list(self.targets_to_recall),
            "skip_round_2": self.skip_round_2,
            "rationale": self.rationale,
        }


@dataclass(slots=True)
class Trade:
    subject: str
    delta_pct: float
    rationale: str = ""

    def to_dict(self) -> dict[str, object]:
        return {"subject": self.subject, "delta_pct": self.delta_pct, "rationale": self.rationale}


@dataclass(slots=True)
class UserOption:
    label: str
    supporting_agents: list[str]
    expected_effect: str

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "supporting_agents": list(self.supporting_agents),
            "expected_effect": self.expected_effect,
        }


@dataclass(slots=True)
class FinalDecision:
    decision: DecisionType
    branch: DecisionBranch
    compliance_check: ComplianceCheck
    reasoning: str
    trades: list[Trade] = field(default_factory=list)
    user_question: str | None = None
    user_options: list[UserOption] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "decision": self.decision.value,
            "branch": self.branch.value,
            "trades": [trade.to_dict() for trade in self.trades],
            "compliance_check": self.compliance_check.to_dict(),
            "reasoning": self.reasoning,
            "user_question": self.user_question,
            "user_options": [option.to_dict() for option in self.user_options] if self.user_options else None,
        }
