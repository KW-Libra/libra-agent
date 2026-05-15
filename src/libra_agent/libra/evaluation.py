from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class DecisionEvaluation:
    verdict: str
    direction_accuracy: bool | None
    magnitude_error: float | None
    cost_efficiency: float | None
    note: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "direction_accuracy": self.direction_accuracy,
            "magnitude_error": self.magnitude_error,
            "cost_efficiency": self.cost_efficiency,
            "note": self.note,
        }


def evaluate_decision_outcome(
    *,
    decision: str,
    rebalance_plan: Mapping[str, float],
    signal_score: float,
    user_feedback: str | None,
    realized_return_pct: float,
    cost_pct: float = 0.0,
) -> DecisionEvaluation:
    normalized_decision = decision.strip().upper()
    normalized_feedback = (user_feedback or "").strip().casefold()
    direction_accuracy = _direction_accuracy(
        signal_score=signal_score, realized_return_pct=realized_return_pct
    )
    magnitude_error = round(abs(abs(signal_score) * 10.0 - abs(realized_return_pct)), 4)
    cost_efficiency = (
        round(abs(realized_return_pct) - max(cost_pct, 0.0), 4) if rebalance_plan else None
    )
    has_actionable_recommendation = normalized_decision in {
        "REBALANCE",
        "USER_DECISION_REQUIRED",
    } or bool(rebalance_plan)

    if has_actionable_recommendation and normalized_feedback.startswith("rejected"):
        if direction_accuracy is True:
            return DecisionEvaluation(
                verdict="USER_WRONG",
                direction_accuracy=True,
                magnitude_error=magnitude_error,
                cost_efficiency=cost_efficiency,
                note="User rejected the recommendation, but the realized direction supported the system signal.",
            )
        if direction_accuracy is False:
            return DecisionEvaluation(
                verdict="USER_CORRECT",
                direction_accuracy=False,
                magnitude_error=magnitude_error,
                cost_efficiency=cost_efficiency,
                note="User rejected the recommendation, and the realized direction did not support the system signal.",
            )

    if normalized_decision == "REBALANCE":
        if direction_accuracy is True:
            return DecisionEvaluation(
                verdict="ACCURATE",
                direction_accuracy=True,
                magnitude_error=magnitude_error,
                cost_efficiency=cost_efficiency,
                note="Rebalance recommendation matched the realized direction.",
            )
        return DecisionEvaluation(
            verdict="FALSE_POSITIVE",
            direction_accuracy=direction_accuracy,
            magnitude_error=magnitude_error,
            cost_efficiency=cost_efficiency,
            note="Rebalance recommendation did not match the realized direction.",
        )

    if normalized_decision == "HOLD":
        if abs(realized_return_pct) < 3.0:
            return DecisionEvaluation(
                verdict="HOLD_CORRECT",
                direction_accuracy=None,
                magnitude_error=None,
                cost_efficiency=None,
                note="Hold was reasonable because the realized move stayed below the action threshold.",
            )
        return DecisionEvaluation(
            verdict="HOLD_WRONG",
            direction_accuracy=None,
            magnitude_error=None,
            cost_efficiency=None,
            note="Hold missed a material realized move.",
        )

    return DecisionEvaluation(
        verdict="BLOCKED",
        direction_accuracy=direction_accuracy,
        magnitude_error=magnitude_error,
        cost_efficiency=cost_efficiency,
        note="Decision was blocked or deferred; evaluation is informational only.",
    )


def _direction_accuracy(*, signal_score: float, realized_return_pct: float) -> bool | None:
    if signal_score == 0:
        return None
    return (signal_score > 0 and realized_return_pct > 0) or (
        signal_score < 0 and realized_return_pct < 0
    )
