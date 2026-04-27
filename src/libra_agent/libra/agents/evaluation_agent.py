from __future__ import annotations

from typing import Any, Mapping

from libra_agent.libra.evaluation import evaluate_decision_outcome


class EvaluationAgent:
    agent_id = "evaluation"
    owner_scope = "Evaluation Agent"

    def run(
        self,
        *,
        decision: str,
        rebalance_plan: Mapping[str, float],
        signal_score: float,
        user_feedback: str | None,
        realized_return_pct: float,
        cost_pct: float = 0.0,
        horizon: str = "1w",
    ) -> dict[str, Any]:
        evaluation = evaluate_decision_outcome(
            decision=decision,
            rebalance_plan=rebalance_plan,
            signal_score=signal_score,
            user_feedback=user_feedback,
            realized_return_pct=realized_return_pct,
            cost_pct=cost_pct,
        )
        return {
            "agent_id": self.agent_id,
            "horizon": horizon or "1w",
            **evaluation.to_dict(),
            "metrics": {
                "decision": decision,
                "rebalance_plan": dict(rebalance_plan),
                "signal_score": signal_score,
                "realized_return_pct": realized_return_pct,
                "cost_pct": cost_pct,
                "user_feedback": user_feedback,
            },
        }
