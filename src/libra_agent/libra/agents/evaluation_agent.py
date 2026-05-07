from __future__ import annotations

from typing import Any, Mapping

from libra_agent.libra.evaluation import evaluate_decision_outcome
from libra_agent.libra.llm_clients.base import ChatClientProtocol
from libra_agent.libra.reflection import generate_reflection


class EvaluationAgent:
    agent_id = "evaluation"
    owner_scope = "Evaluation Agent"

    def __init__(self, *, client: ChatClientProtocol | None = None) -> None:
        self.client = client

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
        reflection = self._generate_reflection_safely(
            decision=decision,
            rebalance_plan=rebalance_plan,
            signal_score=signal_score,
            realized_return_pct=realized_return_pct,
            cost_pct=cost_pct,
            user_feedback=user_feedback,
        )
        return {
            "agent_id": self.agent_id,
            "horizon": horizon or "1w",
            **evaluation.to_dict(),
            "reflection": reflection,
            "metrics": {
                "decision": decision,
                "rebalance_plan": dict(rebalance_plan),
                "signal_score": signal_score,
                "realized_return_pct": realized_return_pct,
                "cost_pct": cost_pct,
                "user_feedback": user_feedback,
            },
        }

    def _generate_reflection_safely(
        self,
        *,
        decision: str,
        rebalance_plan: Mapping[str, float],
        signal_score: float,
        realized_return_pct: float,
        cost_pct: float,
        user_feedback: str | None,
    ) -> str:
        if self.client is None:
            return ""
        try:
            return generate_reflection(
                client=self.client,
                decision=decision,
                rebalance_plan=rebalance_plan,
                signal_score=signal_score,
                realized_return_pct=realized_return_pct,
                cost_pct=cost_pct,
                user_feedback=user_feedback,
            )
        except Exception as exc:
            return f"(reflection generation failed: {exc})"
