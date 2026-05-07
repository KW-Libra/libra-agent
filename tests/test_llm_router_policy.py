from __future__ import annotations

import unittest

from libra_agent.domain_agents.base import PortfolioContext
from libra_agent.domain_agents._services.llm_router import LLMRouter, LLMModel
from libra_agent.domain_agents.tax import TaxAgent


class FakeRouter(LLMRouter):
    def __init__(self, *, policy: str) -> None:
        self.calls: list[tuple[str, str]] = []
        super().__init__(anthropic_key="claude-key", gemini_key="gemini-key", policy=policy)

    def _init_claude(self):
        return object()

    def _init_gemini(self):
        return object()

    def _call_claude(self, model: str, system: str, user: str, max_tokens: int) -> str:
        self.calls.append(("claude", model))
        return "claude answer"

    def _call_gemini(self, model: str, system: str, user: str, max_tokens: int) -> str:
        self.calls.append(("gemini", model))
        return "gemini answer"


class FailingGeminiRouter(FakeRouter):
    def _call_gemini(self, model: str, system: str, user: str, max_tokens: int) -> str:
        self.calls.append(("gemini", model))
        raise RuntimeError("quota exceeded")


class RecordingRouter:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def ask(self, **kwargs):
        self.calls.append(dict(kwargs))
        return "tax rationale"

    def model_name_for(self, agent_id: str) -> str:
        return "gemini-2.5-flash"


class LLMRouterPolicyTests(unittest.TestCase):
    def test_gemini_policy_disables_cross_validation_claude_call(self) -> None:
        router = FakeRouter(policy="gemini")

        result = router.ask(
            agent_id="macro",
            system="system",
            user="user",
            cross_validate=True,
        )

        self.assertEqual(result, "gemini answer")
        self.assertEqual(router.calls, [("gemini", LLMModel.GEMINI_FLASH.value)])

    def test_gemini_policy_does_not_fallback_to_claude(self) -> None:
        router = FailingGeminiRouter(policy="gemini")

        result = router.ask(agent_id="macro", system="system", user="user")

        self.assertIn("[LLM 오류]", result)
        self.assertEqual(router.calls, [("gemini", LLMModel.GEMINI_FLASH.value)])

    def test_balanced_cross_validation_keeps_model_labels_correct(self) -> None:
        router = FakeRouter(policy="balanced")

        result = router.ask(
            agent_id="macro",
            system="system",
            user="user",
            cross_validate=True,
        )

        self.assertIn("[Gemini 판단] gemini answer", result)
        self.assertIn("[Claude 판단] claude answer", result)

    def test_model_name_for_reflects_gemini_policy(self) -> None:
        router = FakeRouter(policy="gemini")

        self.assertEqual(router.model_name_for("tax"), LLMModel.GEMINI_FLASH.value)

    def test_tax_agent_uses_context_router_and_records_actual_model(self) -> None:
        router = RecordingRouter()
        ctx = PortfolioContext(
            user_id="user-1",
            holdings=[
                {
                    "symbol": "005930",
                    "quantity": 200,
                    "current_price": 1000,
                    "average_price": 2000,
                    "weight": 1.0,
                }
            ],
            preferences={"tax_bracket": "unknown"},
            total_value=200_000,
            router=router,
        )

        import asyncio

        verdict = asyncio.run(TaxAgent().deliberate(ctx))

        self.assertEqual(verdict.rationale, "tax rationale")
        self.assertEqual(verdict.llm_used, LLMModel.GEMINI_FLASH.value)
        self.assertEqual(router.calls[0]["agent_id"], "tax")


if __name__ == "__main__":
    unittest.main()
