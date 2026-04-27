from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Protocol

from libra_agent.libra_models import AgentResponse, PortfolioSnapshot

if TYPE_CHECKING:
    from libra_agent.libra.llm_clients import ChatClientProtocol
    from libra_agent.libra.prompts import InformationAgentPromptProfile
    from libra_agent.libra_runtime import LocalKnowledgeBase


class InformationAgentProtocol(Protocol):
    agent_id: str
    owner_scope: str

    def run(
        self,
        *,
        query: str,
        context: str | None = None,
        fallback: str | None = None,
        note: str | None = None,
        turn_number: int,
        portfolio: PortfolioSnapshot,
        knowledge_base: "LocalKnowledgeBase",
        depth: str = "medium",
    ) -> AgentResponse:
        ...


class TradeAgentProtocol(Protocol):
    agent_id: str
    owner_scope: str

    def run(self, **kwargs: object) -> AgentResponse:
        ...


class EvaluationAgentProtocol(Protocol):
    agent_id: str
    owner_scope: str

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
        ...


@dataclass(slots=True)
class AgentBundle:
    disclosure: InformationAgentProtocol
    news: InformationAgentProtocol
    report: InformationAgentProtocol
    profit: TradeAgentProtocol
    cost: TradeAgentProtocol
    evaluation: EvaluationAgentProtocol


class DelegatingInformationAgent:
    agent_id = ""
    owner_scope = ""
    prompt_profile: "InformationAgentPromptProfile | None" = None

    def __init__(self, *, client: "ChatClientProtocol") -> None:
        from libra_agent.libra_runtime import LLMAgent

        self._implementation = LLMAgent(
            agent_id=self.agent_id,
            client=client,
            prompt_profile=self.prompt_profile,
        )

    def run(
        self,
        *,
        query: str,
        context: str | None = None,
        fallback: str | None = None,
        note: str | None = None,
        turn_number: int,
        portfolio: PortfolioSnapshot,
        knowledge_base: "LocalKnowledgeBase",
        depth: str = "medium",
    ) -> AgentResponse:
        return self._implementation.run(
            query=query,
            context=context,
            fallback=fallback,
            note=note,
            turn_number=turn_number,
            portfolio=portfolio,
            knowledge_base=knowledge_base,
            depth=depth,
        )


class DelegatingTradeAgent:
    agent_id = ""
    owner_scope = ""
    _implementation_class_name = ""

    def __init__(self) -> None:
        from libra_agent import libra_runtime

        implementation_class = getattr(libra_runtime, self._implementation_class_name)
        self._implementation = implementation_class()

    def run(self, **kwargs: object) -> AgentResponse:
        return self._implementation.run(**kwargs)
