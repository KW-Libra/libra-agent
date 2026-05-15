from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

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
        knowledge_base: LocalKnowledgeBase,
        depth: str = "medium",
    ) -> AgentResponse: ...


class TradeAgentProtocol(Protocol):
    agent_id: str
    owner_scope: str

    def run(self, **kwargs: object) -> AgentResponse: ...


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
    ) -> dict[str, Any]: ...


@dataclass(slots=True)
class AgentBundle:
    disclosure: InformationAgentProtocol
    news: InformationAgentProtocol
    report: InformationAgentProtocol
    profit: TradeAgentProtocol
    cost: TradeAgentProtocol
    evaluation: EvaluationAgentProtocol
    risk: InformationAgentProtocol | None = None
    tax: InformationAgentProtocol | None = None
    compliance: InformationAgentProtocol | None = None
    macro: InformationAgentProtocol | None = None
    sentiment: InformationAgentProtocol | None = None
    execution: InformationAgentProtocol | None = None
    esg: InformationAgentProtocol | None = None
    liquidity: InformationAgentProtocol | None = None
    technical: InformationAgentProtocol | None = None

    def domain_agents(self) -> dict[str, InformationAgentProtocol]:
        """Return enabled domain agents only."""
        return {
            name: agent
            for name, agent in {
                "risk": self.risk,
                "tax": self.tax,
                "compliance": self.compliance,
                "macro": self.macro,
                "sentiment": self.sentiment,
                "execution": self.execution,
                "esg": self.esg,
                "liquidity": self.liquidity,
                "technical": self.technical,
            }.items()
            if agent is not None
        }


@dataclass(slots=True)
class InformationAgentRunRequest:
    query: str
    context: str | None
    fallback: str | None
    note: str | None
    depth: str


class DelegatingInformationAgent:
    agent_id = ""
    owner_scope = ""
    prompt_profile: InformationAgentPromptProfile | None = None
    owner_task_brief = ""

    def __init__(self, *, client: ChatClientProtocol) -> None:
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
        knowledge_base: LocalKnowledgeBase,
        depth: str = "medium",
    ) -> AgentResponse:
        request = self.prepare_request(
            query=query,
            context=context,
            fallback=fallback,
            note=note,
            turn_number=turn_number,
            portfolio=portfolio,
            knowledge_base=knowledge_base,
            depth=depth,
        )
        return self._implementation.run(
            query=request.query,
            context=request.context,
            fallback=request.fallback,
            note=request.note,
            turn_number=turn_number,
            portfolio=portfolio,
            knowledge_base=knowledge_base,
            depth=request.depth,
        )

    def prepare_request(
        self,
        *,
        query: str,
        context: str | None,
        fallback: str | None,
        note: str | None,
        turn_number: int,
        portfolio: PortfolioSnapshot,
        knowledge_base: LocalKnowledgeBase,
        depth: str,
    ) -> InformationAgentRunRequest:
        del turn_number, portfolio, knowledge_base
        return InformationAgentRunRequest(
            query=query,
            context=context,
            fallback=fallback,
            note=self._append_owner_task_brief(note),
            depth=depth,
        )

    def _append_owner_task_brief(self, note: str | None) -> str | None:
        brief = self.owner_task_brief.strip()
        if not brief:
            return note
        if note and note.strip():
            return f"{note.strip()}\nAgent owner task: {brief}"
        return f"Agent owner task: {brief}"
