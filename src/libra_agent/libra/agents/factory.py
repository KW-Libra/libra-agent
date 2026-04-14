from __future__ import annotations

from typing import TYPE_CHECKING

from .base import AgentBundle
from .cost_agent import CostAgent
from .disclosure_agent import DisclosureAgent
from .news_agent import NewsAgent
from .profit_agent import ProfitAgent
from .report_agent import ReportAgent

if TYPE_CHECKING:
    from libra_agent.libra_runtime import ChatClient


def build_default_agent_bundle(*, client: "ChatClient") -> AgentBundle:
    return AgentBundle(
        disclosure=DisclosureAgent(client=client),
        news=NewsAgent(client=client),
        report=ReportAgent(client=client),
        profit=ProfitAgent(),
        cost=CostAgent(),
    )
