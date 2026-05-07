from __future__ import annotations

import os
from typing import TYPE_CHECKING

from .base import AgentBundle
from .cost_agent import CostAgent
from .disclosure_agent import DisclosureAgent
from .evaluation_agent import EvaluationAgent
from .news_agent import NewsAgent
from .profit_agent import ProfitAgent
from .report_agent import ReportAgent

if TYPE_CHECKING:
    from libra_agent.libra_runtime import ChatClient


def build_default_agent_bundle(*, client: "ChatClient") -> AgentBundle:
    bundle_kwargs = dict(
        disclosure=DisclosureAgent(client=client),
        news=NewsAgent(client=client),
        report=ReportAgent(client=client),
        profit=ProfitAgent(),
        cost=CostAgent(),
        evaluation=EvaluationAgent(client=client),
    )
    if os.environ.get("LIBRA_DOMAIN_AGENTS_ENABLED", "false").strip().lower() == "true":
        from libra_agent.domain_agents._adapter import build_domain_agent_adapters

        bundle_kwargs.update(build_domain_agent_adapters())
    return AgentBundle(**bundle_kwargs)
