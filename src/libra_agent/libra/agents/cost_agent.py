from .base import DelegatingTradeAgent


class CostAgent(DelegatingTradeAgent):
    agent_id = "cost"
    owner_scope = "Cost Agent"
    _implementation_class_name = "DeterministicCostAgent"
