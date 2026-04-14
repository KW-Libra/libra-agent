from .base import DelegatingTradeAgent


class ProfitAgent(DelegatingTradeAgent):
    agent_id = "profit"
    owner_scope = "Profit Agent"
    _implementation_class_name = "DeterministicProfitAgent"
