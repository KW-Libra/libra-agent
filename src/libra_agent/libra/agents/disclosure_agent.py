from .base import DelegatingInformationAgent
from ..prompts import DISCLOSURE_PROMPT_PROFILE


class DisclosureAgent(DelegatingInformationAgent):
    agent_id = "disclosure"
    owner_scope = "Disclosure Agent"
    prompt_profile = DISCLOSURE_PROMPT_PROFILE
