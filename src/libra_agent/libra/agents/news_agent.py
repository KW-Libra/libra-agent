from .base import DelegatingInformationAgent
from ..prompts import NEWS_PROMPT_PROFILE


class NewsAgent(DelegatingInformationAgent):
    agent_id = "news"
    owner_scope = "News Agent"
    prompt_profile = NEWS_PROMPT_PROFILE
