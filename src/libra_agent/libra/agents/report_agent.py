from .base import DelegatingInformationAgent
from ..prompts import REPORT_PROMPT_PROFILE


class ReportAgent(DelegatingInformationAgent):
    agent_id = "report"
    owner_scope = "Report Agent"
    prompt_profile = REPORT_PROMPT_PROFILE
