from .base import DelegatingInformationAgent
from ..prompts import REPORT_PROMPT_PROFILE


class ReportAgent(DelegatingInformationAgent):
    agent_id = "report"
    owner_scope = "Report Agent"
    prompt_profile = REPORT_PROMPT_PROFILE
    owner_task_brief = (
        "증권사 리포트, 컨센서스, 해석 차이를 정리한다. 커버리지 부재는 숨기지 말고 한계로 반환한다."
    )
