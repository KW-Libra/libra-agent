from .base import DelegatingInformationAgent
from ..prompts import DISCLOSURE_PROMPT_PROFILE


class DisclosureAgent(DelegatingInformationAgent):
    agent_id = "disclosure"
    owner_scope = "Disclosure Agent"
    prompt_profile = DISCLOSURE_PROMPT_PROFILE
    owner_task_brief = (
        "공시와 실적 사실만 정리한다. 최종 매매 판단, 수익성 평가, 비용 평가는 Judge와 전담 에이전트에 맡긴다."
    )
