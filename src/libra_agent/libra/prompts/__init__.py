from .base import InformationAgentPromptProfile
from .disclosure import DISCLOSURE_PROMPT_PROFILE
from .judge import (
    JUDGE_ACTION_RULES,
    JUDGE_DOMAIN_ACTION_SYSTEM_PROMPT,
    JUDGE_ACTION_SYSTEM_PROMPT,
    JUDGE_NOTIFICATION_LEVELS,
    JUDGE_PHASE_REQUIRED_KEYS,
    JUDGE_PHASE_SYSTEM_PROMPT,
    default_agent_fallback,
    default_agent_note,
    default_agent_query,
)
from .news import NEWS_PROMPT_PROFILE
from .report import REPORT_PROMPT_PROFILE


INFORMATION_AGENT_PROMPTS = {
    "disclosure": DISCLOSURE_PROMPT_PROFILE,
    "news": NEWS_PROMPT_PROFILE,
    "report": REPORT_PROMPT_PROFILE,
}


def get_information_prompt_profile(agent_id: str) -> InformationAgentPromptProfile:
    return INFORMATION_AGENT_PROMPTS[agent_id]


__all__ = [
    "InformationAgentPromptProfile",
    "DISCLOSURE_PROMPT_PROFILE",
    "NEWS_PROMPT_PROFILE",
    "REPORT_PROMPT_PROFILE",
    "INFORMATION_AGENT_PROMPTS",
    "JUDGE_ACTION_RULES",
    "JUDGE_DOMAIN_ACTION_SYSTEM_PROMPT",
    "JUDGE_ACTION_SYSTEM_PROMPT",
    "JUDGE_NOTIFICATION_LEVELS",
    "JUDGE_PHASE_REQUIRED_KEYS",
    "JUDGE_PHASE_SYSTEM_PROMPT",
    "default_agent_fallback",
    "default_agent_note",
    "default_agent_query",
    "get_information_prompt_profile",
]
