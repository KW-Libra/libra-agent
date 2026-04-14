from __future__ import annotations

from .base import (
    InformationAgentPromptProfile,
    build_information_response_template,
    build_information_system_prompt,
)


NEWS_EVIDENCE_SHAPE_HINT = {
    "sub_role": "macro | company_specific | mixed",
    "company_findings": "dict",
    "macro_findings": "dict or null",
    "cross_check_count": "int",
}


NEWS_PROMPT_PROFILE = InformationAgentPromptProfile(
    agent_id="news",
    owner_scope="News Agent",
    system_prompt=build_information_system_prompt("news"),
    focus="Summarize news and market-reaction context, including macro if relevant.",
    evidence_shape_hint=NEWS_EVIDENCE_SHAPE_HINT,
    response_template=build_information_response_template(NEWS_EVIDENCE_SHAPE_HINT),
)
