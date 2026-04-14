from __future__ import annotations

from .base import (
    InformationAgentPromptProfile,
    build_information_response_template,
    build_information_system_prompt,
)


REPORT_EVIDENCE_SHAPE_HINT = {
    "coverage_reports_count": "int",
    "items": "list of broker/report findings",
    "consensus": "dict or null",
}


REPORT_PROMPT_PROFILE = InformationAgentPromptProfile(
    agent_id="report",
    owner_scope="Report Agent",
    system_prompt=build_information_system_prompt("report"),
    focus="Summarize research-report style takeaways, consensus changes, and business-segment clues.",
    evidence_shape_hint=REPORT_EVIDENCE_SHAPE_HINT,
    response_template=build_information_response_template(REPORT_EVIDENCE_SHAPE_HINT),
)
