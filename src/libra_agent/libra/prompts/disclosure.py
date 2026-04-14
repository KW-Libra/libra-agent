from __future__ import annotations

from .base import (
    InformationAgentPromptProfile,
    build_information_response_template,
    build_information_system_prompt,
)


DISCLOSURE_EVIDENCE_SHAPE_HINT = {
    "found_count": "int",
    "items": "list of disclosure facts",
    "upcoming_disclosures": "list",
}


DISCLOSURE_PROMPT_PROFILE = InformationAgentPromptProfile(
    agent_id="disclosure",
    owner_scope="Disclosure Agent",
    system_prompt=build_information_system_prompt("disclosure"),
    focus="Summarize disclosure and earnings implications without making the final investment decision.",
    evidence_shape_hint=DISCLOSURE_EVIDENCE_SHAPE_HINT,
    response_template=build_information_response_template(DISCLOSURE_EVIDENCE_SHAPE_HINT),
)
