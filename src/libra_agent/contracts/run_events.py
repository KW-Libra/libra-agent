"""Run SSE event contract.

This module intentionally contains names and shape-level constants only.
Domain-specific agent decisions, prompts, report contents, and portfolio rules
are still undecided and should not live here.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final


class RunEventType(StrEnum):
    RUN_STARTED = "run_started"
    NODE_STARTED = "node_started"
    NODE_COMPLETED = "node_completed"
    JUDGE_ACTION = "judge_action"
    AGENT_STARTED = "agent_started"
    AGENT_COMPLETED = "agent_completed"
    AGENT_FAILED = "agent_failed"
    TOOL_OBSERVATION = "tool_observation"
    LLM_PROMPT = "llm_prompt"
    LLM_RESPONSE = "llm_response"
    LLM_ERROR = "llm_error"
    LLM_SKIPPED = "llm_skipped"
    MEDIATOR_DECISION = "mediator_decision"
    CONSENSUS_UPDATED = "consensus_updated"
    FINAL_DECISION_DRAFT = "final_decision_draft"
    INTERRUPT_REQUIRED = "interrupt_required"
    RESUME_RECEIVED = "resume_received"
    RESUME_IGNORED = "resume_ignored"
    RUN_COMPLETED = "run_completed"
    RUN_FAILED = "run_failed"


RUN_EVENT_TYPES: Final[tuple[str, ...]] = tuple(item.value for item in RunEventType)

RUN_NODE_NAMES: Final[tuple[str, ...]] = (
    "compliance_before",
    "round1",
    "mediator",
    "final_judge",
    "human_review",
)

TERMINAL_EVENT_TYPES: Final[tuple[str, ...]] = (
    RunEventType.INTERRUPT_REQUIRED.value,
    RunEventType.RESUME_IGNORED.value,
    RunEventType.RUN_COMPLETED.value,
    RunEventType.RUN_FAILED.value,
)
