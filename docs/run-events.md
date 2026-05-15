# RunEvent Contract v0

This contract defines the SSE envelope between `libra-agent`, `libra-backend`, and `libra-frontend`.
It intentionally avoids domain-specific investment logic while product decisions are still open.

Every SSE message uses:

```text
event: <event_name>
data: <JSON payload>
```

## HTTP Payloads

`POST /api/runs` accepts a JSON body with these top-level fields. Unknown top-level fields are rejected by the route model; nested payload objects are additive because the graph passes them through to domain/runtime models.

Required:

- `query`: non-empty string.
- `portfolio`: object. `holdings` may be empty or omitted when `portfolio_definition` supplies target weights.

Optional:

- `knowledge_sources`: object or `null`. Known keys are `events`, `normalized_documents`, and `enriched_documents`; extra keys are preserved as source metadata.
- `knowledge_base`: object or `null` with inline `events`, `documents`, and `source_paths`.
- `portfolio_definition`: object or `null` with target weights for direct-indexing/drift-aware runs.
- `trigger_event`: object or `null` for push-trigger context.
- `governance_v1`: object or `null`; `execution_mode: "primary"` selects the governance v1 committee runtime.
- `trigger`: `"pull"` or `"push"`, default `"pull"`.
- `depth`: `"shallow"`, `"medium"`, or `"deep"`, default `"medium"`.
- `deadline_seconds`: integer >= 1 or `null`.
- `thread_id`: string or `null`; generated when omitted.
- `enable_human_interrupts`: boolean, default `false`.
- `approval_required`: legacy boolean alias for `enable_human_interrupts`.

Human review is enabled when either `enable_human_interrupts` or legacy `approval_required` is true. The SSE `run_started` payload echoes both fields after normalization.

`POST /api/runs/{thread_id}/resume` accepts a JSON body with `approved` required and these optional fields: `decision`, `interrupt_id`, `option_index`, `override_decision`, `override_plan`, `note`, `effective_at`, `responder`, and `metadata`. Extra resume fields are allowed by the route model but are not currently copied into graph state unless listed above.

## Event Names

| Event | Terminal | Purpose |
|---|---:|---|
| `run_started` | no | A run was accepted and graph execution started. |
| `node_started` | no | A stable graph node started. |
| `node_completed` | no | A stable graph node completed. |
| `judge_action` | no | Judge selected the next action or decided to finalize. |
| `agent_started` | no | A core/domain/committee agent began producing an opinion. |
| `agent_completed` | no | An agent submitted a compact opinion for the UI timeline. |
| `agent_failed` | no | An agent failed before submitting an opinion. The run may still emit `run_failed`. |
| `mediator_decision` | no | Governance v1 mediator selected Round 2 targets or skipped Round 2. |
| `consensus_updated` | no | Consensus or domain-council aggregate state was recalculated. |
| `final_decision_draft` | no | Final Judge produced a compact draft decision before HITL/completion. |
| `interrupt_required` | yes | The graph checkpointed and waits for user input. |
| `resume_received` | no | A resume request was accepted. |
| `resume_ignored` | yes | A resume request was received but no interrupt is pending. |
| `run_completed` | yes | The run reached an end state. |
| `run_failed` | yes | The run failed before reaching an end state. |

Terminal means the current SSE stream should close after the event.

## Stable Node Names

- `compliance_before`
- `round1`
- `mediator`
- `final_judge`
- `human_review`

These are workflow checkpoints, not final product UI labels.

## Payloads

`run_started`

```json
{
  "thread_id": "uuid-or-client-id",
  "trigger": "pull | push",
  "query": "user text",
  "approval_required": false,
  "enable_human_interrupts": false
}
```

`node_started`, `node_completed`

```json
{ "node": "human_review" }
```

`judge_action`

```json
{
  "layer": "core",
  "turn_number": 2,
  "action": "CALL_AGENT",
  "agent_id": "news",
  "reason": "뉴스 신호 확인이 필요합니다.",
  "query": "뉴스 관점에서 점검",
  "depth": "shallow",
  "candidate_rebalance_plan": {},
  "called_agents": ["disclosure"],
  "response_count": 1
}
```

`agent_started`

```json
{
  "agent_id": "news",
  "layer": "core",
  "turn_number": 2,
  "query": "뉴스 관점에서 점검",
  "depth": "shallow"
}
```

`agent_completed`

```json
{
  "agent_id": "news",
  "layer": "core",
  "turn_number": 2,
  "verdict": "PARTIAL_ANSWER",
  "opinion": "NEUTRAL",
  "direction": 0.0,
  "strength": 0.2,
  "confidence": 0.7,
  "urgency": "defer",
  "risk_level": "low",
  "focus_tickers": ["005930"],
  "reasoning": "UI에 보여줄 compact reasoning",
  "tools_called": []
}
```

`mediator_decision`, `consensus_updated`, and `final_decision_draft` are compact timeline events.
They are intentionally not the durable audit record; the durable payload remains `run_completed.agent_result`.

`interrupt_required`

```json
{
  "thread_id": "uuid-or-client-id",
  "interrupt_id": "langgraph-interrupt-id",
  "type": "human_approval",
  "reason": "approval_required",
  "message": "최종 결정 적용 전에 사용자 확인이 필요합니다.",
  "decision": "HOLD",
  "branch": "USER_APPROVAL_REQUIRED",
  "options": [
    { "decision": "APPROVE", "label": "승인" },
    { "decision": "REJECT", "label": "거절" },
    { "decision": "REVISE", "label": "수정 요청" }
  ],
  "interrupts": [
    { "id": "langgraph-interrupt-id", "value": {} }
  ]
}
```

The interrupt event includes the raw `interrupts` list and also merges the first interrupt's `value` object into the top-level payload. `interrupt_id` is copied from the first interrupt id when present.

`resume_received`

```json
{
  "thread_id": "uuid-or-client-id",
  "approved": true,
  "interrupt_id": "langgraph-interrupt-id",
  "option_index": 0
}
```

Only `approved`, `interrupt_id`, and `option_index` are echoed by `resume_received`. The full normalized resume payload is sent to the graph and may later appear in `run_completed.approval_response`.

`resume_ignored`

```json
{ "thread_id": "uuid-or-client-id", "reason": "no_pending_interrupt" }
```

`run_completed`

```json
{
  "thread_id": "uuid-or-client-id",
  "decision": "HOLD",
  "branch": "CONSENSUS",
  "run_status": "completed | completed_after_resume",
  "final_decision": {
    "decision": "HOLD",
    "branch": "CONSENSUS",
    "requires_approval": false
  },
  "agent_result": {},
  "approval_response": {
    "approved": true,
    "decision": "APPROVE",
    "interrupt_id": "langgraph-interrupt-id",
    "option_index": 0,
    "override_decision": null,
    "override_plan": null,
    "note": "optional",
    "effective_at": null,
    "responder": "user-id",
    "metadata": {}
  }
}
```

`final_decision`, `agent_result`, and `approval_response` are present only when the graph state contains them.

`run_failed`

```json
{ "thread_id": "uuid-or-client-id", "error": "message" }
```

## Change Policy

- Additive payload fields are allowed.
- Existing event names and existing payload fields should not change without a version bump.
- New product-specific events should be added only after the underlying product behavior is agreed.
