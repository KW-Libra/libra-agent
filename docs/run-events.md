# RunEvent Contract v0

This contract defines the SSE envelope between `libra-agent`, `libra-backend`, and `libra-frontend`.
It intentionally avoids domain-specific investment logic while product decisions are still open.

Every SSE message uses:

```text
event: <event_name>
data: <JSON payload>
```

## Event Names

| Event | Terminal | Purpose |
|---|---:|---|
| `run_started` | no | A run was accepted and graph execution started. |
| `node_started` | no | A stable graph node started. |
| `node_completed` | no | A stable graph node completed. |
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
  "trigger": "pull | push | user_request | string",
  "query": "user text",
  "approval_required": false
}
```

`node_started`, `node_completed`

```json
{ "node": "human_review" }
```

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

`resume_received`

```json
{ "thread_id": "uuid-or-client-id", "approved": true, "option_index": 0 }
```

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
  "run_status": "completed",
  "approval_response": {
    "approved": true,
    "decision": "APPROVE",
    "option_index": 0,
    "override_plan": null,
    "note": "optional"
  }
}
```

`run_failed`

```json
{ "thread_id": "uuid-or-client-id", "error": "message" }
```

## Change Policy

- Additive payload fields are allowed.
- Existing event names and existing payload fields should not change without a version bump.
- New product-specific events should be added only after the underlying product behavior is agreed.
