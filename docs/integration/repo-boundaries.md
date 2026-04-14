# Repo Boundaries

## Recommended Repository Names

- `libra-ingest`
- `libra-agent`
- `libra-api` (Spring Boot)
- `libra-web`

## Ownership

### `libra-agent`

- Judge orchestration
- sub-agent logic
- local or remote LLM adapter layer
- decision trace generation
- runtime-side decision persistence

### `libra-api`

- Spring Boot product backend
- user auth and tenant model
- portfolio storage
- user preferences and approval settings
- scheduling and job dispatch
- persistence for run history and audit logs
- broker or execution adapters
- calling `libra-agent` and storing its outputs

### `libra-ingest`

- RSS, disclosure, and report collection
- normalization
- entity linking
- event generation
- ingest-side schedules and source-specific retries
- producing `events` and `normalized_documents` payloads for downstream systems

### `libra-web`

- dashboard
- portfolio settings
- approval and override UI
- decision trace viewer
- alerts and notification center
- run history and feedback screens

## Call Direction

- `libra-web -> libra-api`
- `libra-api -> libra-agent`
- `libra-api -> libra-ingest`

`libra-web` should not call `libra-agent` directly.

`libra-agent` should not absorb product backend responsibilities. Judge orchestration stays in Python; Spring Boot owns application-facing workflows.

## Shared Contract Rule

Anything exchanged between `libra-agent` and `libra-api` should be versioned under `contracts/`.

Recommended first contract files:

- `contracts/judge-run-request.schema.json`
- `contracts/judge-run-result.schema.json`
- `contracts/push-trigger-event.schema.json`
- `contracts/user-approval-response.schema.json`
- `contracts/knowledge-events.schema.json`
- `contracts/knowledge-documents.schema.json`

See `spring-boot-agent-boundary.md` for the backend integration split.
