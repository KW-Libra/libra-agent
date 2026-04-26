# Spring Boot and Agent Boundary

## Decision

- `libra-backend`: Spring Boot backend repository
- `libra-agent`: Python LangGraph decision engine repository

Do not move Judge orchestration into Spring Boot.

The backend owns product workflows. The agent repo owns reasoning workflows.

## Why This Split Fits LIBRA

The current LIBRA structure is not a fixed pipeline. The Judge decides which sub-agent to call, observes the result, and changes the next action.

That orchestration logic is already centered on:

- LangGraph state transitions
- Python LLM adapters
- agent-specific prompt/runtime behavior

Rewriting that core into Spring Boot now would slow the team down and blur ownership.

## Repository Responsibilities

### `libra-backend` (Spring Boot)

- user auth, session, tenant, RBAC
- portfolio CRUD
- user preference and approval policy storage
- schedule registration and job dispatch
- run history and audit persistence
- frontend-facing REST API
- broker/execution integration
- calling `libra-agent`
- calling `libra-ingest`

### `libra-agent` (Python)

- Judge orchestration
- Disclosure, News, Report, Profit, Cost agent logic
- prompt definitions
- LLM backend abstraction
- decision trace generation
- follow-up reservation and checkpoint persistence

### `libra-ingest`

- source collection
- normalization
- entity linking
- event/document payload production

## Call Direction

- `libra-frontend -> libra-backend`
- `libra-backend -> libra-agent`
- `libra-backend -> libra-ingest`

`libra-frontend` does not call the agent directly.

## Recommended Integration Shape

Start with a contract-first boundary.

Spring Boot sends a `judge-run-request` payload to the agent and receives a `judge-run-result` payload back.

Recommended first transport:

1. internal HTTP JSON
2. same schemas as `contracts/*.schema.json`
3. synchronous run for regular checks
4. async queue or job wrapper later for long-running flows

CLI subprocess coupling is acceptable for local development, but it should not be the long-term production boundary.

## Spring Boot Repo Structure

Recommended package direction for `libra-backend`:

```text
src/main/java/com/libra/api
  auth/
  portfolio/
  preferences/
  approval/
  runs/
  scheduling/
  execution/
  integration/
    agent/
      LibraAgentClient.java
      dto/
    ingest/
      LibraIngestClient.java
      dto/
  common/
```

Key point: keep the agent client under `integration/agent/` and keep agent DTOs generated or mapped from `contracts/`.

## Runtime Flow

### Regular check

1. Spring Boot loads portfolio, preferences, and latest knowledge references.
2. Spring Boot builds `judge-run-request`.
3. Spring Boot calls `libra-agent`.
4. `libra-agent` runs Judge-centered orchestration.
5. Spring Boot stores `judge-run-result`.
6. Spring Boot decides whether to notify, request approval, or trigger execution.

### Push event

1. `libra-ingest` or a market event source sends a push trigger.
2. Spring Boot validates relevance against owned portfolios and policies.
3. Spring Boot calls `libra-agent` with the trigger context.
4. Spring Boot stores the result and handles notification or approval.

## Contract Rule

The JSON schemas in `contracts/` are the source of truth between Spring Boot and Python.

That means:

- Spring Boot DTOs must match `contracts/`
- Python request/response models must match `contracts/`
- shape changes start from schema update, not ad hoc code edits

## Team Split

- backend team: `libra-backend` Spring Boot, persistence, scheduler, execution, auth
- agent team: `libra-agent` Judge and sub-agents
- ingest team: `libra-ingest` collectors and normalization
- frontend team: `libra-frontend`

This keeps each team working in parallel without mixing framework concerns.
