# agent-core / agent-domain integration ledger

This file tracks the split-repo agent work absorbed into the official
`libra-agent` repository. The goal is to preserve functionality while keeping
the production boundary clear:

- `libra-backend` owns users, KIS credentials, portfolio state, order/audit
  records, decision history, and report metadata.
- `libra-agent` owns LangGraph orchestration, LLM/domain-agent judgment,
  runtime checkpoints, and SSE run events.
- `libra-ingest` owns collection jobs and promoted knowledge artifacts.

## Imported from `agent-core`

Status: imported into this repository.

- Core runtime: `libra_runtime.py`, `libra_graph.py`, `libra_models.py`,
  `libra_validation.py`, `libra_store.py`
- LLM clients: Anthropic, Gemini, Ollama, llama.cpp
- Core agents: disclosure, news, report, profit, cost, evaluation
- Domain council: risk, tax, compliance, macro, sentiment, execution, ESG
- Governance/runtime helpers: committee, compliance, mediator, final judge,
  direct indexing, signal scoring, reflection
- Contracts and examples: root `contracts/`, `examples/`, `benchmarks/`
- Tests from `agent-core`

## Adapted for official `libra-agent`

Status: wired through the existing FastAPI/SSE surface.

- Existing `/api/runs` SSE route remains the deployed interface.
- Existing `AsyncPostgresSaver` remains the checkpoint mechanism for the
  official service.
- The LangGraph `final_judge` node now calls `JudgeOrchestrator` when a
  portfolio snapshot with holdings is supplied.
- Empty-portfolio smoke runs keep the deterministic HOLD path so CI and health
  checks do not require an LLM key.
- Knowledge is read from promoted artifacts through `KnowledgeReader`; direct
  ingest refresh is disabled on this path.

## Kept but not used by the production API

Status: available for tests or local tooling, not part of the deployed
backend-owned data boundary.

- `libra.portfolio_sources.kis`: retained from `agent-core` for historical
  tests/local bootstrap, but the deployed API must receive portfolio snapshots
  from `libra-backend` instead of reading KIS credentials directly.
- `libra_api.py`: retained as the original Team A standalone API for parity
  and tests. The deployed entrypoint remains `libra_agent.main:app`.
- `LocalKnowledgeBase.refresh_from_ingest`: retained for compatibility, but
  the official SSE path sets `ingest_refresh_enabled=false`. In production,
  ingest is requested through the worker/job contract and promoted artifacts.

## Imported/adapted from `agent-domain`

Status: selectively imported.

- Imported candidates: liquidity agent and technical-analysis agent, adapted
  to the official `PortfolioContext -> AgentVerdict -> AgentResponse` path.
- Remaining useful references: aggregator, mediator/final/report-writer ideas.
- Not imported wholesale because the repo is not standalone: its runner imports
  missing `backend.agents.core.*`, `backend.compliance.*`, and
  `backend.data.providers.*` modules.
- The original LLM prompt-file dependency was not copied because the official
  domain-agent package uses router-based prompts and deterministic fallbacks.
