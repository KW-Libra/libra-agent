# Agent Work Items

## Goal

This repository is currently a team-distributable skeleton for `libra-agent`.

The immediate goal is not full product completion. The goal is to let multiple teammates implement agent behavior in parallel without breaking the shared Judge flow.

## Frozen Boundaries

These boundaries should stay stable unless the whole team agrees on a contract change.

- Judge orchestration stays in `src/libra_agent/libra_runtime.py`
- LangGraph flow stays in `src/libra_agent/libra_graph.py`
- shared input/output models stay in `src/libra_agent/libra_models.py`
- response normalization stays in `src/libra_agent/libra_validation.py`
- repo-to-repo payloads stay under `contracts/`

Per-agent owners should work inside their own agent file and prompt file first.

## Shared Inputs

All agents receive the same portfolio and local knowledge boundary.

- portfolio fixture: `examples/portfolio.sample.json`
- events fixture: `examples/events.sample.json`
- normalized documents fixture: `examples/normalized-documents.sample.json`

The agent repo does not collect data directly. It consumes already prepared knowledge inputs.

## Shared Output Rule

Every sub-agent must return an `AgentResponse` shape.

Reference:

- runtime model: `src/libra_agent/libra_models.py`
- sanitized examples: `examples/agent-responses/`

Do not invent a custom output shape per teammate.

## Ownership Map

### Disclosure Agent

- owner files:
  - `src/libra_agent/libra/agents/disclosure_agent.py`
  - `src/libra_agent/libra/prompts/disclosure.py`
- sample output:
  - `examples/agent-responses/disclosure.sample.json`
- responsibility:
  - disclosure lookup
  - earnings/disclosure fact summary
  - upcoming disclosure hints
- should not do:
  - final rebalance decision
  - cost/profit evaluation
  - direct external collection integration in this repo
- done when:
  - evidence items are consistent
  - focus tickers are correct
  - reasoning helps Judge decide whether News or Report is needed

### News Agent

- owner files:
  - `src/libra_agent/libra/agents/news_agent.py`
  - `src/libra_agent/libra/prompts/news.py`
- sample output:
  - `examples/agent-responses/news.sample.json`
- responsibility:
  - company-specific news
  - market reaction
  - cross-check count
  - macro context only when relevant
- should not do:
  - final investment decision
  - pretend every run is a push alert
- done when:
  - company findings are easy for Judge to compare
  - push and pull contexts both remain readable

### Report Agent

- owner files:
  - `src/libra_agent/libra/agents/report_agent.py`
  - `src/libra_agent/libra/prompts/report.py`
- sample output:
  - `examples/agent-responses/report.sample.json`
- responsibility:
  - sell-side interpretation
  - consensus clues
  - preview/coverage distinction
- should not do:
  - fabricate report coverage when no evidence exists
  - override Judge with a direct portfolio action
- done when:
  - conflicting signals become easier to interpret
  - lack of report coverage still returns useful fallback context

### Profit Agent

- owner files:
  - `src/libra_agent/libra/agents/profit_agent.py`
  - current implementation lives in `src/libra_agent/libra_runtime.py`
- sample output:
  - `examples/agent-responses/profit.sample.json`
- responsibility:
  - evaluate candidate rebalance plan upside and risk
  - return plan-relative metrics only
- should not do:
  - choose whether to notify the user
  - bypass candidate plan requirement
- done when:
  - same input plan always produces stable structured output
  - Judge can compare expected payoff versus Cost output

### Cost Agent

- owner files:
  - `src/libra_agent/libra/agents/cost_agent.py`
  - current implementation lives in `src/libra_agent/libra_runtime.py`
- sample output:
  - `examples/agent-responses/cost.sample.json`
- responsibility:
  - transaction cost
  - slippage
  - spread and turnover friction
- should not do:
  - directionally vote on thesis by itself
  - operate without a candidate plan
- done when:
  - execution friction is clear enough for Judge to approve, defer, or request user input

### Judge

- owner files:
  - `src/libra_agent/libra_runtime.py`
  - `src/libra_agent/libra_graph.py`
  - `src/libra_agent/libra/prompts/judge.py`
- sample final output:
  - `contracts/judge-run-result.schema.json`
- responsibility:
  - choose which agent to call next
  - decide when to stop
  - combine agent responses into HOLD, DEFER, USER_DECISION_REQUIRED, REBALANCE
  - record decision trace
- should not do:
  - absorb collection logic
  - turn into a fixed pipeline again

## First Delivery Checklist Per Teammate

1. keep the agent response shape stable
2. edit only the assigned agent file and matching prompt file first
3. compare output with the sample file under `examples/agent-responses/`
4. add or update one focused test if shared behavior changes
5. avoid changing `contracts/` without team agreement

## Scenario Targets For The Next Pass

These are the first scenario names that should become scenario tests.

- `regular_check_basic_hold`
- `regular_check_conflicting_signals_defer`
- `regular_check_report_needed`
- `push_risk_event_user_decision_required`
- `rebalance_candidate_profit_cost_review`

When these are added, keep them under `tests/scenarios/`.
