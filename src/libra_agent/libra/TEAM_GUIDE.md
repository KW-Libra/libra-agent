# LIBRA Team Work Guide

## Ownership Split

- `agents/disclosure_agent.py`: Disclosure Agent owner
- `prompts/disclosure.py`: Disclosure Agent prompt owner
- `agents/news_agent.py`: News Agent owner
- `prompts/news.py`: News Agent prompt owner
- `agents/report_agent.py`: Report Agent owner
- `prompts/report.py`: Report Agent prompt owner
- `agents/profit_agent.py`: Profit Agent owner
- `agents/cost_agent.py`: Cost Agent owner
- `agents/evaluation_agent.py`: Evaluation Agent owner
- `prompts/judge.py`: Judge planner/decision prompt owner
- `config.py`: LIBRA runtime/backend config owner
- `llm_clients/`: provider adapter and client factory owner
- `../libra_runtime.py`: Judge orchestration and shared decision policy owner
- `../libra_graph.py`: LangGraph flow owner

## Working Rule

Each agent owner should change the matching `agents/*.py` and `prompts/*.py` pair first.

The runtime owns orchestration. Sub-agent behavior lives in `agents/`, and prompt text plus response-shape hints live under `prompts/`.

Information agents may override `prepare_request()` in their own file when they need agent-specific query shaping before the shared LLM/tool loop runs. Trade agents own their deterministic plan evaluators directly in their own files.

Evaluation Agent is separate from the Judge-time loop. It scores a stored decision after a feedback checkpoint or realized-return observation and is exposed through the evaluation API.

## Handoff References

- `../../../docs/implementation/agent-work-items.md`: team work split and done criteria
- `../../../examples/agent-responses/`: sanitized sample outputs per agent
- `../../../tests/scenarios/README.md`: planned scenario-level test skeleton

Do not change Judge/runtime files for a normal sub-agent task unless the shared contract must change.
