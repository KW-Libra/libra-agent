# Agent Module Split

- `disclosure_agent.py`: Disclosure Agent
- `news_agent.py`: News Agent
- `report_agent.py`: Report Agent
- `profit_agent.py`: Profit Agent
- `cost_agent.py`: Cost Agent
- `factory.py`: one place where Judge wires the default agent bundle
- `base.py`: shared protocols and delegation helpers
- `../prompts/`: agent-specific prompt profiles and Judge prompt text

Each agent file is meant to be owned and changed independently by one teammate, together with its matching prompt file.

Reference before editing:

- `../../../../docs/implementation/agent-work-items.md`
- `../../../../examples/agent-responses/`

The default rule is:

- agent owner edits `agents/<agent>.py`
- same owner edits `prompts/<agent>.py`
- shared Judge files move only when the contract really changed
