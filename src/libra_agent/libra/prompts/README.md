# Prompt Split

- `disclosure.py`: Disclosure Agent prompt profile
- `news.py`: News Agent prompt profile
- `report.py`: Report Agent prompt profile
- `judge.py`: Judge planner and decision prompt text
- `base.py`: shared prompt profile dataclass and builders

Prompt owners should change these files first before touching runtime orchestration code.

Each prompt should stay consistent with:

- `../../../../docs/implementation/agent-work-items.md`
- `../../../../examples/agent-responses/`
