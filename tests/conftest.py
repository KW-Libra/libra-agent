from __future__ import annotations

import asyncio
import os
import sys

import pytest

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


# Env vars that backtest/replay scripts mutate via os.environ.setdefault.
# Without isolation, a replay-options test that loads/runs those scripts leaks
# e.g. LIBRA_DISABLE_AGENT_FALLBACKS=true into later tests and breaks the
# agent local-fallback tests. Snapshot and restore them per test.
_VOLATILE_ENV_VARS = (
    "LIBRA_DISABLE_AGENT_FALLBACKS",
    "LIBRA_DOMAIN_AGENTS_ENABLED",
    "LIBRA_LLM_PROVIDER",
    "LIBRA_GEMINI_MODEL",
    "GEMINI_MODEL",
    "LLM_ROUTING_POLICY",
    "LIBRA_ANTHROPIC_MODEL",
    "LIBRA_SENTIMENT_PHASE2_ENABLED",
    "LIBRA_LLM_TIMEOUT_SECONDS",
    "LIBRA_LLM_REQUEST_TIMEOUT_SECONDS",
)


@pytest.fixture(autouse=True)
def _isolate_volatile_env():
    saved = {key: os.environ.get(key) for key in _VOLATILE_ENV_VARS}
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
