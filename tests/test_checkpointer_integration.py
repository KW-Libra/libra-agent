from __future__ import annotations

import os

import pytest

DATABASE_URL_ENV = "LIBRA_INTEGRATION_DATABASE_URL"


def test_app_lifespan_initializes_postgres_checkpointer(monkeypatch: pytest.MonkeyPatch) -> None:
    database_url = os.environ.get(DATABASE_URL_ENV)
    if not database_url:
        pytest.skip(f"set {DATABASE_URL_ENV} to run Postgres checkpointer integration tests")

    from fastapi.testclient import TestClient

    from libra_agent import main as main_module
    from libra_agent.config import settings
    from libra_agent.runtime.checkpointer import get_checkpointer

    init_checkpointer = main_module.init_checkpointer
    init_called = False

    async def observed_init_checkpointer() -> None:
        nonlocal init_called
        init_called = True
        await init_checkpointer()

    monkeypatch.setattr(settings, "database_url", database_url)
    monkeypatch.setattr(main_module, "init_checkpointer", observed_init_checkpointer)

    with TestClient(main_module.app) as client:
        response = client.get("/health")

        assert init_called
        assert get_checkpointer() is not None
        assert response.status_code == 200
        assert response.json()["status"] == "UP"

    with pytest.raises(RuntimeError, match="checkpointer not initialized"):
        get_checkpointer()
