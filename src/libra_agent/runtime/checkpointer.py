"""LangGraph AsyncPostgresSaver lifecycle.

- 앱 시작 시 ConnectionPool 열고 setup() 호출 → 테이블 4개 자동 생성
  (checkpoints, checkpoint_blobs, checkpoint_writes, checkpoint_migrations)
- 앱 종료 시 풀 close

interrupt() resume 시: 같은 thread_id 로 graph.invoke 하면 saver 가
마지막 state 로드해서 이어서 실행. 프로세스 죽었다 살아도 보존.

This stores LangGraph runtime checkpoints only. Business/domain data belongs to
the backend API and must not be persisted here.
"""
from __future__ import annotations

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from libra_agent.common.logging import get_logger
from libra_agent.config import settings

log = get_logger(__name__)

_pool: AsyncConnectionPool | None = None
_saver: AsyncPostgresSaver | None = None


async def init_checkpointer() -> None:
    """앱 lifespan startup 에서 호출."""
    global _pool, _saver
    if _saver is not None:
        return

    _pool = AsyncConnectionPool(
        conninfo=settings.database_url,
        max_size=20,
        kwargs={
            "autocommit": True,
            "prepare_threshold": 0,
            "row_factory": dict_row,
        },
        open=False,
    )
    await _pool.open()

    _saver = AsyncPostgresSaver(_pool)
    await _saver.setup()
    log.info("checkpointer_ready", database_url=_masked_url(settings.database_url))


async def close_checkpointer() -> None:
    """앱 lifespan shutdown 에서 호출."""
    global _pool, _saver
    if _pool is not None:
        await _pool.close()
    _pool = None
    _saver = None
    log.info("checkpointer_closed")


def get_checkpointer() -> AsyncPostgresSaver:
    if _saver is None:
        raise RuntimeError(
            "checkpointer not initialized — call init_checkpointer() in lifespan"
        )
    return _saver


def _masked_url(url: str) -> str:
    """postgresql://user:pass@host/db → postgresql://user:***@host/db"""
    if "://" not in url or "@" not in url:
        return url
    scheme, rest = url.split("://", 1)
    creds, host_part = rest.split("@", 1)
    if ":" in creds:
        user = creds.split(":", 1)[0]
        return f"{scheme}://{user}:***@{host_part}"
    return url
