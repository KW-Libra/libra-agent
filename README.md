# libra-agent

Libra 멀티에이전트 의사결정 거버넌스 — Python LangGraph + FastAPI 서버.

## Stack
- Python 3.12 + uv
- FastAPI + uvicorn + sse-starlette
- LangGraph + `langgraph-checkpoint-postgres` (`AsyncPostgresSaver`)
- Anthropic SDK (Haiku 4.5 / Sonnet 4.6)
- Pydantic v2 + pydantic-settings
- structlog (JSON / console 자동 분기)
- psycopg 3 (async)
- boto3 (S3)

## 의사결정 (왜 이렇게)
- **A-2**: 도메인 데이터 (portfolio, decision history, reports) 소유 책임은 agent 한 곳. backend (Spring) 는 users 만.
- **B-2**: LangGraph `astream_events` 로 노드 전이 emit → SSE 응답. Spring 이 그대로 Vue 까지 passthrough.
- **AsyncPostgresSaver**: SqliteSaver 안 씀. `interrupt()` resume 시 process 죽었다 살아도 thread_id 로 복원 가능. backend Postgres 컨테이너 공유.

## Run (local)

```powershell
# 1. Postgres 띄우기 (libra-backend 의 docker-compose 재사용)
cd ..\libra-backend
docker compose up -d
cd ..\libra-agent

# 2. uv 가상환경 + 의존성
uv sync

# 3. env
copy .env.example .env
# .env 에 ANTHROPIC_API_KEY 채움

# 4. run
uv run uvicorn libra_agent.main:app --reload --host 0.0.0.0 --port 8000
```

- http://localhost:8000/health
- http://localhost:8000/docs (Swagger UI)

## 엔드포인트 (현재 골격)

| Method | Path | 비고 |
|---|---|---|
| GET | `/health` | public |
| POST | `/api/runs` | body=`RunStartRequest`, response = **SSE stream**. `approval_required=true` 이면 HITL interrupt 골격을 태움 |
| POST | `/api/runs/{thread_id}/resume` | body=`ResumeRequest` (사용자 옵션 선택) |

SSE 이벤트 계약 v0 는 `docs/run-events.md` 와 `contracts/run_events.py` 기준.
현재는 제품 판단 로직 없이 `run_started` / `node_started` / `node_completed` / `interrupt_required` / `resume_received` / `resume_ignored` / `run_completed` / `run_failed` 만 고정.

## Ingest Handoff

Agent는 요청 처리 중 `libra-ingest`를 직접 실행하지 않는다. 최신 데이터가 필요하면
`ingest_jobs` 계약에 맞춰 job을 만들고, 별도 `libra-ingest-worker`가 성공적으로
promote한 knowledge artifact만 읽는다.

- 계약 문서: `docs/ingest-job-contract.md`
- 코드 계약: `contracts/ingest_jobs.py`
- 기본 cache 위치: `/opt/libra/knowledge/current`
- S3 prefix: `knowledge/current`
- reader: `knowledge/reader.py`
- smoke endpoint: `GET /internal/knowledge/current`

## 횡단

| 항목 | 위치 |
|---|---|
| traceId | `common/correlation.py` — `contextvars.ContextVar`. Spring 의 `X-Trace-Id` 헤더로 들어오거나 새로 발급 |
| 로깅 | `common/logging.py` — structlog. `LOG_FORMAT=json` 이면 JSON, 아니면 console |
| 에러 응답 | `common/errors.py` — `ApiError` → RFC 7807 ProblemDetail (Spring 과 같은 포맷) |
| Checkpointer | `runtime/checkpointer.py` — lifespan 에서 `setup()` 1회, 종료 시 풀 close |
| 그래프 | `runtime/graph.py` — 노드 stub (compliance_before / round1 / mediator / final_judge) |
| SSE | `api/sse.py` — `graph.astream_events()` → `sse_starlette` 이벤트 dict |
| RunEvent contract | `docs/run-events.md`, `contracts/run_events.py` — 제품 로직과 분리된 SSE 계약 |

## 다음 작업
- `schemas/` — spec §2 (AgentOpinion / Vote / FinalDecision / ComplianceCheck) + contracts 차용 (portfolioSnapshot, push-trigger-event, user-approval-response)
- `agents/` — 11 LLM 에이전트 prompt (Risk reference 부터, `prompts_v1.md` §5)
- `mediator.py` / `final_judge.py` — Anthropic `tool_use` 강제
- `compliance/` — 10 룰 (코드, LLM 아님)
- 제품 결정 후 RunEvent v1 확장 — compliance_check / agent_started / mediator_completed 등
- 실제 agent 판단 로직 — 지금은 `approval_required` 플래그로 HITL/checkpoint/resume 배선만 검증
