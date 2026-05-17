# libra-agent

Libra 멀티에이전트 의사결정 거버넌스 — Python LangGraph + FastAPI 서버.

## Stack
- Python 3.12 + uv
- FastAPI + uvicorn + sse-starlette
- LangGraph + `langgraph-checkpoint-postgres` (`AsyncPostgresSaver`)
- Anthropic SDK (Haiku 4.5 / Sonnet 4.6)
- agent-core runtime: core 5 agents, domain council, governance helpers
- Pydantic v2 + pydantic-settings
- structlog (JSON / console 자동 분기)
- psycopg 3 (async)
- boto3 (S3)

## 의사결정 (왜 이렇게)
- **Agent is not the domain DB owner**: portfolio, broker account/order state, decision history, report metadata 같은 비즈니스 영속 데이터의 system of record 는 backend 이다. agent 는 판단 워크플로우와 SSE 이벤트 생성을 맡고, 필요한 입력은 backend/knowledge artifact 를 통해 받는다.
- **B-2**: LangGraph `astream_events` 로 노드 전이 emit → SSE 응답. Spring 이 그대로 Vue 까지 passthrough.
- **AsyncPostgresSaver**: LangGraph checkpoint 전용 저장소다. process 죽었다 살아도 `thread_id` 로 resume 하기 위한 런타임 상태만 저장하며, 포트폴리오/주문/리포트 같은 도메인 데이터는 저장하지 않는다.

## Run (local)

```powershell
# 1. 로컬 Postgres 띄우기 (개발 편의상 libra-backend docker-compose 재사용)
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

## Deploy (EC2 / GitHub Actions)

Agent 배포는 Docker 이미지가 아니라 EC2의 Python 3.12 venv + systemd 서비스를 기준으로 한다.

1. EC2 1회 bootstrap

```bash
# from a checked-out copy of this repository on the EC2 instance
bash scripts/ec2-bootstrap.sh
```

bootstrap 스크립트는 Python 3.12, venv 지원 패키지, AWS CLI, Docker/Compose, `/opt/libra/agent`,
`/opt/libra/knowledge/current`, `/etc/libra/agent.env`, `libra-agent` systemd unit을 준비한다.
GitHub Actions deploy는 이 준비가 끝난 EC2에 SSM으로 접속해 tarball을 풀고 venv를 다시 만든다.

2. EC2 환경 파일 준비

```bash
# from the same checked-out copy used for bootstrap
sudo cp .env.prod.example /etc/libra/agent.env
sudo chown root:libra /etc/libra/agent.env
sudo chmod 0640 /etc/libra/agent.env
sudo editor /etc/libra/agent.env
```

민감정보는 GitHub 저장소나 repo 파일에 넣지 말고 EC2 환경 파일 또는 secret manager에만 둔다.
`.env.prod.example`의 `CHANGE_ME_*` 값은 모두 배포 전에 교체한다.

필수/주의 환경 변수:

| 변수 | 용도 |
|---|---|
| `DATABASE_URL` | **libra-agent LangGraph checkpointer 전용 PostgreSQL URL**. `postgresql://...` 또는 `postgres://...`만 사용한다. portfolio/order/report 같은 도메인 데이터 DB가 아니며, backend용 RDS MySQL `LIBRA_DB_*` 값을 대신 넣으면 안 된다. |
| `ANTHROPIC_API_KEY` | `LIBRA_LLM_PROVIDER=anthropic` 배포에서 Claude 호출에 필요하다. |
| `LIBRA_LLM_TIMEOUT_SECONDS`, `LIBRA_LLM_REQUEST_TIMEOUT_SECONDS` | 외부 LLM 호출 제한 시간. 기본 45초이며, 초과 시 하위 에이전트는 로컬 근거 기반 폴백 응답으로 진행한다. |
| `LIBRA_DOMAIN_AGENTS_ENABLED` | `true`이면 domain council adapter를 켠다. venv 설치도 반드시 `.[domain-agents]` extra로 해야 한다. |
| `KNOWLEDGE_CACHE_DIR` | agent가 먼저 읽는 promote 완료 knowledge cache 경로. 기본값은 `/opt/libra/knowledge/current`. |
| `S3_BUCKET`, `AWS_REGION`, `KNOWLEDGE_S3_PREFIX` | local cache에 필수 artifact가 없을 때 읽는 S3 fallback. `s3://$S3_BUCKET/$KNOWLEDGE_S3_PREFIX/...` 형태로 `manifest.json`, `normalized_documents.json`, `events.json` 등을 찾는다. |
| `LIBRA_DB_HOST`, `LIBRA_DB_PORT`, `LIBRA_DB_NAME`, `LIBRA_DB_USER`, `LIBRA_DB_PASSWORD` | backend용 RDS MySQL 설정이다. agent checkpointer의 `DATABASE_URL`과 별개다. |

3. GitHub Actions deploy 변수

`.github/workflows/deploy.yml`은 Python 3.12에서 `.[dev,domain-agents]`로 테스트하고,
artifact를 S3에 올린 뒤 SSM으로 EC2에서 `/opt/libra/agent/.venv`를 재생성한다.
Repository variables는 `AWS_REGION`, `AWS_ROLE_ARN`, `DEPLOY_BUCKET`, `AGENT_INSTANCE_ID`가 필요하다.
EC2 instance profile에는 deploy artifact S3 read 권한과 SSM 실행 권한이 있어야 한다.

## 엔드포인트

| Method | Path | 비고 |
|---|---|---|
| GET | `/health` | public |
| POST | `/api/runs` | body=`RunStartRequest`, response = **SSE stream**. portfolio holdings 가 있으면 `JudgeOrchestrator` 실제 런타임 실행 |
| POST | `/api/runs/{thread_id}/resume` | body=`ResumeRequest` (사용자 옵션 선택) |

SSE 이벤트 계약 v0 는 `docs/run-events.md` 와 `contracts/run_events.py` 기준.
이벤트 타입은 `run_started` / `node_started` / `node_completed` / `interrupt_required` / `resume_received` / `resume_ignored` / `run_completed` / `run_failed` 로 고정하고, `run_completed.data.agent_result` 에 agent-core 판단 결과를 포함한다.

## Test / E2E

기본 검증은 외부 LLM, broker, Postgres 없이 로컬에서 끝나야 한다.

```powershell
uv run ruff check src tests
uv run pytest -q
uv run python -m unittest discover -s tests
```

HTTP/SSE E2E는 FastAPI `TestClient`와 fake graph로 `/api/runs` 시작, `interrupt_required`,
`/api/runs/{thread_id}/resume`, `run_completed` 계약을 검증한다.

```powershell
uv run pytest tests/test_agent_sse_e2e.py -q
```

Postgres checkpointer는 LangGraph checkpoint 전용 DB가 준비된 경우에만 opt-in으로 실행한다.

```powershell
$env:LIBRA_INTEGRATION_DATABASE_URL="postgresql://libra:libra@localhost:5432/libra"
uv run pytest tests/test_checkpointer_integration.py -q
```

실제 외부 서비스를 호출하는 live E2E는 repo에 커밋되지 않는 `.env.live.local`에 키를 넣고 명시적으로 켠다.
이 테스트는 실제 Postgres checkpointer, 실제 LLM, `/api/runs` SSE, human interrupt, resume까지 검증한다.

```powershell
@"
LIBRA_LIVE_E2E=1
LIBRA_LIVE_DATABASE_URL=postgresql://USER:PASSWORD@HOST:5432/DB
LIBRA_LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=...
LIBRA_ANTHROPIC_MODEL=...
LIBRA_DOMAIN_AGENTS_ENABLED=false
"@ | Set-Content .env.live.local -Encoding UTF8

.\scripts\run-live-e2e.ps1
```

KIS 잔고 조회 smoke는 읽기 전용이며 별도 플래그로 켠다.

```powershell
@"
LIBRA_LIVE_KIS_E2E=1
LIBRA_KIS_ENV=demo
LIBRA_KIS_APP_KEY=...
LIBRA_KIS_APP_SECRET=...
LIBRA_KIS_ACCOUNT_NO=12345678
LIBRA_KIS_PRODUCT_CODE=01
"@ | Add-Content .env.live.local -Encoding UTF8

.\scripts\run-live-e2e.ps1
```

모의투자 계좌에 테스트용 보유 종목을 만들 때는 KIS demo 전용 주문 스크립트를 사용한다.
이 스크립트는 주문을 실제 전송하지만 `demo` 환경으로만 동작하며 실전 환경은 코드에서 거절한다.

```powershell
.\.venv\Scripts\python.exe scripts\kis_paper_buy.py --fixture

# 또는 개별 종목 지정
.\.venv\Scripts\python.exe scripts\kis_paper_buy.py --order 005930:1 --order 000660:1
```

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
| 그래프 | `runtime/graph.py` — compliance_before / round1 / mediator / final_judge / human_review. `final_judge` 에서 agent-core `JudgeOrchestrator` 호출 |
| SSE | `api/sse.py` — `graph.astream_events()` → `sse_starlette` 이벤트 dict |
| RunEvent contract | `docs/run-events.md`, `contracts/run_events.py` — 제품 로직과 분리된 SSE 계약 |
| Integration ledger | `docs/agent-core-domain-integration.md` — split repo 흡수/adapter/보류 기준 |

## 다음 작업
- `agent-domain` 의 liquidity / technical-analysis 를 현 `AgentResponse` adapter 에 맞춰 추가
- backend 가 넘기는 portfolio/knowledge/governance DTO 와 `RunStartRequest` 계약 고정
- 제품 결정 후 RunEvent v1 확장 — compliance_check / agent_started / mediator_completed 등
