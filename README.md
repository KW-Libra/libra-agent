# Libra Agent

`libra-agent` is the standalone decision engine repository for LIBRA.

This repository owns:

- Judge-centered LangGraph orchestration
- agent modules for Disclosure, News, Report, Profit, and Cost
- local or remote LLM adapter layer
- decision trace generation
- local runtime persistence for runs and follow-up checkpoints

This repository does not own:

- market/news/disclosure collection
- normalization or event generation
- user accounts, portfolios, schedules, broker execution
- frontend UI

Recommended repository split:

- `libra-ingest`
- `libra-agent`
- `libra-backend` (Spring Boot)
- `libra-frontend`

## Directory Layout

- `src/libra_agent/`: Python package for the agent runtime
- `src/libra_agent/libra/agents/`: team-owned agent modules
- `src/libra_agent/libra/prompts/`: team-owned prompt profiles and Judge prompt text
- `docs/specs/`: product and scenario specs
- `docs/architecture/`: architecture and agent design docs
- `docs/integration/`: repo boundaries and integration contracts
- `docs/implementation/`: team handoff and work-item docs
- `examples/`: agent input fixtures such as portfolio, events, and normalized documents
- `contracts/`: versioned payload contracts shared with `libra-backend`
- `scripts/`: helper scripts for local development
- `tests/`: test suite
- `models/`: local model weights kept out of source control
- `tools/`: local binaries such as `llama.cpp`
- `outputs/`: local runtime outputs and checkpoint state
- `tmp/`: disposable local scratch space

## Key Files

- `docs/specs/libra-scenarios-spec-v1.md`
- `docs/architecture/libra-agent-detailed-design.md`
- `docs/integration/repo-boundaries.md`
- `docs/implementation/agent-work-items.md`
- `src/libra_agent/libra/TEAM_GUIDE.md`
- `examples/portfolio.sample.json`
- `examples/events.sample.json`
- `examples/normalized-documents.sample.json`
- `examples/agent-responses/README.md`

## Installation

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -U pip
pip install -e .
```

## Local Docker Stack

For Windows development with Docker data on `D:`, use the local stack in [local-docker-stack.md](docs/integration/local-docker-stack.md).

Main files:

- `docker-compose.local.yml`
- `.env.docker.example`
- `scripts/prepare-local-docker-data.ps1`

## Low-Cost AWS Deployment

For the first AWS deployment, use one EC2 host with Docker Compose, RDS MySQL, and S3.

Start with:

- `docker-compose.prod.yml`
- `.env.prod.example`
- `infra/caddy/Caddyfile`
- `scripts/ec2-bootstrap.sh`
- `docs/deployment/ec2-rds-s3.md`

This path intentionally avoids ECS, ALB, and NAT Gateway until the project needs them.

GitHub Actions are included for:

- CI: `.github/workflows/ci.yml`
- EC2 deploy: `.github/workflows/deploy-ec2.yml`

## Local SuperGemma Demo

For the shortest local Judge run with `supergemma4-26b`, use [local-demo.md](docs/local-demo.md).

```powershell
.\scripts\start-supergemma-llama.ps1
.\scripts\run-local-judge.ps1
```

## Agent HTTP API

For Spring Boot integration, run the Python decision engine as a small HTTP service:

```powershell
.\scripts\start-agent-api.ps1
```

Then call:

```text
POST http://127.0.0.1:8010/v1/judge-runs
```

The API reads its provider from environment variables:

- `LIBRA_LLM_PROVIDER=anthropic` with `ANTHROPIC_API_KEY` for Claude API. This is the default path for normal development and AWS deployment.
- `LIBRA_LLM_PROVIDER=llama_cpp` for the local `supergemma4-26b` server
- `LIBRA_LLM_PROVIDER=ollama` for temporary Ollama testing

`scripts/start-agent-api.ps1` loads `.env` from the repo root if it exists. Keep `.env` local only; it is ignored by git.

## Runtime Entry Point

```powershell
python -m libra_agent.libra_cli `
  --query "포트폴리오 점검" `
  --portfolio examples\portfolio.sample.json `
  --events examples\events.sample.json `
  --normalized-documents examples\normalized-documents.sample.json `
  --pretty
```

KIS portfolio bootstrap example:

```powershell
python -m libra_agent.libra_cli `
  --query "포트폴리오 점검" `
  --portfolio-source kis `
  --kis-config "$HOME\\KIS\\config\\kis_devlp.yaml" `
  --kis-env real `
  --events examples\events.sample.json `
  --normalized-documents examples\normalized-documents.sample.json `
  --pretty
```

`--portfolio-source kis` is a local bootstrap path for development before `libra-backend` owns broker integration. Long-term broker integration still belongs in `libra-backend`.

Template config: `examples/kis_devlp.template.yaml`

Direct `llama.cpp` example:

```powershell
python -m libra_agent.libra_cli `
  --backend llama_cpp `
  --query "포트폴리오 점검" `
  --portfolio examples\portfolio.sample.json `
  --events examples\events.sample.json `
  --normalized-documents examples\normalized-documents.sample.json `
  --llama-server-path tools\llama.cpp\b8783\bin\llama-server.exe `
  --llama-model-path models\supergemma4-26b\supergemma4-26b-abliterated-multimodal-Q4_K_M.gguf `
  --llama-mmproj-path models\supergemma4-26b\mmproj-supergemma4-26b-abliterated-multimodal-f16.gguf `
  --llama-alias supergemma4-26b `
  --pretty
```

Direct Claude API example:

```powershell
$env:ANTHROPIC_API_KEY = "<your key>"
python -m libra_agent.libra_cli `
  --backend anthropic `
  --anthropic-model claude-sonnet-4-5 `
  --query "포트폴리오 점검" `
  --portfolio examples\portfolio.sample.json `
  --events examples\events.sample.json `
  --normalized-documents examples\normalized-documents.sample.json `
  --pretty
```

## Required Inputs

`libra-agent` expects agent-ready knowledge inputs, typically produced by `libra-ingest`.

- events file: `events.json` or `events.jsonl`
- normalized documents file: `normalized_documents.json` or `normalized_documents.jsonl`
- portfolio file: `portfolio.sample.json` shape

## Current Scope

- Judge-centered orchestration
- dynamic sub-agent calling
- local LLM runtime via Ollama or `llama.cpp`
- LangGraph checkpointing
- structured decision output
- follow-up and feedback checkpoint persistence
- optional local KIS domestic-stock portfolio bootstrap for development

## Contract Direction

- `libra-frontend -> libra-backend`
- `libra-backend -> libra-agent`
- `libra-backend -> libra-ingest`

`libra-agent` should receive structured inputs from `libra-backend`, not call product systems directly.

## Backend Note

`libra-backend` is the product backend and should be implemented as a Spring Boot repository.

`libra-agent` stays as the Python decision engine repository:

- LangGraph orchestration remains here
- LLM provider adapters remain here
- agent logic remains here
- Spring Boot calls this repo through a versioned contract boundary

See `docs/integration/spring-boot-agent-boundary.md` for the recommended split.
