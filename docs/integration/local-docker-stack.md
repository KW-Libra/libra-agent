# Local Docker Stack

This local stack is intended for Windows development on `D:` while keeping the runtime shape close to a later AWS deployment.

## Services

- `mysql`: local MySQL 8.4 container for future `RDS MySQL` parity
- `minio`: local S3-compatible object storage for future `Amazon S3` parity
- `api`: Spring Boot backend from `D:\libra-backend`
- `agent-cli`: on-demand LIBRA agent CLI container

## Files

- `docker-compose.local.yml`
- `.env.docker.example`
- `scripts/prepare-local-docker-data.ps1`

## First-Time Setup

```powershell
cd D:\libra-agent
powershell -ExecutionPolicy Bypass -File .\scripts\prepare-local-docker-data.ps1
Copy-Item .\.env.docker.example .\.env.docker
docker compose --env-file .\.env.docker -f .\docker-compose.local.yml up -d mysql minio minio-init api
```

## Check Running Services

```powershell
docker compose --env-file .\.env.docker -f .\docker-compose.local.yml ps
```

- API: `http://localhost:8080`
- MinIO API: `http://localhost:9000`
- MinIO Console: `http://localhost:9001`
- MySQL: `localhost:3307`

## Run Agent CLI Through Docker

The agent is not a long-running HTTP service yet. Run it as an on-demand container.

```powershell
docker compose --env-file .\.env.docker -f .\docker-compose.local.yml --profile manual run --rm agent-cli `
  --backend ollama `
  --ollama-host http://host.docker.internal:11434 `
  --model dolphin-llama3:8b `
  --query "포트폴리오 점검" `
  --portfolio /app/examples/portfolio.sample.json `
  --events /app/examples/events.sample.json `
  --normalized-documents /app/examples/normalized-documents.sample.json `
  --pretty
```

## Notes

- `LIBRA_DOCKER_DATA_ROOT` is set to `D:/docker-data/libra` by default.
- `agent-cli` mounts `./models` to `/models` as read-only for future local GGUF usage.
- `libra-backend` still uses in-memory portfolio state today. MySQL is already present in the stack so the persistence layer can move in without changing the local infrastructure shape.
