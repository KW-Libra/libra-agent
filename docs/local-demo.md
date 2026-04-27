# Local Demo

This is the shortest local path for checking that LIBRA can run from portfolio input to a Judge result JSON.

It intentionally avoids Docker, MySQL, MinIO, and the Spring Boot backend. Those pieces can be added after the local decision loop is stable.

## Prerequisites

- `D:\libra-agent\models\supergemma4-26b\supergemma4-26b-abliterated-multimodal-Q4_K_M.gguf`
- `D:\libra-agent\models\supergemma4-26b\mmproj-supergemma4-26b-abliterated-multimodal-f16.gguf`
- `D:\libra-agent\tools\llama.cpp\b8783\bin\llama-server.exe`
- Python environment with `libra-agent` installed

`models/` and `tools/` are local-only directories and are intentionally ignored by git.

## Run

From `D:\libra-agent`:

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
```

```powershell
.\scripts\start-supergemma-llama.ps1
.\scripts\run-local-judge.ps1
```

The first command starts a local OpenAI-compatible `llama.cpp` server on `http://127.0.0.1:8091`.

The second command runs the LangGraph Judge flow against:

- `examples\portfolio.sample.json`
- `examples\events.sample.json`
- `examples\normalized-documents.sample.json`

Outputs and checkpoints are written under:

```text
outputs\local-demo
```

The latest verified run used `supergemma4-26b` through `llama.cpp` and returned a LangGraph result with `decision: DEFER`.

## Run as Local Agent API

Start the same Judge runtime as an HTTP service. Claude API is the default provider when `.env` contains `ANTHROPIC_API_KEY`:

```powershell
.\scripts\start-agent-api.ps1
```

In another terminal:

```powershell
.\scripts\run-agent-api-demo.ps1
```

This calls:

```text
POST http://127.0.0.1:8010/v1/judge-runs
```

To force the local `llama.cpp` runtime instead:

```powershell
.\scripts\start-supergemma-llama.ps1
.\scripts\start-agent-api.ps1 -Provider llama_cpp
```

## Claude API Mode

For AWS deployment, keep the Python agent API shape the same and switch only the LLM provider:

```powershell
$env:LIBRA_LLM_PROVIDER = "anthropic"
$env:ANTHROPIC_API_KEY = "<your key>"
$env:LIBRA_ANTHROPIC_MODEL = "claude-sonnet-4-5"
.\scripts\start-agent-api.ps1 -Provider anthropic
```

Spring Boot should call `POST /v1/judge-runs` and should not know whether the agent is using local `supergemma4-26b` or Claude behind the boundary.

## Useful Checks

```powershell
Invoke-RestMethod http://127.0.0.1:8091/health
Invoke-RestMethod http://127.0.0.1:8091/v1/models
Invoke-RestMethod http://127.0.0.1:8010/health
```

## Stop llama.cpp

```powershell
Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process
```
