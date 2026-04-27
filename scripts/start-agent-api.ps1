param(
  [ValidateSet("llama_cpp", "ollama", "anthropic")]
  [string]$Provider = "llama_cpp",
  [string]$HostName = "127.0.0.1",
  [int]$Port = 8010,
  [string]$StateDir = "outputs\agent-api-local"
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
  $SharedPython = "D:\Libra\.venv\Scripts\python.exe"
  if (Test-Path $SharedPython) {
    $Python = $SharedPython
  } else {
    $Python = "python"
  }
}

$env:LIBRA_LLM_PROVIDER = $Provider
$env:LIBRA_AGENT_HOST = $HostName
$env:LIBRA_AGENT_PORT = [string]$Port
$env:LIBRA_AGENT_STATE_DIR = $StateDir

if ($Provider -eq "llama_cpp") {
  $env:LIBRA_LLAMA_HOST = "127.0.0.1"
  $env:LIBRA_LLAMA_PORT = "8091"
  $env:LIBRA_LLAMA_ALIAS = "supergemma4-26b"
  $env:LIBRA_LLAMA_LAUNCH_SERVER = "false"
}

& $Python -m libra_agent.libra_api
