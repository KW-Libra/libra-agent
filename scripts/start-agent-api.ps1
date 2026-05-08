param(
  [ValidateSet("llama_cpp", "ollama", "anthropic", "gemini")]
  [string]$Provider = "anthropic",
  [string]$HostName = "127.0.0.1",
  [int]$Port = 8010,
  [string]$StateDir = "outputs\agent-api-local",
  [string]$EnvFile = ".env"
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$EnvPath = Join-Path $Root $EnvFile
if (Test-Path -LiteralPath $EnvPath) {
  Get-Content -LiteralPath $EnvPath | ForEach-Object {
    $Line = $_.Trim()
    if (-not $Line -or $Line.StartsWith("#") -or -not $Line.Contains("=")) {
      return
    }
    $Parts = $Line -split "=", 2
    $Name = $Parts[0].Trim()
    $Value = $Parts[1]
    if ($Value.StartsWith('"') -and $Value.EndsWith('"')) {
      $Value = $Value.Substring(1, $Value.Length - 2)
    }
    Set-Item -Path "Env:$Name" -Value $Value
  }
}

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
if (-not $env:LIBRA_DOMAIN_AGENTS_ENABLED) {
  $env:LIBRA_DOMAIN_AGENTS_ENABLED = "true"
}

if ($Provider -eq "llama_cpp") {
  $env:LIBRA_LLAMA_HOST = "127.0.0.1"
  $env:LIBRA_LLAMA_PORT = "8091"
  $env:LIBRA_LLAMA_ALIAS = "supergemma4-26b"
  $env:LIBRA_LLAMA_LAUNCH_SERVER = "false"
}

& $Python -m libra_agent.libra_api
