param(
    [string]$OutDir = "D:\Libra\outputs\backtests\kr-objective-2020-2023-opendart-googlenews",
    [string]$EnvFile = "D:\libra-agent\.env.live.local",
    [string]$Model = "claude-sonnet-4-6",
    [string]$RunId = "",
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python"
}

if (Test-Path $EnvFile) {
    Get-Content $EnvFile | ForEach-Object {
        $line = $_.Trim()
        if (-not $line -or $line.StartsWith("#") -or -not $line.Contains("=")) {
            return
        }
        $parts = $line.Split("=", 2)
        $name = $parts[0].Trim()
        $value = $parts[1].Trim().Trim('"').Trim("'")
        if ($name) {
            Set-Item -Path "Env:$name" -Value $value
        }
    }
}

if (-not $env:ANTHROPIC_API_KEY) {
    throw "ANTHROPIC_API_KEY is required. Put it in $EnvFile or the current environment."
}

if (-not $RunId) {
    $modelSlug = $Model.ToLowerInvariant() -replace "[^a-z0-9]+", "-"
    $RunId = "article-$modelSlug-service-v1-committee-full-official"
}

$env:PYTHONPATH = "src"
$env:LIBRA_LLM_PROVIDER = "anthropic"
$env:LIBRA_ANTHROPIC_MODEL = $Model
$env:LIBRA_DOMAIN_AGENTS_ENABLED = "true"
$env:LLM_ROUTING_POLICY = "claude"
$env:LIBRA_DISABLE_AGENT_FALLBACKS = "true"
$env:LIBRA_SENTIMENT_PHASE2_ENABLED = "false"
$env:LIBRA_LLM_TIMEOUT_SECONDS = "180"
$env:LIBRA_LLM_REQUEST_TIMEOUT_SECONDS = "180"
$env:LIBRA_COMMITTEE_ROUND1_MAX_WORKERS = "11"
$env:LIBRA_COMMITTEE_ROUND2_MAX_WORKERS = "4"
$env:LIBRA_COMMITTEE_LLM_REPAIR_ATTEMPTS = "1"
$env:LIBRA_DROP_INVALID_MEDIATOR_TARGETS = "true"
$env:LIBRA_COMMITTEE_OPINION_REASONING_CHARS = "420"

$Fixture = Join-Path $OutDir "comparison-fixture.json"
$BundlesDir = Join-Path $OutDir "ingest-bundles-article"
$RawOut = Join-Path $OutDir "libra-replay-results.$RunId.jsonl"
$DecisionsOut = Join-Path $OutDir "libra-decisions.$RunId.json"
$SummaryOut = Join-Path $OutDir "$RunId.summary.json"
$UsageLog = Join-Path $OutDir "anthropic-$RunId.usage.jsonl"
$TraceOut = Join-Path $OutDir "$RunId.trace.jsonl"
$StdoutLog = Join-Path $OutDir "$RunId.stdout.log"
$StderrLog = Join-Path $OutDir "$RunId.stderr.log"
$PidJson = Join-Path $OutDir "$RunId.pid.json"

foreach ($path in @($Fixture, (Join-Path $BundlesDir "index.json"))) {
    if (-not (Test-Path $path)) {
        throw "Required input is missing: $path"
    }
}

$outputPaths = @($RawOut, $DecisionsOut, $SummaryOut, $UsageLog, $TraceOut, $StdoutLog, $StderrLog, $PidJson)
$existing = @($outputPaths | Where-Object { Test-Path $_ })
if ($existing.Count -gt 0 -and -not $Force) {
    throw "Output already exists for run '$RunId'. Re-run with -Force or choose another -RunId."
}
if ($Force) {
    foreach ($path in $existing) {
        Remove-Item -LiteralPath $path -Force
    }
}

$expectedRows = (Get-Content $Fixture -Raw | ConvertFrom-Json).prices.Count
$args = @(
    (Join-Path $RepoRoot "scripts\replay_full_committee_backtest.py"),
    "--fixture", $Fixture,
    "--bundles-dir", $BundlesDir,
    "--out", $DecisionsOut,
    "--summary-out", $SummaryOut,
    "--raw-out", $RawOut,
    "--usage-log", $UsageLog,
    "--trace-out", $TraceOut,
    "--backend", "anthropic",
    "--anthropic-model", $Model,
    "--fail-on-fallback-events",
    "--progress-every", "10"
)

$process = Start-Process `
    -FilePath $Python `
    -ArgumentList $args `
    -WorkingDirectory $RepoRoot `
    -RedirectStandardOutput $StdoutLog `
    -RedirectStandardError $StderrLog `
    -WindowStyle Hidden `
    -PassThru

$pidPayload = [ordered]@{
    run_id = $RunId
    pid = $process.Id
    started_at = (Get-Date).ToString("o")
    cwd = $RepoRoot
    python = $Python
    script = (Join-Path $RepoRoot "scripts\replay_full_committee_backtest.py")
    fixture = $Fixture
    bundles_dir = $BundlesDir
    raw_out = $RawOut
    decisions_out = $DecisionsOut
    summary_out = $SummaryOut
    usage_log = $UsageLog
    trace_out = $TraceOut
    stdout_log = $StdoutLog
    stderr_log = $StderrLog
    expected_rows = $expectedRows
    backend = "anthropic"
    model = $Model
    runtime = "JudgeOrchestrator.run_v1_committee"
    domain_agents_enabled = $env:LIBRA_DOMAIN_AGENTS_ENABLED
    disable_agent_fallbacks = $env:LIBRA_DISABLE_AGENT_FALLBACKS
    sentiment_phase2_enabled = $env:LIBRA_SENTIMENT_PHASE2_ENABLED
    committee_round1_max_workers = $env:LIBRA_COMMITTEE_ROUND1_MAX_WORKERS
    committee_round2_max_workers = $env:LIBRA_COMMITTEE_ROUND2_MAX_WORKERS
}
$pidPayload | ConvertTo-Json -Depth 4 | Set-Content -Path $PidJson -Encoding UTF8
$pidPayload | ConvertTo-Json -Depth 4
