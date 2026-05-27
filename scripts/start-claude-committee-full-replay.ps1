param(
    [string]$OutDir = "D:\Libra\outputs\backtests\kr-objective-2020-2023-opendart-googlenews",
    [string]$FixtureFile = "comparison-fixture.pykrx-volume.strict.json",
    [string]$EnvFile = "D:\libra-agent\.env.live.local",
    [string]$Model = "claude-sonnet-4-6",
    [string]$GovernancePreset = "",
    [string]$PromptVariant = "",
    [string]$ExecutionPolicyMode = "",
    [string]$ExecutionParticipationRate = "",
    [string]$ExecutionMaxAbsDeltaPct = "",
    [string]$ExecutionResolveTickerConflicts = "",
    [string]$IssueStateEnabled = "",
    [string]$IssueStateCooldownObservations = "",
    [string]$StartDate = "",
    [string]$EndDate = "",
    [ValidateSet("daily", "every-n-trading-days", "weekly")]
    [string]$DecisionFrequency = "daily",
    [int]$DecisionInterval = 1,
    [int]$Limit = 0,
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
    $presetSlug = if ($GovernancePreset) { $GovernancePreset.ToLowerInvariant() -replace "[^a-z0-9]+", "-" } else { "default" }
    $cadenceSlug = switch ($DecisionFrequency) {
        "daily" { "daily" }
        "weekly" { "weekly" }
        "every-n-trading-days" { "every-$DecisionInterval-trading-days" }
    }
    $suffix = if ($Limit -gt 0) { "smoke-$Limit" } else { "full-official" }
    if ($cadenceSlug -ne "daily") {
        $suffix = "$cadenceSlug-$suffix"
    }
    $RunId = "article-$modelSlug-$presetSlug-service-v1-committee-$suffix"
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
if ($GovernancePreset) {
    $env:LIBRA_GOVERNANCE_PRESET = $GovernancePreset
} else {
    Remove-Item Env:LIBRA_GOVERNANCE_PRESET -ErrorAction SilentlyContinue
}
if ($PromptVariant) {
    $env:LIBRA_PROMPT_VARIANT = $PromptVariant
} else {
    Remove-Item Env:LIBRA_PROMPT_VARIANT -ErrorAction SilentlyContinue
}
if ($ExecutionPolicyMode) {
    $env:LIBRA_EXECUTION_POLICY_MODE = $ExecutionPolicyMode
}
if ($ExecutionParticipationRate) {
    $env:LIBRA_EXECUTION_PARTICIPATION_RATE = $ExecutionParticipationRate
}
if ($ExecutionMaxAbsDeltaPct) {
    $env:LIBRA_EXECUTION_MAX_ABS_DELTA_PCT = $ExecutionMaxAbsDeltaPct
}
if ($ExecutionResolveTickerConflicts) {
    $env:LIBRA_EXECUTION_RESOLVE_TICKER_CONFLICTS = $ExecutionResolveTickerConflicts
}
if ($IssueStateEnabled) {
    $env:LIBRA_ISSUE_STATE_ENABLED = $IssueStateEnabled
}
if ($IssueStateCooldownObservations) {
    $env:LIBRA_ISSUE_STATE_COOLDOWN_OBSERVATIONS = $IssueStateCooldownObservations
}

$Fixture = if ([System.IO.Path]::IsPathRooted($FixtureFile)) { $FixtureFile } else { Join-Path $OutDir $FixtureFile }
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

$fixturePrices = (Get-Content $Fixture -Raw | ConvertFrom-Json).prices
$sourceFixtureRows = $fixturePrices.Count
$selectedFixturePrices = @($fixturePrices | Where-Object {
    (-not $StartDate -or [string]$_.date -ge $StartDate) -and
    (-not $EndDate -or [string]$_.date -le $EndDate)
})
$selectedFixtureRows = $selectedFixturePrices.Count
$expectedRows = if ($Limit -gt 0) { [Math]::Min($Limit, $selectedFixtureRows) } else { $selectedFixtureRows }
if ($DecisionInterval -lt 1) {
    throw "DecisionInterval must be >= 1."
}
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
    "--decision-frequency", $DecisionFrequency,
    "--decision-interval", [string]$DecisionInterval,
    "--progress-every", "10"
)
if ($Limit -gt 0) {
    $args += @("--limit", [string]$Limit)
}
if ($StartDate) {
    $args += @("--start-date", $StartDate)
}
if ($EndDate) {
    $args += @("--end-date", $EndDate)
}

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
    source_fixture_rows = $sourceFixtureRows
    expected_rows = $expectedRows
    requested_limit = $Limit
    start_date = if ($StartDate) { $StartDate } else { $null }
    end_date = if ($EndDate) { $EndDate } else { $null }
    decision_frequency = $DecisionFrequency
    decision_interval = $DecisionInterval
    backend = "anthropic"
    model = $Model
    governance_preset = if ($GovernancePreset) { $GovernancePreset } else { "default" }
    prompt_variant = if ($PromptVariant) { $PromptVariant } else { "default" }
    runtime = "JudgeOrchestrator.run_v1_committee"
    domain_agents_enabled = $env:LIBRA_DOMAIN_AGENTS_ENABLED
    disable_agent_fallbacks = $env:LIBRA_DISABLE_AGENT_FALLBACKS
    sentiment_phase2_enabled = $env:LIBRA_SENTIMENT_PHASE2_ENABLED
    committee_round1_max_workers = $env:LIBRA_COMMITTEE_ROUND1_MAX_WORKERS
    committee_round2_max_workers = $env:LIBRA_COMMITTEE_ROUND2_MAX_WORKERS
    execution_policy_mode = $env:LIBRA_EXECUTION_POLICY_MODE
    execution_participation_rate = $env:LIBRA_EXECUTION_PARTICIPATION_RATE
    execution_max_abs_delta_pct = $env:LIBRA_EXECUTION_MAX_ABS_DELTA_PCT
    execution_resolve_ticker_conflicts = $env:LIBRA_EXECUTION_RESOLVE_TICKER_CONFLICTS
    issue_state_enabled = $env:LIBRA_ISSUE_STATE_ENABLED
    issue_state_cooldown_observations = $env:LIBRA_ISSUE_STATE_COOLDOWN_OBSERVATIONS
}
$pidPayload | ConvertTo-Json -Depth 4 | Set-Content -Path $PidJson -Encoding UTF8
$pidPayload | ConvertTo-Json -Depth 4
