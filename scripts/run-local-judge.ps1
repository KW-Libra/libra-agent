param(
    [string]$Query = "포트폴리오 점검",
    [string]$Depth = "shallow",
    [int]$Port = 8091,
    [string]$HostAddress = "127.0.0.1",
    [string]$PythonPath = "",
    [switch]$Pretty = $true
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$VenvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$ResolvedPython = if ($PythonPath) {
    $PythonPath
} elseif (Test-Path $VenvPython) {
    $VenvPython
} else {
    "python"
}

& $ResolvedPython -c "import libra_agent" 2>$null
if ($LASTEXITCODE -ne 0) {
    throw "libra_agent is not installed in the selected Python environment. Run: $ResolvedPython -m pip install -e $RepoRoot"
}

$PortfolioPath = Join-Path $RepoRoot "examples\portfolio.sample.json"
$EventsPath = Join-Path $RepoRoot "examples\events.sample.json"
$DocumentsPath = Join-Path $RepoRoot "examples\normalized-documents.sample.json"
$StateDir = Join-Path $RepoRoot "outputs\local-demo"
$ModelPath = Join-Path $RepoRoot "models\supergemma4-26b\supergemma4-26b-abliterated-multimodal-Q4_K_M.gguf"
$MmprojPath = Join-Path $RepoRoot "models\supergemma4-26b\mmproj-supergemma4-26b-abliterated-multimodal-f16.gguf"

New-Item -ItemType Directory -Force -Path $StateDir | Out-Null

$arguments = @(
    "-m", "libra_agent.libra_cli",
    "--backend", "llama_cpp",
    "--llama-no-launch",
    "--llama-host", $HostAddress,
    "--llama-port", "$Port",
    "--llama-model-path", $ModelPath,
    "--llama-mmproj-path", $MmprojPath,
    "--llama-alias", "supergemma4-26b",
    "--query", $Query,
    "--portfolio", $PortfolioPath,
    "--events", $EventsPath,
    "--normalized-documents", $DocumentsPath,
    "--depth", $Depth,
    "--state-dir", $StateDir
)

if ($Pretty) {
    $arguments += "--pretty"
}

& $ResolvedPython @arguments
