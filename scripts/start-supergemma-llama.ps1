param(
    [int]$Port = 8091,
    [string]$HostAddress = "127.0.0.1",
    [int]$ContextSize = 2048,
    [string]$GpuLayers = "auto",
    [int]$StartupTimeoutSeconds = 600,
    [switch]$ForceRestart
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$ServerPath = Join-Path $RepoRoot "tools\llama.cpp\b8783\bin\llama-server.exe"
$ModelPath = Join-Path $RepoRoot "models\supergemma4-26b\supergemma4-26b-abliterated-multimodal-Q4_K_M.gguf"
$MmprojPath = Join-Path $RepoRoot "models\supergemma4-26b\mmproj-supergemma4-26b-abliterated-multimodal-f16.gguf"
$LogDir = Join-Path $RepoRoot "outputs\llama-server"
$Alias = "supergemma4-26b"
$HealthHost = if ($HostAddress -eq "0.0.0.0") { "127.0.0.1" } else { $HostAddress }
$HealthUrl = "http://${HealthHost}:${Port}/health"

function Test-Health {
    param([string]$Url)
    try {
        $response = Invoke-RestMethod -Uri $Url -TimeoutSec 5
        return $response.status -eq "ok"
    } catch {
        return $false
    }
}

function Get-PortOwner {
    param([int]$TargetPort)
    Get-NetTCPConnection -LocalPort $TargetPort -ErrorAction SilentlyContinue |
        Where-Object { $_.State -eq "Listen" } |
        Select-Object -First 1
}

function Get-LogTail {
    param([string]$Path)
    if (-not (Test-Path $Path)) {
        return ""
    }
    return (Get-Content $Path -Tail 40) -join [Environment]::NewLine
}

foreach ($path in @($ServerPath, $ModelPath, $MmprojPath)) {
    if (-not (Test-Path $path)) {
        throw "Required file does not exist: $path"
    }
}

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

if (Test-Health -Url $HealthUrl) {
    Write-Output "llama.cpp server is already healthy at $HealthUrl"
    return
}

$owner = Get-PortOwner -TargetPort $Port
if ($owner) {
    $process = Get-Process -Id $owner.OwningProcess -ErrorAction SilentlyContinue
    if ($ForceRestart -and $process -and $process.ProcessName -eq "llama-server") {
        Stop-Process -Id $process.Id -Force
        Start-Sleep -Seconds 2
    } else {
        throw "Port $Port is already in use by process $($owner.OwningProcess). Use -ForceRestart only if it is a stale llama-server process."
    }
}

$stdoutPath = Join-Path $LogDir "supergemma4-26b-${Port}.stdout.log"
$stderrPath = Join-Path $LogDir "supergemma4-26b-${Port}.stderr.log"

$arguments = @(
    "-m", $ModelPath,
    "--host", $HostAddress,
    "--port", "$Port",
    "-c", "$ContextSize",
    "-ngl", $GpuLayers,
    "--alias", $Alias,
    "--reasoning", "off",
    "--no-webui",
    "--jinja",
    "-mm", $MmprojPath
)

$process = Start-Process `
    -FilePath $ServerPath `
    -ArgumentList $arguments `
    -WorkingDirectory (Split-Path $ServerPath -Parent) `
    -RedirectStandardOutput $stdoutPath `
    -RedirectStandardError $stderrPath `
    -WindowStyle Hidden `
    -PassThru

$deadline = (Get-Date).AddSeconds($StartupTimeoutSeconds)
while ((Get-Date) -lt $deadline) {
    if (Test-Health -Url $HealthUrl) {
        Write-Output "Started supergemma4-26b llama.cpp server"
        Write-Output "PID: $($process.Id)"
        Write-Output "Health: $HealthUrl"
        Write-Output "OpenAI-compatible chat API: http://${HealthHost}:${Port}/v1/chat/completions"
        Write-Output "stdout: $stdoutPath"
        Write-Output "stderr: $stderrPath"
        return
    }

    if ($process.HasExited) {
        $stdoutTail = Get-LogTail -Path $stdoutPath
        $stderrTail = Get-LogTail -Path $stderrPath
        throw "llama-server exited with code $($process.ExitCode).`nstdout:`n$stdoutTail`nstderr:`n$stderrTail"
    }

    Start-Sleep -Seconds 2
}

$stdoutTail = Get-LogTail -Path $stdoutPath
$stderrTail = Get-LogTail -Path $stderrPath
throw "Timed out waiting for llama-server at $HealthUrl.`nstdout:`n$stdoutTail`nstderr:`n$stderrTail"
