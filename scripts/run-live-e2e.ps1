param(
    [string]$EnvFile = ".env.live.local",
    [string]$TestPath = "tests/test_live_e2e.py"
)

$ErrorActionPreference = "Stop"
$tunnelProcess = $null

if (Test-Path -LiteralPath $EnvFile) {
    Get-Content -LiteralPath $EnvFile | ForEach-Object {
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

if (-not $env:LIBRA_INTEGRATION_DATABASE_URL -and $env:LIBRA_LIVE_DATABASE_URL) {
    $env:LIBRA_INTEGRATION_DATABASE_URL = $env:LIBRA_LIVE_DATABASE_URL
}

function Test-PortOpen {
    param(
        [string]$HostName,
        [int]$Port
    )
    try {
        $client = [System.Net.Sockets.TcpClient]::new()
        $task = $client.ConnectAsync($HostName, $Port)
        if (-not $task.Wait(1500)) {
            $client.Dispose()
            return $false
        }
        $client.Dispose()
        return $true
    } catch {
        return $false
    }
}

function Resolve-CommandPath {
    param(
        [string]$Name,
        [string]$Override
    )
    if ($Override) {
        if (-not (Test-Path -LiteralPath $Override)) {
            throw "$Name not found at override path: $Override"
        }
        return (Resolve-Path -LiteralPath $Override).Path
    }
    $cmd = Get-Command $Name -ErrorAction SilentlyContinue
    if ($cmd) {
        return $cmd.Source
    }
    foreach ($candidate in @(
        "C:\Windows\System32\OpenSSH\$Name.exe",
        "C:\Program Files\Git\usr\bin\$Name.exe"
    )) {
        if (Test-Path -LiteralPath $candidate) {
            return $candidate
        }
    }
    throw "$Name not found"
}

if ($env:LIBRA_LIVE_DB_TUNNEL -eq "ssh") {
    $localPort = if ($env:LIBRA_LIVE_DB_LOCAL_PORT) { [int]$env:LIBRA_LIVE_DB_LOCAL_PORT } else { 15432 }
    if (-not (Test-PortOpen -HostName "127.0.0.1" -Port $localPort)) {
        $ssh = Resolve-CommandPath -Name "ssh" -Override $env:LIBRA_SSH_PATH
        $keyPath = if ($env:LIBRA_SSH_KEY_PATH) { $env:LIBRA_SSH_KEY_PATH } else { "D:\libra-infra\libra-key.pem" }
        $apiHost = if ($env:LIBRA_API_HOST) { $env:LIBRA_API_HOST } else { "3.34.80.58" }
        $sshUser = if ($env:LIBRA_SSH_USER) { $env:LIBRA_SSH_USER } else { "ubuntu" }
        $forward = "${localPort}:127.0.0.1:5432"
        $args = @(
            "-i", $keyPath,
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", "ExitOnForwardFailure=yes",
            "-N",
            "-L", $forward,
            "$sshUser@$apiHost"
        )
        $stdout = Join-Path $env:TEMP "libra-live-db-tunnel.out"
        $stderr = Join-Path $env:TEMP "libra-live-db-tunnel.err"
        Remove-Item -LiteralPath $stdout, $stderr -ErrorAction SilentlyContinue
        $tunnelProcess = Start-Process -FilePath $ssh -ArgumentList $args -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru -WindowStyle Hidden
        Start-Sleep -Seconds 3
        if (-not (Test-PortOpen -HostName "127.0.0.1" -Port $localPort)) {
            $err = if (Test-Path -LiteralPath $stderr) { Get-Content -LiteralPath $stderr -Raw } else { "" }
            throw "DB SSH tunnel did not open on 127.0.0.1:$localPort. $err"
        }
    }
}

$python = ".\.venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    $python = "python"
}

try {
    & $python -m pytest $TestPath -q -s
    exit $LASTEXITCODE
} finally {
    if ($null -ne $tunnelProcess -and -not $tunnelProcess.HasExited) {
        Stop-Process -Id $tunnelProcess.Id -Force
    }
}
