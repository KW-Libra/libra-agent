param(
    [string]$DataRoot = "D:/docker-data/libra"
)

$targets = @(
    $DataRoot,
    "$DataRoot/mysql",
    "$DataRoot/minio",
    "$DataRoot/agent-state"
)

foreach ($target in $targets) {
    New-Item -ItemType Directory -Force -Path $target | Out-Null
}

Write-Output "Prepared local Docker data directories under $DataRoot"
