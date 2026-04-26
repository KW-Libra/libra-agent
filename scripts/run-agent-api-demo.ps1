param(
  [string]$BaseUrl = "http://127.0.0.1:8010"
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$Portfolio = Get-Content -Raw (Join-Path $Root "examples\portfolio.sample.json") | ConvertFrom-Json

$Payload = @{
  query = "포트폴리오를 현재 근거 기준으로 점검해줘. 필요한 에이전트만 호출하고 판단 과정을 남겨줘."
  portfolio = $Portfolio
  knowledge_sources = @{
    events = (Join-Path $Root "examples\events.sample.json")
    normalized_documents = (Join-Path $Root "examples\normalized-documents.sample.json")
  }
  depth = "medium"
  trigger = "pull"
}

$Json = $Payload | ConvertTo-Json -Depth 20
Invoke-RestMethod -Method Post -Uri "$BaseUrl/v1/judge-runs" -ContentType "application/json; charset=utf-8" -Body $Json
