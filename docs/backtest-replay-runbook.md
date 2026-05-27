# Backtest Replay Runbook

This runbook describes how to run the official committee replay without editing code.
It supports two operator-controlled dimensions:

- Date range: replay any contiguous segment inside the prepared fixture.
- Decision cadence: run the LLM committee daily, weekly, or every N trading days.

Scheduled non-decision days are recorded as `SCHEDULED_SKIP` rows. They are not mock
decisions and they do not call an LLM. They keep the price path continuous while making
the replay cadence explicit in raw, compact, and summary outputs.

## Inputs

Default prepared data:

```powershell
$OutDir = "D:\Libra\outputs\backtests\kr-objective-2020-2023-opendart-googlenews"
```

Required files under that directory:

```text
comparison-fixture.json
ingest-bundles-article\index.json
```

The launcher reads `ANTHROPIC_API_KEY` from `D:\libra-agent\.env.live.local` unless
another `-EnvFile` is provided.

## Daily Full Replay

```powershell
cd D:\libra-agent
.\scripts\start-claude-committee-full-replay.ps1 `
  -Model claude-haiku-4-5-20251001 `
  -GovernancePreset aggressive `
  -RunId article-haiku-aggressive-daily-full `
  -Force
```

This preserves the previous behavior: one committee judgement per trading day.

## Arbitrary Date Range

```powershell
cd D:\libra-agent
.\scripts\start-claude-committee-full-replay.ps1 `
  -Model claude-haiku-4-5-20251001 `
  -GovernancePreset aggressive `
  -StartDate 2021-01-04 `
  -EndDate 2021-12-30 `
  -RunId article-haiku-aggressive-2021-only `
  -Force
```

The replay initializes the portfolio at the first selected price row, so a mid-fixture
range behaves like a standalone backtest for that period.

## Every N Trading Days

```powershell
cd D:\libra-agent
.\scripts\start-claude-committee-full-replay.ps1 `
  -Model claude-haiku-4-5-20251001 `
  -GovernancePreset aggressive `
  -DecisionFrequency every-n-trading-days `
  -DecisionInterval 5 `
  -RunId article-haiku-aggressive-every-5td `
  -Force
```

The first selected trading day is a judgement day. After that, every fifth selected
trading day runs the committee.

## Weekly Replay

```powershell
cd D:\libra-agent
.\scripts\start-claude-committee-full-replay.ps1 `
  -Model claude-haiku-4-5-20251001 `
  -GovernancePreset aggressive `
  -DecisionFrequency weekly `
  -RunId article-haiku-aggressive-weekly `
  -Force
```

Weekly cadence uses the first observed trading day in each ISO week.

## Evaluate Outputs

Full-source replay:

```powershell
python scripts\evaluate_replay_strategies.py `
  --raw D:\Libra\outputs\backtests\kr-objective-2020-2023-opendart-googlenews\libra-replay-results.article-haiku-aggressive-daily-full.jsonl `
  --fixture D:\Libra\outputs\backtests\kr-objective-2020-2023-opendart-googlenews\comparison-fixture.json `
  --out-json D:\Libra\outputs\backtests\kr-objective-2020-2023-opendart-googlenews\comparison-results.article-haiku-aggressive-daily-full.json `
  --out-csv D:\Libra\outputs\backtests\kr-objective-2020-2023-opendart-googlenews\comparison-results.article-haiku-aggressive-daily-full.csv `
  --out-md D:\Libra\outputs\backtests\kr-objective-2020-2023-opendart-googlenews\comparison-results.article-haiku-aggressive-daily-full.md `
  --require-full
```

Date-range replay:

```powershell
python scripts\evaluate_replay_strategies.py `
  --raw D:\Libra\outputs\backtests\kr-objective-2020-2023-opendart-googlenews\libra-replay-results.article-haiku-aggressive-2021-only.jsonl `
  --fixture D:\Libra\outputs\backtests\kr-objective-2020-2023-opendart-googlenews\comparison-fixture.json `
  --start-date 2021-01-04 `
  --end-date 2021-12-30 `
  --out-json D:\Libra\outputs\backtests\kr-objective-2020-2023-opendart-googlenews\comparison-results.article-haiku-aggressive-2021-only.json `
  --out-csv D:\Libra\outputs\backtests\kr-objective-2020-2023-opendart-googlenews\comparison-results.article-haiku-aggressive-2021-only.csv `
  --out-md D:\Libra\outputs\backtests\kr-objective-2020-2023-opendart-googlenews\comparison-results.article-haiku-aggressive-2021-only.md `
  --require-full
```

When `--start-date` or `--end-date` is passed, `--require-full` means the raw output
must cover that selected range, not the entire source fixture.

## Output Fields

The replay summary now includes:

```json
{
  "start_date": "2021-01-04",
  "end_date": "2021-12-30",
  "decision_frequency": "weekly",
  "decision_interval": 1,
  "decision_count": 246,
  "llm_decision_count": 52,
  "scheduled_skip_count": 194
}
```

Each raw row also includes `backtest_schedule.decision_executed` so downstream tooling
can distinguish actual committee judgements from scheduled no-review days.
