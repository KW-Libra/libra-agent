# Ingest Job Contract

`libra-agent` may need fresher market knowledge while a judgment run is in progress. The
agent must not crawl or run `libra-ingest` inside the request handler. It creates an ingest
job and consumes immutable artifacts after the worker succeeds.

## Roles

```text
libra-agent
  - decides whether cached knowledge is fresh enough
  - creates ingest job records
  - waits, resumes, or falls back based on job status
  - reads only promoted artifacts

libra-ingest-worker
  - claims queued jobs
  - runs libra-ingest
  - writes immutable run artifacts
  - promotes successful output to current
  - updates job status
```

## Job Status

```text
QUEUED -> RUNNING -> SUCCEEDED
                  -> FAILED
                  -> CANCELED
```

`current` must only be updated after a successful run. Failed jobs leave the previous
`current` artifacts intact.

## Domains

```text
NEWS        normalized news documents/events
DISCLOSURE  DART disclosure documents/events
REPORT      report PDF documents/events
CONSENSUS   consensus_snapshot.json
FINANCIAL   financial_statement.json
```

V1 may refresh all source fixtures for a ticker even when a narrower domain is requested.
The requested domains still matter for cache TTL, logging, and later worker optimization.

## Artifact Layout

```text
knowledge/
  current/
    manifest.json
    normalized_documents.json
    events.json
    consensus_snapshot.json
    financial_statement.json
    push_candidates.json
  runs/
    <job_id>/
      manifest.json
      normalized_documents.json
      events.json
      consensus_snapshot.json
      financial_statement.json
      push_candidates.json
```

S3 uses the same layout under `s3://<bucket>/knowledge/`.

## Minimal Job Record

```json
{
  "id": "ing_20260514_005930_001",
  "status": "QUEUED",
  "requested_by_thread_id": "thread-123",
  "ticker": "005930",
  "domains": ["NEWS", "DISCLOSURE", "REPORT", "CONSENSUS", "FINANCIAL"],
  "freshness_policy": {
    "max_age_seconds": 900,
    "force_refresh": false,
    "allow_stale_on_failure": true
  },
  "priority": 100,
  "output_s3_prefix": "s3://libra-reports-xxxx/knowledge/runs/ing_20260514_005930_001/",
  "output_local_path": "/opt/libra/knowledge/runs/ing_20260514_005930_001",
  "error_message": null,
  "created_at": "2026-05-14T00:00:00Z",
  "started_at": null,
  "finished_at": null
}
```

## Cache Policy Defaults

```text
NEWS        15 minutes
DISCLOSURE  10 minutes
REPORT      1 day
CONSENSUS   1 day
FINANCIAL   1 day
```

If a matching job is already `QUEUED` or `RUNNING`, the agent should wait for that job
instead of enqueuing a duplicate. If a recent job failed, the agent may use stale `current`
artifacts only when `allow_stale_on_failure` is true.
